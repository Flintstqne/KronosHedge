"""
Kronos Hedge — main run cycle.
Usage:
  python main.py                     # live paper trading
  python main.py --dry-run           # no orders, logs everything
  python main.py --date 2024-06-01   # historical backfill date
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import structlog
import yaml
from dotenv import load_dotenv

load_dotenv()

log = structlog.get_logger()


def load_config(path: str = "config/settings.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


_RISK_STATE_PATH = Path("./data/risk_state.json")


def _load_risk_state() -> dict:
    if _RISK_STATE_PATH.exists():
        try:
            return json.loads(_RISK_STATE_PATH.read_text())
        except Exception:
            pass
    return {"peak_equity": 0.0, "price_peaks": {}, "circuit_active": False}


def _save_risk_state(state: dict) -> None:
    _RISK_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RISK_STATE_PATH.write_text(json.dumps(state, indent=2))


def _apply_trailing_stops(
    weights: dict[str, float],
    position_prices: dict[str, float],
    price_peaks: dict[str, float],
    vol_window: dict,
    trailing_stop: float,
) -> tuple[dict[str, float], dict[str, float]]:
    if trailing_stop <= 0:
        return weights, price_peaks
    result = dict(weights)
    peaks = dict(price_peaks)
    for ticker, price in position_prices.items():
        peak = max(peaks.get(ticker, price), price)
        peaks[ticker] = peak
        if ticker in vol_window:
            daily_std = float(vol_window[ticker]["close"].pct_change().tail(20).std())
            stop = float(np.clip(3.0 * daily_std * np.sqrt(5), trailing_stop, 0.25))
        else:
            stop = trailing_stop
        if (price / peak - 1) < -stop:
            log.info("trailing_stop_triggered", ticker=ticker,
                     drawdown=f"{price/peak-1:.2%}", stop=f"{stop:.1%}")
            result[ticker] = 0.0
            peaks.pop(ticker, None)
    return result, peaks


def _apply_circuit_breaker(
    weights: dict[str, float],
    equity: float,
    state: dict,
    risk_cfg: dict,
) -> tuple[dict[str, float], bool]:
    drawdown_stop      = risk_cfg.get("drawdown_stop", 0.07)
    recovery_threshold = risk_cfg.get("recovery_threshold", 0.02)
    cash_reserve_pct   = risk_cfg.get("cash_reserve_pct", 0.30)
    spy_reserve        = risk_cfg.get("spy_reserve", True)

    state["peak_equity"] = max(state["peak_equity"], equity)
    drawdown = equity / state["peak_equity"] - 1.0

    if not state["circuit_active"] and drawdown < -drawdown_stop:
        state["circuit_active"] = True
        log.info("circuit_breaker_engaged", drawdown=f"{drawdown:.2%}",
                 peak=round(state["peak_equity"], 2))
    elif state["circuit_active"] and drawdown > -recovery_threshold:
        state["circuit_active"] = False
        log.info("circuit_breaker_released", drawdown=f"{drawdown:.2%}")

    if not state["circuit_active"]:
        return weights, False

    scale = 1.0 - cash_reserve_pct
    result = {t: w * scale for t, w in weights.items()}
    if spy_reserve:
        result.pop("SPY", None)
        result["SPY"] = cash_reserve_pct
    return result, True


def run_cycle(cfg: dict, target_date: date | None = None, dry_run: bool = False, mode: str = "daily") -> None:
    from kronos_bridge import KronosSignalBridge, PredictionCache
    from qlib_pipeline.data import fetch_ohlcv_universe
    from qlib_pipeline.alpha import KronosAlphaFactor
    from agents import run_agent_pipeline
    from reconciliation import SignalReconciler
    from execution.alpaca import AlpacaAdapter
    from execution.executor import PortfolioExecutor
    from monitoring.logger import AuditLogger

    run_id = datetime.now(timezone.utc).isoformat()
    log.info("cycle_start", run_id=run_id, dry_run=dry_run, mode=mode)

    tickers: list[str] = cfg["universe"]["tickers"]
    kronos_cfg = cfg["kronos"]
    agent_cfg = cfg["agents"]
    rec_cfg = cfg["reconciliation"]
    exec_cfg = cfg["execution"]

    alpha_cfg = cfg.get("alpha", {})

    # ── 1. Fetch OHLCV ───────────────────────────────────────────────────────
    # Fetch 420 cal days so 12M-1M momentum factor has enough history (needs 273 trading days).
    momentum_cal_days = 420
    fetch_days = max(kronos_cfg["lookback_days"], momentum_cal_days)
    log.info("fetching_ohlcv", tickers=tickers)
    universe = fetch_ohlcv_universe(
        tickers=tickers,
        lookback_days=fetch_days,
        end_date=target_date,
    )
    log.info("ohlcv_fetched", count=len(universe))

    # ── 1b. Intraday price override (intraday mode only) ─────────────────────
    if mode == "intraday":
        from data.intraday import market_is_open, is_past_signal_time, fetch_intraday_prices
        exec_cfg_local = cfg.get("execution", {})
        signal_time = exec_cfg_local.get("intraday_signal_time", "10:00")
        if not market_is_open():
            log.info("market_closed_skipping")
            return
        if not is_past_signal_time(signal_time):
            log.info("before_signal_time_skipping", signal_time=signal_time)
            return
        intraday_px = fetch_intraday_prices(tickers, as_of_time=signal_time)
        log.info("intraday_prices_fetched", count=len(intraday_px))
        # Splice live prices into the tail of each ticker's OHLCV frame
        for ticker, px in intraday_px.items():
            if ticker in universe and px > 0:
                universe[ticker] = universe[ticker].copy()
                universe[ticker].loc[universe[ticker].index[-1], "close"] = px

    # ── 2. Kronos inference ──────────────────────────────────────────────────
    cache = PredictionCache(cache_dir=os.getenv("KRONOS_CACHE_DIR", "./data/kronos_cache"))
    _ms = kronos_cfg.get("model_source", "chronos")
    print(f"[kronos] model_size={kronos_cfg['model_size']} model_source={_ms}", flush=True)
    bridge = KronosSignalBridge(
        model_size=kronos_cfg["model_size"],
        horizon=kronos_cfg["horizon"],
        device=kronos_cfg["device"],
        model_source=_ms,
    )

    last_date = (target_date or date.today()).isoformat()
    kronos_signals = {}
    for ticker, df in universe.items():
        cached = cache.get(ticker, last_date, kronos_cfg["horizon"])
        if cached:
            kronos_signals[ticker] = cached
        else:
            sig = bridge.generate(ticker, df)
            cache.set(sig, last_date)
            kronos_signals[ticker] = sig

    log.info("kronos_done", signals=len(kronos_signals))

    # ── 3a. News context + market signals ────────────────────────────────────
    from data.news import build_news_context, apply_macro_risk, fetch_vix, fetch_pead_signals
    log.info("fetching_news")
    news_context = build_news_context(tickers)
    vix = fetch_vix()
    pead_signals = fetch_pead_signals(tickers)
    log.info("news_fetched",
             macro_events=len(news_context["macro_events"]),
             earnings=len(news_context["earnings"]),
             vix=round(vix, 1),
             pead_beaters=list(pead_signals.keys()))

    # ── 3b. Qlib alpha + Agent pipeline ──────────────────────────────────────
    # Build 60-day vol window and 273-day momentum window from the long fetch.
    vol_window: dict = {t: df.tail(kronos_cfg["lookback_days"]) for t, df in universe.items()}
    mom_window: dict = {t: df.tail(273) for t, df in universe.items()}

    alpha_factor = KronosAlphaFactor(
        momentum_blend=alpha_cfg.get("momentum_blend", 0.80),
        top_n=alpha_cfg.get("top_n", 10),
        short_n=alpha_cfg.get("short_n", 3),
        pead_boost=alpha_cfg.get("pead_boost", 0.20),
    )
    qlib_weights = alpha_factor.compute_weights(
        kronos_signals,
        max_position=exec_cfg["max_position_pct"],
        prices_history=vol_window,
        momentum_history=mom_window,
        pead_signals=pead_signals,
    )
    log.info("qlib_weights_computed", n=len(qlib_weights))

    log.info("running_agents")
    agent_state = run_agent_pipeline(
        kronos_signals=kronos_signals,
        llm_provider=agent_cfg["llm_provider"],
        llm_model=agent_cfg["llm_model"],
        enabled_agents=agent_cfg.get("enabled", []),
        max_position_pct=exec_cfg["max_position_pct"],
        news_context=news_context,
    )
    log.info("agents_done")

    # ── 4. Reconcile + macro risk filter ─────────────────────────────────────
    sf_cfg = cfg.get("sector_filter", {})
    reconciler = SignalReconciler(
        qlib_weight=rec_cfg["qlib_weight"],
        agent_weight=rec_cfg["agent_weight"],
        sector_filter=sf_cfg.get("enabled", True),
        max_sector_pct=sf_cfg.get("max_sector_pct", 0.40),
        max_per_sector=sf_cfg.get("max_per_sector", 4),
    )
    raw_weights = reconciler.reconcile(qlib_weights, agent_state["portfolio_decisions"])
    post_macro = apply_macro_risk(raw_weights, news_context)
    if post_macro != raw_weights:
        log.info("macro_risk_applied", reason="FOMC or earnings event within risk window")

    # ── 5. Risk management + execute ─────────────────────────────────────────
    risk_cfg = cfg.get("risk", {})

    # VIX regime filter: scale all weights down linearly above vix_threshold
    vix_threshold = risk_cfg.get("vix_threshold", 25)
    vix_scale = min(1.0, vix_threshold / vix) if vix > vix_threshold else 1.0
    if vix_scale < 1.0:
        final_weights = {t: w * vix_scale for t, w in post_macro.items()}
        log.info("vix_filter_applied", vix=round(vix, 1), scale=round(vix_scale, 2))
    else:
        final_weights = post_macro

    # Gate shorts: only short in elevated-VIX regimes
    if vix <= vix_threshold:
        final_weights = {t: w for t, w in final_weights.items() if w >= 0}

    # Macro regime filter
    from data.macro_regime import get_regime_series, apply_regime_filter
    import pandas as _pd
    _regime_series = get_regime_series()
    _today_ts = _pd.Timestamp(target_date or date.today())
    regime = str(_regime_series.asof(_today_ts)) if len(_regime_series) and _today_ts >= _regime_series.index[0] else "risk_on"
    final_weights = apply_regime_filter(final_weights, regime, spy_in_prices="SPY" in tickers)
    log.info("regime", regime=regime)

    log.info("reconciled", weights=final_weights)
    broker   = AlpacaAdapter(paper=exec_cfg.get("paper", True))
    equity   = broker.get_equity()
    positions = broker.get_positions()

    # Current price per held position (market_value / qty)
    position_prices = {
        t: p.market_value / p.qty for t, p in positions.items() if p.qty
    }

    risk_state = _load_risk_state()
    if risk_state["peak_equity"] == 0.0:
        risk_state["peak_equity"] = equity  # first run bootstrap

    final_weights, risk_state["price_peaks"] = _apply_trailing_stops(
        final_weights, position_prices, risk_state["price_peaks"],
        vol_window, risk_cfg.get("trailing_stop", 0.07),
    )
    final_weights, cb_active = _apply_circuit_breaker(
        final_weights, equity, risk_state, risk_cfg,
    )
    log.info("risk_applied", circuit_breaker=cb_active, equity=round(equity, 2))

    # Prune peaks for positions fully exited
    held = set(positions)
    risk_state["price_peaks"] = {
        t: v for t, v in risk_state["price_peaks"].items() if t in held
    }

    executor = PortfolioExecutor(
        broker=broker,
        min_order_usd=exec_cfg["min_order_usd"],
        dry_run=dry_run,
    )
    close_prices = {t: float(df["close"].iloc[-1]) for t, df in vol_window.items() if len(df)}
    orders = executor.execute(final_weights, prices=close_prices)
    equity = broker.get_equity()
    _save_risk_state(risk_state)
    log.info("orders_placed", count=len(orders), equity=equity)

    # ── 6. Audit log ─────────────────────────────────────────────────────────
    audit = AuditLogger(log_dir=os.getenv("AUDIT_LOG_DIR", "./logs/audit"))
    log_path = audit.write(
        run_id=run_id,
        kronos_signals=kronos_signals,
        agent_state=agent_state,
        qlib_weights=qlib_weights,
        final_weights=final_weights,
        orders=orders,
        portfolio_equity=equity,
        news_context=news_context,
        signal_attribution=getattr(alpha_factor, "last_breakdown", {}),
    )
    log.info("audit_written", path=str(log_path))

    # ── 7. Alerts ────────────────────────────────────────────────────────────
    from monitoring.alerts import run_all_checks
    from monitoring.metrics import PerformanceMetrics
    all_records = audit.load_all()
    m = PerformanceMetrics(all_records)
    acc_df = m.kronos_accuracy()
    valid  = acc_df.dropna(subset=["correct"])
    kronos_acc = float(valid["correct"].mean()) if len(valid) > 0 else None
    ec = m.equity_curve()
    peak_equity = float(ec["equity"].max()) if len(ec) > 0 else equity
    run_all_checks(
        current_equity=equity,
        peak_equity=peak_equity,
        orders=orders,
        kronos_accuracy=kronos_acc,
        cfg=cfg,
    )

    log.info("cycle_complete", run_id=run_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos Hedge run cycle")
    parser.add_argument("--config", default="config/settings.yaml")
    parser.add_argument("--dry-run", action="store_true", help="No real orders placed")
    parser.add_argument("--date", help="Target date YYYY-MM-DD (default: today)")
    parser.add_argument("--mode", choices=["daily", "intraday"], default="daily",
                        help="daily=EOD signals, intraday=10am live prices")
    args = parser.parse_args()

    cfg = load_config(args.config)
    target_date = date.fromisoformat(args.date) if args.date else None
    run_cycle(cfg, target_date=target_date, dry_run=args.dry_run, mode=args.mode)


if __name__ == "__main__":
    main()
