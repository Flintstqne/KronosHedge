"""
Historical backtester.
Walks day-by-day over a date range, running the full Kronos → agent → reconcile cycle
on data available up to each date. Writes one audit log per day.
No real orders — uses a virtual portfolio to track P&L.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd
import structlog

log = structlog.get_logger()


@dataclass
class VirtualPortfolio:
    initial_cash: float = 100_000.0
    drawdown_stop: float = 0.07       # 3E: raised from 5% — 5% fires in normal noise
    recovery_threshold: float = 0.02
    cash_reserve_pct: float = 0.30
    min_trade_pct: float = 0.005
    trailing_stop: float = 0.07       # floor for vol-scaled stops (0 = disabled)
    spy_reserve: bool = True          # 3F: park circuit-breaker cash in SPY instead of idle
    stop_cooldown_days: int = 5       # rebalance days to block re-entry after a stop fires
    slippage_bps: float = 5.0        # one-way fill slippage in basis points

    cash: float = field(init=False)
    positions: dict[str, float] = field(default_factory=dict)   # negative = short
    equity_history: list[tuple[date, float]] = field(default_factory=list)
    _peak_equity: float = field(init=False)
    _circuit_active: bool = field(init=False, default=False)
    _price_peaks: dict[str, float] = field(default_factory=dict)    # long high-water marks
    _price_troughs: dict[str, float] = field(default_factory=dict)  # short low-water marks
    _stop_cooldowns: dict[str, int] = field(default_factory=dict)   # ticker -> rebalance days remaining

    def __post_init__(self):
        self.cash = self.initial_cash
        self._peak_equity = self.initial_cash

    def _vol_stop(self, ticker: str, prices_history: dict | None) -> float:
        """1A: vol-scaled stop = 3σ over 5-day horizon. Floor=trailing_stop, cap=25%."""
        if self.trailing_stop <= 0:
            return 0.0
        if prices_history and ticker in prices_history:
            daily_std = float(
                prices_history[ticker]["close"].pct_change().tail(20).std()
            )
            scaled = 3.0 * daily_std * np.sqrt(5)
            return float(np.clip(scaled, self.trailing_stop, 0.25))
        return self.trailing_stop

    def _apply_trailing_stops(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        prices_history: dict | None = None,
    ) -> dict[str, float]:
        if self.trailing_stop <= 0:
            return target_weights
        result = dict(target_weights)
        stop_cache: dict[str, float] = {}

        for ticker, shares in list(self.positions.items()):
            if shares == 0 or ticker not in prices:
                self._price_peaks.pop(ticker, None)
                self._price_troughs.pop(ticker, None)
                continue
            px = prices[ticker]
            stop = stop_cache.setdefault(ticker, self._vol_stop(ticker, prices_history))

            if shares > 0:  # long — stop if price drops below peak by stop%
                peak = max(self._price_peaks.get(ticker, px), px)
                self._price_peaks[ticker] = peak
                if stop > 0 and (px / peak - 1) < -stop:
                    log.info("trailing_stop_long", ticker=ticker,
                             drawdown=f"{px/peak-1:.2%}", stop=f"{stop:.1%}")
                    result[ticker] = 0.0
                    self._price_peaks.pop(ticker, None)
                    self._stop_cooldowns[ticker] = self.stop_cooldown_days
            else:           # short — stop if price rises above trough by stop%
                trough = min(self._price_troughs.get(ticker, px), px)
                self._price_troughs[ticker] = trough
                if stop > 0 and (px / trough - 1) > stop:
                    log.info("trailing_stop_short", ticker=ticker,
                             rise=f"{px/trough-1:.2%}", stop=f"{stop:.1%}")
                    result[ticker] = 0.0
                    self._price_troughs.pop(ticker, None)
                    self._stop_cooldowns[ticker] = self.stop_cooldown_days

        return result

    def _apply_circuit_breaker(
        self, target_weights: dict[str, float], equity: float
    ) -> dict[str, float]:
        self._peak_equity = max(self._peak_equity, equity)
        drawdown = (equity / self._peak_equity) - 1.0
        if not self._circuit_active and drawdown < -self.drawdown_stop:
            self._circuit_active = True
            log.info("circuit_breaker_engaged",
                     drawdown=f"{drawdown:.2%}", peak=round(self._peak_equity, 2))
        if self._circuit_active and drawdown > -self.recovery_threshold:
            self._circuit_active = False
            log.info("circuit_breaker_released", drawdown=f"{drawdown:.2%}")
        if self._circuit_active:
            scale = 1.0 - self.cash_reserve_pct
            return {t: w * scale for t, w in target_weights.items()}
        return target_weights

    def rebalance(
        self,
        target_weights: dict[str, float],
        prices: dict[str, float],
        prices_history: dict | None = None,
    ) -> list[dict]:
        equity = self.equity(prices)
        stopped = self._apply_trailing_stops(target_weights, prices, prices_history)
        adjusted = self._apply_circuit_breaker(stopped, equity)

        # 3F: redirect idle cash reserve into SPY when circuit breaker is active
        if self._circuit_active and self.spy_reserve and "SPY" in prices:
            adjusted.pop("SPY", None)   # remove any signal-driven SPY weight
            adjusted["SPY"] = self.cash_reserve_pct

        # Tick down cooldowns; blocked tickers can't be re-entered this rebalance
        self._stop_cooldowns = {t: d - 1 for t, d in self._stop_cooldowns.items() if d > 1}
        adjusted = {t: w for t, w in adjusted.items() if t not in self._stop_cooldowns}

        min_notional = max(100.0, equity * self.min_trade_pct)
        orders = []

        # Compute deltas for all tickers with a target or an existing position.
        # Negative delta = sell/go-short; positive delta = buy/cover-short.
        all_tickers = set(adjusted) | set(self.positions)
        sells, buys = [], []
        for ticker in all_tickers:
            if ticker not in prices:
                continue
            current_val = self.positions.get(ticker, 0.0) * prices[ticker]
            target_val  = equity * adjusted.get(ticker, 0.0)
            delta       = target_val - current_val
            if delta < -min_notional:
                sells.append((ticker, delta))
            elif delta > min_notional:
                buys.append((ticker, delta))

        slip = self.slippage_bps / 10_000

        # Sells first: reduces longs OR opens/increases shorts → always frees or receives cash
        for ticker, delta in sells:
            px = prices[ticker] * (1 - slip)              # sell into bid
            share_delta = delta / prices[ticker]          # shares based on mid
            self.positions[ticker] = self.positions.get(ticker, 0.0) + share_delta
            fill_notional = abs(share_delta) * px
            self.cash += fill_notional                    # receive slippage-reduced proceeds
            if self.positions[ticker] < 0 and ticker not in self._price_troughs:
                self._price_troughs[ticker] = px          # init short trough
            orders.append({"ticker": ticker, "side": "sell",
                           "notional_usd": abs(delta), "status": "filled"})

        # Buys: increases longs OR covers shorts → costs cash
        for ticker, delta in buys:
            px = prices[ticker] * (1 + slip)              # buy at ask
            shares = delta / prices[ticker]               # shares based on mid
            fill_cost = shares * px
            if self.cash < fill_cost:
                continue
            self.positions[ticker] = self.positions.get(ticker, 0.0) + shares
            self.cash -= fill_cost
            if self.positions[ticker] > 0 and ticker not in self._price_peaks:
                self._price_peaks[ticker] = px            # init long peak
            orders.append({"ticker": ticker, "side": "buy",
                           "notional_usd": delta, "status": "filled"})

        # Prune effectively-zero positions
        self.positions = {t: s for t, s in self.positions.items() if abs(s) > 1e-6}

        return orders

    def equity(self, prices: dict[str, float]) -> float:
        pos_val = sum(shares * prices.get(t, 0) for t, shares in self.positions.items())
        return self.cash + pos_val

    def record(self, d: date, prices: dict[str, float]) -> None:
        self.equity_history.append((d, self.equity(prices)))


def trading_days(start: date, end: date) -> list[date]:
    return [d.date() for d in pd.bdate_range(start, end)]


class Backtester:
    def __init__(
        self,
        tickers: list[str],
        start_date: date,
        end_date: date,
        lookback_days: int = 60,
        kronos_model_size: str = "small",
        kronos_horizon: int = 5,
        kronos_device: str = "cpu",
        kronos_model_source: str = "chronos",  # "chronos" | "sy-kronos"
        qlib_weight: float = 0.60,
        agent_weight: float = 0.40,
        llm_provider: str = "anthropic",
        llm_model: str = "claude-sonnet-4-6",
        initial_cash: float = 100_000.0,
        run_agents: bool = False,
        audit_logger=None,
        # Risk management
        drawdown_stop: float = 0.07,    # 3E: raised from 5%
        recovery_threshold: float = 0.02,
        cash_reserve_pct: float = 0.30,
        min_trade_pct: float = 0.005,
        trailing_stop: float = 0.07,
        spy_reserve: bool = True,       # 3F: park circuit-breaker cash in SPY
        stop_cooldown_days: int = 5,    # rebalance days to block re-entry after stop fires
        slippage_bps: float = 5.0,
        # Alpha settings
        momentum_blend: float = 0.80,
        top_n: int = 10,
        short_n: int = 3,
        pead_boost: float = 0.20,
        vix_threshold: float = 25.0,
        rebalance_frequency: str = "daily",
    ):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.lookback_days = lookback_days
        self.kronos_model_size = kronos_model_size
        self.kronos_horizon = kronos_horizon
        self.kronos_device = kronos_device
        self.kronos_model_source = kronos_model_source
        self.qlib_weight = qlib_weight
        self.agent_weight = agent_weight
        self.llm_provider = llm_provider
        self.llm_model = llm_model
        self.run_agents = run_agents
        self.audit_logger = audit_logger
        self.momentum_blend = momentum_blend
        self.top_n = top_n
        self.short_n = short_n
        self.pead_boost = pead_boost
        self.vix_threshold = vix_threshold
        self.rebalance_frequency = rebalance_frequency
        self.portfolio = VirtualPortfolio(
            initial_cash=initial_cash,
            drawdown_stop=drawdown_stop,
            recovery_threshold=recovery_threshold,
            cash_reserve_pct=cash_reserve_pct,
            min_trade_pct=min_trade_pct,
            trailing_stop=trailing_stop,
            spy_reserve=spy_reserve,
            stop_cooldown_days=stop_cooldown_days,
            slippage_bps=slippage_bps,
        )

    def run(self) -> pd.DataFrame:
        from kronos_bridge import KronosSignalBridge, PredictionCache
        from qlib_pipeline.data import fetch_ohlcv_universe
        from qlib_pipeline.alpha import KronosAlphaFactor
        from reconciliation.merger import SignalReconciler

        bridge = KronosSignalBridge(
            model_size=self.kronos_model_size,
            horizon=self.kronos_horizon,
            device=self.kronos_device,
            model_source=self.kronos_model_source,
        )
        cache = PredictionCache()
        alpha = KronosAlphaFactor(
            momentum_blend=self.momentum_blend,
            top_n=self.top_n,
            short_n=self.short_n,
            pead_boost=self.pead_boost,
        )
        reconciler = SignalReconciler(
            self.qlib_weight,
            self.agent_weight,
            sector_filter=True,
            max_sector_pct=0.40,
            max_per_sector=4,
        )

        eval_days = trading_days(self.start_date, self.end_date)

        log.info("backtest_start", start=str(self.start_date), end=str(self.end_date),
                 eval_days=len(eval_days), run_agents=self.run_agents)

        # Fetch enough history for both 60-day vol and 12M-1M momentum (273 trading days ≈ 400 cal days).
        # SPY is fetched alongside the universe for the circuit-breaker SPY reserve (3F).
        momentum_cal_days = 420  # ~295 trading days — covers 252 + 21 + buffer
        total_fetch_days = max(self.lookback_days, momentum_cal_days) + \
                           (self.end_date - self.start_date).days + 30

        fetch_tickers = list(self.tickers)
        if self.portfolio.spy_reserve and "SPY" not in fetch_tickers:
            fetch_tickers = fetch_tickers + ["SPY"]

        all_data = fetch_ohlcv_universe(
            tickers=fetch_tickers,
            lookback_days=total_fetch_days,
            end_date=self.end_date,
        )

        # VIX history for VIX filter — fetch once, look up by date in _run_day
        import yfinance as yf
        _vix_raw = yf.download("^VIX", period="max", progress=False, auto_adjust=True)
        vix_series = _vix_raw["Close"].squeeze().dropna()
        vix_series.index = pd.to_datetime(vix_series.index).normalize()

        # Macro regime series — risk_on / transition / risk_off per trading day
        from data.macro_regime import get_regime_series
        regime_series = get_regime_series()

        # Historical EPS beats for PEAD — yfinance gives ~5y of quarterly history
        historical_earnings: dict[str, pd.DataFrame] = {}
        for ticker in self.tickers:
            try:
                df = yf.Ticker(ticker).get_earnings_dates(limit=20)
                if df is None or df.empty:
                    continue
                df = df.dropna(subset=["EPS Estimate", "Reported EPS"])
                df.index = pd.to_datetime(df.index).tz_localize(None).normalize()
                historical_earnings[ticker] = df.sort_index(ascending=False)
            except Exception:
                pass

        for d in eval_days:
            try:
                self._run_day(d, all_data, bridge, cache, alpha, reconciler,
                              vix_series=vix_series, historical_earnings=historical_earnings,
                              regime_series=regime_series)
            except Exception as exc:
                log.warning("day_failed", date=str(d), error=str(exc))

        log.info("backtest_done", days_run=len(self.portfolio.equity_history))
        return self.results_df()

    def _is_rebalance_day(self, d: date) -> bool:
        if self.rebalance_frequency == "weekly":
            return d.weekday() == 4  # Friday
        return True  # daily (default)

    def _run_day(
        self,
        d: date,
        all_data: dict[str, pd.DataFrame],
        bridge,
        cache,
        alpha,
        reconciler,
        vix_series: pd.Series | None = None,
        historical_earnings: dict[str, pd.DataFrame] | None = None,
        regime_series: pd.Series | None = None,
    ) -> None:
        # Build universe (60-day vol window) and prices dict.
        # SPY is added to prices only — excluded from signal generation, used for SPY reserve.
        universe: dict[str, pd.DataFrame] = {}
        momentum_history: dict[str, pd.DataFrame] = {}  # 273-day for 12M-1M momentum
        prices: dict[str, float] = {}

        for ticker, df in all_data.items():
            mask = df.index.date <= d  # type: ignore[attr-defined]
            full_slice = df[mask]

            # SPY: prices only (circuit-breaker reserve), not signal universe
            if ticker == "SPY" and self.portfolio.spy_reserve:
                if len(full_slice) > 0:
                    prices["SPY"] = float(full_slice["close"].iloc[-1])
                continue

            sliced = full_slice.tail(self.lookback_days)
            if len(sliced) >= 10:
                universe[ticker] = sliced
                prices[ticker] = float(sliced["close"].iloc[-1])

            # 12M-1M momentum needs 273 trading days; use whatever we have (≥22)
            mom_slice = full_slice.tail(273)
            if len(mom_slice) >= 22:
                momentum_history[ticker] = mom_slice

        if not universe:
            return

        # On non-rebalance days: enforce trailing stops, record equity, skip trading
        if not self._is_rebalance_day(d):
            if self.portfolio.trailing_stop > 0 and self.portfolio.positions:
                stopped = self.portfolio._apply_trailing_stops(
                    {t: 0.0 for t in universe}, prices, universe
                )
                for ticker, w in stopped.items():
                    shares = self.portfolio.positions.get(ticker, 0)
                    if w == 0.0 and shares != 0 and ticker in prices:
                        self.portfolio.cash += shares * prices[ticker]  # works for shorts too
                        self.portfolio.positions[ticker] = 0.0
            self.portfolio.record(d, prices)
            return

        # Kronos inference (with cache)
        kronos_signals = {}
        for ticker, df in universe.items():
            cached = cache.get(ticker, str(d), self.kronos_horizon)
            if cached:
                kronos_signals[ticker] = cached
            else:
                sig = bridge.generate(ticker, df)
                cache.set(sig, str(d))
                kronos_signals[ticker] = sig

        # PEAD signals: check which tickers beat EPS in the last 60 days as of d
        pead_signals: dict[str, float] = {}
        if historical_earnings:
            cutoff = d - timedelta(days=60)
            for ticker, df in historical_earnings.items():
                past = df[df.index.date <= d]
                recent = past[past.index.date >= cutoff]
                if recent.empty:
                    continue
                row = recent.iloc[0]
                est, rep = float(row["EPS Estimate"]), float(row["Reported EPS"])
                if abs(est) >= 0.01:
                    surprise = (rep - est) / abs(est)
                    if surprise > 0:
                        pead_signals[ticker] = min(surprise, 1.0)

        # Alpha weights — vol sizing from 60-day universe, momentum from 273-day history
        qlib_weights = alpha.compute_weights(
            kronos_signals,
            prices_history=universe,
            momentum_history=momentum_history,
            pead_signals=pead_signals or None,
        )

        # Agents (optional)
        agent_decisions: dict[str, dict] = {}
        agent_state: dict[str, Any] = {}
        if self.run_agents:
            from agents import run_agent_pipeline
            agent_state = run_agent_pipeline(
                kronos_signals=kronos_signals,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
            )
            agent_decisions = agent_state.get("portfolio_decisions", {})
        else:
            for ticker, sig in kronos_signals.items():
                agent_decisions[ticker] = {
                    "action": sig.directional_signal,
                    "quantity_pct": qlib_weights.get(ticker, 0.0),
                }

        final_weights = reconciler.reconcile(qlib_weights, agent_decisions)

        # VIX regime filter
        vix_val = 20.0
        if vix_series is not None:
            d_ts = pd.Timestamp(d)
            vix_val = float(vix_series.asof(d_ts)) if d_ts >= vix_series.index[0] else 20.0
            if vix_val > self.vix_threshold:
                vix_scale = self.vix_threshold / vix_val
                final_weights = {t: w * vix_scale for t, w in final_weights.items()}

        # Gate shorts: only allow negative weights in elevated-VIX regimes
        if vix_val <= self.vix_threshold:
            final_weights = {t: w for t, w in final_weights.items() if w >= 0}

        # Macro regime filter
        if regime_series is not None and len(regime_series):
            d_ts = pd.Timestamp(d)
            regime = str(regime_series.asof(d_ts)) if d_ts >= regime_series.index[0] else "risk_on"
            from data.macro_regime import apply_regime_filter
            final_weights = apply_regime_filter(final_weights, regime, spy_in_prices="SPY" in prices)
            if regime != "risk_on":
                log.info("regime_filter", regime=regime, date=str(d))

        # Pre-earnings halve: zero-cost, uses historical earnings dates
        if historical_earnings:
            for ticker, df in historical_earnings.items():
                future = df[df.index.date > d].sort_index(ascending=True)
                if future.empty:
                    continue
                days_away = (future.index[0].date() - d).days
                if 0 <= days_away <= 2 and ticker in final_weights:
                    final_weights[ticker] *= 0.50

        orders = self.portfolio.rebalance(final_weights, prices, prices_history=universe)
        self.portfolio.record(d, prices)

        equity = self.portfolio.equity(prices)
        log.debug("day_done", date=str(d), equity=round(equity, 2), orders=len(orders))

        if self.audit_logger:
            from execution.base import Order
            order_objs = [
                Order(
                    ticker=o["ticker"], side=o["side"],
                    notional_usd=o["notional_usd"], status=o["status"]
                )
                for o in orders
            ]
            self.audit_logger.write(
                run_id=f"backtest_{d.isoformat()}",
                kronos_signals=kronos_signals,
                agent_state=agent_state,
                qlib_weights=qlib_weights,
                final_weights=final_weights,
                orders=order_objs,
                portfolio_equity=equity,
                signal_attribution=getattr(alpha, "last_breakdown", {}),
            )

    def results_df(self) -> pd.DataFrame:
        if not self.portfolio.equity_history:
            return pd.DataFrame(columns=["date", "equity", "return", "cumulative_return"])
        df = pd.DataFrame(self.portfolio.equity_history, columns=["date", "equity"])
        df["return"] = df["equity"].pct_change()
        df["cumulative_return"] = (df["equity"] / self.portfolio.initial_cash) - 1
        return df

    def summary(self) -> dict[str, float]:
        df = self.results_df()
        if len(df) < 2:
            return {}
        returns = df["return"].dropna()
        equity = df["equity"].values
        peak = np.maximum.accumulate(equity)
        drawdowns = (equity - peak) / peak
        sharpe = (returns.mean() / (returns.std() + 1e-10)) * np.sqrt(252)
        return {
            "initial_cash": self.portfolio.initial_cash,
            "final_equity": float(equity[-1]),
            "total_return": float(df["cumulative_return"].iloc[-1]),
            "sharpe_ratio": float(sharpe),
            "max_drawdown": float(drawdowns.min()),
            "win_rate": float((returns > 0).sum() / len(returns)),
            "trading_days": len(df),
        }
