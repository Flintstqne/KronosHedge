"""
Kronos Hedge Dashboard
  Data:        Today's picks, Kronos forecasts, agent consensus, portfolio
  Performance: Report card, Kronos vs SPY, prediction tracker, accuracy
  Backtest:    Historical simulation vs SPY
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st
# Bridge Streamlit Cloud secrets into os.environ so dotenv-based code works unchanged
try:
    for _k, _v in st.secrets.items():
        os.environ.setdefault(_k, str(_v))
except Exception:
    pass  # no secrets configured locally — env vars expected via .env or system

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf
from dotenv import load_dotenv

# Load .env from the project root so API keys are available when Streamlit
# launches the dashboard directly (outside of main.py's load_dotenv call).
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from monitoring.logger import AuditLogger
from monitoring.metrics import PerformanceMetrics

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Kronos Hedge",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.grade-pill {
    display:inline-block; padding:4px 14px; border-radius:20px;
    font-weight:700; font-size:18px; color:white; margin-left:8px;
}
.kpi-label { font-size:12px; color:#9e9e9e; margin-bottom:2px; }
.kpi-value { font-size:22px; font-weight:700; }
.kpi-sub   { font-size:12px; color:#9e9e9e; margin-top:2px; }
.pick-card {
    background:#1e1e1e; border-radius:10px; padding:16px 20px;
    border-left:4px solid; margin-bottom:8px;
}
</style>
""", unsafe_allow_html=True)

# ─── Live data helpers ────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def _fetch_live_prices(tickers: tuple[str, ...]) -> dict[str, float]:
    """Latest close price for each ticker via yfinance (cached 60 s)."""
    prices: dict[str, float] = {}
    for ticker in tickers:
        try:
            raw = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if raw.empty:
                continue
            col = raw["Close"]
            if isinstance(col, pd.DataFrame):
                col = col.iloc[:, 0]
            prices[ticker] = float(col.dropna().iloc[-1])
        except Exception:
            pass
    return prices


@st.cache_data(ttl=30)
def _fetch_alpaca_live() -> dict | None:
    """Returns live account data from Alpaca or None if not configured."""
    if not os.getenv("ALPACA_API_KEY"):
        return None
    try:
        from execution.alpaca import AlpacaAdapter
        paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        adapter = AlpacaAdapter(paper=paper)
        equity    = adapter.get_equity()
        positions = adapter.get_positions()
        history   = adapter.get_portfolio_history("1M")
        return {
            "equity":    equity,
            "mode":      "Paper" if paper else "Live",
            "positions": positions,
            "history":   history,
        }
    except Exception as exc:
        return {"error": str(exc)}


# ─── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.title("Kronos Hedge")
page = st.sidebar.radio("Page", ["Data", "Performance", "Backtest", "News", "Trade Log", "Regime & Risk"])
log_dir = st.sidebar.text_input("Audit log directory", value="./logs/audit")
st.sidebar.button("Refresh")

# Live account panel
_alpaca = _fetch_alpaca_live()
if _alpaca and "error" not in _alpaca:
    st.sidebar.divider()
    mode_color = "#ffa726" if _alpaca["mode"] == "Paper" else "#ef5350"
    st.sidebar.markdown(
        f'<span style="background:{mode_color};color:white;padding:2px 8px;'
        f'border-radius:4px;font-size:12px;font-weight:700">{_alpaca["mode"]}</span>',
        unsafe_allow_html=True,
    )
    st.sidebar.metric("Account equity", f"${_alpaca['equity']:,.2f}")
    n_pos = len(_alpaca["positions"])
    st.sidebar.caption(f"{n_pos} open position{'s' if n_pos != 1 else ''}")
elif _alpaca and "error" in _alpaca:
    st.sidebar.warning(f"Alpaca: {_alpaca['error'][:80]}")
else:
    st.sidebar.caption("Set ALPACA_API_KEY to see live account.")

st.sidebar.divider()
if st.sidebar.button("Clear Demo Data", help="Deletes seeded demo logs so real data takes over"):
    import glob
    demo_files = glob.glob(f"{log_dir}/run_demo_*.json")
    for f in demo_files:
        Path(f).unlink(missing_ok=True)
    st.sidebar.success(f"Deleted {len(demo_files)} demo file(s).")
    st.rerun()

logger = AuditLogger(log_dir=log_dir)
records = logger.load_all()

if not records:
    if page not in ("Backtest", "News", "Trade Log", "Regime & Risk"):
        st.warning("No audit records found. Run `python scripts/seed_demo_data.py` to generate demo data.")
        st.stop()
    latest = {}
    metrics = None
else:
    latest = records[-1]
    metrics = PerformanceMetrics(records)

# ─── Shared helpers ───────────────────────────────────────────────────────────

C_BULL   = "#26a69a"
C_BEAR   = "#ef5350"
C_NEUT   = "#78909c"
C_GOLD   = "#ffa726"
C_BLUE   = "#42a5f5"

GRADE_COLORS = {"A+": "#00897b", "A": "#26a69a", "B": "#42a5f5",
                "C": "#ffa726", "D": "#ef9a9a", "F": "#ef5350"}


def grade_chip(grade: str) -> str:
    color = GRADE_COLORS.get(grade, "#78909c")
    return (f'<span class="grade-pill" style="background:{color}">{grade}</span>')


def sharpe_grade(s: float) -> tuple[str, str]:
    if s >= 2.0:   return "A+", "Exceptional risk-adjusted return"
    if s >= 1.5:   return "A",  "Strong risk-adjusted return"
    if s >= 1.0:   return "B",  "Good risk-adjusted return"
    if s >= 0.5:   return "C",  "Moderate risk-adjusted return"
    if s >= 0.0:   return "D",  "Weak risk-adjusted return"
    return "F", "Negative risk-adjusted return"


def accuracy_grade(a: float) -> tuple[str, str]:
    if a >= 0.75:  return "A+", "Highly accurate direction calls"
    if a >= 0.65:  return "A",  "Accurate direction calls"
    if a >= 0.55:  return "B",  "Mostly correct direction calls"
    if a >= 0.50:  return "C",  "Slightly better than random"
    if a >= 0.45:  return "D",  "Near coin-flip accuracy"
    return "F", "Below random accuracy - review model"


def alpha_grade(alpha: float) -> tuple[str, str]:
    if alpha >= 0.10:  return "A+", f"Outperforming SPY by {alpha:+.1%}"
    if alpha >= 0.05:  return "A",  f"Outperforming SPY by {alpha:+.1%}"
    if alpha >= 0.00:  return "B",  f"Slightly ahead of SPY ({alpha:+.1%})"
    if alpha >= -0.05: return "C",  f"Slightly behind SPY ({alpha:+.1%})"
    if alpha >= -0.10: return "D",  f"Underperforming SPY ({alpha:+.1%})"
    return "F", f"Significantly underperforming SPY ({alpha:+.1%})"


def winrate_grade(w: float) -> tuple[str, str]:
    if w >= 0.65:  return "A+", "Positive most trading days"
    if w >= 0.55:  return "A",  "More winning days than losing"
    if w >= 0.50:  return "B",  "Slightly more wins than losses"
    if w >= 0.45:  return "C",  "Near even wins and losses"
    return "D", "More losing days than winning"


def _fetch_spy_returns(n_days: int) -> pd.Series | None:
    """Fetch SPY % returns normalized to 0 start for the last n_days.
    Always returns a plain 1-D Series regardless of yfinance version."""
    try:
        import yfinance as yf
        from datetime import date, timedelta
        end = date.today()
        start = end - timedelta(days=n_days + 60)
        raw = yf.download("SPY", start=start.isoformat(), end=end.isoformat(),
                          progress=False, auto_adjust=True)
        if raw.empty:
            return None
        closes = raw["Close"]
        # yfinance ≥ 0.2.x may return a DataFrame with ticker as column level
        if isinstance(closes, pd.DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.squeeze().dropna().tail(n_days)
        if len(closes) < 2:
            return None
        return (closes / float(closes.iloc[0]) - 1)
    except Exception:
        return None


def _consensus(ticker: str, record: dict) -> dict:
    """Count bull/bear/neutral votes across all agents for a ticker."""
    counts = {"bullish": 0, "bearish": 0, "neutral": 0}
    agent_types = ["technical_signals", "fundamental_signals",
                   "sentiment_signals", "valuation_signals"]
    for at in agent_types:
        sig = record.get(at, {}).get(ticker)
        if isinstance(sig, dict):
            counts[sig.get("signal", "neutral")] = counts.get(sig.get("signal", "neutral"), 0) + 1
    for investor, per_ticker in (record.get("investor_signals") or {}).items():
        sig = per_ticker.get(ticker)
        if isinstance(sig, dict):
            counts[sig.get("signal", "neutral")] = counts.get(sig.get("signal", "neutral"), 0) + 1
    total = sum(counts.values())
    return {**counts, "total": total}


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Data
# ═════════════════════════════════════════════════════════════════════════════

if page == "Data":
    st.title("Today's Overview")
    ts = latest.get("timestamp", "")[:19]

    _is_demo_run = str(latest.get("run_id", "")).startswith("demo_")
    if _is_demo_run:
        st.warning(
            "Showing **demo/seeded signals** - not real Kronos predictions. "
            "Run `python scripts/run_live.py` (or `python main.py`) to generate "
            "live signals. Prices below are still live from the market."
        )
    else:
        st.caption(f"Kronos signals from: {ts} UTC")

    tickers = list(latest.get("kronos_signals", {}).keys())

    # Always fetch live prices from yfinance regardless of signal source
    live_prices = _fetch_live_prices(tuple(tickers)) if tickers else {}

    # ── Today's Picks ─────────────────────────────────────────────────────────
    st.subheader("Top Signals")
    st.caption("Tickers with the strongest agreement between Kronos and the agent panel.")

    pick_rows = []
    for ticker in tickers:
        sig = latest["kronos_signals"].get(ticker, {})
        cons = _consensus(ticker, latest)
        total = cons["total"] or 1
        bull_frac = cons["bullish"] / total
        bear_frac = cons["bearish"] / total
        pick_rows.append({
            "ticker": ticker,
            "direction": sig.get("directional_signal", "HOLD"),
            "predicted_return": sig.get("predicted_return", 0),
            "confidence": sig.get("confidence", 0),
            "bull": cons["bullish"],
            "bear": cons["bearish"],
            "neut": cons["neutral"],
            "total": total,
            "score": bull_frac if sig.get("directional_signal") == "BUY"
                     else (bear_frac if sig.get("directional_signal") == "SELL" else 0),
        })

    pick_rows.sort(key=lambda x: x["score"], reverse=True)
    top3 = pick_rows[:3]

    cols = st.columns(3)
    for col, pick in zip(cols, top3):
        direction = pick["direction"]
        color = C_BULL if direction == "BUY" else (C_BEAR if direction == "SELL" else C_NEUT)
        arrow = "▲" if direction == "BUY" else ("▼" if direction == "SELL" else "─")
        with col:
            st.markdown(f"""
<div class="pick-card" style="border-color:{color}">
  <div style="font-size:22px;font-weight:700;color:{color}">{arrow} {pick['ticker']}</div>
  <div style="font-size:28px;font-weight:700;margin:4px 0">{pick['predicted_return']:+.2%}</div>
  <div style="color:#9e9e9e;font-size:13px">Kronos predicted return</div>
  <div style="margin-top:10px;font-size:13px">
    🟢 {pick['bull']} bull &nbsp; 🔴 {pick['bear']} bear &nbsp; ⚪ {pick['neut']} neutral
  </div>
  <div style="color:#9e9e9e;font-size:12px;margin-top:4px">
    Kronos confidence: {pick['confidence']:.0%}
  </div>
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── Kronos Forecasts ──────────────────────────────────────────────────────
    st.subheader("Kronos Forecasts - All Tickers")
    st.caption("Kronos is a foundation model trained on candlestick data from 45+ global exchanges.")

    krows = []
    for ticker in tickers:
        sig = latest["kronos_signals"].get(ticker, {})
        direction = sig.get("directional_signal", "HOLD")
        ret = sig.get("predicted_return", 0)
        arrow = "▲" if direction == "BUY" else ("▼" if direction == "SELL" else "─")
        live_px = live_prices.get(ticker)
        price_str = f"${live_px:.2f}" if live_px else f"${sig.get('last_close', 0):.2f}"
        krows.append({
            "": arrow,
            "Ticker": ticker,
            "Forecast": direction,
            "Predicted Return (5d)": f"{ret:+.2%}",
            "Confidence": f"{sig.get('confidence', 0):.0%}",
            "Expected Volatility": f"{sig.get('predicted_volatility', 0):.2%}",
            "Worst-case Drop": f"{sig.get('predicted_max_drawdown', 0):.2%}",
            "Price (live)": price_str,
        })

    st.dataframe(pd.DataFrame(krows), use_container_width=True, hide_index=True)

    # Return bar chart
    fig_bar = go.Figure()
    for row in krows:
        ticker = row["Ticker"]
        ret_val = float(row["Predicted Return (5d)"].replace("%", "").replace("+", ""))
        color = C_BULL if ret_val > 0 else C_BEAR
        fig_bar.add_bar(x=[ticker], y=[ret_val], marker_color=color, name=ticker,
                        showlegend=False)
    fig_bar.add_hline(y=0, line_color="white", line_width=1, opacity=0.3)
    fig_bar.update_layout(
        title="Predicted 5-day Return by Ticker",
        yaxis_title="Predicted Return (%)", height=300,
        plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
        font_color="white", margin=dict(t=40, b=20),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # ── Agent Consensus Bars ──────────────────────────────────────────────────
    st.subheader("Agent Consensus")
    st.caption(
        "How many of the AI agents agree on the direction. "
        "Higher agreement = stronger conviction."
    )

    for ticker in tickers:
        sig = latest["kronos_signals"].get(ticker, {})
        cons = _consensus(ticker, latest)
        total = cons["total"] or 1
        bull_pct = cons["bullish"] / total
        bear_pct = cons["bearish"] / total
        neut_pct = cons["neutral"] / total
        direction = sig.get("directional_signal", "HOLD")
        ret = sig.get("predicted_return", 0)

        # Color the ticker label by Kronos signal
        label_color = C_BULL if direction == "BUY" else (C_BEAR if direction == "SELL" else C_NEUT)

        c_label, c_bar = st.columns([1, 4])
        with c_label:
            st.markdown(
                f'<div style="padding-top:6px">'
                f'<span style="font-weight:700;color:{label_color};font-size:16px">{ticker}</span>'
                f'<br><span style="font-size:11px;color:#9e9e9e">{ret:+.2%} predicted</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        with c_bar:
            # Stacked bar: bull | neut | bear
            bar_fig = go.Figure()
            bar_fig.add_bar(
                x=[bull_pct], y=[""], orientation="h",
                marker_color=C_BULL, name="Bullish",
                text=f"{cons['bullish']} bullish", textposition="inside",
                showlegend=False,
            )
            bar_fig.add_bar(
                x=[neut_pct], y=[""], orientation="h",
                marker_color=C_NEUT, name="Neutral",
                text=f"{cons['neutral']} neutral", textposition="inside",
                showlegend=False,
            )
            bar_fig.add_bar(
                x=[bear_pct], y=[""], orientation="h",
                marker_color=C_BEAR, name="Bearish",
                text=f"{cons['bearish']} bearish", textposition="inside",
                showlegend=False,
            )
            bar_fig.update_layout(
                barmode="stack", height=50,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(range=[0, 1], showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            )
            st.plotly_chart(bar_fig, use_container_width=True, key=f"cons_{ticker}")

    st.divider()

    # ── Predicted Candle Chart (per ticker) ───────────────────────────────────
    st.subheader("Predicted Price Candles")
    st.caption("Kronos model's predicted OHLC candles for the next 5 trading days.")
    ticker_sel = st.selectbox("Select ticker", tickers)
    candle_data = latest.get("kronos_signals", {}).get(ticker_sel, {}).get("predicted_candles", [])

    if candle_data:
        cdf = pd.DataFrame(candle_data)
        fig_c = go.Figure(data=[go.Candlestick(
            x=cdf["timestamp"], open=cdf["open"], high=cdf["high"],
            low=cdf["low"], close=cdf["close"],
            increasing_line_color=C_BULL, decreasing_line_color=C_BEAR,
        )])
        last_close = latest["kronos_signals"][ticker_sel].get("last_close", 0)
        fig_c.add_hline(
            y=last_close, line_color=C_GOLD, line_dash="dot",
            annotation_text=f"Today's close ${last_close:.2f}",
            annotation_position="bottom right",
        )
        fig_c.update_layout(
            title=f"{ticker_sel} - Predicted candles (Kronos model)",
            yaxis_title="Price ($)", height=380,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", xaxis_rangeslider_visible=False,
        )
        st.plotly_chart(fig_c, use_container_width=True)

    st.divider()

    # ── Portfolio Weights ─────────────────────────────────────────────────────
    st.subheader("Current Portfolio Weights")
    st.caption("How the portfolio is currently allocated across tickers.")
    final_weights = latest.get("final_weights", {})
    fw_rows = sorted(
        [(t, w) for t, w in final_weights.items() if w > 0],
        key=lambda x: x[1], reverse=True,
    )
    if fw_rows:
        c1, c2 = st.columns([2, 1])
        with c1:
            fig_pie = go.Figure(go.Pie(
                labels=[r[0] for r in fw_rows],
                values=[r[1] for r in fw_rows],
                hole=0.45,
                marker_colors=[C_BULL, C_BLUE, C_GOLD, "#ab47bc",
                               "#ec407a", "#26c6da", "#d4e157", "#ff7043"],
            ))
            fig_pie.update_layout(
                height=320, margin=dict(t=20, b=20),
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white",
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        with c2:
            st.markdown("**Allocation**")
            for ticker, weight in fw_rows:
                bar_w = int(weight * 200)
                st.markdown(
                    f'<div style="margin:6px 0">'
                    f'<span style="font-weight:600;width:55px;display:inline-block">{ticker}</span>'
                    f'<span style="display:inline-block;background:{C_BULL};height:12px;'
                    f'width:{bar_w}px;border-radius:3px;vertical-align:middle;margin:0 8px"></span>'
                    f'<span style="color:#9e9e9e">{weight:.1%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    with st.expander("Raw audit record (JSON)"):
        st.json(latest)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Performance
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Performance":
    st.title("Performance")

    # ── Live account: Alpaca is the ONLY source for portfolio dollar figures ─────
    _alp     = _fetch_alpaca_live()
    _alp_eq  = None   # pd.Series(equity, DatetimeIndex) when Alpaca is live

    if _alp and "error" not in _alp:
        _h = _alp["history"]
        if _h.get("timestamps") and _h.get("equity"):
            _s = pd.Series(
                _h["equity"],
                index=pd.to_datetime(_h["timestamps"], unit="s"),
            ).dropna()
            _s = _s[_s > 0]
            if len(_s) >= 1:
                _alp_eq = _s

    if _alp_eq is not None:
        # Dollar summary entirely from Alpaca history
        _a_curr  = float(_alp["equity"])
        _a_start = float(_alp_eq.iloc[0])
        _a_pnl   = _a_curr - _a_start
        _a_arr   = _alp_eq.values
        _a_pk    = pd.Series(_a_arr).cummax().values
        _a_tri   = int(np.argmin(_a_arr - _a_pk))
        _a_tr    = float(_a_arr[_a_tri])
        _a_pkt   = float(_a_pk[_a_tri])
        _a_pkmax = float(_a_pk[-1])
        _a_dd    = _a_tr - _a_pkt
        _pc      = "#26a69a" if _a_pnl >= 0 else "#ef5350"
        _ps      = "+" if _a_pnl >= 0 else ""
        _badge   = "PAPER" if _alp["mode"] == "Paper" else "LIVE"
        _bcol    = "#ffa726" if _alp["mode"] == "Paper" else "#ef5350"
        st.markdown(f"""
<div style="background:#1a2035;border-radius:10px;padding:16px 22px;margin-bottom:12px;
            border-left:4px solid {_bcol};display:flex;gap:40px;flex-wrap:wrap">
  <div style="width:100%;font-size:11px;color:{_bcol};font-weight:700;margin-bottom:4px">
    {_badge} ALPACA ACCOUNT - live data only
  </div>
  <div>
    <div style="color:#aaa;font-size:12px">Started with</div>
    <div style="font-size:20px;font-weight:700">${_a_start:,.2f}</div>
  </div>
  <div>
    <div style="color:#aaa;font-size:12px">Current value</div>
    <div style="font-size:20px;font-weight:700;color:{_pc}">${_a_curr:,.2f}
      <span style="font-size:14px">&nbsp;({_ps}${_a_pnl:,.2f})</span></div>
  </div>
  <div>
    <div style="color:#aaa;font-size:12px">Worst drop (peak to trough)</div>
    <div style="font-size:20px;font-weight:700;color:#ef5350">${_a_tr:,.2f}
      <span style="font-size:14px">&nbsp;(-${abs(_a_dd):,.2f})</span></div>
  </div>
  <div>
    <div style="color:#aaa;font-size:12px">All-time peak</div>
    <div style="font-size:20px;font-weight:700;color:#ffa726">${_a_pkmax:,.2f}</div>
  </div>
  <div>
    <div style="color:#aaa;font-size:12px">Open positions</div>
    <div style="font-size:20px;font-weight:700">{len(_alp['positions'])}</div>
  </div>
</div>""", unsafe_allow_html=True)

        if len(_alp_eq) >= 2:
            _fig_a = go.Figure()
            _fig_a.add_trace(go.Scatter(
                x=_alp_eq.index, y=_alp_eq.values,
                mode="lines", name=f"Alpaca {_alp['mode']}",
                line=dict(color=_bcol, width=2),
                fill="tozeroy", fillcolor=f"rgba(255,167,38,0.06)",
            ))
            _fig_a.add_trace(go.Scatter(
                x=_alp_eq.index, y=_a_pk,
                mode="lines", name="Running peak",
                line=dict(color="#ef5350", width=1, dash="dot"),
            ))
            _fig_a.update_layout(
                yaxis_title="Portfolio value ($)",
                yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                height=240, plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", legend=dict(x=0.01, y=0.99),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(_fig_a, use_container_width=True)
        st.divider()

    elif _alp and "error" in _alp:
        st.warning(f"Alpaca connection error: {_alp['error']}")
    else:
        st.info(
            "Set `ALPACA_API_KEY` in `.env` to see live portfolio performance here. "
            "Model quality metrics below are still available from audit logs."
        )

    # ── Model quality metrics: audit logs only ────────────────────────────────
    # These measure Kronos prediction quality, not account balance.
    stats    = metrics.summary_stats()
    acc_df   = metrics.kronos_accuracy()
    ec       = metrics.equity_curve()
    dr       = metrics.daily_returns()
    n_days   = len(ec)

    # For SPY comparison and report card dollar context, use Alpaca if available,
    # otherwise fall back to audit log equity (labeled as simulated).
    if _alp_eq is not None:
        start_eq = float(_alp_eq.iloc[0])
        curr_eq  = float(_alp["equity"])
    else:
        start_eq = float(ec["equity"].iloc[0]) if len(ec) > 0 else 100_000.0
        curr_eq  = float(ec["equity"].iloc[-1]) if len(ec) > 0 else 100_000.0

    # Compute Kronos accuracy
    valid_acc   = acc_df.dropna(subset=["correct"]) if not acc_df.empty else pd.DataFrame()
    overall_acc = float(valid_acc["correct"].mean()) if len(valid_acc) > 0 else None

    # Fetch SPY for same window
    spy_returns = _fetch_spy_returns(n_days + 10)
    spy_total   = float(spy_returns.iloc[-1]) if (spy_returns is not None and len(spy_returns) > 0) else None
    our_total   = stats["cumulative_return"]
    alpha       = (our_total - spy_total) if spy_total is not None else None

    # ── Report Card ───────────────────────────────────────────────────────────
    st.subheader("Model Report Card")
    st.caption("Plain-English summary of how the system is performing.")

    rc1, rc2, rc3, rc4 = st.columns(4)

    sharpe = stats["sharpe_ratio"]
    sg, sd = sharpe_grade(sharpe)
    with rc1:
        st.markdown(
            f'<div class="kpi-label">Risk-adjusted return (Sharpe)</div>'
            f'<div class="kpi-value">{sharpe:.2f} {grade_chip(sg)}</div>'
            f'<div class="kpi-sub">{sd}</div>',
            unsafe_allow_html=True,
        )

    if overall_acc is not None:
        ag, ad = accuracy_grade(overall_acc)
        with rc2:
            st.markdown(
                f'<div class="kpi-label">Kronos direction accuracy</div>'
                f'<div class="kpi-value">{overall_acc:.0%} {grade_chip(ag)}</div>'
                f'<div class="kpi-sub">{ad}</div>',
                unsafe_allow_html=True,
            )
    else:
        with rc2:
            st.markdown(
                '<div class="kpi-label">Kronos direction accuracy</div>'
                '<div class="kpi-value">- </div>'
                '<div class="kpi-sub">Need 2+ runs</div>',
                unsafe_allow_html=True,
            )

    if alpha is not None:
        alg, ald = alpha_grade(alpha)
        alpha_usd = start_eq * alpha
        with rc3:
            st.markdown(
                f'<div class="kpi-label">vs SPY (alpha)</div>'
                f'<div class="kpi-value">{alpha:+.2%} {grade_chip(alg)}</div>'
                f'<div class="kpi-sub">{ald} (≈ ${alpha_usd:+,.0f})</div>',
                unsafe_allow_html=True,
            )
    else:
        with rc3:
            st.markdown(
                '<div class="kpi-label">vs SPY (alpha)</div>'
                '<div class="kpi-value">-</div>'
                '<div class="kpi-sub">SPY data unavailable</div>',
                unsafe_allow_html=True,
            )

    wg, wd = winrate_grade(stats["win_rate"])
    n_pos_days = int(round(stats["win_rate"] * len(dr))) if len(dr) > 0 else None
    days_note  = f" ({n_pos_days} of {len(dr)} trading days)" if n_pos_days is not None else ""
    with rc4:
        st.markdown(
            f'<div class="kpi-label">Days with positive return</div>'
            f'<div class="kpi-value">{stats["win_rate"]:.0%} {grade_chip(wg)}</div>'
            f'<div class="kpi-sub">{wd}{days_note}</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── Kronos Hedge vs SPY ───────────────────────────────────────────────────
    st.subheader("Kronos Hedge vs SPY")
    st.caption(
        "Both lines start at 0%. Shows whether the model is beating the market "
        "or if you'd have done better just holding SPY."
    )

    # Use Alpaca history for the "our return" line when connected;
    # fall back to audit log equity (labeled as simulated) otherwise.
    if _alp_eq is not None and len(_alp_eq) >= 2:
        _vs_x   = _alp_eq.index.astype(str)
        _vs_y   = (_alp_eq / float(_alp_eq.iloc[0]) - 1) * 100
        _vs_n   = len(_alp_eq)
        _vs_lbl = f"Kronos Hedge ({_alp['mode']} account)"
    elif len(ec) >= 2:
        _vs_x   = ec["timestamp"]
        _vs_y   = (ec["equity"] / ec["equity"].iloc[0] - 1) * 100
        _vs_n   = len(ec)
        _vs_lbl = "Kronos Hedge (simulated)"
    else:
        _vs_n   = 0

    if _vs_n >= 2:
        our_pct = _vs_y

        fig_vs = go.Figure()
        fig_vs.add_trace(go.Scatter(
            x=_vs_x, y=our_pct,
            mode="lines", name=_vs_lbl,
            line=dict(color=C_BULL, width=2.5),
            fill="tozeroy", fillcolor="rgba(38,166,154,0.08)",
        ))

        if spy_returns is not None and len(spy_returns) >= 2:
            spy_pct = spy_returns * 100
            spy_dates = spy_pct.index.astype(str)
            fig_vs.add_trace(go.Scatter(
                x=spy_dates[-_vs_n:], y=spy_pct.values[-_vs_n:],
                mode="lines", name="SPY (benchmark)",
                line=dict(color=C_GOLD, width=1.5, dash="dash"),
            ))

        fig_vs.add_hline(y=0, line_color="white", line_width=0.8, opacity=0.3)
        fig_vs.update_layout(
            yaxis_title="Return (%)", height=400,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", legend=dict(x=0.01, y=0.99),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_vs, use_container_width=True)

        # Summary row below chart
        dollar_pnl_perf  = start_eq * our_total
        m1, m2, m3, m4 = st.columns(4)
        m1.metric(
            "Our total return",
            f"{our_total:+.2%}",
            delta=f"${dollar_pnl_perf:+,.0f} on ${start_eq:,.0f}",
            delta_color="normal" if dollar_pnl_perf >= 0 else "inverse",
        )
        if spy_total is not None:
            spy_dollar = start_eq * spy_total
            alpha_usd2  = dollar_pnl_perf - spy_dollar
            m2.metric(
                "SPY total return",
                f"{spy_total:+.2%}",
                delta=f"${spy_dollar:+,.0f} if held SPY",
                delta_color="normal" if spy_dollar >= 0 else "inverse",
            )
            m3.metric(
                "Alpha (us − SPY)",
                f"{alpha:+.2%}",
                delta=f"${alpha_usd2:+,.0f} {'ahead' if alpha_usd2 >= 0 else 'behind'}",
                delta_color="normal" if alpha_usd2 >= 0 else "inverse",
            )
        if len(dr) > 0:
            worst_day_pct = float(dr.min())
            worst_day_usd = curr_eq * worst_day_pct
            m4.metric(
                "Worst single day",
                f"{worst_day_pct:.2%}",
                delta=f"${worst_day_usd:,.0f} in one day",
                delta_color="inverse",
            )
        else:
            m4.metric("Worst single day", "-")

    st.divider()

    # ── Kronos Prediction Tracker ─────────────────────────────────────────────
    st.subheader("Kronos Prediction Tracker")
    st.caption(
        "For each past run, compares what Kronos predicted vs what actually happened. "
        "Green = got the direction right. Red = got it wrong."
    )

    if not acc_df.empty and len(acc_df.dropna(subset=["actual_return"])) >= 2:
        tracker = acc_df.dropna(subset=["actual_return"]).copy()
        tracker["date"] = pd.to_datetime(tracker["timestamp"]).dt.date.astype(str)
        tracker["correct_color"] = tracker["correct"].map(
            {True: C_BULL, False: C_BEAR, None: C_NEUT}
        )

        ticker_filter = st.selectbox(
            "Filter by ticker (or All)",
            ["All"] + sorted(tracker["ticker"].unique().tolist()),
            key="tracker_ticker",
        )
        if ticker_filter != "All":
            tracker = tracker[tracker["ticker"] == ticker_filter]

        fig_track = go.Figure()
        fig_track.add_bar(
            x=tracker["date"] + " · " + tracker["ticker"],
            y=tracker["predicted_return"] * 100,
            name="Predicted",
            marker_color=C_BLUE,
            opacity=0.75,
        )
        fig_track.add_bar(
            x=tracker["date"] + " · " + tracker["ticker"],
            y=tracker["actual_return"] * 100,
            name="Actual",
            marker_color=tracker["correct_color"].tolist(),
            opacity=0.90,
        )
        fig_track.add_hline(y=0, line_color="white", line_width=0.8, opacity=0.3)
        fig_track.update_layout(
            barmode="group",
            yaxis_title="Return (%)",
            height=400,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white",
            legend=dict(x=0.01, y=0.99),
            xaxis_tickangle=-45,
            margin=dict(t=20, b=80),
        )
        st.plotly_chart(fig_track, use_container_width=True)

        # Accuracy by ticker summary
        by_ticker = (
            acc_df.dropna(subset=["correct"])
            .groupby("ticker")["correct"].mean()
            .reset_index()
            .rename(columns={"correct": "accuracy"})
            .sort_values("accuracy", ascending=False)
        )
        if not by_ticker.empty:
            st.markdown("**Directional accuracy by ticker**")
            for _, row in by_ticker.iterrows():
                grade, _ = accuracy_grade(row["accuracy"])
                chip = grade_chip(grade)
                bar_w = int(row["accuracy"] * 180)
                st.markdown(
                    f'<div style="margin:5px 0;display:flex;align-items:center;gap:10px">'
                    f'<span style="font-weight:600;width:55px">{row["ticker"]}</span>'
                    f'<span style="display:inline-block;background:{C_BLUE};height:12px;'
                    f'width:{bar_w}px;border-radius:3px"></span>'
                    f'<span>{row["accuracy"]:.0%}</span>'
                    f'{chip}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info(
            "Prediction tracker needs at least 2 runs with the same tickers. "
            "As runs accumulate this chart fills in automatically."
        )

    st.divider()

    # ── Drawdown & Returns ────────────────────────────────────────────────────
    st.subheader("Risk Details")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Daily return distribution**")
        if len(dr) > 0:
            one_pct_usd = curr_eq * 0.01
            st.caption(
                f"How much the portfolio moved each day - clustered near 0 is good. "
                f"Each 1% move = **${one_pct_usd:,.0f}** on your current ${curr_eq:,.0f} portfolio."
            )
            fig_hist = go.Figure()
            fig_hist.add_trace(go.Histogram(
                x=dr * 100, nbinsx=20,
                marker_color=C_BLUE, opacity=0.8, name="Daily return",
            ))
            fig_hist.add_vline(x=0, line_color="white", line_width=1, opacity=0.4)
            fig_hist.update_layout(
                xaxis_title=f"Daily return %  (1% = ${one_pct_usd:,.0f})",
                yaxis_title="Days",
                height=280, showlegend=False,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                margin=dict(t=10, b=20),
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.caption("How much the portfolio moved each day - clustered near 0 is good.")

    with col_b:
        st.markdown("**Portfolio value vs peak ($)**")
        st.caption(
            "Your portfolio in actual dollars. The dotted red line shows the running peak - "
            "any gap below it is real money temporarily lost."
        )
        # Use Alpaca history when connected; audit log equity otherwise
        _dd_src    = _alp_eq if (_alp_eq is not None and len(_alp_eq) >= 2) else None
        _dd_ec_ok  = len(ec) >= 2
        if _dd_src is not None or _dd_ec_ok:
            eq_arr   = _dd_src.values if _dd_src is not None else ec["equity"].values
            dd_x     = (_dd_src.index.astype(str) if _dd_src is not None
                        else ec["timestamp"])
            peak_arr = pd.Series(eq_arr).cummax().values
            tr_idx   = int(np.argmin(eq_arr - peak_arr))
            tr_val   = float(eq_arr[tr_idx])
            pk_at_tr = float(peak_arr[tr_idx])

            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd_x, y=eq_arr,
                mode="lines", name="Portfolio value",
                line=dict(color=C_BULL, width=2),
                fill="tozeroy", fillcolor="rgba(38,166,154,0.07)",
            ))
            fig_dd.add_trace(go.Scatter(
                x=dd_x, y=peak_arr,
                mode="lines", name="Running peak",
                line=dict(color=C_BEAR, width=1, dash="dot"),
            ))
            if tr_val < pk_at_tr:
                fig_dd.add_annotation(
                    x=dd_x[tr_idx] if hasattr(dd_x, '__getitem__') else dd_x.iloc[tr_idx],
                    y=tr_val,
                    text=f"Worst: ${tr_val:,.0f}<br>(-${pk_at_tr - tr_val:,.0f})",
                    showarrow=True, arrowhead=2, arrowcolor=C_BEAR,
                    font=dict(color=C_BEAR, size=11),
                    bgcolor="#0e1117", bordercolor=C_BEAR,
                )
            fig_dd.update_layout(
                yaxis_title="Portfolio value ($)",
                yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                height=280,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117", font_color="white",
                legend=dict(x=0.01, y=0.01, font=dict(size=10)),
                margin=dict(t=10, b=20),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

    dd_val = stats["max_drawdown"]
    _dd_arr = _alp_eq.values if (_alp_eq is not None and len(_alp_eq) >= 2) else (
              ec["equity"].values if len(ec) >= 2 else None)
    if _dd_arr is not None:
        _pk2  = pd.Series(_dd_arr).cummax().values
        _ti2  = int(np.argmin(_dd_arr - _pk2))
        _tr2  = float(_dd_arr[_ti2])
        _pk2v = float(_pk2[_ti2])
        _dd_usd = _tr2 - _pk2v
        st.info(
            f"Worst peak-to-trough loss: portfolio fell from **${_pk2v:,.2f}** "
            f"to **${_tr2:,.2f}** (a drop of **${_dd_usd:,.2f}**). "
            + ("Within acceptable range." if dd_val > -0.10 else
               "Drawdown exceeded 10% - review position sizing.")
        )

    st.divider()

    # ── Run History ───────────────────────────────────────────────────────────
    st.subheader("Run History")
    _eq_hist = [r.get("portfolio_equity", 0) for r in records]
    run_rows = []
    for _i, _r in enumerate(records):
        _eq_i   = _eq_hist[_i]
        _prev_i = _eq_hist[_i - 1] if _i > 0 else _eq_i
        _chg    = _eq_i - _prev_i
        run_rows.append({
            "Date":          _r.get("timestamp", "")[:10],
            "Portfolio ($)": f"${_eq_i:,.0f}",
            "Daily P&L":     f"${_chg:+,.0f}" if _i > 0 else "-",
            "Tickers":       ", ".join(_r.get("kronos_signals", {}).keys()),
            "Orders":        len(_r.get("orders", [])),
        })
    st.dataframe(pd.DataFrame(run_rows), use_container_width=True, hide_index=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Backtest
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Backtest":
    st.title("Backtest")
    st.caption(
        "Simulate how the Kronos + Qlib strategy would have performed on historical data. "
        "Always compared against simply holding SPY."
    )

    bt_csv = Path("logs/backtest_results.csv")

    c1, c2, c3 = st.columns(3)
    with c1:
        _bt_start_raw = st.date_input("Start date",
                                       value=pd.Timestamp.today() - pd.Timedelta(days=365))
    with c2:
        _bt_end_raw = st.date_input("End date", value=pd.Timestamp.today())
    with c3:
        bt_cash = st.number_input("Starting cash ($)", value=100_000, step=10_000)

    from datetime import date as _date_type
    bt_start = _bt_start_raw if isinstance(_bt_start_raw, _date_type) else _bt_start_raw.date()
    bt_end   = _bt_end_raw   if isinstance(_bt_end_raw,   _date_type) else _bt_end_raw.date()

    bt_agents = st.checkbox("Include LLM agents (slow, uses API credits)", value=False)

    with st.expander("Strategy & risk settings"):
        st.caption(
            "Tune signal blend, drawdown protection, and trade frequency. "
            "Defaults calibrated from the year-long backtest."
        )
        _rc1, _rc2, _rc3 = st.columns(3)
        with _rc1:
            st.markdown("**Signal**")
            bt_momentum_blend = st.slider(
                "Momentum blend (%)",
                min_value=0, max_value=100, value=80, step=10,
                help="How much weight to give cross-sectional 12M-1M momentum vs Kronos model score. "
                     "80% recommended — minimises noise from the Chronos predictor.",
            ) / 100.0
            bt_top_n = st.slider(
                "Top-N concentration",
                min_value=3, max_value=20, value=10, step=1,
                help="Hold only the top N ranked stocks. Remaining are zeroed out. "
                     "Lower = more concentrated, higher = more diversified.",
            )
            bt_rebalance_freq = st.selectbox(
                "Rebalance frequency",
                ["daily", "weekly"],
                index=0,
                help="Weekly (Friday) rebalancing cuts transaction costs ~5× vs daily.",
            )
        with _rc2:
            st.markdown("**Drawdown protection**")
            bt_dd_stop = st.slider(
                "Circuit breaker threshold (%)",
                min_value=1, max_value=20, value=7, step=1,
                help="Engage when portfolio is down this far from peak. "
                     "7% avoids false triggers from normal daily noise.",
            ) / 100.0
            bt_recovery = st.slider(
                "Recovery threshold (%)",
                min_value=1, max_value=10, value=2, step=1,
                help="Release breaker once back within this % of peak.",
            ) / 100.0
            bt_cash_res = st.slider(
                "Cash reserve when active (%)",
                min_value=10, max_value=60, value=30, step=5,
                help="Fraction redirected while circuit breaker is engaged.",
            ) / 100.0
            bt_spy_reserve = st.checkbox(
                "Park reserve in SPY (not cash)",
                value=True,
                help="When the circuit breaker fires, redirect the cash reserve into SPY "
                     "instead of idle cash. Prevents drag during bull-market drawdowns.",
            )
        with _rc3:
            st.markdown("**Position & trading**")
            bt_trailing_stop = st.slider(
                "Trailing stop per position (%)",
                min_value=0, max_value=20, value=7, step=1,
                help="Force-sell a position if it falls this far from its personal price peak. 0 = disabled.",
            ) / 100.0
            bt_min_trade = st.slider(
                "Minimum trade size (% of equity)",
                min_value=0.1, max_value=2.0, value=0.5, step=0.1,
                help="Only rebalance if position change exceeds this threshold. Reduces churn.",
            ) / 100.0

    bt_range_days = (bt_end - bt_start).days
    if bt_range_days < 5:
        st.warning("Select at least 5 calendar days to get meaningful results.")

    if st.button("Run Backtest", type="primary"):
        import traceback as _tb
        import yaml
        try:
            with open("config/settings.yaml") as f:
                cfg = yaml.safe_load(f)
        except FileNotFoundError:
            st.error("config/settings.yaml not found. Make sure Streamlit is launched from the project root.")
            st.stop()
        from qlib_pipeline.backtest import Backtester

        _bt_err = None
        _bt_tb  = None
        results = None
        with st.spinner("Running backtest... this takes a few minutes."):
            try:
                bt = Backtester(
                    tickers=cfg["universe"]["tickers"],
                    start_date=bt_start,
                    end_date=bt_end,
                    lookback_days=cfg["kronos"]["lookback_days"],
                    kronos_model_size=cfg["kronos"]["model_size"],
                    kronos_horizon=cfg["kronos"]["horizon"],
                    kronos_device=cfg["kronos"]["device"],
                    qlib_weight=cfg["reconciliation"]["qlib_weight"],
                    agent_weight=cfg["reconciliation"]["agent_weight"],
                    llm_provider=cfg["agents"]["llm_provider"],
                    llm_model=cfg["agents"]["llm_model"],
                    initial_cash=bt_cash,
                    run_agents=bt_agents,
                    drawdown_stop=bt_dd_stop,
                    recovery_threshold=bt_recovery,
                    cash_reserve_pct=bt_cash_res,
                    min_trade_pct=bt_min_trade,
                    trailing_stop=bt_trailing_stop,
                    momentum_blend=bt_momentum_blend,
                    rebalance_frequency=bt_rebalance_freq,
                    spy_reserve=bt_spy_reserve,
                    top_n=bt_top_n,
                )
                results = bt.run()
                stats_bt = bt.summary()
                Path("logs").mkdir(exist_ok=True)
                results.to_csv(bt_csv, index=False)
            except Exception as _e:
                _bt_err = _e
                _bt_tb  = _tb.format_exc()

        if _bt_err is not None:
            st.error(f"Backtest failed: {_bt_err}")
            with st.expander("Full traceback"):
                st.code(_bt_tb)
            st.stop()

        if len(results) == 0:
            st.warning("Backtest ran but produced no results. No trading days found in the selected range.")
            st.stop()

        st.success(f"Backtest complete - {len(results)} trading days processed.")

        sg, sd = sharpe_grade(stats_bt.get("sharpe_ratio", 0))
        wg, wd = winrate_grade(stats_bt.get("win_rate", 0))
        _bt_ret = stats_bt.get("total_return", 0)
        _bt_dd  = stats_bt.get("max_drawdown", 0)
        _bt_pnl = bt_cash * _bt_ret
        _bt_dd_usd = bt_cash * _bt_dd   # negative
        b1, b2, b3, b4 = st.columns(4)
        b1.metric(
            "Total return",
            f"{_bt_ret:+.2%}",
            delta=f"${_bt_pnl:+,.0f} on ${bt_cash:,.0f}",
            delta_color="normal" if _bt_pnl >= 0 else "inverse",
        )
        b2.metric("Sharpe ratio", f"{stats_bt.get('sharpe_ratio', 0):.2f} ({sg})")
        b3.metric(
            "Max drawdown",
            f"{_bt_dd:.2%}",
            delta=f"${_bt_dd_usd:,.0f} worst drop",
            delta_color="inverse",
        )
        b4.metric("Win rate", f"{stats_bt.get('win_rate', 0):.0%} ({wg})")

    # ── Chart: Kronos Hedge % return vs SPY % return ──────────────────────────
    if not bt_csv.exists():
        st.info("No backtest results yet. Pick a date range above and click **Run Backtest**.")
    elif bt_csv.exists():
        bt_df = pd.read_csv(bt_csv, parse_dates=["date"])
        if len(bt_df) < 2:
            st.info("No backtest results yet. Configure the date range above and click 'Run Backtest'.")
            st.stop()

        st.subheader("Kronos Hedge vs SPY - % Return")
        st.caption(
            f"Historical simulation: {bt_df['date'].iloc[0].strftime('%b %d, %Y')} to "
            f"{bt_df['date'].iloc[-1].strftime('%b %d, %Y')} "
            f"({len(bt_df)} trading days). "
            "Both lines start at 0%."
        )

        bt_pct = (bt_df["equity"] / bt_df["equity"].iloc[0] - 1) * 100

        fig_bt = go.Figure()
        fig_bt.add_trace(go.Scatter(
            x=bt_df["date"], y=bt_pct,
            mode="lines", name="Kronos Hedge",
            line=dict(color=C_BULL, width=2.5),
            fill="tozeroy", fillcolor="rgba(38,166,154,0.08)",
        ))

        # SPY - fetch enough history, trim to match bt_df length, normalize to 0%
        spy = _fetch_spy_returns(len(bt_df) + 60)
        spy_pct_arr: np.ndarray | None = None
        spy_alpha: float | None = None

        if spy is not None and len(spy) >= 2:
            # Take last N points matching bt_df length
            n_take = min(len(spy), len(bt_df))
            spy_vals = np.asarray(spy.values[-n_take:], dtype=float).flatten()
            # Re-normalize so it starts at 0% (consistent with Kronos curve)
            spy_pct_arr = (spy_vals - spy_vals[0]) * 100

            # Pad/trim x-axis to match
            bt_dates = bt_df["date"].values[-n_take:]

            fig_bt.add_trace(go.Scatter(
                x=bt_dates, y=spy_pct_arr,
                mode="lines", name="SPY (buy & hold)",
                line=dict(color=C_GOLD, width=1.5, dash="dash"),
            ))

            spy_end  = float(spy_pct_arr[-1])
            kron_end = float(bt_pct.iloc[-1])
            spy_alpha = kron_end - spy_end

        fig_bt.add_hline(y=0, line_color="white", line_width=0.8, opacity=0.3)
        fig_bt.update_layout(
            yaxis_title="Return from start (%)",
            height=420,
            plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
            font_color="white", legend=dict(x=0.01, y=0.99),
            margin=dict(t=20, b=20),
        )
        st.plotly_chart(fig_bt, use_container_width=True)

        # Summary metrics
        bt_start_eq   = float(bt_df["equity"].iloc[0])
        bt_end_eq     = float(bt_df["equity"].iloc[-1])
        bt_dollar_pnl = bt_end_eq - bt_start_eq
        spy_end_pct2  = float(spy_pct_arr[-1]) if spy_pct_arr is not None else None

        s1, s2, s3 = st.columns(3)
        s1.metric(
            "Kronos Hedge total return",
            f"{float(bt_pct.iloc[-1]):+.2f}%",
            delta=f"${bt_dollar_pnl:+,.0f}  (${bt_start_eq:,.0f} → ${bt_end_eq:,.0f})",
            delta_color="normal" if bt_dollar_pnl >= 0 else "inverse",
        )
        if spy_end_pct2 is not None:
            spy_bt_usd = bt_start_eq * spy_end_pct2 / 100
            s2.metric(
                "SPY total return",
                f"{spy_end_pct2:+.2f}%",
                delta=f"${spy_bt_usd:+,.0f} if held SPY",
                delta_color="normal" if spy_bt_usd >= 0 else "inverse",
            )
        if spy_alpha is not None and spy_end_pct2 is not None:
            alpha_usd_bt = bt_dollar_pnl - (bt_start_eq * spy_end_pct2 / 100)
            s3.metric(
                "Alpha vs SPY",
                f"{spy_alpha:+.2f}%",
                delta=f"${alpha_usd_bt:+,.0f} {'ahead' if alpha_usd_bt >= 0 else 'behind'}",
                delta_color="normal" if alpha_usd_bt >= 0 else "inverse",
            )

        st.divider()

        # Drawdown + cumulative return side by side
        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("**Cumulative return over backtest**")
            fig_ret = go.Figure()
            fig_ret.add_trace(go.Scatter(
                x=bt_df["date"], y=bt_pct,
                fill="tozeroy", mode="lines",
                line=dict(color=C_BLUE, width=2),
                fillcolor="rgba(66,165,245,0.1)",
            ))
            if spy_pct_arr is not None:
                fig_ret.add_trace(go.Scatter(
                    x=bt_df["date"].values[-len(spy_pct_arr):], y=spy_pct_arr,
                    mode="lines", name="SPY",
                    line=dict(color=C_GOLD, width=1, dash="dot"),
                ))
            fig_ret.update_layout(
                yaxis_title="Return (%)", height=300, showlegend=False,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white", margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_ret, use_container_width=True)

        with col_b:
            st.markdown("**Portfolio value vs peak - backtest ($)**")
            st.caption("Actual dollars in the portfolio. Dotted line = running high-water mark.")
            eq_arr3   = bt_df["equity"].values
            pk_arr3   = pd.Series(eq_arr3).cummax().values
            tri3      = int(np.argmin(eq_arr3 - pk_arr3))
            trv3      = float(eq_arr3[tri3])
            pkat3     = float(pk_arr3[tri3])

            fig_dd3 = go.Figure()
            fig_dd3.add_trace(go.Scatter(
                x=bt_df["date"], y=eq_arr3,
                mode="lines", name="Portfolio value",
                line=dict(color=C_BULL, width=2),
                fill="tozeroy", fillcolor="rgba(38,166,154,0.07)",
            ))
            fig_dd3.add_trace(go.Scatter(
                x=bt_df["date"], y=pk_arr3,
                mode="lines", name="Running peak",
                line=dict(color=C_BEAR, width=1, dash="dot"),
            ))
            if trv3 < pkat3:
                fig_dd3.add_annotation(
                    x=bt_df["date"].iloc[tri3],
                    y=trv3,
                    text=f"Worst: ${trv3:,.0f}<br>(−${pkat3 - trv3:,.0f})",
                    showarrow=True, arrowhead=2, arrowcolor=C_BEAR,
                    font=dict(color=C_BEAR, size=11),
                    bgcolor="#0e1117", bordercolor=C_BEAR,
                )
            fig_dd3.update_layout(
                yaxis_title="Portfolio value ($)",
                yaxis_tickprefix="$", yaxis_tickformat=",.0f",
                height=300,
                plot_bgcolor="#0e1117", paper_bgcolor="#0e1117",
                font_color="white",
                legend=dict(x=0.01, y=0.01, font=dict(size=10)),
                margin=dict(t=10, b=10),
            )
            st.plotly_chart(fig_dd3, use_container_width=True)

        with st.expander("Raw backtest data"):
            st.dataframe(bt_df, use_container_width=True, hide_index=True)
    else:
        st.info("No backtest results yet. Click 'Run Backtest' above.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: News
# ═════════════════════════════════════════════════════════════════════════════

elif page == "News":
    import yaml as _yaml
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    from data.news import (
        upcoming_macro_events as _get_macro,
        fetch_earnings_calendar as _get_earnings,
        fetch_ticker_news as _get_news,
        keyword_sentiment as _kw_sent,
        fetch_fed_outlook as _get_fed_outlook,
        FOMC_DATES as _FOMC_DATES,
    )

    st.title("News & Events")

    # ── Load tickers from config (instant, no network) ────────────────────────
    try:
        with open("config/settings.yaml") as _f:
            _news_cfg = _yaml.safe_load(_f)
        _news_tickers: list[str] = _news_cfg["universe"]["tickers"]
    except Exception:
        _news_tickers = list(latest.get("kronos_signals", {}).keys()) if latest else []

    if not _news_tickers:
        st.warning("No tickers configured. Check config/settings.yaml.")
        st.stop()

    # ── Calendars + Fed outlook (load in parallel via cached fns) ────────────
    _macro_all = _get_macro(window_days=60)
    _today = pd.Timestamp.today().date()

    @st.cache_data(ttl=1800)
    def _load_fed_outlook() -> dict:
        try:
            return dict(_get_fed_outlook())
        except Exception:
            return {"implied_move": "unknown", "implied_bps": 0,
                    "confidence": "low", "rationale": "Data unavailable.",
                    "current_rate": None, "tbill_3m": None,
                    "cpi_yoy": None, "unemployment": None}

    def _event_card(label: str, date_str: str, days: int, risk: str, note: str = "") -> None:
        if days == 0:
            _when, _col = "TODAY", "#ef5350"
        elif days == 1:
            _when, _col = "Tomorrow", "#ffa726"
        elif days <= 7:
            _when, _col = f"In {days} days", "#ffa726"
        elif days < 0:
            _when, _col = f"{abs(days)}d ago", "#78909c"
        else:
            _when, _col = f"In {days} days", "#42a5f5"
        _badge = (
            f'<span style="background:{_col}22;color:{_col};font-size:11px;'
            f'padding:2px 7px;border-radius:10px;font-weight:600">{_when}</span>'
        )
        _note_html = f'<br><span style="color:#aaa;font-size:12px">{note}</span>' if note else ""
        st.markdown(
            f'<div style="background:#111827;border-left:3px solid {_col};'
            f'padding:10px 14px;border-radius:5px;margin:5px 0">'
            f'<span style="font-weight:600;color:#e0e0e0">{label}</span> '
            f'&nbsp;{_badge}&nbsp;'
            f'<span style="color:#888;font-size:12px">{date_str}</span>'
            f'{_note_html}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Fetch 60-day earnings once; filter into week/ahead below
    @st.cache_data(ttl=3600)
    def _load_earnings_ahead(tickers: tuple) -> list:
        return _get_earnings(list(tickers), window_days=60)

    with st.spinner("Loading earnings calendar..."):
        _earn_ahead_all = _load_earnings_ahead(tuple(_news_tickers))

    _earn_week  = [e for e in _earn_ahead_all if e["days_away"] <= 7]
    _earn_ahead = [e for e in _earn_ahead_all if e["days_away"] > 7]

    # ═══════════════════════════════════════════════════════════════
    # SECTION 1 — This Week (next 7 days)
    # ═══════════════════════════════════════════════════════════════
    st.subheader("This Week")

    # FOMC this week — fetch Fed outlook alongside
    _fomc_this_week = [e for e in _macro_all if 0 <= e["days_away"] <= 7]

    if _fomc_this_week:
        with st.spinner("Fetching Fed rate outlook..."):
            _fed = _load_fed_outlook()

        for _ev in _fomc_this_week:
            _risk_note = "Macro risk filter active: all positions scaled to 65%" if _ev["days_away"] <= 2 else ""
            _event_card(_ev["event"], _ev["date"], _ev["days_away"], _ev["risk_level"], _risk_note)

            # Fed rate projection panel
            _move  = _fed.get("implied_move", "unknown")
            _bps   = _fed.get("implied_bps", 0)
            _conf  = _fed.get("confidence", "low")
            _rat   = _fed.get("rationale", "")
            _rate  = _fed.get("current_rate")
            _tbill = _fed.get("tbill_3m")
            _cpi   = _fed.get("cpi_yoy")
            _unemp = _fed.get("unemployment")

            _move_col = {"cut": "#26a69a", "hike": "#ef5350", "hold": "#ffa726"}.get(_move, "#78909c")
            _move_label = {
                "cut":  f"CUT {_bps}bp" if _bps else "CUT",
                "hike": f"HIKE {_bps}bp" if _bps else "HIKE",
                "hold": "HOLD",
            }.get(_move, "UNKNOWN")
            _conf_badge = (
                f'<span style="background:{_move_col}22;color:{_move_col};font-size:11px;'
                f'padding:1px 6px;border-radius:8px">{_conf} confidence</span>'
            )

            # Economic indicator pills
            _pills = []
            if _rate is not None:
                _pills.append(f'<span style="color:#aaa">Fed Funds&nbsp;<b style="color:#e0e0e0">{_rate:.2f}%</b></span>')
            if _tbill is not None:
                _pills.append(f'<span style="color:#aaa">3m T-Bill&nbsp;<b style="color:#e0e0e0">{_tbill:.2f}%</b></span>')
            if _cpi is not None:
                _cpi_col = "#ef5350" if _cpi > 3.0 else ("#ffa726" if _cpi > 2.0 else "#26a69a")
                _pills.append(f'<span style="color:#aaa">CPI YoY&nbsp;<b style="color:{_cpi_col}">{_cpi:.1f}%</b></span>')
            if _unemp is not None:
                _u_col = "#ef5350" if _unemp > 5.0 else ("#ffa726" if _unemp > 4.0 else "#26a69a")
                _pills.append(f'<span style="color:#aaa">Unemployment&nbsp;<b style="color:{_u_col}">{_unemp:.1f}%</b></span>')
            _pills_html = '&nbsp;&nbsp;&bull;&nbsp;&nbsp;'.join(_pills)

            _no_fred_hint = (
                '<br><span style="color:#555;font-size:11px">'
                'Set FRED_API_KEY in .env for exact CPI + unemployment data.</span>'
                if not _pills or _cpi is None else ""
            )

            st.markdown(
                f'<div style="background:#0d1f33;border:1px solid {_move_col}44;'
                f'padding:14px 16px;border-radius:6px;margin:4px 0 12px 0">'
                f'<span style="color:#888;font-size:11px;font-weight:600;letter-spacing:0.05em">'
                f'MARKET-IMPLIED FOMC OUTCOME</span><br>'
                f'<span style="font-size:26px;font-weight:700;color:{_move_col}">{_move_label}</span>'
                f'&nbsp;&nbsp;{_conf_badge}<br>'
                f'<span style="color:#bbb;font-size:13px">{_rat}</span>'
                + (f'<br><br><span style="font-size:12px">{_pills_html}</span>' if _pills_html else "")
                + _no_fred_hint
                + '</div>',
                unsafe_allow_html=True,
            )

    for _ev in _earn_week:
        _note = "Position halved by macro risk filter" if _ev["days_away"] <= 1 else ""
        _kron_dir = latest.get("kronos_signals", {}).get(_ev["ticker"], {}).get("directional_signal", "") if latest else ""
        _kron_hint = f"Kronos: {_kron_dir}" if _kron_dir else ""
        _full_note = " | ".join(filter(None, [_note, _kron_hint]))
        _event_card(f"{_ev['ticker']} Earnings", _ev["earnings_date"], _ev["days_away"], "medium", _full_note)

    if not _fomc_this_week and not _earn_week:
        st.caption("No scheduled macro events or earnings this week.")

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 2 — Looking Ahead (8-60 days)
    # ═══════════════════════════════════════════════════════════════
    st.subheader("Looking Ahead")

    _fomc_ahead = [e for e in _macro_all if 7 < e["days_away"] <= 60]

    if _fomc_ahead or _earn_ahead:
        # Merge and sort by days_away
        _combined: list[dict] = []
        for _ev in _fomc_ahead:
            _combined.append({"label": _ev["event"], "date": _ev["date"],
                               "days": _ev["days_away"], "type": "FOMC"})
        for _ev in _earn_ahead:
            _combined.append({"label": f"{_ev['ticker']} Earnings", "date": _ev["earnings_date"],
                               "days": _ev["days_away"], "type": "earnings"})
        _combined.sort(key=lambda x: x["days"])

        # Fed outlook for "Looking Ahead" FOMC annotation (reuse cached value)
        _fed2 = _load_fed_outlook() if _fomc_ahead else {}
        _f2_move  = _fed2.get("implied_move", "unknown")
        _f2_bps   = _fed2.get("implied_bps", 0)
        _f2_col   = {"cut": "#26a69a", "hike": "#ef5350", "hold": "#ffa726"}.get(_f2_move, "#555")
        _f2_lbl   = {
            "cut":  f"Market implies cut {_f2_bps}bp" if _f2_bps else "Market implies cut",
            "hike": f"Market implies hike {_f2_bps}bp" if _f2_bps else "Market implies hike",
            "hold": "Market implies hold",
        }.get(_f2_move, "Outcome unknown")

        # Group into rows of 2 columns for a compact calendar look
        for _i in range(0, len(_combined), 2):
            _row = _combined[_i:_i+2]
            _rcols = st.columns(2)
            for _ci, _item in enumerate(_row):
                _is_fomc = _item["type"] == "FOMC"
                _col2 = _f2_col if _is_fomc else "#78909c"
                _d2 = _item["days"]
                _when2 = f"In {_d2} days" if _d2 > 0 else f"{abs(_d2)}d ago"
                _proj_line = (
                    f'<br><span style="color:{_f2_col};font-size:11px;font-weight:600">{_f2_lbl}</span>'
                    if _is_fomc and _f2_move != "unknown" else ""
                )
                _rcols[_ci].markdown(
                    f'<div style="background:#111827;border-left:3px solid {_col2};'
                    f'padding:8px 12px;border-radius:5px">'
                    f'<span style="font-weight:600;color:#e0e0e0;font-size:14px">{_item["label"]}</span><br>'
                    f'<span style="color:#888;font-size:12px">{_item["date"]} &bull; {_when2}</span>'
                    f'{_proj_line}'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.caption("No FOMC meetings or earnings in the next 60 days for the universe.")

    st.divider()

    # ═══════════════════════════════════════════════════════════════
    # SECTION 3 — Ticker Headlines & Kronos View
    # ═══════════════════════════════════════════════════════════════
    st.subheader("Ticker Headlines & Kronos View")
    st.caption("News fetched on demand for the selected ticker. Kronos forecast from the last run cycle.")

    _sel_ticker = st.selectbox("Select ticker", _news_tickers)

    # Per-ticker news — cached per ticker so changing dropdown is fast after first load
    @st.cache_data(ttl=300)
    def _load_ticker_news(ticker: str) -> list:
        return _get_news(ticker, max_items=12)

    with st.spinner(f"Fetching headlines for {_sel_ticker}..."):
        _items = _load_ticker_news(_sel_ticker)

    # Kronos signal from latest audit log
    _kron_sig = (latest.get("kronos_signals", {}).get(_sel_ticker) if latest else None)
    _llm_sent = (latest.get("news_sentiment_signals", {}).get(_sel_ticker) if latest else None)

    # Earnings for this ticker
    _tick_earn = [e for e in (_earn_ahead_all + _earn_week) if e["ticker"] == _sel_ticker]

    # ── Three info cards ──────────────────────────────────────────
    _ck, _cs, _ce = st.columns(3)

    with _ck:
        st.markdown("**Kronos Forecast**")
        if _kron_sig:
            _kret  = _kron_sig.get("predicted_return", 0)
            _kdir  = _kron_sig.get("directional_signal", "HOLD")
            _kconf = _kron_sig.get("confidence", 0)
            _kcol  = C_BULL if _kdir == "BUY" else (C_BEAR if _kdir == "SELL" else C_NEUT)
            st.markdown(
                f'<div style="background:#111827;padding:12px;border-radius:6px;'
                f'border-left:4px solid {_kcol}">'
                f'<span style="font-size:22px;font-weight:700;color:{_kcol}">{_kdir}</span><br>'
                f'<span style="color:#ccc;font-size:13px">{_kret:+.2%} predicted &bull; {_kconf:.0%} conf</span><br>'
                f'<span style="color:#666;font-size:11px">from last run cycle</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("Run a live cycle to see Kronos forecast.")

    with _cs:
        st.markdown("**News Sentiment**")
        if _items:
            _sl, _ss = _kw_sent([i["title"] for i in _items])
            _sc = C_BULL if _sl == "bullish" else (C_BEAR if _sl == "bearish" else C_NEUT)
            st.markdown(
                f'<div style="background:#111827;padding:12px;border-radius:6px;'
                f'border-left:4px solid {_sc}">'
                f'<span style="font-size:22px;font-weight:700;color:{_sc}">{_sl.upper()}</span><br>'
                f'<span style="color:#ccc;font-size:13px">Score {_ss:+.2f} &bull; {len(_items)} headlines</span><br>'
                f'<span style="color:#666;font-size:11px">keyword model</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.caption("No headlines found.")

    with _ce:
        st.markdown("**Next Earnings**")
        if _tick_earn:
            _te = min(_tick_earn, key=lambda e: abs(e["days_away"]))
            _td = _te["days_away"]
            _tc = "#ef5350" if _td <= 1 else ("#ffa726" if _td <= 7 else "#42a5f5")
            _tw = "TODAY" if _td == 0 else ("Tomorrow" if _td == 1 else (f"In {_td}d" if _td > 0 else f"{abs(_td)}d ago"))
            st.markdown(
                f'<div style="background:#111827;padding:12px;border-radius:6px;'
                f'border-left:4px solid {_tc}">'
                f'<span style="font-size:22px;font-weight:700;color:{_tc}">{_tw}</span><br>'
                f'<span style="color:#ccc;font-size:13px">{_te["earnings_date"]}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#111827;padding:12px;border-radius:6px;'
                'border-left:4px solid #444">'
                '<span style="color:#666">No earnings in next 60 days</span>'
                '</div>',
                unsafe_allow_html=True,
            )

    st.markdown("")

    # ── Kronos + News synthesis ───────────────────────────────────
    if _llm_sent and _llm_sent.get("kronos_news_view"):
        _ll_sig = _llm_sent.get("signal", "neutral")
        _ll_col = C_BULL if _ll_sig == "bullish" else (C_BEAR if _ll_sig == "bearish" else C_NEUT)
        st.markdown(
            f'<div style="background:#0d1f33;padding:14px;border-radius:8px;'
            f'border:1px solid {_ll_col}55;margin-bottom:10px">'
            f'<span style="color:#888;font-size:11px;font-weight:600">KRONOS + NEWS SYNTHESIS</span><br>'
            f'<span style="color:{_ll_col};font-weight:700">{_ll_sig.upper()} &bull; '
            f'{_llm_sent.get("confidence",50)}% confidence</span><br>'
            f'<span style="color:#ddd;font-size:14px">{_llm_sent["kronos_news_view"]}</span>'
            + (f'<br><br><span style="color:#aaa;font-size:12px">{_llm_sent.get("reasoning","")}</span>'
               if _llm_sent.get("reasoning") else "")
            + '</div>',
            unsafe_allow_html=True,
        )
    elif _kron_sig and _items:
        _s2, _ = _kw_sent([i["title"] for i in _items])
        _kd2 = _kron_sig.get("directional_signal", "HOLD")
        _kr2 = _kron_sig.get("predicted_return", 0)
        _aligned = (_kd2 == "BUY" and _s2 == "bullish") or (_kd2 == "SELL" and _s2 == "bearish")
        _diverged = (_kd2 == "BUY" and _s2 == "bearish") or (_kd2 == "SELL" and _s2 == "bullish")
        if _aligned:
            _synth = f"Kronos ({_kr2:+.2%}) and news sentiment both point {_s2} - signals aligned."
        elif _diverged:
            _synth = f"Divergence: Kronos predicts {_kr2:+.2%} but news sentiment is {_s2}. Elevated uncertainty."
        else:
            _synth = f"Kronos predicts {_kr2:+.2%}. News sentiment is {_s2}. No strong signal divergence."
        st.info(f"Kronos + News view (rule-based): {_synth}")

    # ── Headlines ─────────────────────────────────────────────────
    if _items:
        st.markdown(f"**Recent headlines — {_sel_ticker}**")
        for _item in _items:
            _pub = _item.get("published_at", "")
            try:
                if _pub:
                    _pdt = _dt.fromisoformat(str(_pub).replace("Z", "+00:00"))
                    _age = _dt.now(_tz.utc) - _pdt
                    _pub_str = (
                        f"{_age.days}d ago" if _age.days > 0
                        else f"{_age.seconds//3600}h ago" if _age.seconds > 3600
                        else f"{_age.seconds//60}m ago"
                    )
                else:
                    _pub_str = ""
            except Exception:
                _pub_str = str(_pub)[:10]

            _url = _item.get("url", "")
            _title_md = f"[{_item['title']}]({_url})" if _url else _item["title"]
            _meta = " &bull; ".join(filter(None, [_item.get("publisher", ""), _pub_str]))
            _rs, _ = _kw_sent([_item["title"]])
            _rc = C_BULL if _rs == "bullish" else (C_BEAR if _rs == "bearish" else "#444")
            st.markdown(
                f'<div style="border-left:3px solid {_rc};padding:7px 14px;margin:3px 0;'
                f'background:#0a0a0a;border-radius:0 4px 4px 0">'
                f'{_title_md}<br>'
                f'<span style="color:#666;font-size:11px">{_meta}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
    else:
        st.caption(f"No recent headlines found for {_sel_ticker}.")

# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Trade Log
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Trade Log":
    from datetime import datetime as _dt, timezone as _tz

    st.title("Trade Log")

    # ── Flatten all orders from audit records ────────────────────────────────
    _rows = []
    for _rec in records:
        _ts = _rec.get("timestamp", "")
        _date = _ts[:10] if _ts else "—"
        _eq = _rec.get("portfolio_equity", 0.0)
        _fw = _rec.get("final_weights", {})
        for _ord in _rec.get("orders", []):
            _ticker  = _ord.get("ticker", "")
            _side    = _ord.get("side", "")
            _notional = float(_ord.get("notional_usd", 0))
            _status  = _ord.get("status", "")
            _weight  = _fw.get(_ticker, 0.0)
            _rows.append({
                "Date":       _date,
                "Ticker":     _ticker,
                "Side":       _side.upper(),
                "Notional ($)": round(_notional, 2),
                "Target Weight": f"{_weight:.1%}" if _weight else "—",
                "Status":     _status,
                "Equity ($)": round(_eq, 2),
                "Run ID":     _rec.get("run_id", "")[:16],
            })

    if not _rows:
        st.info("No trades recorded yet. Orders will appear here after the first live run cycle.")
        st.stop()

    _tlog = pd.DataFrame(_rows)

    # ── Filters ──────────────────────────────────────────────────────────────
    _fc1, _fc2, _fc3 = st.columns(3)
    _sel_tickers = _fc1.multiselect(
        "Ticker", sorted(_tlog["Ticker"].unique()), default=[]
    )
    _sel_side = _fc2.selectbox("Side", ["All", "BUY", "SELL"])
    _sel_status = _fc3.selectbox("Status", ["All"] + sorted(_tlog["Status"].unique().tolist()))

    _view = _tlog.copy()
    if _sel_tickers:
        _view = _view[_view["Ticker"].isin(_sel_tickers)]
    if _sel_side != "All":
        _view = _view[_view["Side"] == _sel_side]
    if _sel_status != "All":
        _view = _view[_view["Status"] == _sel_status]

    # ── Summary metrics ───────────────────────────────────────────────────────
    _m1, _m2, _m3, _m4 = st.columns(4)
    _m1.metric("Total orders", len(_view))
    _m2.metric("Buys", int((_view["Side"] == "BUY").sum()))
    _m3.metric("Sells", int((_view["Side"] == "SELL").sum()))
    _m4.metric("Total notional", f"${_view['Notional ($)'].sum():,.0f}")

    st.divider()

    # ── Styled table ──────────────────────────────────────────────────────────
    def _color_side(val: str) -> str:
        if val == "BUY":
            return f"color: {C_BULL}; font-weight:600"
        if val == "SELL":
            return f"color: {C_BEAR}; font-weight:600"
        return ""

    _styled = (
        _view.sort_values("Date", ascending=False)
        .reset_index(drop=True)
        .style.applymap(_color_side, subset=["Side"])
        .format({"Notional ($)": "${:,.2f}", "Equity ($)": "${:,.2f}"})
    )
    st.dataframe(_styled, use_container_width=True, height=500)

    # ── Notional by ticker bar chart ──────────────────────────────────────────
    st.subheader("Notional traded per ticker")
    _by_ticker = (
        _view.groupby(["Ticker", "Side"])["Notional ($)"]
        .sum()
        .reset_index()
    )
    _fig_tl = go.Figure()
    for _s, _col in [("BUY", C_BULL), ("SELL", C_BEAR)]:
        _d = _by_ticker[_by_ticker["Side"] == _s]
        if not _d.empty:
            _fig_tl.add_trace(go.Bar(
                x=_d["Ticker"], y=_d["Notional ($)"],
                name=_s, marker_color=_col,
            ))
    _fig_tl.update_layout(
        barmode="group", template="plotly_dark",
        paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(t=20, b=20), height=300,
        legend=dict(orientation="h"),
    )
    st.plotly_chart(_fig_tl, use_container_width=True)

    # ── Activity over time ────────────────────────────────────────────────────
    st.subheader("Order count by date")
    _by_date = _view.groupby("Date").size().reset_index(name="Orders")
    _fig_act = go.Figure(go.Bar(
        x=_by_date["Date"], y=_by_date["Orders"],
        marker_color=C_BLUE,
    ))
    _fig_act.update_layout(
        template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
        margin=dict(t=20, b=20), height=250,
    )
    st.plotly_chart(_fig_act, use_container_width=True)


# ═════════════════════════════════════════════════════════════════════════════
# PAGE: Regime & Risk Monitor
# ═════════════════════════════════════════════════════════════════════════════

elif page == "Regime & Risk":
    import json as _json

    st.title("Regime & Risk Monitor")

    _rc1, _rc2 = st.columns([1, 2])

    # ── Macro regime ──────────────────────────────────────────────────────────
    with _rc1:
        st.subheader("Macro Regime")

        @st.cache_data(ttl=3600)
        def _load_regime_inputs():
            try:
                from data.macro_regime import fetch_regime_inputs, classify_regime
                _df = fetch_regime_inputs(lookback_days=252)
                _df["regime"] = _df.apply(classify_regime, axis=1)
                return _df
            except Exception:
                return pd.DataFrame()

        _rdf = _load_regime_inputs()
        if not _rdf.empty:
            _today_regime = _rdf["regime"].iloc[-1]
            _regime_color = {"risk_on": C_BULL, "transition": C_GOLD, "risk_off": C_BEAR}.get(
                _today_regime, C_NEUT
            )
            st.markdown(
                f'<div style="background:{_regime_color}22;border:1px solid {_regime_color};'
                f'border-radius:8px;padding:16px 20px;text-align:center;margin-bottom:12px">'
                f'<div style="font-size:11px;color:{_regime_color};letter-spacing:1px;text-transform:uppercase">Current Regime</div>'
                f'<div style="font-size:28px;font-weight:700;color:{_regime_color}">'
                f'{_today_regime.replace("_", " ").title()}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            _latest = _rdf.iloc[-1]
            _signals = [
                ("Yield Curve (10Y−3M)", float(_latest.get("yield_spread", 1.0)),
                 -0.5, "% spread", True),
                ("HYG Momentum (vs 20d SMA)", float(_latest.get("hyg_mom", 0.0)) * 100,
                 -2.0, "%", True),
                ("SPY vs 200d SMA", float(_latest.get("spy_vs_200", 0.0)) * 100,
                 -5.0, "%", True),
            ]
            for _label, _val, _thresh, _unit, _higher_good in _signals:
                _bearish = _val < _thresh
                _sc = C_BEAR if _bearish else C_BULL
                _icon = "▼" if _bearish else "▲"
                st.markdown(
                    f'<div style="display:flex;justify-content:space-between;'
                    f'align-items:center;padding:6px 0;border-bottom:1px solid #1e1e1e">'
                    f'<span style="font-size:13px;color:#aaa">{_label}</span>'
                    f'<span style="color:{_sc};font-weight:600">{_icon} {_val:+.2f}{_unit}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Regime history sparkline
            st.markdown("<br>", unsafe_allow_html=True)
            _regime_num = _rdf["regime"].map({"risk_on": 1, "transition": 0, "risk_off": -1})
            _fig_reg = go.Figure(go.Scatter(
                x=_rdf.index, y=_regime_num,
                mode="lines", fill="tozeroy",
                line=dict(color=C_BLUE, width=1),
                fillcolor=f"{C_BLUE}22",
            ))
            _fig_reg.add_hline(y=0, line_color=C_GOLD, line_dash="dot", line_width=1)
            _fig_reg.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(t=10, b=10, l=0, r=0), height=120,
                yaxis=dict(tickvals=[-1, 0, 1],
                           ticktext=["Risk Off", "Transition", "Risk On"],
                           showgrid=False),
                xaxis=dict(showgrid=False),
                showlegend=False,
            )
            st.plotly_chart(_fig_reg, use_container_width=True)
        else:
            st.warning("Regime data unavailable — check network access to yfinance.")

    # ── Live risk state ───────────────────────────────────────────────────────
    with _rc2:
        st.subheader("Portfolio Risk State")

        _risk_path = Path("data/risk_state.json")
        if _risk_path.exists():
            try:
                _rs = _json.loads(_risk_path.read_text())
            except Exception:
                _rs = {}
        else:
            _rs = {}

        _live_eq = _alpaca.get("equity") if (_alpaca and "error" not in _alpaca) else None
        _peak_eq = _rs.get("peak_equity", 0.0)
        _cb_active = _rs.get("circuit_active", False)

        _rm1, _rm2, _rm3 = st.columns(3)
        _rm1.metric("Peak equity", f"${_peak_eq:,.2f}" if _peak_eq else "—")
        if _live_eq and _peak_eq:
            _dd_now = (_live_eq / _peak_eq - 1) * 100
            _rm2.metric("Current drawdown", f"{_dd_now:.2f}%",
                        delta=f"{_dd_now:.2f}%",
                        delta_color="inverse")
        else:
            _rm2.metric("Current drawdown", "—")
        _cb_label = "ACTIVE" if _cb_active else "Off"
        _cb_color = C_BEAR if _cb_active else C_BULL
        st.markdown(
            f'<div style="padding:8px 0"><span style="color:#aaa;font-size:13px">Circuit Breaker: </span>'
            f'<span style="color:{_cb_color};font-weight:700">{_cb_label}</span></div>',
            unsafe_allow_html=True,
        )
        _rm3.metric("Open positions", len(_alpaca.get("positions", {})) if _alpaca else "—")

        # Price peaks / stop proximity
        _peaks = _rs.get("price_peaks", {})
        if _peaks and _alpaca and "positions" in _alpaca:
            st.markdown("**Trailing stop proximity** — distance from peak price")
            _stop_rows = []
            for _pos in _alpaca.get("positions", {}).values():
                _t = _pos.get("symbol") or _pos.get("ticker", "")
                if _t in _peaks:
                    _px = float(_pos.get("current_price") or _pos.get("avg_entry_price", 0))
                    _pk = float(_peaks[_t])
                    _drawdown = (_px / _pk - 1) * 100 if _pk else 0
                    _stop_rows.append({
                        "Ticker": _t,
                        "Current": _px,
                        "Peak": _pk,
                        "Drawdown from Peak": f"{_drawdown:.2f}%",
                    })
            if _stop_rows:
                _spdf = pd.DataFrame(_stop_rows).sort_values("Drawdown from Peak")
                st.dataframe(_spdf, use_container_width=True, hide_index=True)
            else:
                st.caption("No peak data for current positions.")
        elif not _peaks:
            st.caption("Risk state will populate after the first live run.")

        # ── VIX gauge ─────────────────────────────────────────────────────────
        st.subheader("VIX")

        @st.cache_data(ttl=1800)
        def _fetch_vix_hist():
            try:
                _raw = yf.download("^VIX", period="6mo", progress=False, auto_adjust=True)
                _s = _raw["Close"].squeeze().dropna()
                return _s
            except Exception:
                return pd.Series(dtype=float)

        _vix_hist = _fetch_vix_hist()
        if not _vix_hist.empty:
            _vix_now = float(_vix_hist.iloc[-1])
            _vix_color = C_BEAR if _vix_now > 30 else (C_GOLD if _vix_now > 20 else C_BULL)
            _vc1, _vc2 = st.columns([1, 3])
            _vc1.markdown(
                f'<div style="text-align:center;padding-top:10px">'
                f'<div style="font-size:36px;font-weight:700;color:{_vix_color}">{_vix_now:.1f}</div>'
                f'<div style="font-size:11px;color:#aaa">Current VIX</div>'
                f'<div style="font-size:11px;color:#666;margin-top:4px">'
                f'Shorts gate: {"open" if _vix_now > 25 else "closed"}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            _fig_vix = go.Figure(go.Scatter(
                x=_vix_hist.index, y=_vix_hist.values,
                mode="lines", line=dict(color=_vix_color, width=1.5),
                fill="tozeroy", fillcolor=f"{_vix_color}22",
            ))
            _fig_vix.add_hline(y=25, line_color=C_GOLD, line_dash="dot",
                               line_width=1, annotation_text="shorts gate (25)")
            _fig_vix.add_hline(y=30, line_color=C_BEAR, line_dash="dot",
                               line_width=1, annotation_text="high vol (30)")
            _fig_vix.update_layout(
                template="plotly_dark", paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                margin=dict(t=10, b=10, l=0, r=0), height=150,
                yaxis=dict(showgrid=False), xaxis=dict(showgrid=False),
                showlegend=False,
            )
            _vc2.plotly_chart(_fig_vix, use_container_width=True)
        else:
            st.caption("VIX data unavailable.")
