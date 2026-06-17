"""
News, earnings, and macro event data.
All free — yfinance for company news/earnings, hardcoded FOMC calendar.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import TypedDict

import yfinance as yf

# ── FOMC announcement dates 2025-2026 ────────────────────────────────────────
# Source: federalreserve.gov — the second day of each two-day meeting.
FOMC_DATES: list[date] = [
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7),
    date(2025, 6, 18), date(2025, 7, 30), date(2025, 9, 17),
    date(2025, 10, 29), date(2025, 12, 10),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 5, 6),
    date(2026, 6, 17), date(2026, 7, 29), date(2026, 9, 16),
    date(2026, 10, 28), date(2026, 12, 9),
]

# Positive / negative word lists for lightweight dashboard sentiment scoring
_POS_WORDS = {
    "beat", "beats", "record", "surge", "surges", "soar", "soars", "rally",
    "bullish", "upgrade", "upgraded", "outperform", "profit", "growth",
    "strong", "boost", "raised", "raise", "positive", "exceeds", "exceed",
    "buy", "higher", "gain", "gains", "top", "wins", "win",
}
_NEG_WORDS = {
    "miss", "misses", "missed", "fall", "falls", "drop", "drops", "decline",
    "bearish", "downgrade", "downgraded", "underperform", "loss", "losses",
    "weak", "cut", "cuts", "negative", "below", "warning", "sell", "lower",
    "concern", "concerns", "debt", "layoff", "layoffs", "recall", "fine",
}


# ── TypedDicts ────────────────────────────────────────────────────────────────

class NewsItem(TypedDict):
    title: str
    publisher: str
    url: str
    published_at: str   # ISO string (may be epoch int as str if raw)


class EarningsEvent(TypedDict):
    ticker: str
    earnings_date: str  # ISO date
    days_away: int


class MacroEvent(TypedDict):
    event: str
    date: str           # ISO date
    days_away: int
    risk_level: str     # "high" | "medium"


class NewsContext(TypedDict):
    ticker_news: dict[str, list[NewsItem]]
    earnings: list[EarningsEvent]
    macro_events: list[MacroEvent]
    as_of: str


# ── Fetchers ──────────────────────────────────────────────────────────────────

def fetch_ticker_news(ticker: str, max_items: int = 10) -> list[NewsItem]:
    try:
        raw = yf.Ticker(ticker).news or []
        items: list[NewsItem] = []
        for n in raw[:max_items]:
            # yfinance ≥0.2.40 wraps in a 'content' key; older versions don't
            content = n.get("content", n)
            title = (
                content.get("title")
                or content.get("headline")
                or n.get("title", "")
            )
            publisher = (
                content.get("provider", {}).get("displayName", "")
                or content.get("publisher", "")
                or n.get("publisher", "")
            )
            url = (
                content.get("canonicalUrl", {}).get("url", "")
                or content.get("url", "")
                or n.get("link", "")
            )
            pub_at = content.get("pubDate") or n.get("providerPublishTime", "")
            # epoch int → ISO string
            if isinstance(pub_at, (int, float)):
                pub_at = datetime.fromtimestamp(pub_at, tz=timezone.utc).isoformat()
            items.append({
                "title": str(title),
                "publisher": str(publisher),
                "url": str(url),
                "published_at": str(pub_at),
            })
        return [i for i in items if i["title"]]
    except Exception:
        return []


def fetch_earnings_calendar(tickers: list[str], window_days: int = 14) -> list[EarningsEvent]:
    today = date.today()
    events: list[EarningsEvent] = []
    for ticker in tickers:
        try:
            cal = yf.Ticker(ticker).calendar
            if not cal:
                continue
            ed = cal.get("Earnings Date")
            if ed is None:
                continue
            candidates = list(ed) if hasattr(ed, "__iter__") and not isinstance(ed, str) else [ed]
            for dt in candidates:
                d = dt.date() if hasattr(dt, "date") else dt
                if not isinstance(d, date):
                    continue
                days_away = (d - today).days
                if -3 <= days_away <= window_days:
                    events.append({
                        "ticker": ticker,
                        "earnings_date": d.isoformat(),
                        "days_away": days_away,
                    })
        except Exception:
            pass
    return sorted(events, key=lambda e: e["days_away"])


def upcoming_macro_events(window_days: int = 21) -> list[MacroEvent]:
    today = date.today()
    events: list[MacroEvent] = []
    for fomc_date in FOMC_DATES:
        days_away = (fomc_date - today).days
        if -1 <= days_away <= window_days:
            events.append({
                "event": "FOMC Rate Decision",
                "date": fomc_date.isoformat(),
                "days_away": days_away,
                "risk_level": "high",
            })
    return sorted(events, key=lambda e: e["days_away"])


class FedOutlook(TypedDict):
    current_rate: float | None       # Fed Funds target midpoint, e.g. 4.375
    tbill_3m: float | None           # 3-month T-bill yield (market expectation proxy)
    cpi_yoy: float | None            # latest CPI year-over-year %
    unemployment: float | None       # latest unemployment rate %
    implied_move: str                # "hike" | "cut" | "hold" | "unknown"
    implied_bps: int                 # estimated basis points of move (0 if hold)
    confidence: str                  # "high" | "medium" | "low"
    rationale: str                   # one-sentence explanation


def fetch_fed_outlook() -> FedOutlook:
    """
    Infer the likely FOMC outcome from publicly available data.

    Primary source (no API key): 3-month T-bill yield vs implied Fed Funds rate.
    Enriched source (optional FRED_API_KEY env var): exact current rate, CPI, unemployment.
    """
    import os

    current_rate: float | None = None
    cpi_yoy: float | None = None
    unemployment: float | None = None
    tbill_3m: float | None = None

    # ── FRED (optional) ────────────────────────────────────────────────────────
    fred_key = os.getenv("FRED_API_KEY", "")
    if fred_key:
        import urllib.request, json as _json

        def _fred(series: str) -> float | None:
            url = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id={series}&api_key={fred_key}"
                f"&limit=1&sort_order=desc&file_type=json"
            )
            try:
                with urllib.request.urlopen(url, timeout=6) as r:
                    data = _json.loads(r.read())
                val = data["observations"][0]["value"]
                return float(val) if val != "." else None
            except Exception:
                return None

        current_rate = _fred("DFEDTARU")  # upper bound of target range
        lower = _fred("DFEDTARL")         # lower bound
        if current_rate is not None and lower is not None:
            current_rate = (current_rate + lower) / 2   # midpoint
        cpi_yoy = _fred("CPIAUCSL")       # level — we'll compute YoY below
        # Fetch two CPI readings to compute YoY ourselves
        try:
            url2 = (
                f"https://api.stlouisfed.org/fred/series/observations"
                f"?series_id=CPIAUCSL&api_key={fred_key}"
                f"&limit=13&sort_order=desc&file_type=json"
            )
            with urllib.request.urlopen(url2, timeout=6) as r:
                data2 = _json.loads(r.read())
            obs = [o for o in data2["observations"] if o["value"] != "."]
            if len(obs) >= 13:
                latest_cpi = float(obs[0]["value"])
                year_ago_cpi = float(obs[12]["value"])
                cpi_yoy = (latest_cpi - year_ago_cpi) / year_ago_cpi * 100
            else:
                cpi_yoy = None
        except Exception:
            cpi_yoy = None
        unemployment = _fred("UNRATE")

    # ── T-bill yield from yfinance (always available, no key needed) ───────────
    try:
        raw = yf.download("^IRX", period="5d", progress=False, auto_adjust=True)
        closes = raw["Close"] if "Close" in raw.columns else raw.iloc[:, 0]
        if isinstance(closes, __import__("pandas").DataFrame):
            closes = closes.iloc[:, 0]
        closes = closes.dropna()
        if not closes.empty:
            tbill_3m = float(closes.iloc[-1])   # annualised %, e.g. 4.32
    except Exception:
        pass

    # ── Infer current rate from T-bill if FRED unavailable ────────────────────
    if current_rate is None and tbill_3m is not None:
        # 3-month T-bill trades close to the Fed Funds effective rate
        current_rate = round(tbill_3m, 2)

    # ── Derive implied move ────────────────────────────────────────────────────
    implied_move = "unknown"
    implied_bps = 0
    confidence = "low"
    rationale = "Insufficient data to project FOMC outcome."

    if current_rate is not None and tbill_3m is not None:
        spread = tbill_3m - current_rate   # positive = market pricing in hike

        if spread > 0.30:
            implied_move, implied_bps, confidence = "hike", 50, "medium"
        elif spread > 0.15:
            implied_move, implied_bps, confidence = "hike", 25, "medium"
        elif spread < -0.30:
            implied_move, implied_bps, confidence = "cut", 50, "medium"
        elif spread < -0.15:
            implied_move, implied_bps, confidence = "cut", 25, "medium"
        else:
            implied_move, implied_bps, confidence = "hold", 0, "medium"

        # Upgrade confidence when CPI + unemployment add corroborating signal
        if cpi_yoy is not None and unemployment is not None:
            if implied_move == "cut" and cpi_yoy < 2.5 and unemployment > 4.5:
                confidence = "high"
            elif implied_move == "hold" and 2.0 <= cpi_yoy <= 3.0:
                confidence = "high"
            elif implied_move == "hike" and cpi_yoy > 3.5:
                confidence = "high"

        # Build rationale sentence
        cpi_str = f"CPI {cpi_yoy:.1f}%  " if cpi_yoy is not None else ""
        unemp_str = f"unemployment {unemployment:.1f}%  " if unemployment is not None else ""
        spread_str = f"T-bill spread {spread:+.2f}%"
        if implied_move == "hold":
            rationale = f"Market pricing a hold. {cpi_str}{unemp_str}{spread_str}."
        elif implied_move == "cut":
            rationale = (
                f"Market implies a {implied_bps}bp cut. {cpi_str}{unemp_str}{spread_str}."
            )
        else:
            rationale = (
                f"Market implies a {implied_bps}bp hike. {cpi_str}{unemp_str}{spread_str}."
            )

    return FedOutlook(
        current_rate=current_rate,
        tbill_3m=tbill_3m,
        cpi_yoy=cpi_yoy,
        unemployment=unemployment,
        implied_move=implied_move,
        implied_bps=implied_bps,
        confidence=confidence,
        rationale=rationale,
    )


def fetch_vix() -> float:
    """Latest VIX close from yfinance. Returns 20.0 (neutral) on failure."""
    try:
        raw = yf.download("^VIX", period="2d", progress=False, auto_adjust=True)
        return float(raw["Close"].squeeze().dropna().iloc[-1])
    except Exception:
        return 20.0


def fetch_pead_signals(tickers: list[str], lookback_days: int = 60) -> dict[str, float]:
    """
    Post-earnings announcement drift: EPS surprise magnitude for tickers that
    beat estimates within the last lookback_days. Free via yfinance earnings history.
    Returns {ticker: surprise_pct capped at 1.0}. Absent = no recent beat.
    """
    cutoff = date.today() - timedelta(days=lookback_days)
    signals: dict[str, float] = {}
    import pandas as pd
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).get_earnings_dates(limit=8)
            if df is None or df.empty:
                continue
            df = df.dropna(subset=["EPS Estimate", "Reported EPS"])
            df.index = pd.to_datetime(df.index).tz_localize(None)
            recent = df[df.index.date >= cutoff]
            if recent.empty:
                continue
            row = recent.iloc[0]
            est, rep = float(row["EPS Estimate"]), float(row["Reported EPS"])
            if abs(est) < 0.01:
                continue
            surprise = (rep - est) / abs(est)
            if surprise > 0:
                signals[ticker] = min(surprise, 1.0)
        except Exception:
            pass
    return signals


def build_news_context(tickers: list[str]) -> NewsContext:
    ticker_news: dict[str, list[NewsItem]] = {t: fetch_ticker_news(t) for t in tickers}
    return {
        "ticker_news": ticker_news,
        "earnings": fetch_earnings_calendar(tickers),
        "macro_events": upcoming_macro_events(),
        "as_of": datetime.now(timezone.utc).isoformat(),
    }


# ── Keyword sentiment ─────────────────────────────────────────────────────────

def keyword_sentiment(headlines: list[str]) -> tuple[str, float]:
    """
    Fast, dependency-free sentiment from word matching.
    Returns (signal, score) where signal is 'bullish'|'bearish'|'neutral'
    and score is -1..+1.
    """
    pos = neg = 0
    for h in headlines:
        words = set(h.lower().split())
        pos += len(words & _POS_WORDS)
        neg += len(words & _NEG_WORDS)
    total = pos + neg or 1
    score = (pos - neg) / total
    if score > 0.15:
        return "bullish", score
    if score < -0.15:
        return "bearish", score
    return "neutral", score


# ── Macro risk filter ─────────────────────────────────────────────────────────

def apply_macro_risk(
    weights: dict[str, float],
    news_context: NewsContext,
    fomc_window_days: int = 2,
    earnings_window_days: int = 2,
) -> dict[str, float]:
    """
    Shrinks position sizes near high-risk events:
      - FOMC in <= fomc_window_days  → scale all weights by 0.65
      - Earnings in <= earnings_window_days → scale that ticker by 0.50
    """
    result = dict(weights)

    fomc_imminent = any(
        0 <= ev["days_away"] <= fomc_window_days
        for ev in news_context.get("macro_events", [])
        if ev["risk_level"] == "high"
    )
    if fomc_imminent:
        result = {t: w * 0.65 for t, w in result.items()}

    for ev in news_context.get("earnings", []):
        if 0 <= ev["days_away"] <= earnings_window_days and ev["ticker"] in result:
            result[ev["ticker"]] *= 0.50

    return result


# ── NewsFetcher convenience class ─────────────────────────────────────────────

class NewsFetcher:
    """Thin wrapper so callers can instantiate once and call methods."""

    def __init__(self, tickers: list[str]):
        self.tickers = tickers

    def build_context(self) -> NewsContext:
        return build_news_context(self.tickers)

    def macro_events(self) -> list[MacroEvent]:
        return upcoming_macro_events()

    def earnings(self) -> list[EarningsEvent]:
        return fetch_earnings_calendar(self.tickers)
