"""
Sector-balanced post-filter for portfolio weights.

After the reconciler produces raw weights, this trims over-concentration:
  1. Within each sector, keep only the top `max_per_sector` positions by weight.
  2. Cap any sector's total allocation to `max_sector_pct` of the portfolio.
  3. Re-normalize so weights still sum to 1.
"""

from __future__ import annotations

SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL":  "Technology",
    "MSFT":  "Technology",
    "NVDA":  "Technology",
    "GOOGL": "Technology",
    "META":  "Technology",
    "AVGO":  "Technology",
    # Consumer Discretionary
    "AMZN":  "ConsumerDiscretionary",
    "TSLA":  "ConsumerDiscretionary",
    "HD":    "ConsumerDiscretionary",
    "MCD":   "ConsumerDiscretionary",
    # Healthcare
    "UNH":   "Healthcare",
    "LLY":   "Healthcare",
    "JNJ":   "Healthcare",
    "ABBV":  "Healthcare",
    # Financials
    "JPM":   "Financials",
    "V":     "Financials",
    "MA":    "Financials",
    "GS":    "Financials",
    # Energy
    "XOM":   "Energy",
    "CVX":   "Energy",
    # Industrials
    "CAT":   "Industrials",
    "RTX":   "Industrials",
    "HON":   "Industrials",
    # Consumer Staples
    "COST":  "ConsumerStaples",
    "WMT":   "ConsumerStaples",
    "PG":    "ConsumerStaples",
    # Communication Services
    "NFLX":  "Communication",
    "T":     "Communication",
    # Materials
    "LIN":   "Materials",
    # Utilities
    "NEE":   "Utilities",
}


def apply_sector_filter(
    weights: dict[str, float],
    sector_map: dict[str, str] | None = None,
    max_sector_pct: float = 0.40,
    max_per_sector: int = 4,
) -> dict[str, float]:
    """
    Return a sector-balanced copy of `weights`.

    Tickers not in `sector_map` pass through without modification.
    """
    smap = sector_map or SECTOR_MAP

    # Separate positive weights by sector
    by_sector: dict[str, list[tuple[str, float]]] = {}
    passthrough: dict[str, float] = {}
    for ticker, w in weights.items():
        if w <= 0:
            continue
        sector = smap.get(ticker)
        if sector is None:
            passthrough[ticker] = w
        else:
            by_sector.setdefault(sector, []).append((ticker, w))

    # Step 1: keep only top max_per_sector per sector
    kept: dict[str, float] = dict(passthrough)
    for pairs in by_sector.values():
        pairs.sort(key=lambda x: x[1], reverse=True)
        for ticker, w in pairs[:max_per_sector]:
            kept[ticker] = w

    total = sum(kept.values()) or 1.0

    # Step 2: cap sector share at max_sector_pct
    sector_totals: dict[str, float] = {}
    for ticker, w in kept.items():
        s = smap.get(ticker, "__none__")
        sector_totals[s] = sector_totals.get(s, 0.0) + w

    capped: dict[str, float] = {}
    for ticker, w in kept.items():
        s = smap.get(ticker, "__none__")
        s_total = sector_totals[s]
        s_share = s_total / total
        if s_share > max_sector_pct:
            capped[ticker] = w * (max_sector_pct * total / s_total)
        else:
            capped[ticker] = w

    # Step 3: re-normalize
    total_capped = sum(capped.values()) or 1.0
    return {t: w / total_capped for t, w in capped.items()}
