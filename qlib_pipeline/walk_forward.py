"""Walk-forward validation: sequential OOS backtests across non-overlapping windows."""

from __future__ import annotations

from datetime import date

import pandas as pd
from dateutil.relativedelta import relativedelta


def walk_forward(
    tickers: list[str],
    full_start: date,
    full_end: date,
    window_months: int = 2,
    progress_cb=None,   # callable(current: int, total: int, label: str)
    **backtest_kwargs,
) -> pd.DataFrame:
    from qlib_pipeline.backtest import Backtester

    # Pre-count windows so progress bar has a denominator
    _cur, _total = full_start, 0
    while _cur < full_end:
        _end = min(_cur + relativedelta(months=window_months), full_end)
        if (_end - _cur).days < 20:
            break
        _total += 1
        _cur = _end

    windows: list[dict] = []
    current = full_start
    idx = 0
    while current < full_end:
        window_end = min(current + relativedelta(months=window_months), full_end)
        if (window_end - current).days < 20:
            break
        if progress_cb:
            progress_cb(idx, _total, f"Window {idx+1}/{_total}: {current} → {window_end}")
        bt = Backtester(tickers=tickers, start_date=current, end_date=window_end,
                        **backtest_kwargs)
        bt.run()
        s = bt.summary()
        windows.append({
            "start":        str(current),
            "end":          str(window_end),
            "total_return": s["total_return"],
            "sharpe_ratio": s["sharpe_ratio"],
            "max_drawdown": s["max_drawdown"],
            "win_rate":     s["win_rate"],
            "trading_days": int(s["trading_days"]),
        })
        current = window_end
        idx += 1
    if progress_cb:
        progress_cb(_total, _total, "Complete")

    if not windows:
        return pd.DataFrame()

    df = pd.DataFrame(windows)

    numeric = ["total_return", "sharpe_ratio", "max_drawdown", "win_rate", "trading_days"]
    means = df[numeric].mean()
    stds  = df[numeric].std()

    _fmt_row(df, "Window results")
    print()
    print(f"{'Mean':>10}  " + "  ".join(f"{means[c]:>+.2%}" if "return" in c or "drawdown" in c or "rate" in c
                                         else f"{means[c]:>6.2f}" for c in numeric))
    print(f"{'Std':>10}  " + "  ".join(f"{stds[c]:>+.2%}"  if "return" in c or "drawdown" in c or "rate" in c
                                         else f"{stds[c]:>6.2f}" for c in numeric))
    print()
    print(f"Consistency (% windows positive return): "
          f"{(df['total_return'] > 0).mean():.0%}  "
          f"({(df['total_return'] > 0).sum()}/{len(df)} windows)")

    summary_row = {"start": "MEAN", "end": "", **{c: means[c] for c in numeric}}
    return pd.concat([df, pd.DataFrame([summary_row])], ignore_index=True)


def _fmt_row(df: pd.DataFrame, title: str) -> None:
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}")
    header = f"  {'Start':<12} {'End':<12} {'Return':>8} {'Sharpe':>7} {'MaxDD':>8} {'WinRate':>8} {'Days':>5}"
    print(header)
    print(f"  {'─'*65}")
    for _, r in df.iterrows():
        print(
            f"  {str(r['start']):<12} {str(r['end']):<12} "
            f"{r['total_return']:>+8.2%} {r['sharpe_ratio']:>7.2f} "
            f"{r['max_drawdown']:>+8.2%} {r['win_rate']:>8.2%} {int(r['trading_days']):>5}"
        )
    print(f"{'─'*70}")


if __name__ == "__main__":
    from datetime import date as _date
    TICKERS = ["AAPL", "MSFT", "NVDA", "GOOGL", "META", "AMZN", "TSLA", "V", "JPM", "SPY"]
    walk_forward(
        tickers=TICKERS,
        full_start=_date(2025, 1, 1),
        full_end=_date(2026, 6, 1),
        window_months=2,
        kronos_model_size="tiny",
        initial_cash=10_000,
        run_agents=False,
        momentum_blend=0.80,
        top_n=5,
    )
