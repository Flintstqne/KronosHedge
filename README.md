# KronosHedge

Quantitative long/short equity system. Combines Amazon Chronos time-series forecasting, cross-sectional momentum, PEAD signals, and a multi-agent LLM pipeline into a daily rebalanced portfolio. Executes via Alpaca (paper or live).

**Backtest (Jun 2025 – Jun 2026):** +32.7% return, 1.80 Sharpe, -9.2% max drawdown vs SPY +27.0%.

---

## Prerequisites

- Python 3.11+
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/) package manager
- [Alpaca](https://alpaca.markets/) account (paper trading is free)
- [Anthropic](https://console.anthropic.com/) API key (for LLM agent pipeline)
- Optional: [FRED API key](https://fred.stlouisfed.org/docs/api/api_key.html) for macro data enrichment

---

## Installation

```bash
git clone https://github.com/Flintstqne/KronosHedge.git
cd KronosHedge

# Install dependencies
uv sync

# Copy and fill in your credentials
cp .env.example .env
$EDITOR .env
```

`.env` requires at minimum:
```
ANTHROPIC_API_KEY=sk-ant-...
ALPACA_API_KEY=PK...
ALPACA_SECRET_KEY=...
```

---

## Configuration

All strategy parameters live in `config/settings.yaml`. Key sections:

| Section | What it controls |
|---------|-----------------|
| `universe.tickers` | Stock universe (31 large-caps by default) |
| `alpha` | Momentum blend (80%), top-N concentration (10), PEAD boost |
| `risk` | Trailing stop floor (7%), circuit breaker (7%), cooldown (2 days), VIX threshold (25) |
| `regime` | Macro regime thresholds (yield curve, HYG momentum, SPY 200d trend) |
| `execution` | Alpaca paper/live, max position size, intraday signal time |
| `agents` | Which LLM agents run per cycle |

Set `execution.paper: false` and ensure `ALPACA_PAPER=false` in `.env` before live trading.

---

## Usage

### Backtest
```bash
uv run python -c "
from qlib_pipeline.backtest import Backtester
from datetime import date
import yaml

cfg = yaml.safe_load(open('config/settings.yaml'))
bt = Backtester(
    tickers=cfg['universe']['tickers'],
    start_date=date(2025, 1, 1),
    end_date=date(2026, 6, 1),
    kronos_model_size='tiny',
    initial_cash=10_000,
    run_agents=False,
    **{k: cfg['alpha'][k] for k in ('momentum_blend','top_n','short_n','pead_boost')},
    **{k: cfg['risk'][k] for k in ('trailing_stop','drawdown_stop','recovery_threshold',
                                    'cash_reserve_pct','spy_reserve','stop_cooldown_days')},
    vix_threshold=cfg['risk']['vix_threshold'],
)
bt.run()
print(bt.summary())
"
```

Use `kronos_model_size='small'` for better signal quality (slower). `'tiny'` is fast for development.

### Walk-Forward Validation
```bash
uv run python -c "
from qlib_pipeline.walk_forward import walk_forward
from datetime import date
import yaml

cfg = yaml.safe_load(open('config/settings.yaml'))
walk_forward(
    tickers=cfg['universe']['tickers'],
    full_start=date(2024, 1, 1),
    full_end=date(2026, 6, 1),
    window_months=2,
    kronos_model_size='tiny',
    run_agents=False,
    momentum_blend=0.80,
    top_n=10,
)
"
```

### Live Run Cycle (daily, paper)
```bash
# Single run — daily signals at market close
uv run python main.py --dry-run        # verify without placing orders
uv run python main.py                  # place orders via Alpaca

# Intraday mode — signals at 10am ET, prices updated live
uv run python main.py --mode intraday
```

### Dashboard
```bash
uv run streamlit run monitoring/dashboard.py
# Opens at http://localhost:8501
```

### Check current macro regime
```bash
uv run python -m data.macro_regime
```

---

## Automated Scheduling

Add to crontab (`crontab -e`) to run daily after market close:

```cron
# Daily close — Mon-Fri at 4:05pm ET
5 16 * * 1-5 cd /path/to/KronosHedge && source .venv/bin/activate && python main.py >> logs/run_live.log 2>&1

# OR: intraday mode at 10:00am ET
0 10 * * 1-5 cd /path/to/KronosHedge && source .venv/bin/activate && python main.py --mode intraday >> logs/run_live.log 2>&1
```

---

## Architecture

```
main.py                   — live run cycle (daily + intraday modes)
config/settings.yaml      — all strategy parameters

qlib_pipeline/
  backtest.py             — VirtualPortfolio, Backtester, full simulation engine
  alpha.py                — KronosAlphaFactor: momentum blend + PEAD + short generation
  walk_forward.py         — sequential OOS validation across rolling windows

kronos_bridge/            — wraps Amazon Chronos (chronos-t5-tiny/small) for forecasts
agents/                   — LLM agent pipeline (technicals, valuation, sentiment, …)
reconciliation/           — blends Kronos weights with agent decisions
execution/                — Alpaca broker adapter + order executor

data/
  news.py                 — earnings calendar, VIX, PEAD signals, FOMC calendar
  macro_regime.py         — yield curve + credit spread + trend regime classifier
  intraday.py             — live 5-min bar fetch, market hours utilities

monitoring/
  dashboard.py            — Streamlit dashboard (backtest, live P&L, signals, news)
  alerts.py               — Discord / webhook / email alerts on drawdown + trades
```

### Signal flow (live cycle)
```
OHLCV (420 days) → Kronos inference → momentum + PEAD scores
                 → LLM agents → reconcile (60% quant / 40% agent)
                 → macro risk filter → VIX filter → regime filter
                 → trailing stops → circuit breaker → Alpaca execute
```

---

## Risk state

On first run `data/risk_state.json` is created automatically, bootstrapping the circuit breaker peak to current equity. Delete it to reset risk tracking (e.g. after depositing funds).

---

## Development

```bash
uv sync --extra dev
uv run pytest tests/
```
