"""Fetch recent Form 4 insider transactions from SEC EDGAR."""

from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path

import httpx
import yaml

INSIDER_PATH = Path("logs/insider_data.json")
_HEADERS = {"User-Agent": "kronos-hedge contact@kronoshedge.local"}
_CIK_URL = "https://www.sec.gov/files/company_tickers.json"
_SUBS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"

_cik_cache: dict[str, str] = {}


def _cik_map() -> dict[str, str]:
    global _cik_cache
    if _cik_cache:
        return _cik_cache
    try:
        r = httpx.get(_CIK_URL, headers=_HEADERS, timeout=15)
        _cik_cache = {
            v["ticker"]: str(v["cik_str"]).zfill(10)
            for v in r.json().values()
        }
    except Exception:
        pass
    return _cik_cache


def fetch_insider_transactions(tickers: list[str], days_back: int = 60) -> list[dict]:
    cik = _cik_map()
    cutoff = date.today() - timedelta(days=days_back)
    rows: list[dict] = []

    for ticker in tickers:
        cik_num = cik.get(ticker)
        if not cik_num:
            continue
        try:
            r = httpx.get(_SUBS_URL.format(cik=cik_num), headers=_HEADERS, timeout=10)
            recent = r.json().get("filings", {}).get("recent", {})
            forms       = recent.get("form", [])
            filed_dates = recent.get("filingDate", [])
            accessions  = recent.get("accessionNumber", [])

            for form, filed, acc in zip(forms, filed_dates, accessions):
                if form != "4":
                    continue
                if date.fromisoformat(filed) < cutoff:
                    continue
                acc_clean = acc.replace("-", "")
                rows.append({
                    "ticker": ticker,
                    "filed":  filed,
                    "url":    f"https://www.sec.gov/Archives/edgar/data/{int(cik_num)}/{acc_clean}/{acc}-index.htm",
                })
        except Exception:
            pass

    rows.sort(key=lambda x: x["filed"], reverse=True)
    return rows


def run() -> dict:
    cfg = yaml.safe_load(open("config/settings.yaml"))
    txns = fetch_insider_transactions(cfg["universe"]["tickers"])
    result = {"date": str(date.today()), "transactions": txns}
    INSIDER_PATH.parent.mkdir(exist_ok=True)
    INSIDER_PATH.write_text(json.dumps(result, indent=2))
    print(f"Insider: {len(txns)} Form 4 filings found")
    return result


if __name__ == "__main__":
    run()
