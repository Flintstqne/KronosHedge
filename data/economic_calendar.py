"""Upcoming major US economic events. Hardcoded from official schedules."""

from datetime import date, timedelta

# Federal Reserve FOMC decision dates (remaining 2026 — from Fed's published calendar)
FOMC_DECISIONS = [
    "2026-07-29",
    "2026-09-16",
    "2026-10-28",
    "2026-12-16",
]

# BLS CPI release dates (remaining 2026 — typically 2nd or 3rd Wed of month)
CPI_RELEASES = [
    "2026-07-15",
    "2026-08-12",
    "2026-09-10",
    "2026-10-14",
    "2026-11-12",
    "2026-12-10",
]

# BLS Non-Farm Payrolls (first Friday of month — remaining 2026)
NFP_RELEASES = [
    "2026-07-02",
    "2026-08-07",
    "2026-09-04",
    "2026-10-02",
    "2026-11-06",
    "2026-12-04",
]


def upcoming_events(days_ahead: int = 60) -> list[dict]:
    today = date.today()
    cutoff = today + timedelta(days=days_ahead)
    events: list[dict] = []

    for d in FOMC_DECISIONS:
        dt = date.fromisoformat(d)
        if today <= dt <= cutoff:
            events.append({"date": d, "type": "FOMC", "description": "Fed rate decision"})

    for d in CPI_RELEASES:
        dt = date.fromisoformat(d)
        if today <= dt <= cutoff:
            events.append({"date": d, "type": "CPI", "description": "US CPI release (BLS)"})

    for d in NFP_RELEASES:
        dt = date.fromisoformat(d)
        if today <= dt <= cutoff:
            events.append({"date": d, "type": "NFP", "description": "Non-Farm Payrolls (BLS)"})

    events.sort(key=lambda x: x["date"])
    return events
