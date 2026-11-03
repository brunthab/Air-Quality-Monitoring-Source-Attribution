"""Estimated pollution score from CPCB-style sub-index breakpoints.

With only PM2.5, PM10 and CO available this is an *estimated pollution score*,
not an official CPCB AQI (which needs more pollutants and averaging rules).
"""
import numpy as np

# (conc_low, conc_high, index_low, index_high)
_BREAKS = {
    "pm25": [(0, 30, 0, 50), (30, 60, 51, 100), (60, 90, 101, 200),
             (90, 120, 201, 300), (120, 250, 301, 400), (250, 500, 401, 500)],
    "pm10": [(0, 50, 0, 50), (50, 100, 51, 100), (100, 250, 101, 200),
             (250, 350, 201, 300), (350, 430, 301, 400), (430, 800, 401, 500)],
    # CO in mg/m3
    "co": [(0, 1, 0, 50), (1, 2, 51, 100), (2, 10, 101, 200),
           (10, 17, 201, 300), (17, 34, 301, 400), (34, 60, 401, 500)],
}

CATEGORIES = [(50, "Good"), (100, "Satisfactory"), (200, "Moderate"),
              (300, "Poor"), (400, "Very Poor"), (10_000, "Severe")]


def sub_index(pollutant: str, value: float) -> float:
    if value is None or np.isnan(value):
        return np.nan
    for lo, hi, ilo, ihi in _BREAKS[pollutant]:
        if value <= hi:
            return ilo + (ihi - ilo) * (max(value, lo) - lo) / (hi - lo)
    return 500.0


def pollution_score(readings: dict):
    """readings: any of pm25, pm10, co. Returns (score, category, driver)."""
    subs = {p: sub_index(p, readings[p]) for p in _BREAKS if p in readings}
    subs = {p: v for p, v in subs.items() if not np.isnan(v)}
    if not subs:
        return np.nan, "n/a", None
    driver = max(subs, key=subs.get)
    score = subs[driver]
    return score, category(score), driver


def category(score: float) -> str:
    for limit, name in CATEGORIES:
        if score <= limit:
            return name
    return "Severe"
