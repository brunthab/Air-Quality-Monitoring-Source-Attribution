"""Rule-based, explainable source attribution.

For a location and moment, each *active* known source is scored by
  plume exposure (wind alignment + distance + dilution)
  x signature match (does the pollutant mix look like that source type?)
and scores are normalised into ranked shares. Output is a ranked list of
LIKELY contributors with a confidence label, never proof against one site.
"""
from dataclasses import dataclass, field

import numpy as np

from .config import SOURCES, SIGNATURE
from .geo import angle_diff_deg, bearing_deg, plume_factor

DISPLAY = {"pm25": "PM2.5", "pm10": "PM10", "no2": "NO2", "so2": "SO2", "co": "CO"}


@dataclass
class Attribution:
    source: str
    kind: str
    share: float
    confidence: str
    reasons: list = field(default_factory=list)


def excess_ratios(readings: dict, baseline: dict) -> dict:
    """How far above the typical-clean level each pollutant is (0 = at baseline)."""
    return {k: max(0.0, v / baseline[k] - 1.0) for k, v in readings.items()
            if k in baseline and v is not None and not np.isnan(v) and baseline[k] > 0}


def _is_active(kind, hour, ctx):
    if kind == "traffic":
        return ctx.get("traffic_idx", 0) >= 0.4
    if kind == "construction":
        return 8 <= hour < 18
    if kind == "biomass":
        return bool(ctx.get("fire_hotspot", False))
    return True


def attribute(x, y, wind_dir, wind_speed, excess: dict, hour: int, ctx: dict):
    """Return (ranked list[Attribution], note)."""
    if not excess or max(excess.values()) < 0.15:
        return [], "Pollution is close to baseline; no meaningful source signal."

    rows = []
    for s in SOURCES:
        if not _is_active(s.kind, hour, ctx):
            continue
        plume = float(s.strength * plume_factor(x, y, s.x, s.y, wind_dir, wind_speed))
        sig = SIGNATURE[s.kind]
        num = sum(sig[k] * min(v, 3.0) for k, v in excess.items())
        den = sum(sig[k] for k in excess)
        match = num / den if den else 0.0
        rows.append((s, plume, match, plume * match))

    total = sum(r[3] for r in rows)
    if total <= 0:
        return [], "No active source is plausibly upwind of this location."

    out = []
    for s, plume, match, raw in sorted(rows, key=lambda r: -r[3]):
        share = raw / total
        d = float(np.hypot(s.x - x, s.y - y))
        diff = float(angle_diff_deg(bearing_deg(x, y, s.x, s.y), wind_dir))
        reasons = [f"{d:.1f} km away; wind from {wind_dir:.0f}° is "
                   f"{'aligned' if diff < 45 else 'not aligned'} with it ({diff:.0f}° off)"]
        for k, w in SIGNATURE[s.kind].items():
            if w >= 0.8 and excess.get(k, 0) > 0.3:
                reasons.append(f"elevated {DISPLAY[k]} (+{excess[k] * 100:.0f}% over baseline) "
                               f"matches {s.kind} signature")
        if s.kind == "biomass":
            reasons.append("active fire hotspot detected")
        if s.kind == "traffic":
            reasons.append(f"traffic index {ctx.get('traffic_idx', 0):.2f}")
        conf = "High" if share >= 0.5 and plume >= 0.15 else \
               "Medium" if share >= 0.3 else "Low"
        out.append(Attribution(s.name, s.kind, share, conf, reasons))
    return out, ("Ranked likely contributors only; factory-level claims need "
                 "validated emissions data.")
