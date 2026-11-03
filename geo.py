"""Geometry and wind helpers shared by the simulator, model and attribution."""
import numpy as np

from .config import ORIGIN_LAT, ORIGIN_LON


def km_to_latlon(x_km, y_km):
    lat = ORIGIN_LAT + np.asarray(y_km) / 110.574
    lon = ORIGIN_LON + np.asarray(x_km) / (111.320 * np.cos(np.radians(ORIGIN_LAT)))
    return lat, lon


def bearing_deg(tx, ty, sx, sy):
    """Compass bearing (0=N, 90=E) from target (tx,ty) to source (sx,sy)."""
    return np.degrees(np.arctan2(np.asarray(sx) - tx, np.asarray(sy) - ty)) % 360


def angle_diff_deg(a, b):
    """Smallest absolute difference between two compass angles."""
    return np.abs((np.asarray(a) - np.asarray(b) + 180) % 360 - 180)


def plume_factor(tx, ty, sx, sy, wind_dir, wind_speed):
    """Simple wind-aware exposure factor in [0, ~1].

    wind_dir is the meteorological direction the wind blows FROM. A source is
    upwind of the target when the bearing target->source matches wind_dir.
    Combines alignment, distance decay and dilution by wind speed. This is a
    screening heuristic, not a full dispersion model.
    """
    d = np.hypot(np.asarray(sx) - tx, np.asarray(sy) - ty) + 0.3
    diff = np.radians(angle_diff_deg(bearing_deg(tx, ty, sx, sy), wind_dir))
    align = np.clip(np.cos(diff), 0, None)
    spread = 0.15 + 0.85 * align ** 2
    decay = np.exp(-d / (2.5 + 0.4 * np.asarray(wind_speed)))
    dilution = 1.0 / (1.0 + 0.25 * np.asarray(wind_speed))
    return spread * decay * dilution * 1.75
