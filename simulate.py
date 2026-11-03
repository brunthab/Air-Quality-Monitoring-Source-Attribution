"""Synthetic multi-source dataset so the whole pipeline runs without hardware.

Produces hourly ground truth on a grid, weather, IoT sensor readings at a few
cells, and coarse, cloud-gapped satellite-like columns (one overpass per day,
~2.4 km blocks, last valid value carried forward with its age).

Replace this module with real loaders (Sentinel-5P / MODIS / Landsat / IoT
database / weather API) that return the same columns.,
"""
import numpy as np
import pandas as pd

from .config import (GRID_N, CELL_KM, N_SENSORS, POLLUTANTS, SOURCES,
                     SIGNATURE, SIM_SCALE, SIM_BACKGROUND)
from .geo import km_to_latlon, plume_factor

BLOCK = 3  # satellite pixel = 3x3 grid cells


def _block_mean(arr):
    g = arr.reshape(GRID_N, GRID_N)
    b = g.reshape(GRID_N // BLOCK, BLOCK, GRID_N // BLOCK, BLOCK).mean(axis=(1, 3))
    return np.repeat(np.repeat(b, BLOCK, axis=0), BLOCK, axis=1).ravel()


def _weather(T, hours, rng):
    drift = np.convolve(rng.normal(0, 40, T), np.ones(6) / 6, mode="same")
    wdir = (225 + drift + 50 * np.sin(np.arange(T) / 37)) % 360
    wspd = np.clip(3 + 1.5 * np.sin(2 * np.pi * (hours - 14) / 24)
                   + rng.normal(0, 0.5, T), 0.5, 8)
    temp = 28 + 4 * np.sin(2 * np.pi * (hours - 9) / 24) + rng.normal(0, 0.3, T)
    rh = np.clip(70 - 15 * np.sin(2 * np.pi * (hours - 9) / 24)
                 + rng.normal(0, 3, T), 30, 95)
    return wdir, wspd, temp, rh


def _activity(kind, hour, day):
    if kind == "traffic":
        return 0.3 + 0.7 * (np.exp(-((hour - 9) / 2) ** 2) + np.exp(-((hour - 18) / 2.5) ** 2))
    if kind == "construction":
        return 1.0 if 8 <= hour < 18 else 0.05
    if kind == "biomass":
        return 1.0 if (day in (5, 6, 7, 11) and 16 <= hour <= 23) else 0.0
    return 0.9  # industrial


def generate_dataset(days: int = 14, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    xs = (np.arange(GRID_N) + 0.5) * CELL_KM
    gx, gy = [a.ravel() for a in np.meshgrid(xs, xs)]
    n = gx.size
    sensor_cells = np.random.default_rng(7).choice(n, N_SENSORS, replace=False)
    is_sensor = np.zeros(n, bool)
    is_sensor[sensor_cells] = True

    total_days = days + 1  # first day only warms up the satellite carry-forward
    times = pd.date_range("2026-10-01", periods=total_days * 24, freq="h")
    T = len(times)
    hours = times.hour.values
    day_idx = np.arange(T) // 24
    wdir, wspd, temp, rh = _weather(T, hours, rng)

    truth = {p: np.zeros((T, n)) for p in POLLUTANTS}
    traffic_idx = np.zeros(T)
    fire = np.zeros(T, bool)
    for t in range(T):
        plumes = {}
        acts = {}
        for s in SOURCES:
            plumes[s.name] = plume_factor(gx, gy, s.x, s.y, wdir[t], wspd[t])
            acts[s.name] = _activity(s.kind, hours[t], day_idx[t])
            if s.kind == "traffic":
                traffic_idx[t] = max(traffic_idx[t], acts[s.name])
            if s.kind == "biomass":
                fire[t] = acts[s.name] > 0
        for p in POLLUTANTS:
            v = np.full(n, SIM_BACKGROUND[p]) * (1 + 0.15 * rng.normal())
            for s in SOURCES:
                v = v + (acts[s.name] * s.strength * SIGNATURE[s.kind][p]
                         * SIM_SCALE[p] * plumes[s.name])
            if p == "pm25":
                v = v * (1 + 0.004 * (rh[t] - 60))
            truth[p][t] = v * (1 + 0.05 * rng.normal(size=n))

    # --- satellite-like columns: one overpass/day at 13:00, 30% cloud gaps ---
    sat = {k: np.zeros((T, n)) for k in ("sat_aod", "sat_no2", "sat_so2")}
    sat_age = np.zeros((T, n))
    last = None
    last_t = np.zeros(n)
    for t in range(T):
        if hours[t] == 13:
            new = {
                "sat_aod": _block_mean(truth["pm25"][t]) * 0.012 * (1 + (rh[t] - 50) / 100),
                "sat_no2": _block_mean(truth["no2"][t]),
                "sat_so2": _block_mean(truth["so2"][t]),
            }
            for k in new:
                new[k] = new[k] * (1 + 0.08 * rng.normal(size=n))
            valid = np.repeat(np.repeat(rng.random((GRID_N // BLOCK,) * 2) > 0.3,
                                        BLOCK, axis=0), BLOCK, axis=1).ravel()
            if last is None:
                last, valid = new, np.ones(n, bool)
            for k in last:
                last[k] = np.where(valid, new[k], last[k])
            last_t = np.where(valid, t, last_t)
        if last is not None:
            for k in sat:
                sat[k][t] = last[k]
            sat_age[t] = t - last_t

    lat, lon = km_to_latlon(gx, gy)
    df = pd.DataFrame({
        "time": np.repeat(times, n),
        "cell_id": np.tile(np.arange(n), T),
        "x": np.tile(gx, T), "y": np.tile(gy, T),
        "lat": np.tile(lat, T), "lon": np.tile(lon, T),
        "is_sensor": np.tile(is_sensor, T),
        "wind_dir": np.repeat(wdir, n), "wind_speed": np.repeat(wspd, n),
        "temp": np.repeat(temp, n), "rh": np.repeat(rh, n),
        "traffic_idx": np.repeat(traffic_idx, n),
        "fire_hotspot": np.repeat(fire, n),
        **{p: truth[p].ravel() for p in POLLUTANTS},
        **{k: v.ravel() for k, v in sat.items()},
        "sat_age_h": sat_age.ravel(),
    })
    df = df[df["time"] >= times[24]].reset_index(drop=True)
    # Ground truth exists only where a sensor sits; elsewhere it is hidden
    # from the model (kept in *_true columns for evaluation only).
    for p in POLLUTANTS:
        df[f"{p}_true"] = df[p]
        df.loc[~df["is_sensor"], p] = np.nan
    return df
