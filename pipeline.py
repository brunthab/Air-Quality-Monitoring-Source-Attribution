"""End-to-end pipeline: data -> model -> estimates -> attribution.

Run:  python -m aerotrace.pipeline
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .aqi import pollution_score
from .attribution import attribute, excess_ratios
from .model import Estimator, evaluate, forecast, train
from .simulate import generate_dataset


@dataclass
class Bundle:
    df: pd.DataFrame          # all rows, with pm25_est column
    estimator: Estimator
    metrics: dict
    baseline_ground: dict
    baseline_sat: dict


def build_bundle(days: int = 14, with_metrics: bool = True) -> Bundle:
    df = generate_dataset(days=days)
    est = train(df)
    df["pm25_est"] = est.predict(df)
    # where a sensor exists, show the measurement instead of the estimate
    df["pm25_best"] = np.where(df["is_sensor"], df["pm25"], df["pm25_est"])
    sensors = df[df["is_sensor"]]
    base_g = {p: float(sensors[p].quantile(0.2)) for p in ("pm25", "pm10", "no2", "so2", "co")}
    base_s = {"pm25": float(df["pm25_est"].quantile(0.2)),
              "no2": float(df["sat_no2"].quantile(0.2)),
              "so2": float(df["sat_so2"].quantile(0.2))}
    metrics = evaluate(df) if with_metrics else {}
    return Bundle(df, est, metrics, base_g, base_s)


def attribute_cell(b: Bundle, time, cell_id: int):
    """Attribution for one cell/time. Uses full readings if the cell has a
    sensor; otherwise PM2.5 estimate + satellite NO2/SO2 (coarser evidence)."""
    row = b.df[(b.df["time"] == time) & (b.df["cell_id"] == cell_id)].iloc[0]
    if row["is_sensor"]:
        readings = {p: row[p] for p in ("pm25", "pm10", "no2", "so2", "co")}
        base, mode = b.baseline_ground, "measured (IoT sensor)"
    else:
        readings = {"pm25": row["pm25_est"], "no2": row["sat_no2"], "so2": row["sat_so2"]}
        base, mode = b.baseline_sat, "estimated (satellite + model)"
    ctx = {"traffic_idx": row["traffic_idx"], "fire_hotspot": bool(row["fire_hotspot"])}
    ranked, note = attribute(row["x"], row["y"], row["wind_dir"], row["wind_speed"],
                             excess_ratios(readings, base), pd.Timestamp(time).hour, ctx)
    return row, ranked, note, mode


def main():
    b = build_bundle()
    print("== Model validation (PM2.5, ug/m3) ==")
    for name, m in b.metrics.items():
        print(f"{name:18s} MAE={m['MAE']:.2f}  RMSE={m['RMSE']:.2f}  R2={m['R2']:.3f}")
    print("\nTop features:\n", b.estimator.importances().head(6).round(3).to_string())

    last = b.df["time"].max()
    snap = b.df[b.df["time"] == last]
    worst = snap.loc[snap["pm25_best"].idxmax()]
    row, ranked, note, mode = attribute_cell(b, last, int(worst["cell_id"]))
    score, cat, driver = pollution_score({"pm25": row["pm25_best"]})
    print(f"\n== Hotspot at {last}: cell {int(row['cell_id'])} ({mode}) ==")
    print(f"PM2.5 {row['pm25_best']:.0f} ug/m3 -> estimated score {score:.0f} ({cat})")
    for a in ranked[:3]:
        print(f"  {a.share * 100:4.0f}%  {a.source}  [{a.confidence}]")
        for r in a.reasons:
            print(f"        - {r}")
    print(note)
    print("\n== 6h forecast of mean sensor PM2.5 ==")
    fc = forecast(b.df)
    fc["pm25_forecast"] = fc["pm25_forecast"].round(1)
    print(fc.to_string(index=False))


if __name__ == "__main__":
    main()
