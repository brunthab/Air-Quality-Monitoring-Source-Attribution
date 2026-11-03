"""Ground-level PM2.5 estimation (Random Forest) + short-term forecast."""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import SOURCES
from .geo import plume_factor

KINDS = ["industrial", "traffic", "construction", "biomass"]

FEATURES = (
    ["sat_aod", "sat_no2", "sat_so2", "sat_age_h",
     "wind_speed", "wind_u", "wind_v", "rh", "temp",
     "hour_sin", "hour_cos", "fire_hotspot", "traffic_idx"]
    + [f"dist_{k}_km" for k in KINDS]
    + [f"plume_{k}" for k in KINDS]
)


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    h = out["time"].dt.hour
    out["hour_sin"] = np.sin(2 * np.pi * h / 24)
    out["hour_cos"] = np.cos(2 * np.pi * h / 24)
    rad = np.radians(out["wind_dir"])
    # u/v components of the wind vector (direction wind blows TOWARD)
    out["wind_u"] = -out["wind_speed"] * np.sin(rad)
    out["wind_v"] = -out["wind_speed"] * np.cos(rad)
    out["fire_hotspot"] = out["fire_hotspot"].astype(int)
    for kind in KINDS:
        dists, plumes = [], []
        for s in (s for s in SOURCES if s.kind == kind):
            dists.append(np.hypot(out["x"] - s.x, out["y"] - s.y))
            plumes.append(s.strength * plume_factor(
                out["x"], out["y"], s.x, s.y, out["wind_dir"], out["wind_speed"]))
        out[f"dist_{kind}_km"] = np.minimum.reduce(dists)
        out[f"plume_{kind}"] = np.sum(plumes, axis=0)
    return out


def _metrics(y, p):
    return {"MAE": mean_absolute_error(y, p),
            "RMSE": float(np.sqrt(mean_squared_error(y, p))),
            "R2": r2_score(y, p)}


def _rf():
    return RandomForestRegressor(n_estimators=150, min_samples_leaf=3,
                                 n_jobs=-1, random_state=0)


def evaluate(df: pd.DataFrame) -> dict:
    """Two honest checks, both on sensor rows only:
    - spatial: leave-one-sensor-out (how well do we estimate an unmonitored spot?)
    - temporal: train on first 80% of time, test on the last 20%.
    """
    s = add_features(df[df["is_sensor"]].dropna(subset=["pm25"]))
    X, y = s[FEATURES], s["pm25"]

    pred = pd.Series(index=s.index, dtype=float)
    for cid in s["cell_id"].unique():
        te = s["cell_id"] == cid
        m = _rf().fit(X[~te], y[~te])
        pred[te] = m.predict(X[te])
    spatial = _metrics(y, pred)

    cut = s["time"].quantile(0.8)
    tr = s["time"] <= cut
    m = _rf().fit(X[tr], y[tr])
    temporal = _metrics(y[~tr], m.predict(X[~tr]))
    baseline = _metrics(y, np.full(len(y), y.mean()))
    return {"spatial_loso": spatial, "temporal_holdout": temporal,
            "mean_baseline": baseline}


@dataclass
class Estimator:
    model: RandomForestRegressor

    def importances(self) -> pd.Series:
        return pd.Series(self.model.feature_importances_, index=FEATURES).sort_values(ascending=False)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.predict(add_features(df)[FEATURES])


def train(df: pd.DataFrame) -> Estimator:
    s = add_features(df[df["is_sensor"]].dropna(subset=["pm25"]))
    return Estimator(_rf().fit(s[FEATURES], s["pm25"]))


# ---------------- short-term forecast of city-mean PM2.5 ----------------
def _lag_frame(series: pd.Series, wind: pd.Series) -> pd.DataFrame:
    f = pd.DataFrame({"y": series})
    for k in (1, 2, 3, 24):
        f[f"lag{k}"] = series.shift(k)
    h = series.index.hour
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    f["wind_speed"] = wind
    return f


def forecast(df: pd.DataFrame, horizon: int = 6) -> pd.DataFrame:
    """Recursive forecast of mean sensor PM2.5 (wind held at last value)."""
    s = df[df["is_sensor"]].groupby("time").agg(pm25=("pm25", "mean"),
                                               wind=("wind_speed", "mean"))
    frame = _lag_frame(s["pm25"], s["wind"]).dropna()
    cols = [c for c in frame.columns if c != "y"]
    model = RandomForestRegressor(n_estimators=100, min_samples_leaf=3,
                                  n_jobs=-1, random_state=0).fit(frame[cols], frame["y"])
    hist = s["pm25"].copy()
    out = []
    for _ in range(horizon):
        t = hist.index[-1] + pd.Timedelta(hours=1)
        row = {f"lag{k}": hist.iloc[-k] for k in (1, 2, 3)}
        row["lag24"] = hist.iloc[-24]
        row["hour_sin"], row["hour_cos"] = np.sin(2 * np.pi * t.hour / 24), np.cos(2 * np.pi * t.hour / 24)
        row["wind_speed"] = s["wind"].iloc[-1]
        yhat = float(model.predict(pd.DataFrame([row])[cols])[0])
        hist.loc[t] = yhat
        out.append({"time": t, "pm25_forecast": yhat})
    return pd.DataFrame(out)
