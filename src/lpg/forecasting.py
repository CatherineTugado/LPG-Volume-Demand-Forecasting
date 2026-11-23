"""Weekly demand forecasting by plant and business line (LPG B2C dealers, canister distributors)."""
from __future__ import annotations

import holidays
import lightgbm as lgb
import numpy as np
import pandas as pd

HORIZON = 4  # weeks: production is planned up to a month ahead
SERIES = ["plant", "segment"]


def weekly_demand(visits: pd.DataFrame) -> pd.DataFrame:
    """Weekly ordered (unconstrained) and withdrawn tonnes per plant x segment.
    Weeks end on Sunday."""
    v = visits.copy()
    v["week"] = v["date"].dt.to_period("W-SUN").dt.end_time.dt.normalize()
    w = (v.groupby(SERIES + ["week"])
           .agg(order_t=("order_kg", "sum"), withdrawn_t=("total_kg", "sum"), visits=("visit_id", "size"))
           .reset_index())
    w[["order_t", "withdrawn_t"]] /= 1000.0
    # drop a final, incomplete week
    return w[w["week"] <= v["date"].max()].reset_index(drop=True)


def week_calendar(weeks: pd.Series) -> pd.DataFrame:
    """Calendar features of each week (known in advance)."""
    ph = holidays.Philippines(years=range(weeks.dt.year.min(), weeks.dt.year.max() + 2))
    hol = set(pd.to_datetime(list(ph.keys())))
    rows = []
    for end in weeks:
        days = pd.date_range(end - pd.Timedelta(days=6), end)
        open_days = [d for d in days if d.dayofweek < 6 and d not in hol]
        rows.append({
            "week": end,
            "weekofyear": int(end.isocalendar().week),
            "month": end.month,
            "open_days": len(open_days),
            "payday_days": sum(d.day in (14, 15, 16, 29, 30, 31, 1) for d in open_days),
            "holidays": sum(d in hol for d in days),
            "christmas_run": int(end.month == 12 and end.day >= 8),
        })
    return pd.DataFrame(rows)


def add_features(w: pd.DataFrame, target: str = "order_t", horizon: int = HORIZON) -> pd.DataFrame:
    """Lag features at least `horizon` weeks old, so one model covers the whole horizon."""
    w = w.sort_values(SERIES + ["week"]).copy()
    g = w.groupby(SERIES)[target]
    w["level"] = g.transform(lambda s: s.shift(horizon).rolling(8, min_periods=4).mean())
    for lag in (horizon, horizon + 1, horizon + 4, 52):
        w[f"lag_{lag}_rel"] = g.shift(lag) / w["level"]
    w["yoy_growth"] = g.transform(lambda s: s.shift(horizon).rolling(8).mean() / s.shift(horizon + 52).rolling(8).mean())
    w = w.merge(week_calendar(w["week"].drop_duplicates()), on="week", how="left")
    w["plant_code"] = w["plant"].astype("category").cat.codes
    w["is_canister"] = (w["segment"] == "Canister").astype(int)
    return w


FEATURES = ["lag_4_rel", "lag_5_rel", "lag_8_rel", "lag_52_rel", "yoy_growth", "weekofyear", "month",
            "open_days", "payday_days", "holidays", "christmas_run", "plant_code", "is_canister"]


def fit_lgbm(train: pd.DataFrame, target: str = "order_t") -> lgb.LGBMRegressor:
    """Predict demand relative to the recent level (handles growth and different plant sizes)."""
    d = train.dropna(subset=["level", "lag_52_rel"])
    model = lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=8, min_child_samples=10,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1,
                              random_state=42)
    return model.fit(d[FEATURES], d[target] / d["level"])


def predict_lgbm(model, frame: pd.DataFrame) -> np.ndarray:
    return model.predict(frame[FEATURES]) * frame["level"].values


def mape(actual, forecast) -> float:
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.mean(np.abs(forecast - actual) / actual) * 100)


def wape(actual, forecast) -> float:
    """Weighted APE: total absolute error / total volume (fair across big and small series)."""
    actual, forecast = np.asarray(actual), np.asarray(forecast)
    return float(np.abs(forecast - actual).sum() / actual.sum() * 100)
