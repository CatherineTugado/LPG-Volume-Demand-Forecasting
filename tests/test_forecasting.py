from pathlib import Path

import pandas as pd

from lpg.forecasting import HORIZON, add_features, wape, weekly_demand

DATA = Path(__file__).resolve().parents[1] / "data" / "simulated"
visits = pd.read_csv(DATA / "plant_visits.csv", parse_dates=["date"])


def test_weekly_features_do_not_leak():
    """Changing the last HORIZON-1 weeks must not change any feature of the last week."""
    w = weekly_demand(visits)
    f1 = add_features(w)
    w2 = w.copy()
    last_weeks = sorted(w2["week"].unique())[-(HORIZON - 1):]
    w2.loc[w2["week"].isin(last_weeks), "order_t"] *= 10
    f2 = add_features(w2)
    cols = ["lag_4_rel", "lag_52_rel", "level", "yoy_growth"]
    last = w["week"].max()
    pd.testing.assert_frame_equal(f1.loc[f1.week == last, cols].reset_index(drop=True),
                                  f2.loc[f2.week == last, cols].reset_index(drop=True))


def test_wape():
    assert wape([100, 100], [110, 90]) == 10.0
