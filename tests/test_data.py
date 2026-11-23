from pathlib import Path

import numpy as np
import pandas as pd

from lpg.simulate import SIZES

DATA = Path(__file__).resolve().parents[1] / "data" / "simulated"
visits = pd.read_csv(DATA / "plant_visits.csv", parse_dates=["date"])
accounts = pd.read_csv(DATA / "accounts.csv", parse_dates=["onboard_date"])


def test_gate_log_is_consistent():
    kg = sum(visits[c] * w for c, w in SIZES.items())
    assert np.allclose(kg, visits["total_kg"], atol=0.11)
    nw = visits[visits["withdrew"] == 0]
    assert (nw["total_kg"] == 0).all()
    assert nw["reason_no_withdrawal"].notna().all()
    assert (visits["total_kg"] <= visits["order_kg"] + 0.11).all()


def test_business_lines_are_separate():
    assert set(visits["product_line"]) == {"LPG B2C", "LPG B2B", "Canister"}
    can = visits[visits["segment"] == "Canister"]
    assert can[["qty_2_7kg", "qty_11kg", "qty_22kg", "qty_50kg"]].sum().sum() == 0
    assert visits.loc[visits["segment"] != "Canister", "qty_canister_case"].sum() == 0
