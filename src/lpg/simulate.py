"""Simulate an LPG distribution business in the Philippines (2023-2025).

All data is SYNTHETIC. It copies the *structure* of real plant data (gate logs,
production, dealer and business accounts) so the analysis can be shared publicly.

Three business lines:
    LPG B2C   dealers withdrawing household cylinders (2.7 / 11 / 22 / 50 kg)
    LPG B2B   businesses (restaurants, hotels, industry); managed through the SPANCOP sales pipeline
    Canister  canister distributors withdrawing cases of butane canisters
              (1 case = 28 cans x 220 g = 6.16 kg); separate stock from bulk LPG
Swap in real tables with the same columns to run the notebooks on company data.

Outputs (data/simulated/):
    regions.csv           region, island group, serving plant, households, distance
    accounts.csv          LPG B2C dealers, LPG B2B businesses and canister distributors
    plant_visits.csv      gate log: one row per plant visit, incl. visits with NO withdrawal;
                          total_kg = withdrawn, order_kg = requested (lost if no stock)
    plant_production.csv  daily production and inventory per plant and product (LPG / Canister), kg
    household_refills.csv refill purchases of a household sample (e.g. delivery-app data)

Usage:
    python -m lpg.simulate --out data/simulated --seed 7
"""
from __future__ import annotations

import argparse
from pathlib import Path

import holidays
import numpy as np
import pandas as pd

START, END = pd.Timestamp("2023-01-01"), pd.Timestamp("2025-12-31")
CYL_SIZES = {"qty_2_7kg": 2.7, "qty_11kg": 11.0, "qty_22kg": 22.0, "qty_50kg": 50.0}
CANISTER_CASE_KG = 28 * 0.22
SIZES = {**CYL_SIZES, "qty_canister_case": CANISTER_CASE_KG}  # every withdrawn unit, in kg
LINE = {"B2C": "LPG B2C", "B2B": "LPG B2B", "Canister": "Canister"}

# region, island group, plant, households ('000, simulated), avg distance to plant (km),
# dealer coverage (dealers per 100k households, relative), annual growth
REGIONS = [
    ("NCR",                 "Luzon",    "PLANT-LUZ", 3500,  60, 1.20, 0.03),
    ("CALABARZON",          "Luzon",    "PLANT-LUZ", 4000,  45, 1.15, 0.07),
    ("Central Luzon",       "Luzon",    "PLANT-LUZ", 3000, 120, 1.00, 0.06),
    ("Ilocos",              "Luzon",    "PLANT-LUZ", 1300, 380, 0.55, 0.05),
    ("Bicol",               "Luzon",    "PLANT-LUZ", 1500, 330, 0.45, 0.09),
    ("Western Visayas",     "Visayas",  "PLANT-VIS", 1900, 220, 0.70, 0.06),
    ("Central Visayas",     "Visayas",  "PLANT-VIS", 2100,  40, 1.20, 0.05),
    ("Eastern Visayas",     "Visayas",  "PLANT-VIS", 1100, 260, 0.40, 0.08),
    ("Davao Region",        "Mindanao", "PLANT-MIN", 1400,  50, 1.10, 0.08),
    ("Northern Mindanao",   "Mindanao", "PLANT-MIN", 1200, 250, 0.65, 0.07),
    ("SOCCSKSARGEN",        "Mindanao", "PLANT-MIN", 1200, 180, 0.70, 0.07),
    ("Zamboanga Peninsula", "Mindanao", "PLANT-MIN",  900, 420, 0.35, 0.10),
]

B2B_TYPES = {  # share, mean days between withdrawals, qty means per size
    "Restaurant":  (0.55, 9,  {"qty_11kg": 20, "qty_50kg": 8}),
    "Hotel":       (0.15, 14, {"qty_50kg": 18, "qty_22kg": 10}),
    "Industrial":  (0.20, 12, {"qty_50kg": 60}),
    "Institution": (0.10, 18, {"qty_11kg": 25, "qty_50kg": 12}),
}
MONTH_FACTOR = {1: 0.95, 2: 0.95, 3: 0.97, 4: 0.98, 5: 1.00, 6: 0.98, 7: 0.96,
                8: 0.96, 9: 0.98, 10: 1.02, 11: 1.08, 12: 1.22}
TYPHOON_MONTHS = {7, 8, 9, 10}
# Canisters: portable stoves for Noche Buena gatherings, and brownouts in the typhoon season
CANISTER_MONTH_FACTOR = {1: 0.85, 2: 0.85, 3: 0.90, 4: 0.95, 5: 0.95, 6: 1.00, 7: 1.12,
                         8: 1.15, 9: 1.15, 10: 1.12, 11: 1.15, 12: 1.50}


def make_typhoons(rng) -> list[tuple]:
    """Three typhoons a year (Jul-Nov), each hitting one island group for ~10 days."""
    events = []
    for year in range(START.year, END.year + 1):
        for _ in range(3):
            start = pd.Timestamp(year=year, month=int(rng.integers(7, 12)), day=int(rng.integers(1, 28)))
            events.append((rng.choice(["Luzon", "Visayas", "Mindanao"]), start, start + pd.Timedelta(days=10)))
    return events


def plant_closed_days() -> set:
    ph = holidays.Philippines(years=range(START.year, END.year + 1))
    closed = {pd.Timestamp(d) for d, name in ph.items()
              if name in ("New Year's Day", "Maundy Thursday", "Good Friday", "Christmas Day")}
    closed |= set(pd.date_range(START, END, freq="W-SUN"))
    return closed


def make_regions() -> pd.DataFrame:
    return pd.DataFrame(REGIONS, columns=["region", "island_group", "plant", "households_k",
                                          "distance_km", "h_coverage", "h_growth"])


def make_accounts(regions: pd.DataFrame, rng) -> pd.DataFrame:
    rows = []
    span = (END - START).days
    for r in regions.itertuples():
        n_dealers = max(4, int(round(r.households_k / 100 * 1.6 * r.h_coverage)))
        n_b2b = max(2, int(round(r.households_k / 100 * 0.5 * (0.6 + 0.4 * r.h_coverage))))
        n_can = max(2, int(round(r.households_k / 100 * 0.8 * (0.5 + 0.5 * r.h_coverage))))
        for seg, n in (("B2C", n_dealers), ("B2B", n_b2b), ("Canister", n_can)):
            for _ in range(n):
                onboard = START if rng.random() < 0.75 else START + pd.Timedelta(days=int(rng.integers(30, span - 120)))
                churn = pd.NaT
                if rng.random() < {"B2C": 0.12, "B2B": 0.08, "Canister": 0.10}[seg]:
                    lo = (onboard - START).days + 120
                    if lo < span - 30:
                        churn = START + pd.Timedelta(days=int(rng.integers(lo, span - 30)))
                if seg == "B2C":
                    btype = "Dealer"
                    # households served per dealer: fewer dealers -> each serves more, but far ones lose share
                    size_factor = np.clip(rng.lognormal(0, 0.35) / (0.6 + 0.4 * r.h_coverage) ** 0.3, 0.4, 3.0)
                    cycle = float(np.clip(rng.normal(7, 2) * (1 + r.distance_km / 600), 3, 21))
                elif seg == "Canister":
                    btype = "Canister distributor"
                    size_factor = float(np.clip(rng.lognormal(0, 0.4), 0.3, 3.0))
                    cycle = float(np.clip(rng.normal(14, 4) * (1 + r.distance_km / 800), 5, 35))
                else:
                    btype = rng.choice(list(B2B_TYPES), p=[v[0] for v in B2B_TYPES.values()])
                    size_factor = float(np.clip(rng.lognormal(0, 0.4), 0.3, 3.0))
                    cycle = float(np.clip(rng.normal(B2B_TYPES[btype][1], 3), 4, 35))
                rows.append(dict(region=r.region, island_group=r.island_group, plant=r.plant, segment=seg,
                                 product_line=LINE[seg], business_type=btype,
                                 distance_km=max(3, int(rng.normal(r.distance_km, r.distance_km * 0.25))),
                                 onboard_date=onboard, h_churn_date=churn, h_cycle_days=cycle,
                                 h_size=size_factor, h_growth=r.h_growth))
    acc = pd.DataFrame(rows)
    prefix = {"B2C": "D", "B2B": "B", "Canister": "C"}
    acc.insert(0, "account_id", [f"{prefix[s]}{i:04d}" for i, s in enumerate(acc.segment, 1)])
    return acc


def dealer_mix(rng) -> dict:
    """Cylinder mix of a B2C dealer (mostly 11 kg household cylinders)."""
    base = 90 * rng.lognormal(0, 0.2)
    return {"qty_2_7kg": base * rng.uniform(0.10, 0.35), "qty_11kg": base,
            "qty_22kg": base * rng.uniform(0.0, 0.05), "qty_50kg": base * rng.uniform(0.0, 0.03)}


def in_typhoon(day, island, typhoons) -> bool:
    return any(island == isl and s <= day <= e for isl, s, e in typhoons)


def simulate_demand(acc: pd.DataFrame, closed: set, typhoons: list, rng) -> pd.DataFrame:
    """Planned plant visits and the quantity each account wants to withdraw."""
    rows = []
    for a in acc.itertuples():
        if a.segment == "B2C":
            mix = dealer_mix(rng)
        elif a.segment == "Canister":
            mix = {"qty_canister_case": 110 * rng.lognormal(0, 0.25)}
        else:
            mix = {k: float(v) for k, v in B2B_TYPES[a.business_type][2].items()}
        month_factor = CANISTER_MONTH_FACTOR if a.segment == "Canister" else MONTH_FACTOR
        growth = a.h_growth + (0.04 if a.segment == "Canister" else 0.0)
        last_end = a.h_churn_date if pd.notna(a.h_churn_date) else END
        day = a.onboard_date + pd.Timedelta(days=int(rng.integers(0, max(1, int(a.h_cycle_days)))))
        while day <= min(last_end, END):
            while day in closed:
                day += pd.Timedelta(days=1)
            if day > END:
                break
            years = (day - START).days / 365.25
            mult = month_factor[day.month] * (1 + growth) ** years * a.h_size
            if a.segment == "B2C" and (day.day in (14, 15, 16) or day.day >= 29 or day.day == 1):
                mult *= 1.12  # payday
            typhoon = in_typhoon(day, a.island_group, typhoons)
            if typhoon and a.segment == "Canister":
                mult *= 1.8  # brownouts: households cook on portable butane stoves
            days_to_churn = (a.h_churn_date - day).days if pd.notna(a.h_churn_date) else 999
            fading = days_to_churn < 90
            if fading:
                mult *= 0.6
            qty = {k: int(rng.poisson(mix.get(k, 0.0) * mult)) for k in SIZES}
            rows.append(dict(date=day, account_id=a.account_id, **qty))
            gap = a.h_cycle_days / month_factor[day.month] ** 0.5
            if typhoon and a.segment == "Canister":
                gap *= 0.6
            if day.month in TYPHOON_MONTHS and a.distance_km > 200:
                gap *= 1.15
            if fading:
                gap *= 1.7
            day += pd.Timedelta(days=max(1, int(round(rng.gamma(6, gap / 6)))))
    return pd.DataFrame(rows)


def run_plants(demand: pd.DataFrame, acc: pd.DataFrame, closed: set, rng):
    """Day-by-day plant operation. Production follows the plant's current planning rule:
    produce the trailing 4-week average daily withdrawal, corrected towards a target stock of
    1.5 days of demand (6 days for canisters), with production capped at 106% (110%) of that average. When inventory runs short,
    visits leave without withdrawing ('Stock unavailable'). Bulk LPG (cylinders) and canisters
    have separate stock."""
    demand = demand.merge(acc[["account_id", "plant", "segment", "product_line", "distance_km"]], on="account_id")
    demand["want_kg"] = sum(demand[c] * kg for c, kg in SIZES.items())
    demand["stock"] = np.where(demand["segment"] == "Canister", "Canister", "LPG")
    out_visits, prod_rows = [], []
    for (plant, stock), dp in demand.groupby(["plant", "stock"]):
        by_day = dict(tuple(dp.groupby("date")))
        history = []  # kg withdrawn per open day
        inventory = None
        for day in pd.date_range(START, END):
            if day in closed:
                continue
            today = by_day.get(day, dp.iloc[0:0])
            if inventory is None:
                inventory = today["want_kg"].sum() * (8.0 if stock == "Canister" else 2.5)
            avg = np.mean(history[-24:]) if history else today["want_kg"].sum()
            # canisters are packed goods kept in a warehouse: hold ~6 days of stock, more flexible supply
            target_days, cap = (6.0, 1.10) if stock == "Canister" else (1.5, 1.06)
            plan = np.clip(avg + 0.5 * (target_days * avg - inventory), 0, cap * avg)
            produced = plan * rng.normal(1.0, 0.03)
            opening = inventory
            inventory += produced
            withdrawn = 0.0
            for v in today.sample(frac=1, random_state=int(rng.integers(1e9))).itertuples():
                rec = v._asdict()
                # visits that come in without withdrawing for non-stock reasons
                p_other = {"B2C": 0.07, "B2B": 0.03, "Canister": 0.04}[v.segment] * (1.6 if v.distance_km < 60 else 0.8)
                if rng.random() < p_other:
                    reason = rng.choice(["Returned empties only", "Payment / collection",
                                         "Order not ready / documents"], p=[0.5, 0.3, 0.2])
                    rec.update(withdrew=0, reason_no_withdrawal=reason)
                elif v.want_kg <= inventory:
                    inventory -= v.want_kg
                    withdrawn += v.want_kg
                    rec.update(withdrew=1, reason_no_withdrawal="")
                else:
                    rec.update(withdrew=0, reason_no_withdrawal="Stock unavailable")
                out_visits.append(rec)
            history.append(withdrawn)
            prod_rows.append(dict(date=day, plant=plant, product=stock, produced_kg=round(produced),
                                  opening_inventory_kg=round(opening), withdrawn_kg=round(withdrawn),
                                  closing_inventory_kg=round(inventory)))
    visits = pd.DataFrame(out_visits).drop(columns=["Index", "distance_km", "stock"])
    qty_cols = list(SIZES)
    # order_kg = what the customer came to withdraw (unconstrained demand). It is lost when
    # the plant has no stock; visits for other reasons carry no order.
    visits["order_kg"] = np.where((visits["withdrew"] == 1) |
                                  (visits["reason_no_withdrawal"] == "Stock unavailable"),
                                  visits["want_kg"], 0.0).round(1)
    visits.loc[visits["withdrew"] == 0, qty_cols] = 0
    visits["total_kg"] = sum(visits[c] * kg for c, kg in SIZES.items()).round(1)
    visits = visits.sort_values(["date", "plant", "account_id"]).reset_index(drop=True)
    visits.insert(0, "visit_id", [f"V{i:06d}" for i in range(1, len(visits) + 1)])
    cols = ["visit_id", "date", "plant", "account_id", "segment", "product_line", "withdrew",
            "reason_no_withdrawal", *qty_cols, "total_kg", "order_kg"]
    production = pd.DataFrame(prod_rows).sort_values(["date", "plant", "product"]).reset_index(drop=True)
    return visits[cols], production


def simulate_households(regions: pd.DataFrame, rng, n: int = 4000) -> pd.DataFrame:
    """Refill purchases of a household sample (Jan 2024 - Dec 2025)."""
    start, end = pd.Timestamp("2024-01-01"), END
    w = regions["households_k"] / regions["households_k"].sum()
    rows = []
    for i in range(1, n + 1):
        region = rng.choice(regions["region"], p=w)
        cyl = "11kg" if rng.random() < 0.85 else "2.7kg"
        size = int(np.clip(rng.poisson(4.2), 1, 10))
        use_kg_month = (2.2 + 1.1 * size) * rng.lognormal(0, 0.25)  # cooking only
        if rng.random() < 0.08:
            use_kg_month *= 2.5  # small home business (sari-sari / carinderia)
        cyl_kg = 11.0 if cyl == "11kg" else 2.7
        mean_days = cyl_kg / use_kg_month * 30.4
        stop = end if rng.random() > 0.15 else start + pd.Timedelta(days=int(rng.integers(90, 700)))
        day = start + pd.Timedelta(days=int(rng.integers(0, int(mean_days) + 1)))
        while day <= stop:
            rows.append(dict(household_id=f"H{i:05d}", region=region, cylinder=cyl,
                             household_size=size, purchase_date=day))
            day += pd.Timedelta(days=max(5, int(round(rng.gamma(9, mean_days / 9)))))
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="data/simulated")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    closed = plant_closed_days()
    regions = make_regions()
    acc = make_accounts(regions, rng)
    typhoons = make_typhoons(rng)
    demand = simulate_demand(acc, closed, typhoons, rng)
    visits, production = run_plants(demand, acc, closed, rng)
    households = simulate_households(regions, rng)

    regions.drop(columns=[c for c in regions if c.startswith("h_")]).to_csv(out / "regions.csv", index=False)
    public_acc = acc.drop(columns=[c for c in acc if c.startswith("h_")])
    public_acc.to_csv(out / "accounts.csv", index=False)
    visits.to_csv(out / "plant_visits.csv", index=False)
    production.to_csv(out / "plant_production.csv", index=False)
    households.to_csv(out / "household_refills.csv", index=False)
    pd.DataFrame(typhoons, columns=["island_group", "start", "end"]).to_csv(out / "typhoon_events.csv", index=False)
    print(f"accounts={len(acc)}, visits={len(visits)}, households={households.household_id.nunique()}, "
          f"refills={len(households)}")


if __name__ == "__main__":
    main()
