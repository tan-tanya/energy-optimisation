"""
DEC validation: modelled year-0 energy intensities against metered Display Energy Certificates.

Addresses the circularity in the electricity arm of Section 3.2.9.1, where the modelled
non-EV electricity IS the CIBSE benchmark used to build the profile, so comparing it against
a benchmark survey tests one benchmark against another. DECs are metered annual consumption
from real buildings, so they sit outside the whole build-up.

Three-way comparison per activity class:
    model  - this study, year 0, across all nine districts
    TM46   - the DEC 'typical_*' fields, i.e. the benchmark the DEC scheme scores against
    DEC    - the DEC 'annual_*' fields, i.e. metered actuals

Conventions matched to Section 3.2.9.1:
  - DEC thermal is DELIVERED FUEL, so modelled useful heat is divided by the boiler
    efficiency before comparison.
  - DEC electricity is whole-building metered, so the model is reported both including
    and excluding its assumed EV charging layer.

Population caveats, which the output restates:
  - DECs are issued to public authority buildings only, so the office and retail samples are
    public-sector stock and are not representative of private commercial premises.
  - DEC applies at or above 250 m2, so smaller premises are absent by design.

Source: Energy Performance of Buildings Register, England and Wales, DEC bulk CSV
        (.admin/display-csv/certificates-<year>.csv).

Usage:
    python scripts/dec_validation.py                      # 2025 lodgements
    python scripts/dec_validation.py --years 2018-2026    # pooled, needed for retail
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import demand_profile_model as dm                    # noqa: E402
import optimisation_engine as oe                     # noqa: E402
from districts import DISTRICTS                      # noqa: E402
from optimisation_config import N_DAYS_OF, S_KEYS    # noqa: E402

DEC_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       ".admin", "display-csv")

# Activity class -> TM46 main_benchmark as lodged on the register (lower-cased), plus whether
# to restrict the sample to air-conditioned stock. The modelled office is an A/C archetype,
# while TM46 'General Office' spans naturally ventilated, mixed-mode and A/C premises alike,
# so the unrestricted sample is not the right comparator for it.
CLASSES = {
    "Health: Health centre":    dict(benchmark="clinic",                           aircon=None),
    "Health: Hospital":         dict(benchmark="hospital - clinical and research", aircon=None),
    "Office: A/C standard":     dict(benchmark="general office",                   aircon="Y"),
    "Retail: Department store": dict(benchmark="large non-food shop",              aircon=None),
}

DEC_COLS = ["main_benchmark", "total_floor_area", "main_heating_fuel", "aircon_present",
            "annual_electrical_fuel_usage", "typical_electrical_fuel_usage",
            "annual_thermal_fuel_usage", "typical_thermal_fuel_usage",
            "uprn", "nominated_date"]

MIN_FLOOR_AREA_M2 = 250          # DEC scope threshold
ELEC_BOUNDS = (1.0, 1000.0)      # guards against unit / floor-area lodgement errors
THERM_BOUNDS = (1.0, 1500.0)


# ---------------------------------------------------------------- model side
def model_intensities() -> pd.DataFrame:
    """Year-0 kWh/m2/yr per activity class across all nine districts."""
    oe._ensure_dm_initialized()
    rows = []
    for activity in CLASSES:
        area = dm.bees_floor_areas[activity]
        ev_int = dm.ev_annual_kwh_per_sqm(activity)      # district-invariant
        for district in DISTRICTS:
            elec = oe.building_demand_kwh(activity, district)
            heat = oe.building_heat_demand_kwh(activity, district)
            elec_kwh = sum(elec[k].sum() * N_DAYS_OF[k] for k in S_KEYS)
            heat_kwh = sum(heat[k].sum() * N_DAYS_OF[k] for k in S_KEYS)
            rows.append({
                "activity": activity, "district": district, "floor_area_m2": area,
                "elec_incl_ev": elec_kwh / area,
                "elec_excl_ev": elec_kwh / area - ev_int,
                "heat_fuel": heat_kwh / area / dm.ETA_BOILER,
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ DEC side
def load_dec(years: list) -> pd.DataFrame:
    frames = []
    for y in years:
        path = os.path.join(DEC_DIR, f"certificates-{y}.csv")
        if not os.path.exists(path):
            print(f"  (skipping missing file: certificates-{y}.csv)")
            continue
        d = pd.read_csv(path, usecols=DEC_COLS, low_memory=False)
        d["benchmark"] = d["main_benchmark"].astype(str).str.strip().str.lower()
        frames.append(d[d["benchmark"].isin(c["benchmark"] for c in CLASSES.values())])
    if not frames:
        raise SystemExit(f"No DEC files found in {DEC_DIR}")
    d = pd.concat(frames, ignore_index=True)

    for c in ("total_floor_area", "annual_electrical_fuel_usage", "typical_electrical_fuel_usage",
              "annual_thermal_fuel_usage", "typical_thermal_fuel_usage"):
        d[c] = pd.to_numeric(d[c], errors="coerce")

    # One record per building: a UPRN can carry several lodgements across years, so keep the
    # most recent assessment period. Records without a UPRN are retained rather than dropped.
    d["nominated_date"] = pd.to_datetime(d["nominated_date"], errors="coerce")
    has = d["uprn"].notna()
    d = pd.concat([d[has].sort_values("nominated_date").drop_duplicates("uprn", keep="last"),
                   d[~has]], ignore_index=True)
    return d[d["total_floor_area"] >= MIN_FLOOR_AREA_M2]


def dec_sample(d: pd.DataFrame, spec: dict) -> pd.DataFrame:
    b = d[d["benchmark"] == spec["benchmark"]]
    if spec["aircon"] is not None:
        b = b[b["aircon_present"] == spec["aircon"]]
    return b


def _dist(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {"n": 0}
    return {"n": len(s), "p10": s.quantile(.10), "median": s.median(),
            "p90": s.quantile(.90), "max": s.max(), "series": s}


# -------------------------------------------------------------------- report
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--years", default="2025",
                    help="lodgement year or range, e.g. 2025 or 2018-2026 (default 2025)")
    args = ap.parse_args()
    if "-" in args.years:
        lo, hi = (int(x) for x in args.years.split("-"))
        years = list(range(lo, hi + 1))
    else:
        years = [int(args.years)]

    dec = load_dec(years)
    mod = model_intensities()

    print(f"\nDEC validation - lodgement years {years[0]}"
          f"{'' if len(years) == 1 else f'-{years[-1]}'}")
    print(f"Records in scope: {len(dec):,} "
          f"(floor area >= {MIN_FLOOR_AREA_M2} m2, deduplicated by UPRN)")
    print(f"Boiler efficiency for the fuel conversion: {dm.ETA_BOILER}")
    print("DEC covers public authority buildings only; the office and retail samples are "
          "public-sector stock.\n")

    for activity, spec in CLASSES.items():
        b = dec_sample(dec, spec)
        m = mod[mod["activity"] == activity]
        ac = "" if spec["aircon"] is None else ", air-conditioned only"
        print("=" * 100)
        print(f"{activity}   ->   TM46 '{spec['benchmark']}'{ac}")

        elec = _dist(b.loc[b["annual_electrical_fuel_usage"].between(*ELEC_BOUNDS),
                           "annual_electrical_fuel_usage"])
        # The modelled counterfactual burns gas, so the thermal comparison is restricted to
        # gas-heated stock; electrically and district-heated premises report no thermal fuel.
        gas = b[b["main_heating_fuel"] == "Natural Gas"]
        therm = _dist(gas.loc[gas["annual_thermal_fuel_usage"].between(*THERM_BOUNDS),
                              "annual_thermal_fuel_usage"])

        print(f"  DEC records {len(b)}   (electricity n={elec['n']}, "
              f"gas-heated thermal n={therm['n']})")
        if len(b):
            print(f"  Floor area   model {m['floor_area_m2'].iloc[0]:>7,.0f} m2   "
                  f"DEC median {b['total_floor_area'].median():>7,.0f} m2")

        tm46_e = b.loc[elec.get("series", pd.Series(dtype=float)).index,
                       "typical_electrical_fuel_usage"].median() if elec["n"] else np.nan
        tm46_t = gas.loc[therm.get("series", pd.Series(dtype=float)).index,
                         "typical_thermal_fuel_usage"].median() if therm["n"] else np.nan

        for label, mvals, dd, tm46 in (
            ("ELECTRICITY (incl EV)", m["elec_incl_ev"], elec, tm46_e),
            ("ELECTRICITY (excl EV)", m["elec_excl_ev"], elec, tm46_e),
            ("THERMAL FUEL",          m["heat_fuel"],    therm, tm46_t),
        ):
            lo, hi, mid = mvals.min(), mvals.max(), mvals.median()
            if dd["n"] == 0:
                print(f"  {label:22s} model {lo:6.0f}-{hi:<6.0f} | no DEC data for this class")
                continue
            pct = (dd["series"] < mid).mean() * 100
            print(f"  {label:22s} model {lo:6.0f}-{hi:<6.0f} "
                  f"| DEC p10 {dd['p10']:5.0f}  med {dd['median']:5.0f}  p90 {dd['p90']:5.0f}  "
                  f"max {dd['max']:5.0f} | TM46 {tm46:5.0f} "
                  f"| model/DEC {mid / dd['median']:5.2f}x | percentile {pct:3.0f}")
        print()


if __name__ == "__main__":
    main()
