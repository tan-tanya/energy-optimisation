"""ASHP technology sensitivity. Runs as part of the pipeline (optimisation_model.main() calls
run_ashp_sensitivity() after the cost rounds; --skip-ashp-sensitivity to opt out), and also
standalone against the most recent completed run:
    python ashp_grant_aerona3_sensitivity.py [--limit N] [--time-limit S] [--jobs N]

Prefer the in-pipeline path. This is a FIXED-SIZING re-solve: it pins n_pv to the saved design, so
running it standalone against a run that predates any change to the roof-area cap (or another
sizing bound) makes every row infeasible. That is exactly what blocked it between 2026-07-23 and
2026-07-27 — all 35 ASHP rows of the then-latest run had a saved n_pv above the current
n_pv_max_for_activity. Driving it from the run that just finished removes the failure mode.

Compares the model's current ASHP COP (CEP Technology Library v1.02, Company 4 R-32 average,
ashp_cop_at_7C=2.7283) against a real market "value" UK product -- the Grant Aerona3 R32
(COP @ A7/W55 = 2.66, per manufacturer's EN14511 datasheet) -- for every ASHP design in the
LATEST deterministic cost-round optimisation run (most-recently-modified outputs/Optimisation (*)
folder).

Method: mirrors pv_panel1_sensitivity.py's fixed-sizing, dispatch-only re-solve. For each
(district, activity) design where ASHP is the winning heating system, first-stage sizing (n_pv,
e_batt, o_batt, q_heat_cap, e_th) is fixed at the saved deterministic-round solution and only the
stage-2 dispatch is re-solved with:
  - ashp_cop_at_7C lowered to Grant Aerona3's 2.66 (COP/temperature SLOPE held at the model's
    existing value, per direction -- no published multi-temperature curve was found for Grant
    Aerona3 in public datasheets [manufacturers only publish the two single-point EN14511 ratings,
    A7/W35 and A7/W55, required for ErP labelling], so the reduction is applied only at the
    reference point; cold-weather sensitivity is assumed identical to the current model).
  - HEAT_COSTS["ASHP"]["capex_per_kwth"] cut by CAPEX_REDUCTION_FRAC, paired with the COP cut
    (2.66 / 2.7283 - 1 = -2.5%; CAPEX_REDUCTION_FRAC set to the requested -2.6%).
Baseline (current-COP) metrics are read straight from the saved CSV rather than re-solved -- same
validated shortcut as pv_panel1_sensitivity.py (fixed sizing + dispatch-only re-solve reproduces a
fresh full re-solve's total_cost_npv_GBP to within a rounding error, confirmed for the PV case on
2026-07-08; the mechanism is identical here, only the perturbed technology differs).

Output: <run folder>/ashp_grant_aerona3_sensitivity.xlsx, one row per ASHP design, in the same
NPV-savings-ranked row order as the source cost_deterministic_results.csv.
"""
import argparse
import glob
import multiprocessing as mp
import os

import pandas as pd
from openpyxl.formatting.rule import CellIsRule
from openpyxl.styles import Font, PatternFill
from openpyxl.utils import get_column_letter

import demand_profile_model as dm
import optimisation_engine as oe

# Grant Aerona3 R32, COP @ A7/W55 (manufacturer EN14511 datasheet) vs the model's current
# Company-4-R32-average (CEP Technology Library v1.02, HP air-water (Heating) sheet, units 12-17).
CURRENT_ASHP_COP_AT_7C = 2.7283
GRANT_COP_AT_7C        = 2.66
CAPEX_REDUCTION_FRAC   = 0.026   # paired capex cut, matching the ~2.5% COP drop (user-specified)

RESULT_COLS = ["status", "q_heat_cap_kwth", "annual_import_kwh", "annual_export_kwh",
               "capex_GBP", "opex_npv_GBP", "total_cost_npv_GBP", "npv_savings_GBP",
               "lifetime_emissions_tco2e"]

BASE_RENAME = {
    "annual_import_kwh": "base_annual_import_kwh", "annual_export_kwh": "base_annual_export_kwh",
    "capex_GBP": "base_capex_GBP", "opex_npv_GBP": "base_opex_npv_GBP",
    "total_cost_npv_GBP": "base_total_cost_npv_GBP", "npv_savings_GBP": "base_npv_savings_GBP",
    "lifetime_emissions_tco2e": "base_lifetime_emissions_tco2e",
}

DISPLAY_COLS = [
    "district", "activity", "grant_status",
    "base_total_cost_npv_GBP", "grant_total_cost_npv_GBP", "total_cost_npv_GBP_delta",
    "base_lifetime_emissions_tco2e", "grant_lifetime_emissions_tco2e", "lifetime_emissions_tco2e_delta",
    "cheap_ashp_pays_off",
]

_RED_FILL   = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
_RED_FONT   = Font(color="9C0006")
_GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
_GREEN_FONT = Font(color="006100")

DELTA_COLS = ["total_cost_npv_GBP_delta", "lifetime_emissions_tco2e_delta"]
BOOL_COL = "cheap_ashp_pays_off"


def apply_conditional_formatting(ws, columns: list, n_rows: int) -> None:
    for col_name in DELTA_COLS:
        col_letter = get_column_letter(columns.index(col_name) + 1)
        cell_range = f"{col_letter}2:{col_letter}{n_rows + 1}"
        ws.conditional_formatting.add(cell_range, CellIsRule(
            operator="greaterThan", formula=["0"], fill=_RED_FILL, font=_RED_FONT))
        ws.conditional_formatting.add(cell_range, CellIsRule(
            operator="lessThan", formula=["0"], fill=_GREEN_FILL, font=_GREEN_FONT))

    col_letter = get_column_letter(columns.index(BOOL_COL) + 1)
    cell_range = f"{col_letter}2:{col_letter}{n_rows + 1}"
    ws.conditional_formatting.add(cell_range, CellIsRule(
        operator="equal", formula=["FALSE"], fill=_RED_FILL, font=_RED_FONT))
    ws.conditional_formatting.add(cell_range, CellIsRule(
        operator="equal", formula=["TRUE"], fill=_GREEN_FILL, font=_GREEN_FONT))


def _latest_run_dir() -> str:
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
    runs = glob.glob(os.path.join(root, "Optimisation (*)"))
    if not runs:
        raise FileNotFoundError(f"No 'Optimisation (*)' run folders found under {root}")
    return max(runs, key=os.path.getmtime)


def _pool_init():
    # Runs once per worker process. Every task this pool ever runs is a Grant-Aerona3 solve, so the
    # ASHP tech-param swap happens once here rather than per task.
    dm.initialize()
    dm.ashp_cop_at_7C = GRANT_COP_AT_7C   # slope (dm.ashp_cop_slope_per_C) left untouched, per direction
    base_capex = oe.HEAT_COSTS["ASHP"]["capex_per_kwth"]
    oe.HEAT_COSTS["ASHP"]["capex_per_kwth"] = base_capex * (1.0 - CAPEX_REDUCTION_FRAC)


def _solve_one(task: tuple) -> dict:
    district, activity, n_pv, e_batt, o_batt, q_heat_cap, e_th, threads, time_limit_s = task
    prob, V = oe.build_milp(district, activity, "ASHP",
                            objective="cost", scenarios=[oe.central_scenario()])
    prob += V["n_pv"]       == n_pv,       "fix_n_pv"
    prob += V["e_batt"]     == e_batt,     "fix_e_batt"
    prob += V["o_batt"]     == o_batt,     "fix_o_batt"
    # Floors, not equalities -- same rationale as pv_panel1_sensitivity.py: the saved CSV rounds
    # both to 1 dp, and thermal capacity is unaffected by the COP change (COP only changes the
    # electrical input needed for a given thermal output), so pinning it exactly risks a spurious
    # infeasibility from rounding rather than reflecting anything about Grant Aerona3 itself.
    prob += V["q_heat_cap"] >= q_heat_cap, "floor_q_heat_cap"
    prob += V["e_th"]       >= e_th,       "floor_e_th"
    solver = oe._make_solver(solver_msg=False, time_limit_s=time_limit_s, threads=threads)
    status = prob.solve(solver)
    res = oe._extract_results(prob, V, status, district, activity, "ASHP", oe.HORIZON_YEARS)
    return {"district": district, "activity": activity,
            **{f"grant_{c}": res.get(c) for c in RESULT_COLS}}


def run_ashp_sensitivity(all_rows: pd.DataFrame, run_dir: str, *, time_limit_s: int = 600,
                         n_jobs: int = None, limit: int = None) -> pd.DataFrame:
    """Library entry point — called by optimisation_model.main() with the fresh deterministic
    cost-round DataFrame, and by this module's own CLI with the same frame read back from
    cost_deterministic_results.csv.

    Passing the in-memory frame is what makes this safe to run in-pipeline: the fixed-sizing
    re-solve pins n_pv to the saved design, so a saved run that predates a change to the roof-area
    cap (or any other sizing bound) goes infeasible on every row. Sourcing the sizing from the run
    that just completed removes that failure mode by construction — see the staleness note in the
    module docstring.
    """
    print(f"Grant Aerona3 COP@7C={GRANT_COP_AT_7C} vs current {CURRENT_ASHP_COP_AT_7C} "
          f"({(GRANT_COP_AT_7C / CURRENT_ASHP_COP_AT_7C - 1) * 100:+.2f}%); "
          f"capex cut applied: {CAPEX_REDUCTION_FRAC * 100:.1f}%")

    base = all_rows[(all_rows["status"] == "Optimal") & (all_rows["heating"] == "ASHP")].reset_index(drop=True)
    base["_rank"] = base.index   # preserves the source file's NPV-savings ranking through the merge
    if limit:
        base = base.iloc[:limit].copy()
    if base.empty:
        print("No Optimal ASHP designs to test — skipping Grant Aerona3 sensitivity.")
        return pd.DataFrame()
    print(f"{len(base)} ASHP designs Optimal (of {len(all_rows)} total rows) -- "
          f"solving Grant Aerona3 dispatch for each")

    logical = os.cpu_count() or 2
    # Physical-core parallelism, matching the rest of the pipeline (see the solver-contention note
    # in optimisation_config): independent large LPs contend for cache/memory bandwidth badly enough
    # to false-fail clean cells if workers x threads is pushed to the logical-core count.
    n_jobs = max(1, n_jobs or logical // 4)
    threads_per_worker = max(1, logical // n_jobs)

    tasks = [(r.district, r.activity, r.n_pv, r.e_batt_kwh, r.o_batt_kw,
              r.q_heat_cap_kwth, r.e_th_kwh, threads_per_worker, time_limit_s)
             for r in base.itertuples()]

    print(f"Solving {len(tasks)} designs across {n_jobs} processes x {threads_per_worker} solver threads...")
    with mp.Pool(processes=n_jobs, initializer=_pool_init) as pool:
        grant_rows = []
        for i, row in enumerate(pool.imap(_solve_one, tasks), 1):
            print(f"  [{i:>3}/{len(tasks)}] {row['activity']!r:27s} {row['district']!r:28s} "
                  f"{row['grant_status']}")
            grant_rows.append(row)
    grant_df = pd.DataFrame(grant_rows)

    base_ren = base.rename(columns=BASE_RENAME)
    keep = ["district", "activity", "_rank", "n_pv", "e_batt_kwh", "o_batt_kw",
            "q_heat_cap_kwth", "e_th_kwh"] + list(BASE_RENAME.values())
    out = base_ren[keep].merge(grant_df, on=["district", "activity"], how="left")
    out = out.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)

    out["annual_import_kwh_delta"]        = out["grant_annual_import_kwh"] - out["base_annual_import_kwh"]
    out["annual_export_kwh_delta"]        = out["grant_annual_export_kwh"] - out["base_annual_export_kwh"]
    out["capex_GBP_delta"]                = out["grant_capex_GBP"] - out["base_capex_GBP"]
    out["opex_npv_GBP_delta"]             = out["grant_opex_npv_GBP"] - out["base_opex_npv_GBP"]
    out["total_cost_npv_GBP_delta"]       = out["grant_total_cost_npv_GBP"] - out["base_total_cost_npv_GBP"]
    out["lifetime_emissions_tco2e_delta"] = out["grant_lifetime_emissions_tco2e"] - out["base_lifetime_emissions_tco2e"]
    # NA (not False) when Grant Aerona3's fixed-sizing dispatch is Infeasible -- distinct from
    # "solved but costs more". Infeasible would mean the lower COP pushes required grid import over
    # the district's DNO connection limit at the frozen sizing.
    is_optimal = out["grant_status"] == "Optimal"
    out["cheap_ashp_pays_off"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[is_optimal, "cheap_ashp_pays_off"] = out.loc[is_optimal, "total_cost_npv_GBP_delta"] < 0

    out_path = os.path.join(run_dir, "ashp_grant_aerona3_sensitivity.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        out[DISPLAY_COLS].to_excel(xw, sheet_name="Grant Aerona3 vs current", index=False)
        apply_conditional_formatting(xw.sheets["Grant Aerona3 vs current"], DISPLAY_COLS, len(out))
        notes = pd.DataFrame({
            "Note": ["Source run", "Current model ASHP COP@7C", "Grant Aerona3 R32 COP@A7/W55",
                     "COP/temperature slope", "Capex reduction applied", "Method"],
            "Value": [
                run_dir,
                f"{CURRENT_ASHP_COP_AT_7C} (CEP Technology Library v1.02, Company 4 R-32 average, units 12-17)",
                f"{GRANT_COP_AT_7C} (manufacturer EN14511 datasheet)",
                "Unchanged from the current model -- no published multi-temperature curve found "
                "for Grant Aerona3 in public datasheets, so cold-weather sensitivity is assumed "
                "identical; only the reference-point COP is lowered.",
                f"{CAPEX_REDUCTION_FRAC * 100:.1f}% cut to HEAT_COSTS['ASHP']['capex_per_kwth'], "
                f"paired with the COP drop (COP itself falls {(GRANT_COP_AT_7C / CURRENT_ASHP_COP_AT_7C - 1) * 100:.2f}%).",
                "First-stage sizing (n_pv, e_batt, o_batt, q_heat_cap, e_th) fixed at the saved "
                "deterministic-round solution; only stage-2 dispatch is re-solved with Grant "
                "Aerona3's COP/capex swapped in. Baseline (base_*) columns are the saved CSV "
                "values, not re-solved. Applies only to designs where ASHP is the heating system.",
            ],
        })
        notes.to_excel(xw, sheet_name="Assumptions", index=False)

    n_infeasible = int((out["grant_status"] != "Optimal").sum())
    n_worse      = int((out["cheap_ashp_pays_off"] == False).sum())
    n_better     = int((out["cheap_ashp_pays_off"] == True).sum())
    print(f"\nWrote {out_path}")
    print(f"{len(out)} ASHP designs total: {n_infeasible} Infeasible on Grant Aerona3 under the "
          f"fixed sizing, {n_worse} solved but net worse off over 15 yr despite the cheaper unit, "
          f"{n_better} solved and net better off.")
    return out


def main():
    ap = argparse.ArgumentParser(description="Standalone re-run against an already-completed "
                                             "optimisation run. The pipeline calls "
                                             "run_ashp_sensitivity() directly instead.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only solve the first N designs (for a quick test run).")
    ap.add_argument("--time-limit", type=int, default=600, dest="time_limit", metavar="S",
                    help="per-design solver time limit in seconds (default: 600)")
    ap.add_argument("--jobs", type=int, default=None, metavar="N",
                    help="parallel worker processes (default: logical cores // 4)")
    args = ap.parse_args()

    run_dir = _latest_run_dir()
    print(f"Latest run: {run_dir}")
    all_rows = pd.read_csv(os.path.join(run_dir, "cost_deterministic_results.csv"))
    run_ashp_sensitivity(all_rows, run_dir, time_limit_s=args.time_limit,
                         n_jobs=args.jobs, limit=args.limit)


if __name__ == "__main__":
    main()
