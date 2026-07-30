"""PV technology sensitivity. Runs as part of the pipeline (optimisation_model.main() calls
run_pv_sensitivity() after the cost rounds; --skip-pv-sensitivity to opt out), and also standalone
against the most recent completed run:
    python pv_panel1_sensitivity.py [--limit N] [--time-limit S] [--jobs N]

Prefer the in-pipeline path. This is a FIXED-SIZING re-solve: it pins n_pv to the saved design, so
running it standalone against a run that predates any change to the roof-area cap (or another
sizing bound) makes every row infeasible — the same trap that blocked the sibling ASHP sensitivity
between 2026-07-23 and 2026-07-27. Driving it from the run that just finished removes the failure
mode. This module covers ALL heating systems, so it is ~4x the design count of the ASHP one.

Compares the model's current PV spec (CEP Technology Library v1.02, Panels 16-18 average) against
Panel 1 (low-end) for every Optimal design in the deterministic cost round.

Method: for each (district, activity, heating) design, first-stage sizing (n_pv, e_batt, o_batt,
q_heat_cap, e_th) is fixed at the saved deterministic-round solution and only the stage-2 dispatch
is re-solved with Panel 1's efficiency / temperature coefficient / module kWp / module area
substituted for Panels 16-18, and capex_per_kwp scaled by Panel 1's relative price discount
(same CEP source/year as Panels 16-18, so internally consistent -- see PRICE_RATIO). This is far
cheaper than a full sizing+dispatch re-optimisation. Baseline (Panels 16-18) metrics are read
straight from the saved CSV rather than re-solved -- validated against a fresh re-solve for one
cell on 2026-07-08 (reproduced the saved total_cost_npv_GBP to within GBP 9 on a GBP 4.47m NPV).

Output: <run folder>/pv_panel1_sensitivity.xlsx, one row per design, in the same NPV-savings-ranked
row order as the source cost_deterministic_results.csv.
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

# CEP Technology Library v1.02, "PV (monocrystalline silicon)" sheet, .DOCS/Technology Library
# Version 1.02.xlsx. Panels 16-18 (current model default) = eff 20.9%, temp coeff -0.26%/K,
# 0.365 kWp, 1.75 m^2, price/Wp mean(GBP0.6389, GBP0.6438, GBP0.7081). Panel 1 (low-end) below.
PANEL_1 = dict(efficiency=0.1752, temp_coeff_per_C=-0.0040, module_kwp=0.285, module_area_m2=1.63)
PANEL_1_PRICE_PER_WP = 0.4737
PANELS_16_18_MEAN_PRICE_PER_WP = (0.6389 + 0.6438 + 0.7081) / 3.0
PRICE_RATIO = PANEL_1_PRICE_PER_WP / PANELS_16_18_MEAN_PRICE_PER_WP   # applied to capex_per_kwp

RESULT_COLS = ["status", "pv_kwp", "annual_pv_gen_kwh", "annual_import_kwh", "annual_export_kwh",
               "capex_GBP", "opex_npv_GBP", "total_cost_npv_GBP", "npv_savings_GBP",
               "lifetime_emissions_tco2e"]

BASE_RENAME = {
    "pv_kwp": "base_pv_kwp", "annual_pv_gen_kwh": "base_annual_pv_gen_kwh",
    "annual_import_kwh": "base_annual_import_kwh", "annual_export_kwh": "base_annual_export_kwh",
    "capex_GBP": "base_capex_GBP", "opex_npv_GBP": "base_opex_npv_GBP",
    "total_cost_npv_GBP": "base_total_cost_npv_GBP", "npv_savings_GBP": "base_npv_savings_GBP",
    "lifetime_emissions_tco2e": "base_lifetime_emissions_tco2e",
}

# Sheet output is trimmed to identifiers + the two headline comparisons (lifetime NPV, lifetime
# carbon) rather than every intermediate sizing/dispatch column computed above -- those stay
# available in `out` for anyone extending this script, just not written to the workbook.
DISPLAY_COLS = [
    "district", "activity", "heating", "panel1_status",
    "base_total_cost_npv_GBP", "panel1_total_cost_npv_GBP", "total_cost_npv_GBP_delta",
    "base_lifetime_emissions_tco2e", "panel1_lifetime_emissions_tco2e", "lifetime_emissions_tco2e_delta",
    "cheap_panel_pays_off",
]

# Excel's standard "Good"/"Bad" cell-style colours.
_RED_FILL   = PatternFill(start_color="FFC7CE", end_color="FFC7CE", fill_type="solid")
_RED_FONT   = Font(color="9C0006")
_GREEN_FILL = PatternFill(start_color="C6EFCE", end_color="C6EFCE", fill_type="solid")
_GREEN_FONT = Font(color="006100")

DELTA_COLS = ["total_cost_npv_GBP_delta", "lifetime_emissions_tco2e_delta"]
BOOL_COL = "cheap_panel_pays_off"


def apply_conditional_formatting(ws, columns: list, n_rows: int) -> None:
    # Delta columns: positive (Panel 1 worse) = red, negative (Panel 1 better) = green.
    # BOOL_COL: TRUE (pays off) = green, FALSE = red. NA cells (Infeasible on Panel 1) match neither.
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
    # Runs once per worker process. Every task this pool ever runs is a Panel-1 solve, so the
    # PV tech-param swap happens once here rather than per task.
    dm.initialize()
    base_capex_per_kwp = oe.TECH_COSTS["pv"]["capex_per_kwp"]
    oe.TECH_COSTS["pv"].update(PANEL_1)
    oe.TECH_COSTS["pv"]["capex_per_kwp"] = base_capex_per_kwp * PRICE_RATIO


def _solve_one(task: tuple) -> dict:
    district, activity, heating, n_pv, e_batt, o_batt, q_heat_cap, e_th, threads, time_limit_s = task
    prob, V = oe.build_milp(district, activity, heating,
                            objective="cost", scenarios=[oe.central_scenario()])
    prob += V["n_pv"]       == n_pv,       "fix_n_pv"
    prob += V["e_batt"]     == e_batt,     "fix_e_batt"
    prob += V["o_batt"]     == o_batt,     "fix_o_batt"
    # q_heat_cap/e_th are floors, not equalities: the saved CSV rounds both to 1 dp
    # (optimisation_engine.py:1075-6), and heat capacity is normally sized to exactly match peak
    # heat demand at the cost optimum -- a rounded-down value can land fractionally below what's
    # physically required, which is infeasible regardless of the PV swap (confirmed by bisecting
    # the fixed constraints on a failing cell: infeasibility appeared as soon as q_heat_cap was
    # pinned by ==, before e_th was even added). A >= floor costs at most a rounding error's worth
    # of extra heat capacity/store if the solver needs it, and is a no-op otherwise.
    prob += V["q_heat_cap"] >= q_heat_cap, "floor_q_heat_cap"
    prob += V["e_th"]       >= e_th,       "floor_e_th"
    solver = oe._make_solver(solver_msg=False, time_limit_s=time_limit_s, threads=threads)
    status = prob.solve(solver)
    res = oe._extract_results(prob, V, status, district, activity, heating, oe.HORIZON_YEARS)
    return {"district": district, "activity": activity, "heating": heating,
            **{f"panel1_{c}": res.get(c) for c in RESULT_COLS}}


def run_pv_sensitivity(all_rows: pd.DataFrame, run_dir: str, *, time_limit_s: int = 600,
                       n_jobs: int = None, limit: int = None) -> pd.DataFrame:
    """Library entry point — called by optimisation_model.main() with the fresh deterministic
    cost-round DataFrame, and by this module's own CLI with the same frame read back from
    cost_deterministic_results.csv.

    Passing the in-memory frame is what makes this safe to run in-pipeline: the fixed-sizing
    re-solve pins n_pv to the saved design, so a saved run that predates a change to the roof-area
    cap (or any other sizing bound) goes infeasible on every row. Sourcing the sizing from the run
    that just completed removes that failure mode by construction — see the module docstring.
    """
    print(f"Panel 1 / Panels 16-18 price ratio applied to capex_per_kwp: {PRICE_RATIO:.4f}")

    base = all_rows[all_rows["status"] == "Optimal"].reset_index(drop=True)
    base["_rank"] = base.index   # preserves the source file's NPV-savings ranking through the merge
    if limit:
        base = base.iloc[:limit].copy()
    if base.empty:
        print("No Optimal designs to test — skipping Panel 1 sensitivity.")
        return pd.DataFrame()
    print(f"{len(base)} of {len(all_rows)} designs Optimal -- solving Panel 1 dispatch for each")

    logical = os.cpu_count() or 2
    # Physical-core parallelism, matching the rest of the pipeline: independent large LPs contend
    # for cache/memory bandwidth badly enough to false-fail clean cells if workers x threads is
    # pushed to the logical-core count.
    n_jobs = max(1, n_jobs or logical // 4)
    threads_per_worker = max(1, logical // n_jobs)

    tasks = [(r.district, r.activity, r.heating, r.n_pv, r.e_batt_kwh, r.o_batt_kw,
              r.q_heat_cap_kwth, r.e_th_kwh, threads_per_worker, time_limit_s)
             for r in base.itertuples()]

    print(f"Solving {len(tasks)} designs across {n_jobs} processes x {threads_per_worker} solver threads...")
    with mp.Pool(processes=n_jobs, initializer=_pool_init) as pool:
        panel1_rows = []
        for i, row in enumerate(pool.imap(_solve_one, tasks), 1):
            print(f"  [{i:>3}/{len(tasks)}] {row['activity']!r:27s} {row['heating']!r:16s} "
                  f"{row['district']!r:28s} {row['panel1_status']}")
            panel1_rows.append(row)
    panel1_df = pd.DataFrame(panel1_rows)

    base_ren = base.rename(columns=BASE_RENAME)
    keep = ["district", "activity", "heating", "_rank", "n_pv", "e_batt_kwh", "o_batt_kw",
            "q_heat_cap_kwth", "e_th_kwh"] + list(BASE_RENAME.values())
    out = base_ren[keep].merge(panel1_df, on=["district", "activity", "heating"], how="left")
    out = out.sort_values("_rank").drop(columns="_rank").reset_index(drop=True)

    out["pv_kwp_pct_change"]              = 100 * (out["panel1_pv_kwp"] - out["base_pv_kwp"]) / out["base_pv_kwp"]
    out["annual_pv_gen_kwh_delta"]        = out["panel1_annual_pv_gen_kwh"] - out["base_annual_pv_gen_kwh"]
    out["annual_pv_gen_pct_change"]       = 100 * out["annual_pv_gen_kwh_delta"] / out["base_annual_pv_gen_kwh"]
    out["annual_import_kwh_delta"]        = out["panel1_annual_import_kwh"] - out["base_annual_import_kwh"]
    out["annual_export_kwh_delta"]        = out["panel1_annual_export_kwh"] - out["base_annual_export_kwh"]
    out["capex_GBP_delta"]                = out["panel1_capex_GBP"] - out["base_capex_GBP"]
    out["opex_npv_GBP_delta"]             = out["panel1_opex_npv_GBP"] - out["base_opex_npv_GBP"]
    out["total_cost_npv_GBP_delta"]       = out["panel1_total_cost_npv_GBP"] - out["base_total_cost_npv_GBP"]
    out["lifetime_emissions_tco2e_delta"] = out["panel1_lifetime_emissions_tco2e"] - out["base_lifetime_emissions_tco2e"]
    # NA (not False) when Panel 1's fixed-sizing dispatch is Infeasible -- distinct from "solved but
    # costs more". Infeasible means the design can't run on Panel 1 without a resize (e.g. reduced
    # daytime generation pushes required grid import over the district's DNO connection limit).
    is_optimal = out["panel1_status"] == "Optimal"
    out["cheap_panel_pays_off"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out.loc[is_optimal, "cheap_panel_pays_off"] = out.loc[is_optimal, "total_cost_npv_GBP_delta"] < 0

    out_path = os.path.join(run_dir, "pv_panel1_sensitivity.xlsx")
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        out[DISPLAY_COLS].to_excel(xw, sheet_name="Panel1 vs 16-18", index=False)
        apply_conditional_formatting(xw.sheets["Panel1 vs 16-18"], DISPLAY_COLS, len(out))
        notes = pd.DataFrame({
            "Note": ["Source run", "Panel 1 (low-end)", "Panels 16-18 (model default)",
                     "Price ratio applied to capex_per_kwp", "Method"],
            "Value": [
                run_dir,
                "eff 17.52%, temp coeff -0.40%/K, 0.285 kWp, 1.63 m^2 -- CEP Technology Library v1.02, Panel 1",
                "eff 20.9%, temp coeff -0.26%/K, 0.365 kWp, 1.75 m^2 -- CEP Technology Library v1.02, mean of Panels 16-18",
                f"{PRICE_RATIO:.4f} (Panel 1 / mean(Panels 16-18) GBP/Wp, same source+year)",
                "First-stage sizing (n_pv, e_batt, o_batt, q_heat_cap, e_th) fixed at the saved "
                "deterministic-round solution; only stage-2 dispatch is re-solved with Panel 1's PV "
                "params swapped in. Baseline (base_*) columns are the saved CSV values, not re-solved.",
            ],
        })
        notes.to_excel(xw, sheet_name="Assumptions", index=False)

    n_infeasible = int((out["panel1_status"] != "Optimal").sum())
    n_worse      = int((out["cheap_panel_pays_off"] == False).sum())
    n_better     = int((out["cheap_panel_pays_off"] == True).sum())
    print(f"\nWrote {out_path}")
    print(f"{len(out)} designs total: {n_infeasible} Infeasible on Panel 1 under the fixed sizing "
          f"(can't run without a resize -- see panel1_status), {n_worse} solved but net worse off "
          f"over 15 yr despite the cheaper panel, {n_better} solved and net better off.")
    return out


def main():
    ap = argparse.ArgumentParser(description="Standalone re-run against an already-completed "
                                             "optimisation run. The pipeline calls "
                                             "run_pv_sensitivity() directly instead.")
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
    run_pv_sensitivity(all_rows, run_dir, time_limit_s=args.time_limit,
                       n_jobs=args.jobs, limit=args.limit)


if __name__ == "__main__":
    main()
