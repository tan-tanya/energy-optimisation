"""
Run: python optimisation_model.py.
Sizes electricity and heating systems per (district, activity, heating), and ranks combos by 15-year NPV vs gas-boiler BAU.

Outputs (in outputs/Optimisation ({timestamp})/):
1. Optimisation Results ({deterministic,stochastic}).xlsx (assembled by optimisation_report); one
   per cost round, each carrying its own sweep rows on the NPV Data sheet   [--skip-stochastic]
2. charts/{deterministic,stochastic}_*.png; every workbook figure as a standalone image
   (for reports / slides), one prefixed set per cost round   [--skip-stochastic]
2b. demand/*.png; the complete demand-side chart set — every district × activity class ×
   heating system × energy type, seasonal / monthly / WD-WE   [demand_report]
3. Policy Recommendations.xlsx                  [--skip-policy]
4. ashp_grant_aerona3_sensitivity.xlsx          [--skip-ashp-sensitivity]
5. pv_panel1_sensitivity.xlsx                   [--skip-pv-sensitivity]

Pipeline:
    1. build_lp()                                   assemble the LP for one (district, activity, heating) [optimisation_engine]
    2. solve_scenario()                             solve and extract sizing + cost + energy metrics      [optimisation_engine]
    3. _cost_round_specs() / _run_merged_sweep()    sweep every (district, activity), rank vs BAU         [optimisation_engine]
    4. write_results_workbook()                     assemble workbook + charts     [optimisation_engine -> optimisation_report]
    5. main()                                       end-to-end run, timestamped output directory
"""

import os
from datetime import datetime

import pandas as pd
from optimisation_config import DEFAULT_TIME_LIMIT_S
from optimisation_engine import (
    COST_ROUNDS, scenarios_for_round, price_scenarios, _ensure_dm_initialized,
    _run_merged_sweep, assemble_pareto, write_results_workbook, _osm_survey_info,
)
# Demand-side charts
from demand_profile_model import generate_all_demand_plots


def _cost_round_specs(rounds: tuple = COST_ROUNDS) -> list:
    specs = []
    for name in rounds:
        scen = scenarios_for_round(name)
        print(f"[cost · {name}] {len(scen)} price scenario(s) "
              f"({'no uncertainty' if name == 'deterministic' else 'import-price uncertainty'}) "
              f"across (district, activity, heating) …")
        specs.append({"tag": name, "objective": "cost", "scenarios": scen, "sort_col": "npv_savings_GBP"})
    return specs


def _read_grid_sensitivity(run_dir: str) -> pd.DataFrame:
    # Reuse a completed run's grid-headroom bisection instead of re-solving it. 
    wb = os.path.join(run_dir, "Optimisation Results (deterministic).xlsx")
    if not os.path.exists(wb):
        raise FileNotFoundError(f"No deterministic workbook in {run_dir} - nothing to reuse the "
                                f"grid-headroom bisection from.")
    if "Grid Sensitivity Data" not in pd.ExcelFile(wb).sheet_names:
        raise ValueError(f"{os.path.basename(wb)} has no 'Grid Sensitivity Data' sheet - that run "
                         f"was solved with --skip-grid-sensitivity, so there is nothing to reuse.")
    df = pd.read_excel(wb, sheet_name="Grid Sensitivity Data")
    return df.drop(columns=["edge_margin_label"], errors="ignore")


def main(argv=None):
    # Entry point - including argument parsing
    import argparse
    p = argparse.ArgumentParser(
        description="Two-stage stochastic optimisation of building PV/battery/heat sizing under "
                    "electricity import-price uncertainty. A full sweep by default: both cost "
                    "rounds (deterministic + stochastic) and the emissions objective, over every "
                    "(district, activity, heating) cell.")
    p.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT_S, dest="time_limit",
                   metavar="S", help=f"per-cell solver time limit in seconds (default: {DEFAULT_TIME_LIMIT_S})")
    p.add_argument("--jobs", type=int, default=None, metavar="N",
                   help="parallel worker processes (default: a quarter of the logical cores, the rest "
                        "given to each worker as solver threads)")
    p.add_argument("--skip-stochastic", action="store_true",
                   help="run the deterministic cost round only")
    p.add_argument("--skip-deterministic", action="store_true",
                   help="run the stochastic cost round only.")
    p.add_argument("--grid-sensitivity-from", metavar="RUN_DIR", default=None,
                   dest="grid_sensitivity_from",
                   help="reuse the grid-headroom bisection from a completed run instead of "
                        "re-solving it, by reading the 'Grid Sensitivity Data' sheet of that run's "
                        "deterministic workbook. Overrides --skip-grid-sensitivity")
    p.add_argument("--skip-policy", action="store_true",
                   help="skip the policy_recommendations.py lever back-calculation after the cost "
                        "rounds: per activity class (4 of them, each at its own best-NPV district)")
    p.add_argument("--skip-grid-sensitivity", action="store_true",
                   help="skip the grid_sensitivity.py DNO ceiling / demand-margin bisection after "
                        "the cost rounds (one bisection per district x activity x heat-pump heating, "
                        "so ~108 extra re-solves on a full sweep)")
    p.add_argument("--skip-ashp-sensitivity", action="store_true",
                   help="skip the Grant Aerona3 ASHP technology sensitivity after the cost rounds "
                        "(one fixed-sizing, dispatch-only re-solve per Optimal ASHP design)")
    p.add_argument("--skip-pv-sensitivity", action="store_true",
                   help="skip the Panel 1 PV technology sensitivity after the cost rounds (one "
                        "fixed-sizing, dispatch-only re-solve per Optimal design, all heating "
                        "systems — so ~4x the ASHP sensitivity's design count)")
    args = p.parse_args(argv)

    _ensure_dm_initialized()

    # One --skip-<round> flag per name in COST_ROUNDS.
    rounds = tuple(n for n in COST_ROUNDS if not getattr(args, f"skip_{n}"))
    if not rounds:
        p.error("--skip-deterministic and --skip-stochastic together leave no cost round to solve.")
    headline = rounds[-1]
    # The technology sensitivities and the grid bisection all pin stage-1 sizing from the
    # deterministic round, so they can only run when it is in memory.
    has_det = "deterministic" in rounds

    print("optimisation_model.py — two-stage stochastic optimisation (PV / battery / heat sizing)")
    if args.skip_stochastic:
        print("Cost rounds: deterministic only  (1 central scenario; --skip-stochastic).")
    elif args.skip_deterministic:
        print(f"Cost rounds: stochastic only  ({len(price_scenarios())} weighted import-price "
              f"scenarios; --skip-deterministic).")
    else:
        print(f"Cost rounds: deterministic + stochastic  (deterministic = 1 central scenario; "
              f"stochastic = {len(price_scenarios())} weighted import-price scenarios).")
    print(f"Scope: every district × activity × heating cell  |  objective=both (NPV + Carbon + Pareto)  "
          f"time_limit={args.time_limit}s")
    print(f"\n{_osm_survey_info()}")

    common = dict(time_limit_s=args.time_limit, n_jobs=args.jobs)
    _timestamp = datetime.now().strftime("%Y%m%d, %H%M")
    run_dir    = os.path.join("outputs", f"Optimisation ({_timestamp})")
    os.makedirs(run_dir, exist_ok=True)

    print("\nRendering demand profile charts (all districts × activity classes) …")
    generate_all_demand_plots(run_dir)

    # Full run (both objectives): NPV + Carbon + Pareto workbook.
    print(f"\nNPV + Carbon objectives — cost rounds ({' + '.join(rounds)}) and carbon sweep "
          "across (district, activity, heating) …")
    cost_specs = _cost_round_specs(rounds)
    # One emissions sweep per round, each priced on that round's scenario set. 
    emis_specs = [{"tag": f"emissions_{name}", "objective": "emissions",
                   "scenarios": scenarios_for_round(name),
                   "sort_col": "emissions_saving_tco2e"} for name in rounds]
    specs = cost_specs + emis_specs
    merged = _run_merged_sweep(specs, **common)
    cost_rounds = {}
    for spec in cost_specs:
        df = merged[spec["tag"]]
        df["round"] = spec["tag"]
        cost_rounds[spec["tag"]] = df
    carbon_rounds = {name: merged[f"emissions_{name}"] for name in rounds}
    df_carbon = carbon_rounds[headline]
    df_npv = cost_rounds[headline]
    if not args.skip_policy:
        import policy_recommendations as polrec
        print(f"\nBack-calculating policy levers per end-user type (battery + HP capex + HP "
              f"elec-price + export price + battery capex sweep, {' + '.join(rounds)}) …")
        polrec.run_policy_recommendations(
            cost_rounds,
            os.path.join(run_dir, "Policy Recommendations.xlsx"),
            time_limit_s=args.time_limit)
    # Technology sensitivities: fixed-sizing, dispatch-only re-solves off the deterministic round.
    import ashp_grant_aerona3_sensitivity as ashpsens
    import pv_panel1_sensitivity as pvsens
    for skip, run_fn, label in (
            (args.skip_ashp_sensitivity, ashpsens.run_ashp_sensitivity, "Grant Aerona3 ASHP"),
            (args.skip_pv_sensitivity, pvsens.run_pv_sensitivity, "Panel 1 PV")):
        if skip:
            continue
        if not has_det:
            print(f"\n(skipping {label} sensitivity - it pins sizing from the deterministic "
                  f"round, which this run did not solve)")
            continue
        print(f"\n{label} sensitivity (fixed sizing, dispatch-only re-solve, deterministic round) …")
        run_fn(cost_rounds["deterministic"], run_dir,
               time_limit_s=args.time_limit, n_jobs=args.jobs)
    df_grid_sens = pd.DataFrame()
    if args.grid_sensitivity_from:
        df_grid_sens = _read_grid_sensitivity(args.grid_sensitivity_from)
        print(f"\nReusing the grid-connection headroom bisection from "
              f"{args.grid_sensitivity_from} ({len(df_grid_sens)} cells) - not re-solving it.")
    elif not args.skip_grid_sensitivity:
        if not has_det:
            print("\n(skipping the grid-headroom bisection - it pins sizing from the "
                  "deterministic round, which this run did not solve. Pass "
                  "--grid-sensitivity-from RUN_DIR to carry a completed run's "
                  "'Grid Sensitivity Data' sheet into this workbook instead)")
        else:
            import grid_sensitivity as gridsens
            print("\nBisecting grid-connection headroom (demand-growth margin / ceiling threshold, "
                  "deterministic scenario, heat-pump cells only) …")
            df_grid_sens = gridsens.run_grid_sensitivity(
                cost_rounds["deterministic"],
                time_limit_s=args.time_limit)
    # Objective 3: NPV/carbon Pareto, assembled from the two completed sweeps.
    df_pareto = assemble_pareto(df_npv, df_carbon)

    # Rank only proven-optimal designs.
    npv_opt    = df_npv[df_npv["status"] == "Optimal"] if "status" in df_npv else df_npv
    carbon_opt = df_carbon[df_carbon["status"] == "Optimal"] if "status" in df_carbon else df_carbon
    n_drop_npv = len(df_npv) - len(npv_opt)
    n_drop_car = len(df_carbon) - len(carbon_opt)
    if n_drop_npv or n_drop_car:
        print(f"\n(excluded from rankings — not proven optimal: {n_drop_npv} NPV, "
              f"{n_drop_car} carbon cells)")

    for name, df_cost in cost_rounds.items():
        df_carb_r = carbon_rounds[name]
        dfp = df_pareto if name == headline else assemble_pareto(df_cost, df_carb_r)
        out = os.path.join(run_dir, f"Optimisation Results ({name}).xlsx")
        print(f"\nBuilding {name} workbook -> {os.path.basename(out)}")
        sheets = {"NPV": df_cost, "Carbon": df_carb_r, "Pareto": dfp}
        write_results_workbook(sheets, out,
                               run_meta={"timestamp": _timestamp, "round": name},
                               scenarios=scenarios_for_round(name),
                               grid_sensitivity=df_grid_sens)


if __name__ == "__main__":
    main()
