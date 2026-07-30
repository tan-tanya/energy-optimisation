"""
Run: python optimisation_model.py.
Sizes electricity and heating systems per (district, activity, heating), and ranks combos by 15-year NPV vs gas-boiler BAU.

Outputs (in outputs/Optimisation ({timestamp})/):
1. Optimisation Results.xlsx (assembled by optimisation_report)
2. charts/{deterministic,stochastic}_*.png; every workbook figure as a standalone image
   (for reports / slides), one prefixed set per cost round   [--skip-stochastic]
3. Policy Recommendations.xlsx                  [--skip-policy]
4. ashp_grant_aerona3_sensitivity.xlsx          [--skip-ashp-sensitivity]
5. pv_panel1_sensitivity.xlsx                   [--skip-pv-sensitivity]
   (grid sensitivity has no standalone file — it lands in the main workbook)

Pipeline:
    1. build_milp()                                 assemble the LP/MILP for one (district, activity)    [optimisation_engine]
    2. solve_scenario()                             solve and extract sizing + cost + energy metrics     [optimisation_engine]
    3. _cost_round_specs() / _run_merged_sweep()    sweep every (district, activity), rank vs BAU
    4. write_results_workbook()                     write the ranking sheets                             [optimisation_engine]
    5. main()                                       end-to-end run, timestamped output directory
"""

import os
from datetime import datetime

import pandas as pd
from optimisation_config import DEFAULT_TIME_LIMIT_S

# The model engine (PV/demand/COP/BAU physics, single-cell build+solve, sweep/pool mechanics).
from optimisation_engine import (
    price_scenarios, central_scenario, set_price_scenarios, _ensure_dm_initialized,
    build_milp, solve_scenario, _make_solver, effective_import_limit_kw,
    building_demand_kwh, building_heat_demand_kwh,
    elec_growth_factors, heat_growth_factors,
    rank_all_combinations, _run_merged_sweep, assemble_pareto, write_results_workbook,
    _osm_survey_info,
)


# Two-round cost optimisation:
# Round 1 (deterministic): a single central price scenario.
# Round 2 (stochastic):    the reduced weighted price-scenario set.
COST_ROUNDS = ("deterministic", "stochastic")

def _scenarios_for_round(name: str) -> list:
    return [central_scenario()] if name == "deterministic" else price_scenarios()

def _cost_round_specs(rounds: tuple = COST_ROUNDS) -> list:
    # Build the _run_merged_sweep spec for the requested cost rounds.
    specs = []
    for name in rounds:
        scen = _scenarios_for_round(name)
        print(f"[cost · {name}] {len(scen)} price scenario(s) "
              f"({'no uncertainty' if name == 'deterministic' else 'import-price uncertainty'}) "
              f"across (district, activity, heating) …")
        specs.append({"tag": name, "objective": "cost", "scenarios": scen, "sort_col": "npv_savings_GBP"})
    return specs

_CMP_METRICS = ["pv_kwp", "e_batt_kwh", "e_th_kwh", "q_heat_cap_kwth", "capex_GBP",
                "total_cost_npv_GBP", "npv_savings_GBP", "payback_years"]

def _compare_cost_rounds(df_det: pd.DataFrame, df_sto: pd.DataFrame) -> pd.DataFrame:
    # Per-cell side-by-side of the deterministic vs stochastic design, with sizing/cost deltas.
    keys = ["district", "activity", "heating"]
    cols = ["status"] + [c for c in _CMP_METRICS if c in df_det.columns and c in df_sto.columns]
    m = df_det[keys + cols].merge(df_sto[keys + cols], on=keys, suffixes=("_det", "_sto"))
    for c in ("pv_kwp", "e_batt_kwh", "e_th_kwh", "q_heat_cap_kwth", "total_cost_npv_GBP"):
        if f"{c}_det" in m and f"{c}_sto" in m:
            m[f"delta_{c}"] = m[f"{c}_sto"] - m[f"{c}_det"]
    return m



def main(argv=None):
    # CLI entry point; always a full run.
    import argparse
    p = argparse.ArgumentParser(
        description="Two-stage stochastic optimisation of building PV/battery/heat sizing under "
                    "electricity import-price uncertainty. A full sweep by default: both cost "
                    "rounds (deterministic + stochastic) and the emissions objective, over every "
                    "(district, activity, heating) cell.")
    p.add_argument("--time-limit", type=int, default=DEFAULT_TIME_LIMIT_S, dest="time_limit",
                   metavar="S", help=f"per-cell solver time limit in seconds (default: {DEFAULT_TIME_LIMIT_S})")
    p.add_argument("--jobs", type=int, default=None, metavar="N",
                   help="parallel worker processes (default: CPU cores - 1; 1 = serial, easier to debug)")
    p.add_argument("--skip-stochastic", action="store_true",
                   help="run the deterministic cost round only (single central price scenario), "
                        "skipping the weighted import-price scenario set. Roughly halves the cost "
                        "sweep; drops the det-vs-sto comparison CSV, the stochastic workbook, and "
                        "the stochastic half of the policy rebates")
    p.add_argument("--skip-policy", action="store_true",
                   help="skip the policy_recommendations.py rebate back-calculation after the cost "
                        "rounds (4 bisections per round on the single best-NPV cell, ~7 re-solves "
                        "each; measured ~3h45m for the stochastic round, ~4-5h30m for both)")
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

    rounds = ("deterministic",) if args.skip_stochastic else COST_ROUNDS
    # Headline round: the robust (stochastic) set when it was solved, else the deterministic one.
    headline = rounds[-1]

    print("optimisation_model.py — two-stage stochastic optimisation (PV / battery / heat sizing)")
    if args.skip_stochastic:
        print("Cost rounds: deterministic only  (1 central scenario; --skip-stochastic).")
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

    # Full run (both objectives) → NPV + Carbon + Pareto workbook.
    print(f"\nNPV + Carbon objectives — cost rounds ({' + '.join(rounds)}) and carbon sweep "
          "across (district, activity, heating) …")
    cost_specs = _cost_round_specs(rounds)
    specs = cost_specs + [{"tag": "emissions", "objective": "emissions", "scenarios": None,
                           "sort_col": "emissions_saving_tco2e"}]
    merged = _run_merged_sweep(specs, **common)
    cost_rounds = {}
    for spec in cost_specs:
        df = merged[spec["tag"]]
        df["round"] = spec["tag"]
        cost_rounds[spec["tag"]] = df
    df_carbon = merged["emissions"]
    df_npv = cost_rounds[headline]   # robust set is the headline when it was solved
    for name, df in cost_rounds.items():
        df.to_csv(os.path.join(run_dir, f"cost_{name}_results.csv"), index=False)
    if len(cost_rounds) > 1:
        cmp = _compare_cost_rounds(cost_rounds["deterministic"], cost_rounds["stochastic"])
        cmp.to_csv(os.path.join(run_dir, "cost_round_comparison.csv"), index=False)
    if not args.skip_policy:
        import policy_recommendations as polrec
        # Every solved round: the HP rebate is the one output where the det/sto delta is substantive
        # (heat pumps are the most import-price-exposed technology), so both are worth back-calculating.
        # run_policy_recommendations keys its output frames by round name and re-derives each round's
        # scenario set from that key, so passing cost_rounds straight through also honours
        # --skip-stochastic. Costs ~4 tasks per round, and n_jobs is capped at logical//4.
        print(f"\nBack-calculating policy rebates (battery + HP capex + HP elec-price, "
              f"{' + '.join(rounds)}) …")
        polrec.run_policy_recommendations(
            cost_rounds,
            os.path.join(run_dir, "Policy Recommendations.xlsx"),
            time_limit_s=args.time_limit)
    # Technology sensitivities: fixed-sizing, dispatch-only re-solves off the deterministic round.
    # Imported lazily — each pulls in optimisation_engine and builds a worker pool of its own.
    import ashp_grant_aerona3_sensitivity as ashpsens
    import pv_panel1_sensitivity as pvsens
    for skip, run_fn, label in (
            (args.skip_ashp_sensitivity, ashpsens.run_ashp_sensitivity, "Grant Aerona3 ASHP"),
            (args.skip_pv_sensitivity, pvsens.run_pv_sensitivity, "Panel 1 PV")):
        if skip:
            continue
        print(f"\n{label} sensitivity (fixed sizing, dispatch-only re-solve, deterministic round) …")
        run_fn(cost_rounds["deterministic"], run_dir,
               time_limit_s=args.time_limit, n_jobs=args.jobs)
    df_grid_sens = pd.DataFrame()
    if not args.skip_grid_sensitivity:
        import grid_sensitivity as gridsens
        print("\nBisecting grid-connection headroom (demand-growth margin / ceiling threshold, "
              "deterministic scenario, heat-pump cells only) …")
        df_grid_sens = gridsens.run_grid_sensitivity(
            cost_rounds["deterministic"],
            time_limit_s=args.time_limit)
    # Objective 3 — NPV/carbon Pareto, assembled from the two completed sweeps.
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
        dfp = df_pareto if df_cost is df_npv else assemble_pareto(df_cost, df_carbon)
        out = os.path.join(run_dir, f"Optimisation Results ({name}).xlsx")
        print(f"\nBuilding {name} workbook -> {os.path.basename(out)}")
        write_results_workbook({"NPV": df_cost, "Carbon": df_carbon, "Pareto": dfp}, out,
                               run_meta={"timestamp": _timestamp, "round": name},
                               scenarios=_scenarios_for_round(name),
                               grid_sensitivity=df_grid_sens)


if __name__ == "__main__":
    main()
