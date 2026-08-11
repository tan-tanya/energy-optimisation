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
# Demand-side charts (re-exported by demand_profile_model from demand_report).
from demand_profile_model import generate_all_demand_plots


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
                   help="skip the policy_recommendations.py lever back-calculation after the cost "
                        "rounds: per activity class (4 of them, each at its own best-NPV district) "
                        "5 bisections of ~7 re-solves plus a 21-step battery capex sweep, so ~285 "
                        "solves per round. Measured ~3h45m for the stochastic round back when it "
                        "was ~49 solves on one cell — budget most of a day per round now")
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

    # Demand charts first: they need no solve, take ~5 min against a multi-hour sweep, and every
    # (district, activity) pair has its own profile — so the whole set is written, not one cell's.
    # Running before the sweep means they survive an interrupted run.
    print("\nRendering demand profile charts (all districts × activity classes) …")
    generate_all_demand_plots(run_dir)

    # Full run (both objectives) → NPV + Carbon + Pareto workbook.
    print(f"\nNPV + Carbon objectives — cost rounds ({' + '.join(rounds)}) and carbon sweep "
          "across (district, activity, heating) …")
    cost_specs = _cost_round_specs(rounds)
    # One emissions sweep PER ROUND, each priced on that round's scenario set. The emissions
    # objective never reads a price, so the design and its emissions come out identical either way —
    # what changes is the cost reported alongside them. Solving it once on central prices left the
    # stochastic workbook comparing 3-scenario costs against 1-scenario costs in assemble_pareto,
    # which knocked genuinely cost-optimal designs off the front for no reason but the price basis.
    emis_specs = [{"tag": f"emissions_{name}", "objective": "emissions",
                   "scenarios": _scenarios_for_round(name),
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
    df_npv = cost_rounds[headline]   # robust set is the headline when it was solved
    # No per-cell det-vs-sto comparison table is written. The two rounds do not share a price
    # basis -- the stochastic scenario set is right-skewed, so its weighted mean import-price level
    # is ~x1.31 against the deterministic x1.000 (see the Cover) -- and a side-by-side invites the
    # cost gap to be read as a cost of ignoring uncertainty when it is mostly that level shift.
    # The comparison that IS valid is stated in the write-up: first-stage sizing is identical in
    # every cell bar sub-51 kWh moves in the thermal store, so the design is robust across the set.
    if not args.skip_policy:
        import policy_recommendations as polrec
        # Every solved round: the HP rebate is the one output where the det/sto delta is substantive
        # (heat pumps are the most import-price-exposed technology), so both are worth back-calculating.
        # run_policy_recommendations keys its output frames by round name and re-derives each round's
        # scenario set from that key, so passing cost_rounds straight through also honours
        # --skip-stochastic. Every lever runs once per activity class (each in its own best-NPV
        # district), so a round is ~104 tasks: 4 classes x (5 bisections + the 21-step battery capex
        # sweep). n_jobs is capped at logical//4.
        print(f"\nBack-calculating policy levers per end-user type (battery + HP capex + HP "
              f"elec-price + export price + battery capex sweep, {' + '.join(rounds)}) …")
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
        # Both sheets of a workbook now come from the same price basis, so the Pareto dominance
        # test compares like with like.
        df_carb_r = carbon_rounds[name]
        dfp = df_pareto if name == headline else assemble_pareto(df_cost, df_carb_r)
        out = os.path.join(run_dir, f"Optimisation Results ({name}).xlsx")
        print(f"\nBuilding {name} workbook -> {os.path.basename(out)}")
        sheets = {"NPV": df_cost, "Carbon": df_carb_r, "Pareto": dfp}
        write_results_workbook(sheets, out,
                               run_meta={"timestamp": _timestamp, "round": name},
                               scenarios=_scenarios_for_round(name),
                               grid_sensitivity=df_grid_sens)


if __name__ == "__main__":
    main()
