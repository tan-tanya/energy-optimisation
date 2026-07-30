"""
Import-only (by optimisation_model.py (main()) after cost rounds).

For every (district, activity) x heat-pump-heating cell, bisects how close the DNO grid-import connection ceiling is to infeasibility:

  - Cells that solve Optimal: bisects a uniform demand-level multiplier upward until the cell turns infeasible.
    Reported as `demand_growth_margin_pct` (how much additional demand the present-day grid ceiling can absorb).
    Each trial solve PINS stage-1 sizing to the baseline design — fixed n_pv, fixed thermal store,
    and NO battery (see _solve_fixed). Without that the margin is meaningless: e_batt is a decision
    variable bounded at 50 MWh, so a free re-solve reaches feasibility at absurd multipliers by
    buying grid-scale storage purely to flatten the winter peak under the ceiling. Closing that
    escape hatch roughly halved the office and department-store margins (2026-07-29).

  - Cells that are Infeasible / No incumbent: bisects the import-ceiling override upward until the cell turns Optimal.
    Reported as `grid_import_threshold_kw` (the ceiling that would be needed) and `grid_shortfall_kw` (how much bigger than today's ceiling that is).

Both bisections are run under the deterministic price scenario rather than the full stochastic set: 
feasibility is a physical property of demand/COP/grid-ceiling data shared across all price scenarios.

Only the three heat-pump technologies are scoped, as Gas Boiler never draws heat-pump electricity.
Its grid margin is set by non-heat demand alone, which the model's own baseline-peak floor already guarantees feasible).

SCOPE OF THE RESULT — the margin is PER BUILDING. The DNO headroom it is measured against is already
net of everything currently connected (headroom = firm capacity - existing load), but the whole of
that spare capacity is allocated to this one site, and no other connection behind the same substation
is assumed to grow. It is therefore an upper bound. 
"""
import os
import multiprocessing as mpr

import numpy as np
import pandas as pd

import model_params as mp

_HP_HEATINGS = ("ASHP", "GSHP (vertical)", "GSHP (horizontal)")


def _bisect_boundary(test, lo: float, hi_start: float, *, tol: float, grow: bool,
                     max_expand: int = 8, max_iter: int = 12, expand_factor: float = 2.0):
    # grow=True:  test(lo) is False (infeasible) and becomes True as x increases. 
    #             Returns the smallest x where test(x) is True, and whether a crossing was actually bracketed.
    # grow=False: test(lo) is True (feasible) and becomes False as x increases. 
    #             Returns the largest x where test(x) is still True, and whether a crossing was actually bracketed.
    if grow:
        a, b = lo, hi_start
        found = test(b)
        tries = 0
        while not found and tries < max_expand:
            a = b
            b = b * expand_factor
            found = test(b)
            tries += 1
        if not found:
            return b, False
        for _ in range(max_iter):
            if b - a <= tol:
                break
            mid = (a + b) / 2.0
            if test(mid):
                b = mid
            else:
                a = mid
        return b, True
    else:
        a, b = lo, hi_start
        still_ok = test(b)
        tries = 0
        while still_ok and tries < max_expand:
            a = b
            b = b * expand_factor
            still_ok = test(b)
            tries += 1
        if still_ok:
            return b, False
        for _ in range(max_iter):
            if b - a <= tol:
                break
            mid = (a + b) / 2.0
            if test(mid):
                a = mid
            else:
                b = mid
        return a, True


def _solve_fixed(om, district, activity, heating, *, scenarios, time_limit_s, threads,
                 demand_multiplier, sizing) -> str:
    # Demand-margin solve with stage-1 sizing PINNED to the baseline design, mirroring the
    # fixed-sizing/dispatch-only pattern of pv_panel1_sensitivity.py.
    #
    # Why this is not just a free re-solve: e_batt is a decision variable bounded by BATT_MAX_KWH
    # (50 MWh). Left free, the optimiser reaches feasibility at absurd demand multipliers by buying
    # grid-scale storage purely to shave the winter peak under the import ceiling — measured
    # 2026-07-29 on Midlands office at 16.1x demand, where e_batt, o_batt AND n_pv all sat exactly
    # on their bounds. The margin then reports "how far demand can grow before a maxed-out 50 MWh
    # battery can no longer hide the peak", not "how far before the grid ceiling binds", and the
    # "how many buildings could connect" reading it is used for does not survive that.
    #
    # What is pinned, and why each:
    #   n_pv    fixed  — roof area is physical and does not grow with demand
    #   e_batt  ZERO   — no storage escape hatch (baseline installs none anywhere in this run)
    #   o_batt  ZERO   — ditto
    #   e_th    fixed  — a thermal store is the same escape hatch by another route, so hold it at
    #                    the design value rather than letting it absorb the peak
    #   q_heat  FREE   — must be allowed to grow, or the bisection finds the heat-capacity edge
    #                    instead of the grid edge and measures the wrong thing entirely
    import pulp
    import optimisation_engine as oe
    n_pv, e_th = sizing
    prob, V = oe.build_milp(district, activity, heating, objective="cost", scenarios=scenarios,
                            demand_multiplier=demand_multiplier)
    prob += V["n_pv"]   == n_pv, "fix_n_pv"
    prob += V["e_batt"] == 0.0,  "no_batt_energy"
    prob += V["o_batt"] == 0.0,  "no_batt_power"
    prob += V["e_th"]   >= e_th, "fix_e_th"      # floor: saved value is rounded to 1 dp
    prob += V["e_th"]   <= e_th + 1.0, "cap_e_th"
    status = prob.solve(oe._make_solver(False, time_limit_s, threads=threads))
    return oe._extract_results(prob, V, status, district, activity, heating,
                               oe.HORIZON_YEARS).get("status")


def _cell_row(om, district: str, activity: str, heating: str, status: str, *, scenarios,
             time_limit_s, threads, max_iter, tol_demand, sizing=None) -> dict:
    # Report against the ceiling the model applies. The engine floors the DNO ceiling at the building's own final-year non-heat peak, 
    # so for large buildings (hospitals, department stores) the nameplate can be far below the operative limit.
    nameplate_kw = mp.GRID_LIMITS[district]["import_kw"]
    current_kw   = om.effective_import_limit_kw(district, activity)
    row = {
        "district": district, "activity": activity, "heating": heating,
        "baseline_status": status,
        "grid_import_limit_kw": round(current_kw, 1),
        "dno_nameplate_limit_kw": round(nameplate_kw, 1),
        "limit_is_floored": bool(current_kw > nameplate_kw + 1e-6),
    }
    if status == "Optimal":
        def test(mult):
            return _solve_fixed(om, district, activity, heating, scenarios=scenarios,
                                time_limit_s=time_limit_s, threads=threads,
                                demand_multiplier=mult, sizing=sizing) == "Optimal"

        # max_expand=11: guarantees the expansion reaches at least 104x demand before conceding "open-ended".
        mult, bounded = _bisect_boundary(test, lo=1.0, hi_start=1.2, tol=tol_demand,
                                         grow=False, expand_factor=1.5, max_iter=max_iter,
                                         max_expand=11)
        margin_pct = (mult - 1.0) * 100.0
        # Approximate kW-equivalent of the demand-growth margin: current_kw scaled by the same %
        # (this assumes import draw grows roughly proportionally with demand — reasonable near the margin).
        row.update({
            "mode": "demand_margin",
            "demand_growth_margin_pct": round(margin_pct, 1),
            "search_bounded": bounded,
            "grid_import_threshold_kw": np.nan,
            "grid_shortfall_kw": np.nan,
            "edge_margin_pct": round(margin_pct, 1),
            "edge_margin_kw": round(current_kw * margin_pct / 100.0, 1),
        })
    else:
        def test(kw):
            r = om.solve_scenario(district, activity, heating, scenarios=scenarios,
                                  time_limit_s=time_limit_s, threads=threads,
                                  import_limit_override_kw=kw)
            return r.get("status") == "Optimal"

        hi_start = max(current_kw * 1.5, current_kw + 50.0)
        tol_kw = max(1.0, current_kw * 0.01)
        kw, bounded = _bisect_boundary(test, lo=current_kw, hi_start=hi_start, tol=tol_kw,
                                       grow=True, expand_factor=2.0, max_iter=max_iter)
        shortfall = (kw - current_kw) if bounded else np.nan
        row.update({
            "mode": "grid_threshold",
            "demand_growth_margin_pct": np.nan,
            "search_bounded": bounded,
            "grid_import_threshold_kw": round(kw, 1) if bounded else np.nan,
            "grid_shortfall_kw": round(shortfall, 1) if bounded else np.nan,
            "edge_margin_pct": round(-(shortfall / current_kw) * 100.0, 1) if bounded and current_kw > 0 else np.nan,
            "edge_margin_kw": round(-shortfall, 1) if bounded else np.nan,
        })
    return row


def _solve_task(task: tuple) -> dict:
    import optimisation_model as om
    (district, activity, heating, status, scenarios, time_limit_s, threads,
     max_iter, tol_demand, sizing) = task
    return _cell_row(om, district, activity, heating, status, scenarios=scenarios,
                     time_limit_s=time_limit_s, threads=threads,
                     max_iter=max_iter, tol_demand=tol_demand, sizing=sizing)


def run_grid_sensitivity(df_deterministic: pd.DataFrame, out_path: str = None, *,
                         districts: list = None, activities: list = None,
                         heatings=_HP_HEATINGS, time_limit_s: int = None,
                         max_iter: int = 10, tol_demand: float = 0.05,
                         n_jobs: int = None, verbose: bool = True) -> pd.DataFrame:
    # Already-solved deterministic-round ranking dataframe (one row per district x activity x heating, carrying "status"). 
    import optimisation_model as om
    time_limit_s = time_limit_s or om.DEFAULT_TIME_LIMIT_S
    scen = om._scenarios_for_round("deterministic")

    sub = df_deterministic[df_deterministic["heating"].isin(heatings)]
    if districts:
        sub = sub[sub["district"].isin(districts)]
    if activities:
        sub = sub[sub["activity"].isin(activities)]

    logical = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs = max(1, logical // 4)
    n_tasks = len(sub)
    n_jobs = max(1, min(n_jobs, n_tasks or 1))
    threads_per_worker = max(1, logical // n_jobs)

    # Baseline stage-1 sizing per cell, pinned during the demand-margin bisection (see _solve_fixed).
    # Sourced from the frame just solved, not a saved CSV — same staleness guard the ASHP/PV
    # sensitivity runs use, since a pinned n_pv above the current roof cap fails every row.
    missing = [c for c in ("n_pv", "e_th_kwh") if c not in sub.columns]
    if missing:
        raise ValueError(f"grid sensitivity needs baseline sizing columns {missing} to pin the "
                         f"demand-margin re-solves; got {list(sub.columns)}")
    tasks = [(r.district, r.activity, r.heating, r.status, scen, time_limit_s, threads_per_worker,
             max_iter, tol_demand, (float(r.n_pv or 0.0), float(r.e_th_kwh or 0.0)))
             for r in sub.itertuples()]

    rows = []
    if n_tasks == 0:
        rows = []
    elif n_jobs == 1:
        for i, t in enumerate(tasks, 1):
            r = _solve_task(t)
            rows.append(r)
            if verbose:
                print(f"[{i}/{n_tasks}] {r['activity']} · {r['district']} · {r['heating']} "
                      f"({r['mode']}) -> edge margin {r['edge_margin_pct']}%")
    else:
        if verbose:
            print(f"Solving {n_tasks} grid-sensitivity bisections across {n_jobs} processes "
                  f"x {threads_per_worker} solver threads (of {logical} logical cores) …")
        with mpr.Pool(processes=n_jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_solve_task, tasks), 1):
                rows.append(r)
                if verbose:
                    print(f"[{i}/{n_tasks}] {r['activity']} · {r['district']} · {r['heating']} "
                          f"({r['mode']}) -> edge margin {r['edge_margin_pct']}%")

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["edge_margin_pct"], na_position="last").reset_index(drop=True)
    if out_path:
        _write_workbook(df, out_path)
    return df


def _write_workbook(df: pd.DataFrame, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="Grid Sensitivity", index=False)
    print(f"Saved: {out_path}")
