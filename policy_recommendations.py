"""
Import-only, by optimisation_model.py (main()) once the cost rounds complete; it is handed
every solved round. Scenarios are re-derived per round key via om._scenarios_for_round(),
so the caller's dict keys decide the price sets the bisections re-solve under.

Back-calculates the required capex rebate (or electricity price cut) for the 
single (district, activity) cell with the best NPV savings across the sweep, until:

  1. Battery storage becomes NPV-recommended.
  2. Heat pumps become the cheapest heating technology via a CAPEX rebate, undercutting the Gas Boiler cost.
  3. Heat pumps become the cheapest heating technology via an electricity import PRICE cut:
     Uniform % reduction applied to the wholesale+DUoS import-price build-up, using two capex anchors — 
     0% and 100% HP capex rebate. Gas Boiler is re-solved under the same price cut to ensure fair comparison.

Run_policy_recommendations(cost_rounds, out_path, *, districts=None, activities=None,
                           time_limit_s=None, max_iter=10, tol=0.05, n_jobs=None, verbose=True)
        -> {"battery": {round: df}, "hp": {round: df}, "hp_elec": {round: df}}
"""
import os
import multiprocessing as mpr
from contextlib import contextmanager

import numpy as np
import pandas as pd

import model_params as mp

_BATT_KEYS = ("energy_capex_per_kwh", "power_capex_per_kw")
_BATT_INSTALL_THRESHOLD_KWH = 1.0   # ignore sub-kWh noise around zero when deciding "installed"
_HP_HEATINGS = ("ASHP", "GSHP (vertical)", "GSHP (horizontal)")


@contextmanager
def _battery_rebate(frac: float):
    base = {k: mp.TECH_COSTS["battery"][k] for k in _BATT_KEYS}
    try:
        for k in _BATT_KEYS:
            mp.TECH_COSTS["battery"][k] = base[k] * (1.0 - frac)
        yield
    finally:
        for k in _BATT_KEYS:
            mp.TECH_COSTS["battery"][k] = base[k]


@contextmanager
def _hp_rebate(heating: str, frac: float):
    base = mp.HEAT_COSTS[heating]["capex_per_kwth"]
    try:
        mp.HEAT_COSTS[heating]["capex_per_kwth"] = base * (1.0 - frac)
        yield
    finally:
        mp.HEAT_COSTS[heating]["capex_per_kwth"] = base


@contextmanager
def _elec_price_cut(frac: float):
    import optimisation_engine as oe
    orig = oe.import_price_slots_central
    if frac == 0.0:
        yield
        return

    def patched(district, band_name):
        raw = orig(district, band_name)
        return {k: v * (1.0 - frac) for k, v in raw.items()}

    oe.import_price_slots_central = patched
    try:
        yield
    finally:
        oe.import_price_slots_central = orig


def _bisect_frac(test, *, lo: float = 0.0, hi: float = 1.0, max_iter: int = 10, tol: float = 0.05):
    # hit=False means even a 100% rebate (hi) fails — the target is blocked by 
    # something rebate can't fix (e.g. a grid-import ceiling), not by capex.
    if test(lo):
        return lo, True
    if not test(hi):
        return hi, False
    for _ in range(max_iter):
        if hi - lo <= tol:
            break
        mid = (lo + hi) / 2.0
        if test(mid):
            hi = mid
        else:
            lo = mid
    return hi, True


def _battery_row(om, district: str, activity: str, heating: str, *, scenarios, time_limit_s,
                 threads, max_iter, tol, baseline_e_batt: float) -> dict:
    base_energy = mp.TECH_COSTS["battery"]["energy_capex_per_kwh"]
    base_power  = mp.TECH_COSTS["battery"]["power_capex_per_kw"]
    cache = {}

    def test(frac):
        with _battery_rebate(frac):
            r = om.solve_scenario(district, activity, heating, scenarios=scenarios,
                                  time_limit_s=time_limit_s, threads=threads)
        cache[frac] = r
        return r.get("status") == "Optimal" and (r.get("e_batt_kwh") or 0.0) > _BATT_INSTALL_THRESHOLD_KWH

    frac, hit = _bisect_frac(test, max_iter=max_iter, tol=tol)
    r = cache[frac]
    return {
        "district": district, "activity": activity, "heating": heating,
        "baseline_e_batt_kwh": baseline_e_batt,
        "rebate_achievable": hit,
        "rebate_pct": round(frac * 100, 2) if hit else np.nan,
        "battery_energy_rebate_GBP_per_kwh": round(base_energy * frac, 1) if hit else np.nan,
        "battery_power_rebate_GBP_per_kw": round(base_power * frac, 1) if hit else np.nan,
        "e_batt_kwh_at_rebate": r.get("e_batt_kwh") if hit else np.nan,
        "total_cost_npv_GBP_at_rebate": r.get("total_cost_npv_GBP") if hit else np.nan,
        "status_at_rebate": r.get("status"),
    }


def _hp_row(om, district: str, activity: str, heating: str, target_cost_npv: float, *, scenarios,
           time_limit_s, threads, max_iter, tol, baseline_cost: float) -> dict:
    base_capex = mp.HEAT_COSTS[heating]["capex_per_kwth"]
    cache = {}

    def test(frac):
        with _hp_rebate(heating, frac):
            r = om.solve_scenario(district, activity, heating, scenarios=scenarios,
                                  time_limit_s=time_limit_s, threads=threads)
        cache[frac] = r
        cost = r.get("total_cost_npv_GBP")
        return r.get("status") == "Optimal" and cost is not None and cost <= target_cost_npv

    frac, hit = _bisect_frac(test, max_iter=max_iter, tol=tol)
    r = cache[frac]
    return {
        "district": district, "activity": activity, "heating": heating,
        "target_cost_npv_GBP": target_cost_npv,
        "baseline_hp_cost_npv_GBP": baseline_cost,
        "rebate_achievable": hit,
        "rebate_pct": round(frac * 100, 2) if hit else np.nan,
        "hp_capex_rebate_GBP_per_kwth": round(base_capex * frac, 1) if hit else np.nan,
        "hp_cost_npv_GBP_at_rebate": r.get("total_cost_npv_GBP") if hit else np.nan,
        "status_at_rebate": r.get("status"),
    }


def _hp_elec_row(om, district: str, activity: str, heating: str, hp_capex_frac: float, *,
                 scenarios, time_limit_s, threads, max_iter, tol, baseline_hp_cost: float,
                 baseline_gas_cost: float) -> dict:
    # hp_capex_frac is a fixed anchor (0.0 = today's HP capex, 1.0 = free HP capex), not searched;
    # the search variable is the electricity-price cut. Gas Boiler is re-solved at every step
    # under the same cut since a price cut isn't HP-specific.
    cache = {}

    def test(frac):
        with _elec_price_cut(frac):
            with _hp_rebate(heating, hp_capex_frac):
                r_hp = om.solve_scenario(district, activity, heating, scenarios=scenarios,
                                         time_limit_s=time_limit_s, threads=threads)
            r_gas = om.solve_scenario(district, activity, "Gas Boiler", scenarios=scenarios,
                                      time_limit_s=time_limit_s, threads=threads)
        cache[frac] = (r_hp, r_gas)
        hp_cost, gas_cost = r_hp.get("total_cost_npv_GBP"), r_gas.get("total_cost_npv_GBP")
        return (r_hp.get("status") == "Optimal" and r_gas.get("status") == "Optimal"
                and hp_cost is not None and gas_cost is not None and hp_cost <= gas_cost)

    frac, hit = _bisect_frac(test, max_iter=max_iter, tol=tol)
    r_hp, r_gas = cache[frac]
    return {
        "district": district, "activity": activity, "heating": heating,
        "hp_capex_rebate_pct": round(hp_capex_frac * 100, 1),
        "baseline_hp_cost_npv_GBP": baseline_hp_cost,
        "baseline_gas_cost_npv_GBP": baseline_gas_cost,
        "rebate_achievable": hit,
        "elec_price_reduction_pct": round(frac * 100, 2) if hit else np.nan,
        "hp_cost_npv_GBP_at_reduction": r_hp.get("total_cost_npv_GBP") if hit else np.nan,
        "gas_cost_npv_GBP_at_reduction": r_gas.get("total_cost_npv_GBP") if hit else np.nan,
        "status_at_reduction": f"{r_hp.get('status')}/{r_gas.get('status')}",
    }


def _winning_cell(df_opt: pd.DataFrame, districts, activities):
    # The single (district, activity) with the best NPV savings, after optional scoping.
    # Returns None if scoping leaves nothing to pick from.
    scoped = df_opt
    if districts:
        scoped = scoped[scoped["district"].isin(districts)]
    if activities:
        scoped = scoped[scoped["activity"].isin(activities)]
    if scoped.empty:
        return None
    best = scoped.loc[scoped["npv_savings_GBP"].idxmax()]
    return best.district, best.activity


def _solve_task(task: tuple) -> dict:
    import optimisation_model as om
    kind = task[0]
    if kind == "battery":
        (_, round_name, district, activity, heating, scenarios, time_limit_s, threads,
         max_iter, tol, baseline_e_batt) = task
        row = _battery_row(om, district, activity, heating, scenarios=scenarios,
                           time_limit_s=time_limit_s, threads=threads, max_iter=max_iter, tol=tol,
                           baseline_e_batt=baseline_e_batt)
    elif kind == "hp":
        (_, round_name, district, activity, meta, scenarios, time_limit_s, threads,
         max_iter, tol, baseline_cost) = task
        heating, target_cost = meta
        row = _hp_row(om, district, activity, heating, target_cost, scenarios=scenarios,
                     time_limit_s=time_limit_s, threads=threads, max_iter=max_iter, tol=tol,
                     baseline_cost=baseline_cost)
    else:  # "hp_elec"
        (_, round_name, district, activity, meta, scenarios, time_limit_s, threads,
         max_iter, tol, baselines) = task
        heating, hp_capex_frac = meta
        baseline_hp_cost, baseline_gas_cost = baselines
        row = _hp_elec_row(om, district, activity, heating, hp_capex_frac, scenarios=scenarios,
                          time_limit_s=time_limit_s, threads=threads, max_iter=max_iter, tol=tol,
                          baseline_hp_cost=baseline_hp_cost, baseline_gas_cost=baseline_gas_cost)
    return {"kind": kind, "round": round_name, **row}


def run_policy_recommendations(cost_rounds: dict, out_path: str, *,
                               districts: list = None, activities: list = None,
                               time_limit_s: int = None, max_iter: int = 10, tol: float = 0.05,
                               n_jobs: int = None, verbose: bool = True) -> dict:
    # cost_rounds: {"stochastic": df, ...} — the already-solved ranking dataframe(s) for the cost objective, 
    # as produced by optimisation_model. By the time this module is imported (in optimisation_model.main()), 
    # optimisation_model has already finished its own module-level setup, so this reverse import is safe.
    import optimisation_model as om
    time_limit_s = time_limit_s or om.DEFAULT_TIME_LIMIT_S

    tasks = []
    for round_name, df in cost_rounds.items():
        scen = om._scenarios_for_round(round_name)
        opt = df[df["status"] == "Optimal"]
        cell = _winning_cell(opt, districts, activities)
        if cell is None:
            continue
        d, a = cell
        sub = opt[(opt["district"] == d) & (opt["activity"] == a)]

        winner = sub.loc[sub["npv_savings_GBP"].idxmax()]
        tasks.append(("battery", round_name, d, a, winner.heating, scen, time_limit_s, None,
                     max_iter, tol, winner.get("e_batt_kwh", 0.0)))

        gas = sub[sub["heating"] == "Gas Boiler"]
        hp_candidates = sub[sub["heating"].isin(_HP_HEATINGS)]
        if gas.empty or hp_candidates.empty:
            continue
        target_cost = gas["total_cost_npv_GBP"].iloc[0]
        cheapest_hp = hp_candidates.loc[hp_candidates["total_cost_npv_GBP"].idxmin()]
        heating = cheapest_hp.heating
        baseline_hp_cost = cheapest_hp.total_cost_npv_GBP
        tasks.append(("hp", round_name, d, a, (heating, target_cost), scen, time_limit_s, None,
                     max_iter, tol, baseline_hp_cost))

        for hp_capex_frac in (0.0, 1.0):
            tasks.append(("hp_elec", round_name, d, a, (heating, hp_capex_frac), scen,
                         time_limit_s, None, max_iter, tol, (baseline_hp_cost, target_cost)))

    logical = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs = max(1, logical // 4)
    n_jobs = max(1, min(n_jobs, len(tasks) or 1))
    threads_per_worker = max(1, logical // n_jobs)
    tasks = [t[:7] + (threads_per_worker,) + t[8:] for t in tasks]

    n_total = len(tasks)
    rows = []
    if n_total == 0:
        rows = []
    elif n_jobs == 1:
        for i, t in enumerate(tasks, 1):
            r = _solve_task(t)
            rows.append(r)
            if verbose:
                pct = r.get("rebate_pct", r.get("elec_price_reduction_pct"))
                print(f"[{i}/{n_total}] {r['kind']} · {r['round']} · {r['activity']} · "
                      f"{r['district']} -> {pct}%")
    else:
        if verbose:
            print(f"Solving {n_total} rebate bisections across {n_jobs} processes "
                  f"x {threads_per_worker} solver threads (of {logical} logical cores) …")
        with mpr.Pool(processes=n_jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_solve_task, tasks), 1):
                rows.append(r)
                if verbose:
                    pct = r.get("rebate_pct", r.get("elec_price_reduction_pct"))
                    print(f"[{i}/{n_total}] {r['kind']} · {r['round']} · {r['activity']} · "
                          f"{r['district']} -> {pct}%")

    battery_frames, hp_frames, hp_elec_frames = {}, {}, {}
    for round_name in cost_rounds:
        batt = [r for r in rows if r["kind"] == "battery" and r["round"] == round_name]
        hp = [r for r in rows if r["kind"] == "hp" and r["round"] == round_name]
        hp_elec = [r for r in rows if r["kind"] == "hp_elec" and r["round"] == round_name]
        battery_frames[round_name] = pd.DataFrame(batt).drop(columns=["kind", "round"], errors="ignore")
        hp_frames[round_name] = pd.DataFrame(hp).drop(columns=["kind", "round"], errors="ignore")
        hp_elec_frames[round_name] = pd.DataFrame(hp_elec).drop(columns=["kind", "round"], errors="ignore")

    _write_workbook(battery_frames, hp_frames, hp_elec_frames, out_path)
    return {"battery": battery_frames, "hp": hp_frames, "hp_elec": hp_elec_frames}


def _write_workbook(battery_frames: dict, hp_frames: dict, hp_elec_frames: dict, out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for name, df in battery_frames.items():
            df.to_excel(xw, sheet_name=f"Battery ({name})"[:31], index=False)
        for name, df in hp_frames.items():
            df.to_excel(xw, sheet_name=f"HP ({name})"[:31], index=False)
        for name, df in hp_elec_frames.items():
            df.to_excel(xw, sheet_name=f"HP Elec ({name})"[:31], index=False)
    print(f"Saved: {out_path}")
