"""
Import-only, by optimisation_model.py (main()) once the cost rounds complete; it is handed
every solved round. Scenarios are re-derived per round key via om._scenarios_for_round(),
so the caller's dict keys decide the price sets the bisections re-solve under.

Back-calculates the required capex rebate (or electricity price cut) ONCE PER ACTIVITY CLASS, each
in that class's own best-NPV district, so a recommendation can be targeted by end-user type and by
location rather than read off one headline cell. For every such cell it searches until:

  1. Battery storage becomes NPV-recommended.
  2. Heat pumps become the cheapest heating technology via a CAPEX rebate, undercutting the Gas Boiler cost.
  3. Heat pumps become the cheapest heating technology via an electricity import PRICE cut:
     Uniform % reduction applied to the wholesale+DUoS import-price build-up, using two capex anchors —
     0% and 100% HP capex rebate. Gas Boiler is re-solved under the same price cut to ensure fair comparison.
  4. Battery storage becomes NPV-recommended on a raised SEG EXPORT price alone (search variable is a
     multiple of today's export price). Deliberately independent of levers 1-3: no battery/HP capex
     rebate and no import-price cut are applied, so the row answers "what can export price do by
     itself?". Because a flat export tariff pays a battery nothing for shifting export to export, the
     search often runs to its ceiling without installing one — so what the higher price DOES buy
     (PV size, export volume, NPV) is recorded at the tested multiple either way.

Plus one step-wise sweep (fixed steps, no search):

  5. Battery capex grant swept 0% -> 100% in batt_sweep_step increments (default 5%, so 21 solves per
     round). Where target 1 stops at the threshold where a battery first appears, this carries on past
     it to show how much storage each further increment buys, and what the grant costs per kWh
     installed. Set batt_sweep_step=0 to skip it — it dominates this module's runtime.

Outputs, per round: one tab per lever (rows = activity classes), a Summary tab pivoting every lever
against activity class, and a Charts tab holding policy_thresholds_<round>.png and
battery_capex_sweep_<round>.png (also written to the charts/ subfolder beside the workbook).

Runtime scales with the number of classes covered: ~285 solves per round for all four. Scope it with
`activities=[...]`, thin the sweep with `batt_sweep_step`, or raise `n_jobs` (the default logical//4
predates the per-class loop and is likely low now that a round holds ~104 tasks).

Run_policy_recommendations(cost_rounds, out_path, *, districts=None, activities=None,
                           time_limit_s=None, max_iter=10, tol=0.05, n_jobs=None, verbose=True,
                           batt_sweep_step=0.05)
        -> {"battery": {round: df}, "hp": {round: df}, "hp_elec": {round: df},
            "export": {round: df}, "batt_sweep": {round: df}, "summary": {round: df}}
"""
import os
import multiprocessing as mpr
from contextlib import contextmanager

import numpy as np
import pandas as pd
from openpyxl.drawing.image import Image as XLImage

import model_params as mp

_BATT_KEYS = ("energy_capex_per_kwh", "power_capex_per_kw")
_BATT_INSTALL_THRESHOLD_KWH = 1.0   # ignore sub-kWh noise around zero when deciding "installed"
_HP_HEATINGS = ("ASHP", "GSHP (vertical)", "GSHP (horizontal)")
# Ceiling of the export-price search, as a multiple of today's active SEG price. 5x (GBP 0.50/kWh at
# the current GBP 0.10 SEG) is already well past any tariff a UK supplier has offered and past the
# import price, so reaching it is the answer "export price cannot do this", not a call to search
# higher. The old 10x ceiling only mattered while export could be resold grid-to-grid; the
# ex_gen_* export cap in optimisation_engine now bounds export by own generation, so the extra
# search range bought nothing but solve time.
_EXPORT_MAX_MULTIPLE = 5.0
# Battery capex sweep: fixed grant steps from 0% to 100%. The bisection levers stop at the threshold
# where the target first flips; this sweep keeps going past it to show how much battery each further
# increment of grant actually buys (and what it costs the grant-giver per kWh installed).
_BATT_SWEEP_STEP = 0.05


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
def _export_price_multiple(mult: float):
    # Scales the year-0 SEG export price; the escalation path and every other price are untouched.
    # optimisation_engine imported TECH_COSTS from model_params by reference, so writing the key here
    # is visible to export_price() in the engine — same mechanism as _battery_rebate above.
    base = mp.TECH_COSTS["elec_export_price"]
    try:
        mp.TECH_COSTS["elec_export_price"] = base * mult
        yield
    finally:
        mp.TECH_COSTS["elec_export_price"] = base


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


def _batt_sweep_row(om, district: str, activity: str, heating: str, frac: float, *, scenarios,
                    time_limit_s, threads, baseline_e_batt: float) -> dict:
    # One fixed grant step — no search. Every step is its own task so the worker pool spreads them.
    base_energy = mp.TECH_COSTS["battery"]["energy_capex_per_kwh"]
    base_power  = mp.TECH_COSTS["battery"]["power_capex_per_kw"]
    with _battery_rebate(frac):
        r = om.solve_scenario(district, activity, heating, scenarios=scenarios,
                              time_limit_s=time_limit_s, threads=threads)
    e_batt = r.get("e_batt_kwh") or 0.0
    o_batt = r.get("o_batt_kw") or 0.0
    installed = r.get("status") == "Optimal" and e_batt > _BATT_INSTALL_THRESHOLD_KWH
    # What the grant costs the giver at this step: the discount, charged on the size actually built.
    # Zero when nothing is built, however deep the discount — a grant nobody takes up costs nothing.
    grant_cost = frac * (base_energy * e_batt + base_power * o_batt)
    return {
        "district": district, "activity": activity, "heating": heating,
        "rebate_pct": round(frac * 100, 1),
        "battery_energy_capex_GBP_per_kwh": round(base_energy * (1.0 - frac), 1),
        "battery_power_capex_GBP_per_kw": round(base_power * (1.0 - frac), 1),
        "baseline_e_batt_kwh": baseline_e_batt,
        "battery_installed": installed,
        "e_batt_kwh": r.get("e_batt_kwh"),
        "o_batt_kw": r.get("o_batt_kw"),
        "grant_cost_GBP": round(grant_cost, 0),
        # Grant spend per kWh of storage it brings into existence — the lever's cost-effectiveness.
        "grant_cost_GBP_per_kwh_installed": round(grant_cost / e_batt, 1) if installed else np.nan,
        "pv_kwp": r.get("pv_kwp"),
        "annual_batt_disc_kwh": r.get("annual_batt_disc_kwh"),
        "annual_export_kwh": r.get("annual_export_kwh"),
        "self_consumption_rate": r.get("self_consumption_rate"),
        "total_cost_npv_GBP": r.get("total_cost_npv_GBP"),
        "npv_savings_GBP": r.get("npv_savings_GBP"),
        "status": r.get("status"),
    }


def _export_row(om, district: str, activity: str, heating: str, *, scenarios, time_limit_s,
                threads, max_iter, tol, baseline_e_batt: float, baseline_npv_savings: float) -> dict:
    # Search variable is a multiple of today's export price, applied on its own — no capex rebate and
    # no import-price cut — so this row is readable as a standalone policy lever.
    base_price = mp.TECH_COSTS["elec_export_price"]
    cache = {}

    def test(mult):
        with _export_price_multiple(mult):
            r = om.solve_scenario(district, activity, heating, scenarios=scenarios,
                                  time_limit_s=time_limit_s, threads=threads)
        cache[mult] = r
        return r.get("status") == "Optimal" and (r.get("e_batt_kwh") or 0.0) > _BATT_INSTALL_THRESHOLD_KWH

    mult, hit = _bisect_frac(test, lo=1.0, hi=_EXPORT_MAX_MULTIPLE, max_iter=max_iter, tol=tol)
    r = cache[mult]
    # Unlike the rebate levers, the endpoint metrics are reported whether or not the target was hit:
    # if export price alone buys no battery, what it does buy is the substantive result.
    return {
        "district": district, "activity": activity, "heating": heating,
        "baseline_export_price_GBP_per_kwh": round(base_price, 4),
        "baseline_e_batt_kwh": baseline_e_batt,
        "baseline_npv_savings_GBP": baseline_npv_savings,
        "battery_achievable": hit,
        "search_ceiling_multiple": _EXPORT_MAX_MULTIPLE,
        "search_ceiling_GBP_per_kwh": round(base_price * _EXPORT_MAX_MULTIPLE, 4),
        # Threshold multiple when the battery was reached, otherwise the ceiling that was tested.
        "export_price_multiple": round(mult, 3),
        "export_price_GBP_per_kwh": round(base_price * mult, 4),
        "e_batt_kwh_at_multiple": r.get("e_batt_kwh"),
        "pv_kwp_at_multiple": r.get("pv_kwp"),
        "annual_export_kwh_at_multiple": r.get("annual_export_kwh"),
        "npv_savings_GBP_at_multiple": r.get("npv_savings_GBP"),
        "status_at_multiple": r.get("status"),
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


def _winning_cells_per_activity(df_opt: pd.DataFrame, districts, activities) -> list:
    # One (district, activity) per activity class: that class's own best-NPV district, so the levers
    # are answered per end-user type and per location rather than for a single headline cell.
    # `districts` narrows the candidate locations, `activities` narrows which classes are covered.
    # Returns [] if scoping leaves nothing to pick from.
    scoped = df_opt
    if districts:
        scoped = scoped[scoped["district"].isin(districts)]
    if activities:
        scoped = scoped[scoped["activity"].isin(activities)]
    if scoped.empty:
        return []
    # Ordered by NPV so the strongest class leads the console log and every output tab.
    best = scoped.loc[scoped.groupby("activity")["npv_savings_GBP"].idxmax()]
    best = best.sort_values("npv_savings_GBP", ascending=False)
    return [(r.district, r.activity) for r in best.itertuples()]


def _solve_task(task: tuple) -> dict:
    import optimisation_model as om
    kind = task[0]
    if kind == "battery":
        (_, round_name, district, activity, heating, scenarios, time_limit_s, threads,
         max_iter, tol, baseline_e_batt) = task
        row = _battery_row(om, district, activity, heating, scenarios=scenarios,
                           time_limit_s=time_limit_s, threads=threads, max_iter=max_iter, tol=tol,
                           baseline_e_batt=baseline_e_batt)
    elif kind == "batt_sweep":
        (_, round_name, district, activity, meta, scenarios, time_limit_s, threads,
         _max_iter, _tol, baseline_e_batt) = task          # fixed step: nothing is searched
        heating, frac = meta
        row = _batt_sweep_row(om, district, activity, heating, frac, scenarios=scenarios,
                              time_limit_s=time_limit_s, threads=threads,
                              baseline_e_batt=baseline_e_batt)
    elif kind == "export":
        (_, round_name, district, activity, heating, scenarios, time_limit_s, threads,
         max_iter, tol, baselines) = task
        baseline_e_batt, baseline_npv_savings = baselines
        row = _export_row(om, district, activity, heating, scenarios=scenarios,
                          time_limit_s=time_limit_s, threads=threads, max_iter=max_iter, tol=tol,
                          baseline_e_batt=baseline_e_batt, baseline_npv_savings=baseline_npv_savings)
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


def _progress_txt(r: dict) -> str:
    # One-line result for the console: thresholds report the % (or price multiple) they landed on,
    # the sweep reports what its fixed step bought.
    if r["kind"] == "export":
        mult = r.get("export_price_multiple")
        return (f"{mult:g}x export price" if r.get("battery_achievable")
                else f">{r.get('search_ceiling_multiple'):g}x export price (no battery)")
    if r["kind"] == "batt_sweep":
        return f"{r.get('rebate_pct')}% grant -> {r.get('e_batt_kwh')} kWh battery"
    pct = r.get("rebate_pct", r.get("elec_price_reduction_pct"))
    return f"{pct}%"


def _batt_sweep_fracs(step: float) -> list:
    # 0.0 -> 1.0 inclusive. The 0% row is the un-grantsed baseline, kept in the tab as the reference
    # every other step is read against.
    if not step or step <= 0:
        return []
    n = int(round(1.0 / step))
    return [round(i * step, 4) for i in range(n + 1)]


def run_policy_recommendations(cost_rounds: dict, out_path: str, *,
                               districts: list = None, activities: list = None,
                               time_limit_s: int = None, max_iter: int = 10, tol: float = 0.05,
                               n_jobs: int = None, verbose: bool = True,
                               batt_sweep_step: float = _BATT_SWEEP_STEP) -> dict:
    # cost_rounds: {"stochastic": df, ...} — the already-solved ranking dataframe(s) for the cost objective, 
    # as produced by optimisation_model. By the time this module is imported (in optimisation_model.main()), 
    # optimisation_model has already finished its own module-level setup, so this reverse import is safe.
    import optimisation_model as om
    time_limit_s = time_limit_s or om.DEFAULT_TIME_LIMIT_S

    tasks = []
    activity_order = {}          # {round: [activity, ...]} — best-NPV class first, for output order
    for round_name, df in cost_rounds.items():
        scen = om._scenarios_for_round(round_name)
        opt = df[df["status"] == "Optimal"]
        cells = _winning_cells_per_activity(opt, districts, activities)
        activity_order[round_name] = [a for _, a in cells]
        # Every lever below runs once per activity class, on that class's own best-NPV district.
        for d, a in cells:
            sub = opt[(opt["district"] == d) & (opt["activity"] == a)]

            winner = sub.loc[sub["npv_savings_GBP"].idxmax()]
            tasks.append(("battery", round_name, d, a, winner.heating, scen, time_limit_s, None,
                         max_iter, tol, winner.get("e_batt_kwh", 0.0)))
            # Queued before the gas/HP guard below so the export lever still runs on a cell that has
            # no Gas Boiler or HP counterpart to compare against — it needs neither.
            tasks.append(("export", round_name, d, a, winner.heating, scen, time_limit_s, None,
                         max_iter, tol, (winner.get("e_batt_kwh", 0.0),
                                         winner.get("npv_savings_GBP", np.nan))))
            for frac in _batt_sweep_fracs(batt_sweep_step):
                tasks.append(("batt_sweep", round_name, d, a, (winner.heating, frac), scen,
                             time_limit_s, None, max_iter, tol, winner.get("e_batt_kwh", 0.0)))

            gas = sub[sub["heating"] == "Gas Boiler"]
            hp_candidates = sub[sub["heating"].isin(_HP_HEATINGS)]
            # Skips only this class's HP levers — the other classes still get theirs.
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
                print(f"[{i}/{n_total}] {r['kind']} · {r['round']} · {r['activity']} · "
                      f"{r['district']} -> {_progress_txt(r)}")
    else:
        if verbose:
            print(f"Solving {n_total} policy tasks (bisections + sweep steps) over "
                  f"{len(set(a for order in activity_order.values() for a in order))} activity "
                  f"classes, across {n_jobs} processes x {threads_per_worker} solver threads "
                  f"(of {logical} logical cores) …")
        with mpr.Pool(processes=n_jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_solve_task, tasks), 1):
                rows.append(r)
                if verbose:
                    print(f"[{i}/{n_total}] {r['kind']} · {r['round']} · {r['activity']} · "
                          f"{r['district']} -> {_progress_txt(r)}")

    frames = {kind: {} for kind in ("battery", "hp", "hp_elec", "export", "batt_sweep")}
    for round_name in cost_rounds:
        for kind in frames:
            picked = [r for r in rows if r["kind"] == kind and r["round"] == round_name]
            df_kind = pd.DataFrame(picked).drop(columns=["kind", "round"], errors="ignore")
            frames[kind][round_name] = _order_rows(df_kind, kind, activity_order.get(round_name, []))

    frames["summary"] = {name: _build_summary(frames, name, activity_order.get(name, []))
                         for name in cost_rounds}
    charts_dir = os.path.join(os.path.dirname(out_path) or ".", "charts")
    charts = {name: _build_charts(frames, name, charts_dir) for name in cost_rounds}
    _write_workbook(frames, out_path, charts)
    return frames


def _order_rows(df: pd.DataFrame, kind: str, activity_order: list) -> pd.DataFrame:
    # imap_unordered returns rows as they finish; every tab has to read in a stable order —
    # activity class (best-NPV class first), then the lever's own step column within each class.
    if df.empty or "activity" not in df.columns:
        return df
    within = {"batt_sweep": "rebate_pct", "hp_elec": "hp_capex_rebate_pct"}.get(kind)
    rank = {a: i for i, a in enumerate(activity_order)}
    df = df.copy()
    df["_ord"] = df["activity"].map(rank).fillna(len(rank))
    by = ["_ord"] + ([within] if within and within in df.columns else [])
    return df.sort_values(by).drop(columns="_ord").reset_index(drop=True)


def _pick(df: pd.DataFrame, activity: str, col: str, where: dict = None):
    # One cell out of a lever frame for one activity class; NaN when that lever produced no row
    # (e.g. a class whose cell has no Gas Boiler counterpart never gets the HP levers).
    if df is None or df.empty or "activity" not in df.columns or col not in df.columns:
        return np.nan
    sub = df[df["activity"] == activity]
    for k, v in (where or {}).items():
        if k in sub.columns:
            sub = sub[sub[k] == v]
    return sub[col].iloc[0] if len(sub) else np.nan


def _build_summary(frames: dict, round_name: str, activity_order: list) -> pd.DataFrame:
    # One row per activity class, pulled from the lever frames already in memory — no extra solves.
    # This is the cross-class table the per-end-user-type recommendation is read off.
    batt  = frames["battery"].get(round_name, pd.DataFrame())
    sweep = frames["batt_sweep"].get(round_name, pd.DataFrame())
    hp    = frames["hp"].get(round_name, pd.DataFrame())
    hpe   = frames["hp_elec"].get(round_name, pd.DataFrame())
    exp   = frames["export"].get(round_name, pd.DataFrame())

    out = []
    for a in activity_order:
        # The sweep's cheapest step that actually builds a battery — the grant a scheme would set.
        s_a = sweep[(sweep["activity"] == a) & sweep["battery_installed"]] if not sweep.empty else pd.DataFrame()
        s_full = sweep[(sweep["activity"] == a) & (sweep["rebate_pct"] == 100.0)] if not sweep.empty else pd.DataFrame()
        out.append({
            "activity": a,
            "district": _pick(batt, a, "district"),
            "heating":  _pick(batt, a, "heating"),
            "baseline_npv_savings_GBP": _pick(exp, a, "baseline_npv_savings_GBP"),
            "baseline_e_batt_kwh":      _pick(batt, a, "baseline_e_batt_kwh"),
            # 1 - battery capex rebate needed before storage is recommended at all
            "battery_rebate_achievable": _pick(batt, a, "rebate_achievable"),
            "battery_rebate_pct":        _pick(batt, a, "rebate_pct"),
            # 5 - what the swept grant buys past that threshold
            "first_install_rebate_pct":         float(s_a["rebate_pct"].min()) if len(s_a) else np.nan,
            "e_batt_kwh_at_100pct":             float(s_full["e_batt_kwh"].iloc[0]) if len(s_full) else np.nan,
            "grant_cost_GBP_per_kwh_at_100pct": (float(s_full["grant_cost_GBP_per_kwh_installed"].iloc[0])
                                                 if len(s_full) else np.nan),
            # 2 - HP capex rebate to undercut the gas boiler
            "hp_rebate_achievable":  _pick(hp, a, "rebate_achievable"),
            "hp_capex_rebate_pct":   _pick(hp, a, "rebate_pct"),
            # 3 - import-price cut to the same end, at both HP-capex anchors
            "elec_price_cut_pct_at_0pct_capex":   _pick(hpe, a, "elec_price_reduction_pct",
                                                        {"hp_capex_rebate_pct": 0.0}),
            "elec_price_cut_pct_at_100pct_capex": _pick(hpe, a, "elec_price_reduction_pct",
                                                        {"hp_capex_rebate_pct": 100.0}),
            # 4 - export price on its own
            "export_battery_achievable": _pick(exp, a, "battery_achievable"),
            "export_price_multiple":     _pick(exp, a, "export_price_multiple"),
        })
    return pd.DataFrame(out)


# Sheet-name stem per lever, in workbook order. Excel caps sheet names at 31 chars.
_SHEET_STEMS = {"summary": "Summary", "battery": "Battery", "batt_sweep": "Battery Sweep",
                "hp": "HP", "hp_elec": "HP Elec", "export": "Export Price"}


def _act_style():
    # Short activity labels + per-class colours, borrowed from the results report so the policy
    # charts read as part of the same deck. Imported inside the function: optimisation_report pulls
    # in the whole model stack, and this module is itself imported from optimisation_model.
    import optimisation_report as orp
    return orp.ACT_SHORT, orp.ACT_COLORS


def _build_charts(frames: dict, round_name: str, charts_dir: str) -> list:
    # Two PNGs per round, saved next to the workbook and embedded in its Charts tab.
    import matplotlib
    matplotlib.use("Agg")                       # headless: no GUI backend
    import matplotlib.pyplot as plt

    act_short, act_colors = _act_style()
    summary = frames["summary"].get(round_name, pd.DataFrame())
    sweep   = frames["batt_sweep"].get(round_name, pd.DataFrame())
    if summary.empty:
        return []
    os.makedirs(charts_dir, exist_ok=True)
    labels = [act_short.get(a, a) for a in summary["activity"]]
    colors = [act_colors.get(act_short.get(a, a), "#4c72b0") for a in summary["activity"]]
    paths = []

    # (1) Thresholds by activity class. The %-denominated levers share a panel; the export-price
    # multiple gets its own because it is a multiple, not a percentage.
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    pct_levers = [("battery_rebate_pct", "Battery capex rebate", "#4c72b0"),
                  ("hp_capex_rebate_pct", "HP capex rebate", "#dd8452"),
                  ("elec_price_cut_pct_at_0pct_capex", "Import-price cut (0% HP capex rebate)", "#55a868")]
    x = np.arange(len(summary))
    width = 0.8 / len(pct_levers)
    for k, (col, lbl, c) in enumerate(pct_levers):
        vals = pd.to_numeric(summary.get(col, pd.Series(dtype=float)), errors="coerce").to_numpy()
        pos  = x - 0.4 + width * (k + 0.5)
        ax1.bar(pos, np.nan_to_num(vals), width, label=lbl, color=c)
        for xi, v in zip(pos, vals):
            # A missing threshold means the lever never reached its target — labelled at the axis so
            # an absent bar cannot be misread as "no support needed".
            if np.isfinite(v):
                ax1.text(xi, v + 1.5, f"{v:.0f}%", ha="center", va="bottom", fontsize=7.5)
            else:
                ax1.text(xi, 1.5, "n/a", ha="center", va="bottom", fontsize=7,
                         style="italic", color="#666666")
    ax1.set_xticks(x, labels, fontsize=9)
    ax1.set_ylabel("Required support (% of capex / price)")
    ax1.set_ylim(0, 105)
    ax1.set_title("Support needed to flip each lever, by end-user type", fontweight="bold", fontsize=10)
    ax1.legend(fontsize=8)

    mult = pd.to_numeric(summary.get("export_price_multiple", pd.Series(dtype=float)),
                         errors="coerce").to_numpy()
    ok = summary.get("export_battery_achievable", pd.Series([False] * len(summary))).fillna(False).to_numpy(dtype=bool)
    # Unachievable cells are drawn hatched at the search ceiling: the bar marks where the search
    # stopped, not a threshold that was found.
    ax2.bar(x, np.nan_to_num(mult), 0.6, color=colors,
            hatch=["" if o else "//" for o in ok],
            edgecolor="black", linewidth=0.6)
    for xi, v, o in zip(x, mult, ok):
        ax2.text(xi, v, f"{v:.1f}x" if o else f">{v:.0f}x\nno battery", ha="center", va="bottom",
                 fontsize=7.5)
    ax2.set_xticks(x, labels, fontsize=9)
    ax2.set_ylabel("Export price (x today's SEG rate)")
    ax2.set_title("Export price needed for a battery\n(hatched = never reached)",
                  fontweight="bold", fontsize=10)
    fig.suptitle(f"Policy support by end-user type — {round_name} round "
                 f"(each class at its own best-NPV district)", fontweight="bold", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    p1 = os.path.join(charts_dir, f"policy_thresholds_{round_name}.png")
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    paths.append(p1)

    # (2) The swept battery grant: what each further increment buys, per class.
    if not sweep.empty:
        fig, (axa, axb) = plt.subplots(1, 2, figsize=(13, 5.5))
        for a in summary["activity"]:
            s = sweep[sweep["activity"] == a].sort_values("rebate_pct")
            if s.empty:
                continue
            lbl = act_short.get(a, a)
            c   = act_colors.get(lbl, "#4c72b0")
            axa.plot(s["rebate_pct"], s["e_batt_kwh"], marker="o", ms=3, color=c, label=lbl)
            inst = s[s["battery_installed"]]
            axb.plot(inst["rebate_pct"], inst["grant_cost_GBP_per_kwh_installed"],
                     marker="o", ms=3, color=c, label=lbl)
        axa.set_xlabel("Battery capex grant (%)")
        axa.set_ylabel("Recommended storage (kWh)")
        axa.set_title("Storage bought by the grant", fontweight="bold", fontsize=10)
        axa.grid(alpha=0.3)
        axa.legend(fontsize=8)
        axb.set_xlabel("Battery capex grant (%)")
        axb.set_ylabel("Grant cost per kWh installed (£/kWh)")
        axb.set_title("What that storage costs the grant-giver", fontweight="bold", fontsize=10)
        axb.grid(alpha=0.3)
        fig.suptitle(f"Battery capex grant sweep by end-user type — {round_name} round",
                     fontweight="bold", fontsize=12)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p2 = os.path.join(charts_dir, f"battery_capex_sweep_{round_name}.png")
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(p2)
    return paths


def _write_workbook(frames: dict, out_path: str, charts: dict = None) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with pd.ExcelWriter(out_path, engine="openpyxl") as xw:
        for kind, stem in _SHEET_STEMS.items():
            for name, df in frames.get(kind, {}).items():
                df.to_excel(xw, sheet_name=f"{stem} ({name})"[:31], index=False)
        for name, pngs in (charts or {}).items():
            if not pngs:
                continue
            ws = xw.book.create_sheet(f"Charts ({name})"[:31])
            row = 2
            for png in pngs:
                img = XLImage(png)
                ws.add_image(img, f"B{row}")
                row += int(img.height / 20) + 3      # ~20 px per Excel row, plus a gap
    print(f"Saved: {out_path}")
