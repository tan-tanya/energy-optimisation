"""
Matplotlib renderers for the optimisation model.

Three figures:
  - plot_dispatch():       the stage-2 optimal half-hourly dispatch for representative day(s).
  - plot_demand_profile(): the underlying heat vs non-heat electricity demand for the same cell/day.
  - plot_cell_front():     the cost/carbon frontier for one (district, activity) cell across heating types.
"""
import os

import numpy as np
import pulp

import demand_profile_model as dm
import optimisation_engine as oe
from model_params import TECH_COSTS
from seasons import HH_PER_DAY
from optimisation_config import T_RES_H, HORIZON_YEARS, DEFAULT_TIME_LIMIT_S


def activity_area_suffix(activity: str) -> str:
    # Append the building's BEES median floor area after an activity name in a title/label. 
    areas = getattr(dm, "bees_floor_areas", None) or {}
    a = areas.get(activity)
    return f" ({a:,.0f} m²)" if a else ""


def _norm_month(m) -> str:
    # Accept 'Jan' / 'jan' / 'January' and return the canonical MONTHS_ORDER name.
    ml = str(m).strip().lower()
    for full in dm.MONTHS_ORDER:
        if full.lower().startswith(ml):
            return full
    raise ValueError(f"unknown month {m!r}; expected one of {dm.MONTHS_ORDER}")


def spec_line(r, wrapped: bool = False) -> str:
    """Compact technical summary of one design: PV size + capacity factor + battery and thermal-store capacity."""
    pv   = getattr(r, "pv_kwp", 0.0) or 0.0
    gen  = getattr(r, "annual_pv_gen_kwh", 0.0) or 0.0
    cf   = (gen / (pv * 8760.0)) if pv > 0 else 0.0
    batt = getattr(r, "e_batt_kwh", 0.0) or 0.0
    tst  = getattr(r, "e_th_kwh", 0.0) or 0.0
    pv_txt = f"PV {pv:.0f}kWp (CF {cf:.0%})" if pv > 0 else "PV 0kWp"
    if wrapped:
        return f"{pv_txt}\nBatt {batt:.0f}kWh  TST {tst:.0f}kWh"
    return f"{pv_txt} · Batt {batt:.0f}kWh · TST {tst:.0f}kWh"


def _save_fig(fig, out_path: str, label: str):
    # Shared figure-save boilerplate.
    import matplotlib.pyplot as plt
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  + {label} -> {out_path}")


# Stage-2 dispatch figure (intra-day charge/discharge behaviour)
def _dispatch_series(V: dict, w, year: int, m: str, d: str) -> dict:
    # Pull the solved half-hourly dispatch for one (scenario, year, month, day-type) and convert
    # every kWh/half-hour flow to average kW (÷ T_RES_H).
    val   = pulp.value
    n_pv  = float(val(V["n_pv"]) or 0.0)
    is_hp = V["is_hp"]
    eg    = V["elec_growth"][year]
    hg    = V["heat_growth"][(year, m)]
    T     = range(HH_PER_DAY)

    def hh(var):  # solved kW profile for a per-(w,y,m,d,t) dispatch variable
        return np.array([float(val(V[var][(w, year, m, d, t)]) or 0.0) for t in T]) / T_RES_H

    pv    = n_pv * V["pv_per_mod"][(year, m)] / T_RES_H              # kW
    dem   = V["demand_kwh"][(m, d)] * eg / T_RES_H                  # non-heat elec demand, kW
    out = {
        "hours":     np.arange(HH_PER_DAY) * T_RES_H,
        "pv":        pv,
        "discharge": hh("e_disc"),
        "import":    hh("e_im"),
        "demand":    dem,
        "charge":    hh("e_chg"),
        "export":    hh("e_ex"),
        "elec_heat": hh("elec_heat") if is_hp else np.zeros(HH_PER_DAY),
        "soc_kwh":   np.array([float(val(V["e_lvl"][(w, year, m, d, t)]) or 0.0) for t in T]),
        "heat_dem":  V["heat_kwh"][(m, d)] * hg / T_RES_H,          # useful heat demand, kW_th
        "th_soc_kwh": np.array([float(val(V["e_th_lvl"][(w, year, m, d, t)]) or 0.0) for t in T]),
    }
    return out


def plot_dispatch(district: str, activity: str, heating: str = "Gas Boiler", *,
                  months=("January", "July"), day_type: str = "WD", year: int = 0,
                  horizon_years: int = HORIZON_YEARS,
                  time_limit_s: int = DEFAULT_TIME_LIMIT_S,
                  objective: str = "cost", emissions_cap: float = None,
                  scenarios: list = None, scenario_index: int = 0,
                  out_path: str = None, solver_msg: bool = False):
    """Builds and solves a single (district, activity, heating) instance, then plots the optimal
    half-hourly dispatch for the chosen representative day(s) as one panel per month:
      - left axis: PV (generation), Grid (+ import / − export), and Load+HP (non-heat demand + heat-pump electricity).
      - right axis: SoC as % of installed capacity. Flat at 0 all day if that technology wasn't sized."""
    import matplotlib
    if out_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    oe._ensure_dm_initialized()
    months = [_norm_month(m) for m in months]
    if scenarios is None:
        scenarios = [oe.central_scenario()]              # deterministic single dispatch by default
    # Re-solve the exact design to be plotted. 
    prob, V = oe.build_lp(district, activity, heating,
                            horizon_years=horizon_years, objective=objective,
                            emissions_cap=emissions_cap, scenarios=scenarios)
    solver = oe._make_solver(solver_msg, time_limit_s)
    status = prob.solve(solver)
    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        raise RuntimeError(f"dispatch solve not optimal: {pulp.LpStatus[status]} "
                           f"({district} · {activity} · {heating})")
    w = list(V["W"])[scenario_index]

    fig, axes = plt.subplots(1, len(months), figsize=(6.4 * len(months), 5.4),
                             squeeze=False, sharey=True)
    dispatch_axes = axes[0]
    LINES    = [("pv", "PV", "#E8A33D"), ("grid", "Grid", "#B22222"), ("load", "Load+HP", "#1a1a1a")]
    SOC_COL  = "#4a4a4a"
    BATT_COL = "#2E8B57"
    TH_COL   = "#6A3D9A"
    e_batt = float(pulp.value(V["e_batt"]) or 0.0)
    e_th   = float(pulp.value(V["e_th"])   or 0.0)

    pct_axes = []
    for i, (ax, m) in enumerate(zip(dispatch_axes, months)):
        s = _dispatch_series(V, w, year, m, day_type)
        x = s["hours"]
        net = {"pv":   s["pv"],
               "grid": s["import"] - s["export"],
               "load": s["demand"] + s["elec_heat"]}
        for key, lbl, col in LINES:
            ax.plot(x, net[key], color=col, lw=1.6, label=lbl, solid_capstyle="round")
        ax.axhline(0, color="k", lw=0.6)
        ax.set_title(f"{m} · {day_type}", fontweight="bold")
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 24); ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(lw=0.3, alpha=0.4)

        # State of charge on the shared right-hand axis, as % of installed capacity.
        ax_pct = ax.twinx()
        batt_pct = 100.0 * s["soc_kwh"]    / e_batt if e_batt > 0 else np.zeros_like(x)
        th_pct   = 100.0 * s["th_soc_kwh"] / e_th   if e_th   > 0 else np.zeros_like(x)
        ax_pct.plot(x, batt_pct, color=BATT_COL, lw=1.6, label="Battery SoC (%)",
                    solid_capstyle="round")
        ax_pct.plot(x, th_pct, color=TH_COL, lw=1.6, ls="--",
                    label="Thermal store SoC (%)", solid_capstyle="round")
        ax_pct.set_ylim(0, 105)
        ax_pct.tick_params(axis="y", labelcolor=SOC_COL,
                           labelright=(i == len(months) - 1))
        pct_axes.append(ax_pct)

    dispatch_axes[0].set_ylabel("Power (kW)")
    pct_axes[-1].set_ylabel("State of charge (% of capacity)", color=SOC_COL)
    # one combined legend under the figure (electrical lines + both SoC lines)
    h, l   = dispatch_axes[0].get_legend_handles_labels()
    h2, l2 = pct_axes[0].get_legend_handles_labels()
    fig.legend(h + h2, l + l2, loc="lower center", ncol=len(l) + len(l2),
               fontsize=9, bbox_to_anchor=(0.5, -0.01))
    n_pv = int(round(float(pulp.value(V["n_pv"])) or 0))
    o_b  = float(pulp.value(V["o_batt"]) or 0.0)
    fig.suptitle(f"Optimal dispatch — {activity}{activity_area_suffix(activity)} · {district} · {heating}\n"
                 f"(year {year}; PV {n_pv * TECH_COSTS['pv']['module_kwp']:.0f} kWp, "
                 f"battery {e_batt:.0f} kWh / {o_b:.0f} kW, thermal store {e_th:.0f} kWh$_{{th}}$)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))

    if out_path is not None:
        _save_fig(fig, out_path, "dispatch figure")
    return fig


# Underlying demand figure — heat vs non-heat electricity
def plot_demand_profile(district: str, activity: str, *,
                        months=("January", "July"), day_type: str = "WD", year: int = 0,
                        horizon_years: int = HORIZON_YEARS, out_path: str = None):
    """Useful heat demand (kW_th) and non-heating electricity (kW)."""
    import matplotlib
    if out_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator

    oe._ensure_dm_initialized()
    months = [_norm_month(m) for m in months]

    # Demand inputs (supply-agnostic): non-heat electricity + useful heat, with the same horizon growth.
    demand_kwh  = oe.building_demand_kwh(activity, district)            # kWh/half-hour per (month, day-type)
    heat_kwh    = oe.building_heat_demand_kwh(activity, district)       # kWh_th/half-hour per (month, day-type)
    elec_growth = oe.elec_growth_factors(horizon_years)
    heat_growth = oe.heat_growth_factors(district, horizon_years)

    x = np.arange(HH_PER_DAY) * T_RES_H
    fig, axes = plt.subplots(1, len(months), figsize=(6.4 * len(months), 5.4),
                             squeeze=False, sharey=True)
    axes = axes[0]
    LINES = [("heat", "Heat demand (thermal)", "#C44E52"),
             ("elec", "Non-heating electricity", "#4C72B0")]
    for ax, m in zip(axes, months):
        series = {
            "heat": heat_kwh[(m, day_type)]   * heat_growth[(year, m)] / T_RES_H,   # kW_th
            "elec": demand_kwh[(m, day_type)] * elec_growth[year]      / T_RES_H,   # kW
        }
        for key, lbl, col in LINES:
            ax.plot(x, series[key], color=col, lw=1.8, label=lbl, solid_capstyle="round")
        ax.set_ylim(bottom=0)
        ax.set_title(f"{m} · {day_type}", fontweight="bold")
        ax.set_xlabel("Hour of day")
        ax.set_xlim(0, 24); ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.grid(lw=0.3, alpha=0.4)

    axes[0].set_ylabel("Power (kW)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(l), fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Underlying demand — {activity}{activity_area_suffix(activity)} · {district}  (year {year})",
                 fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    if out_path is not None:
        _save_fig(fig, out_path, "demand-profile figure")
    return fig


# Cost / carbon cell frontier
def plot_cell_front(df_front, out_path: str = None, *, title: str = None):
    # Cost/carbon frontier for one (district, activity) cell across heating types: 
    # all candidate points coloured by heating, the non-dominated frontier joined with adjacent-point MAC (£/tCO2e) annotations, 
    # and the knee (recommended) design starred.
    import matplotlib
    if out_path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    if df_front.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.5, 6))
    for h, sub in df_front.groupby("heating"):
        ax.scatter(sub["lifetime_emissions_tco2e"], sub["total_cost_npv_GBP"],
                   s=34, alpha=0.55, label=h, zorder=2)
    pf = df_front[df_front["pareto_optimal"]].sort_values("lifetime_emissions_tco2e") \
                                             .reset_index(drop=True)
    ax.plot(pf["lifetime_emissions_tco2e"], pf["total_cost_npv_GBP"],
            "-", color="k", lw=1.3, zorder=3, label="Pareto frontier")
    ax.scatter(pf["lifetime_emissions_tco2e"], pf["total_cost_npv_GBP"],
               facecolors="none", edgecolors="k", s=95, linewidths=1.2, zorder=4)
    # MAC = extra cost per tonne abated, between adjacent non-dominated points
    for i in range(1, len(pf)):
        de = pf.loc[i - 1, "lifetime_emissions_tco2e"] - pf.loc[i, "lifetime_emissions_tco2e"]
        dc = pf.loc[i, "total_cost_npv_GBP"] - pf.loc[i - 1, "total_cost_npv_GBP"]
        if de > 1e-6:
            xm = 0.5 * (pf.loc[i - 1, "lifetime_emissions_tco2e"] + pf.loc[i, "lifetime_emissions_tco2e"])
            ym = 0.5 * (pf.loc[i - 1, "total_cost_npv_GBP"] + pf.loc[i, "total_cost_npv_GBP"])
            ax.annotate(f"£{dc / de:,.0f}/t", (xm, ym), fontsize=7, color="dimgray",
                        ha="center", va="bottom")
    knee = df_front[df_front["is_knee"]] if "is_knee" in df_front else df_front.iloc[0:0]
    if not knee.empty:
        k = knee.iloc[0]
        ax.scatter([k["lifetime_emissions_tco2e"]], [k["total_cost_npv_GBP"]], marker="*",
                   s=320, color="#d62728", edgecolors="k", linewidths=0.6, zorder=5,
                   label=f"Knee — {k['heating']}")
        ax.annotate(spec_line(k), (k["lifetime_emissions_tco2e"], k["total_cost_npv_GBP"]),
                    xytext=(12, -16), textcoords="offset points", fontsize=7.5, zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#d62728", alpha=0.9))
    ax.set_xlabel("Lifetime emissions (tCO₂e, lower → better)")
    ax.set_ylabel("Total cost NPV (lower → better)")
    ax.set_title(title or "Cost / carbon frontier — featured cell", fontweight="bold")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"£{v / 1e3:,.0f}k"))
    ax.grid(lw=0.3, alpha=0.5)
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    if out_path is not None:
        _save_fig(fig, out_path, "cell front figure")
    return fig
