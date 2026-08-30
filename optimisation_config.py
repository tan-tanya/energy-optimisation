"""
Shared configuration for the optimisation model; imports only demand_profile_model + model_params.
"""
import calendar
import os

import demand_profile_model as dm
from model_params import TECH_COSTS
from seasons import DAY_TYPES

# Horizon and operational bounds
HORIZON_YEARS        = 15
T_RES_H              = 0.5        # half-hour timestep
BATT_MAX_KWH         = 50000      # upper bound for sizing decision
BATT_MAX_KW          = int(BATT_MAX_KWH * TECH_COSTS["battery"]["c_rate_chg"])
DEFAULT_TIME_LIMIT_S = 7200
PARALLEL_JOBS        = None       # worker processes per sweep; None → resolve_jobs() default, 1 → serial
SOLVER_THREADS       = None       # threads per worker solver; None → auto-pair with the worker count so (workers × threads) ≈ logical cores

# Wholesale·HH-shape + per-DNO DUoS Red/Amber/Green + CCL + band-specific residual
# DUoS unit rates and the Red/Amber/Green time-band windows are transcribed per district from each DNO's 2025/26 CDCM Schedule of Charges
# Wholesale is real Elexon MID 2025
# Set False to revert to flat DESNZ band price
USE_WHOLESALE_DUOS_BUILDUP = True

# Construction context: new-build vs retrofit
# NEW_BUILD = True (default): every scenario AND the BAU install a fresh heating system
# NEW_BUILD = False (retrofit): the gas boiler carries no upfront capex / replacement; 
# heat-pump scenario pays capex + a one-off boiler decommissioning cost 
NEW_BUILD = True

# Representative-day sets — 12 months × {WD, WE}; weighted by 2025 calendar.
# DAY_TYPES is imported from seasons.py and re-exported here (S_KEYS is built from it).
S_KEYS = [(m, d) for m in dm.MONTHS_ORDER for d in DAY_TYPES]


def wd_we_counts_2025() -> dict:
    # Actual count of weekdays / weekend days in each month of 2025; same counts assumed across the 15-year horizon
    counts = {}
    for m_idx, m_name in enumerate(dm.MONTHS_ORDER, 1):
        n_days = calendar.monthrange(2025, m_idx)[1]
        wd = sum(1 for d in range(1, n_days + 1)
                 if calendar.weekday(2025, m_idx, d) < 5)
        counts[m_name] = {"WD": wd, "WE": n_days - wd}
    return counts


WD_WE_DAYS = wd_we_counts_2025()
N_DAYS_OF  = {(m, d): WD_WE_DAYS[m][d] for (m, d) in S_KEYS}


def resolve_jobs(n_jobs: int = None, n_tasks: int = None) -> tuple:
    logical = os.cpu_count() or 2
    if n_jobs is None:
        n_jobs = PARALLEL_JOBS if PARALLEL_JOBS is not None else max(1, logical // 4)
    n_jobs = max(1, n_jobs)
    if n_tasks:
        n_jobs = min(n_jobs, n_tasks)
    threads = SOLVER_THREADS or max(1, logical // n_jobs)
    return n_jobs, threads
