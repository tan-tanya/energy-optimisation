"""
Import-only. Projects demand-growth and climate trajectories over the optimisation horizon.
  - Non-heat electricity grows per the DESNZ Reference projection (electricity_projection_output.csv).
  - Heat demand falls as climate warms, per the UKCP18 HDD projection (climate_projection_output.csv): scaled by projected_hdd/baseline_hdd per district-month.
COP improves over the horizon as climate warms; heat-pump electricity falls from both lower heat demand and higher COP.

Projection CSVs are cached once per process.
"""
import demand_profile_model as dm
import datasets
from model_params import EMISSIONS_BASE_YEAR

_elec_growth_by_year = None    # {calendar_year: growth_factor}
_climate_hdd_ratio   = None    # {district: {(year, month): projected_hdd / baseline_hdd}}
_climate_delta_t     = None    # {district: {(year, month): delta_T_mean (°C)}} — COP warming shift


def _load_elec_growth() -> dict:
    global _elec_growth_by_year
    if _elec_growth_by_year is None:
        df = datasets.get_electricity_projection()
        _elec_growth_by_year = {int(r.year): float(r.growth_factor) for r in df.itertuples()}
    return _elec_growth_by_year

def _load_climate_ratio() -> dict:
    global _climate_hdd_ratio
    if _climate_hdd_ratio is None:
        df = datasets.get_climate_projection()
        out = {}
        for r in df.itertuples():
            base = float(r.baseline_hdd_per_day)
            out.setdefault(r.district, {})[(int(r.year), int(r.month))] = (
                float(r.projected_hdd_per_day) / base if base > 0 else 1.0)
        _climate_hdd_ratio = out
    return _climate_hdd_ratio

def _load_climate_delta_t() -> dict:
    # {district: {(year, month): delta_T_mean}} — UKCP18 monthly warming anomaly (°C) vs current climate.
    global _climate_delta_t
    if _climate_delta_t is None:
        df = datasets.get_climate_projection()
        out = {}
        for r in df.itertuples():
            out.setdefault(r.district, {})[(int(r.year), int(r.month))] = float(r.delta_T_mean)
        _climate_delta_t = out
    return _climate_delta_t

def cop_delta_factors(district: str, horizon_years: int, *, air_source: bool) -> dict:
    # {(y, month_name): temperature shift (°C)} applied to the COP's driving temperature for warming.
    # ASHP tracks monthly; GSHP tracks annual mean
    dmap = _load_climate_delta_t().get(district, {})
    midx = {name: i for i, name in enumerate(dm.MONTHS_ORDER, 1)}

    def monthly(cal, m):
        if cal <= EMISSIONS_BASE_YEAR:
            return 0.0
        if (cal, midx[m]) in dmap:
            return dmap[(cal, midx[m])]
        yrs = [yy for (yy, mm) in dmap if mm == midx[m] and yy <= cal]   # clamp beyond projection range
        return dmap[(max(yrs), midx[m])] if yrs else 0.0

    out = {}
    for y in range(horizon_years):
        cal = EMISSIONS_BASE_YEAR + y
        if air_source:
            for m in dm.MONTHS_ORDER:
                out[(y, m)] = monthly(cal, m)
        else:
            tot = sum(monthly(cal, m) * dm.MONTH_DAYS[m] for m in dm.MONTHS_ORDER)
            ann = tot / sum(dm.MONTH_DAYS[m] for m in dm.MONTHS_ORDER)
            for m in dm.MONTHS_ORDER:
                out[(y, m)] = ann
    return out

def elec_growth_factors(horizon_years: int) -> list:
    # Per-horizon-year non-heat electricity growth factor.
    g = _load_elec_growth()
    last = g[max(g)]
    out = []
    for y in range(horizon_years):
        cal = EMISSIONS_BASE_YEAR + y
        out.append(1.0 if cal <= EMISSIONS_BASE_YEAR else g.get(cal, last))
    return out

def heat_growth_factors(district: str, horizon_years: int) -> dict:
    # {(y, month_name): projected_hdd/baseline_hdd} for the district.
    ratio = _load_climate_ratio().get(district, {})
    midx  = {name: i for i, name in enumerate(dm.MONTHS_ORDER, 1)}
    out = {}
    for y in range(horizon_years):
        cal = EMISSIONS_BASE_YEAR + y
        for m in dm.MONTHS_ORDER:
            if cal <= EMISSIONS_BASE_YEAR:
                out[(y, m)] = 1.0
            elif (cal, midx[m]) in ratio:
                out[(y, m)] = ratio[(cal, midx[m])]
            else:                                   # clamp to the latest projected year for this month
                yrs = [yy for (yy, mm) in ratio if mm == midx[m] and yy <= cal]
                out[(y, m)] = ratio[(max(yrs), midx[m])] if yrs else 1.0
    return out
