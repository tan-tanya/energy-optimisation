"""
Import-only. 
Builds and solves one cell's PV/battery/heat-pump/thermal-store problem, and sweeps every cell for a run. 
optimisation_model.py is the thin CLI on top, re-exports the names below so existing consumers that 
`import optimisation_model as om` keep working unchanged.

Inputs
1. demand_profile_model; reused module-level state and helpers
    - DISTRICT_STATIONS, MONTHS_ORDER, MONTH_SEASON, WE_LOAD_FACTOR, HDD_BASE, DIURNAL_AMPLITUDE
    - half_hourly_kw_per_sqm()                          (half-hourly electricity or gas demand per m², kW/m²)
    - solar_elevation_profile()                         (peak-shape hourly solar elevation per parent season)
    - hourly_temp_profile()                             (diurnal temperature swing around mean T)
    - daily_hdd_by_district, monthly_dd_by_district     (per-district daily/monthly degree-day stats)
    - bees_floor_areas                                  (BEES median premises floor area per activity, m²)
2. data/sunlighthours/*.txt;                            (Met Office HadUK-Grid monthly sunshine hours per district)

Model layers:
    (a) PV generation     irradiance shape × monthly daily-GHI × temperature derate × SOH (module + inverter)
    (b) PV sizing limits  roof area × usable fraction × inter-row spacing (flat-roof only) × module area;
                          live-load capacity × module weight
    (c) Battery           n_pv·pv_t + e_disc + e_im  ==  dem_t + elec_heat_t + e_chg + e_ex
    (d) Heat pump         elec_heat = heat_prod / COP(y,m,t)
    (e) COP               ASHP: varies with intra-day temperature; GSHP: flat for vertical, month-averaged daily ground-temp for horizontal
    (f) Boiler            gas_im = heat_prod / η_boiler
    (g) Thermal storage   heat_prod + heat_dis == heat_dem + heat_chg
    (h) Grid              flat-rate import/export with annual price escalation, capped by DNO connection
    (i) Economics         capex (incl. BoS + install fixed costs), opex (NPV), maintenance + insurance (escalating, NPV),
                          year-10 replacements (PV inverter + battery); 15-yr horizon at WACC = 5%
    (j) Dispatch          24 representative days/year (12 months × WD/WE), weighted by 2025 WD/WE day counts

Pipeline:
    1. build_milp()              assemble the LP/MILP for one (district, activity, heating)
    2. solve_scenario()          solve and extract sizing + cost + energy metrics
    3. rank_all_combinations() / _run_merged_sweep()   sweep every (district, activity, heating)
    4. write_results_workbook()  write the ranking sheets
"""

import os
import multiprocessing as mp
from datetime import datetime

import numpy as np
import pandas as pd
import pulp

import demand_profile_model as dm
import datasets
import uncertainty as unc
from model_params import (
    TECH_COSTS, HEAT_COSTS, THERMAL_STORE, ROOF_PROPERTIES,
    ROOF_LOAD_KG_PER_M2, ROOF_PITCH_DEG, PITCHED_USABLE_SLOPE_FRAC,
    DISTRICT_MONTHLY_GHI, select_elec_band, select_gas_band, select_grid_limit,
    GAS_EMISSION_FACTOR, elec_emission_factor, carbon_value,
    HORIZONTAL_COLLECTOR_M2_PER_KWTH, SITE_PLOT_RATIO, PARKING_GROSS_M2_PER_SPACE,
    EV_PARKING_DENSITY,
)
from optimisation_config import (
    HORIZON_YEARS, T_RES_H, HH_PER_DAY, BATT_MAX_KWH, BATT_MAX_KW,
    DEFAULT_TIME_LIMIT_S, MIP_GAP_REL, LP_METHOD, PARALLEL_JOBS, SOLVER_THREADS,
    USE_WHOLESALE_DUOS_BUILDUP, NEW_BUILD, S_KEYS, N_DAYS_OF,
)
from growth import cop_delta_factors, elec_growth_factors, heat_growth_factors
from pricing import import_price_slots_central


# TSSP Stage 1: sizing/selection — n_pv, e_batt, o_batt, q_heat_cap, e_th
# TSSP Stage 2: dispatch — e_im, e_ex, e_chg, e_disc, q_heat_prod, q_heat_dis, q_heat_chg
# The only uncertain input in this iteration is the electricity import price; export and gas prices stay central
# Emissions are price-independent, so the emissions objective is solved deterministically on a single central scenario
_PRICE_SCENARIOS = None   

def price_scenarios() -> list:
    # Reduced, weighted electricity import-price scenarios for the cost objective
    global _PRICE_SCENARIOS
    if _PRICE_SCENARIOS is None:
        _PRICE_SCENARIOS = unc.generate_price_scenarios(horizon=HORIZON_YEARS)
    return _PRICE_SCENARIOS

def set_price_scenarios(scenarios: list) -> None:
    # Override the cached reduced price-scenario set. 
    global _PRICE_SCENARIOS
    _PRICE_SCENARIOS = scenarios

def central_scenario() -> "unc.Scenario":
    # Single scenario reproducing the deterministic central case
    g = TECH_COSTS["elec_price_growth"]
    return unc.Scenario(id="central", weight=1.0, level=1.0, growth=g,
                        path=unc.price_multiplier_path(1.0, g, HORIZON_YEARS))

def _scenarios_for(objective: str, scenarios: list = None) -> list:
    # Cost objective uses stochastic scenarios; emissions objective uses single central (deterministic) scenario
    if scenarios is not None:
        return scenarios
    return price_scenarios() if objective == "cost" else [central_scenario()]


def _ensure_dm_initialized():
    # dm.initialize() populates bees_floor_areas, daily_hdd_by_district, monthly_dd_by_district, etc.
    if dm.bees_floor_areas is None:
        dm.initialize()


# 1 - Source data and parameters 

# OSM storey survey output
OSM_STOREYS_XLSX = datasets.OSM_STOREYS_XLSX   # data/api_osm_storeys.xlsx (single path source: datasets)

# Footprint-size bands — must match FOOTPRINT_BINS / FOOTPRINT_LABELS in api_osm_storeys.py.
FOOTPRINT_BINS   = [0, 250, 1000, 5000, float("inf")]
FOOTPRINT_LABELS = ["<250", "250-1,000", "1,000-5,000", ">=5,000"]

_osm_median_storeys   = None  # set up {activity: median storeys}; lazily loaded from OSM_STOREYS_XLSX
_osm_flat_by_footprint = None # set up {(activity, band): flat fraction}; lazily loaded from OSM_STOREYS_XLSX

# Horizon / solver constants, the wholesale+DUoS build-up flags.

# 2 - PV generation model (eq. 1.21–1.24, 1.47, 1.49)

def daily_ghi_kwh_per_m2(district: str, month: str) -> float:
    # Per-district daily GHI (kWh/m²/day): PVGIS-SARAH3 2005–2023 monthly average at the district's reference station.
    return DISTRICT_MONTHLY_GHI[district][month]

def _pv_module_y0(district: str, month: str) -> np.ndarray:
    # Year-0 PV: 48-element half-hourly per-module power (kW/module, AC side)
    pv      = TECH_COSTS["pv"]
    parent  = dm.MONTH_SEASON[month]

    # (a) Half-hourly irradiance profile (kW/m²) — peak-shape × daily-total scaling.
    solar_24h = dm.solar_elevation_profile(parent)      # 24-element, peak = 1
    shape_48  = np.repeat(solar_24h, 2)                 # 48-element
    integral  = shape_48.sum() * T_RES_H                # area under shape, peak=1 kW/m²
    daily_ghi = daily_ghi_kwh_per_m2(district, month)
    irrad_kw_per_m2 = shape_48 * (daily_ghi / integral) if integral > 0 else shape_48 * 0.0

    # (b) Cell temperature: T_amb mean from monthly HDD; diurnal amplitude from parent season
    monthly_HDD = dm.monthly_dd_by_district[district][month]["hdd_per_day"]
    T_amb_mean  = dm.HDD_BASE - monthly_HDD
    T_amb_24h   = dm.hourly_temp_profile(T_amb_mean, parent, district=district)
    T_amb_48    = np.repeat(T_amb_24h, 2)
    T_cell      = T_amb_48 + pv["cell_temp_coeff"] * (irrad_kw_per_m2 * 1000.0)

    # (c) Temperature correction
    temp_factor = 1.0 + pv["temp_coeff_per_C"] * (T_cell - pv["ref_temp_C"])
    temp_factor = np.maximum(temp_factor, 0.0)          # no negative output

    # (d) Per-module power before SOH
    output_kw = irrad_kw_per_m2 * pv["module_area_m2"] * pv["efficiency"] * pv["inverter_eff"] * temp_factor
    return np.maximum(0.0, output_kw)

def load_osm_median_storeys() -> dict:
    # Load median storeys per activity class from the OSM survey.
    df = datasets.get_osm_summary()
    median_col = next(c for c in df.columns if "median" in str(c).lower())
    return {str(a): float(v) for a, v in zip(df["Activity Class"], df[median_col])
            if pd.notna(v)}

def osm_median_storeys() -> dict:
    # Load + cache per-activity median storeys from the OSM survey.
    global _osm_median_storeys
    if _osm_median_storeys is None:
        _osm_median_storeys = load_osm_median_storeys()
    return _osm_median_storeys

def roof_to_floor_ratio(activity: str) -> float:
    # Building footprint / total floor area = 1 / median storeys (OSM building:levels survey).
    return 1.0 / osm_median_storeys()[activity]

def roof_area_m2_for_activity(activity: str) -> float:
    # Roof footprint = floor area / median storeys = roof_to_floor_ratio × BEES median floor area.
    return roof_to_floor_ratio(activity) * dm.bees_floor_areas[activity]

# Horizontal ground-loop land availability
# A trenched (slinky) collector needs open SOFT ground in proportion to its thermal capacity; a
# vertical borefield does not. Chain: plot ratio gives the site, subtract the building footprint,
# subtract committed hardstanding (parking), and what remains is available for the collector.
def site_area_m2(activity: str) -> float:
    # Plot ratio is gross FLOORSPACE / site area, so it divides floor area directly (not footprint).
    return dm.bees_floor_areas[activity] / SITE_PLOT_RATIO


def parking_area_m2(activity: str) -> float:
    # Committed hardstanding = spaces (per m2 GFA) x gross land per space. Reuses the per-activity
    # EV_PARKING_DENSITY already on the workbook so there is one parking assumption, not two.
    return (dm.bees_floor_areas[activity] * EV_PARKING_DENSITY[activity]
            * PARKING_GROSS_M2_PER_SPACE)


def soft_ground_m2(activity: str) -> float:
    # Open land left after the building and its parking. Clamped at 0: for the smallest buildings the
    # maximum parking standard can exceed the plot-ratio-implied site, i.e. the two sourced maxima are
    # mutually inconsistent — that means parking-dominated, so no collector land, not negative land.
    return max(0.0, site_area_m2(activity)
               - roof_area_m2_for_activity(activity) - parking_area_m2(activity))


def horizontal_loop_max_kwth(activity: str) -> float:
    # Ground-loop thermal capacity the available soft ground can support [kW_th].
    return soft_ground_m2(activity) / HORIZONTAL_COLLECTOR_M2_PER_KWTH


def land_limit_kwth(activity: str, heating: str) -> float:
    # Land-imposed ceiling on q_heat_cap. Only GSHP (horizontal) is constrained; everything else
    # (vertical borefield, ASHP, gas boiler) is unconstrained by surface area.
    if heating != "GSHP (horizontal)":
        return float("inf")
    return horizontal_loop_max_kwth(activity)


def load_osm_flat_by_footprint() -> dict:
    # Flat-roof share per (activity class, footprint band). Returns {(activity, band_label): flat fraction in [0, 1]}.
    df = datasets.get_osm_flat_by_footprint()
    flat_col = next(c for c in df.columns if "flat" in str(c).lower() and "%" in str(c))
    return {(str(a), str(b)): float(v) / 100.0
            for a, b, v in zip(df["Activity Class"], df["Footprint band"], df[flat_col])
            if pd.notna(v)}

def osm_flat_by_footprint() -> dict:
    # Load + cache the per-(activity, band) flat share from the OSM survey.
    global _osm_flat_by_footprint
    if _osm_flat_by_footprint is None:
        _osm_flat_by_footprint = load_osm_flat_by_footprint()
    return _osm_flat_by_footprint

def _footprint_band(footprint_m2: float) -> str:
    # Band label for a footprint, using the same bins as api_osm_storeys.py.
    i = int(np.digitize([footprint_m2], FOOTPRINT_BINS, right=False)[0]) - 1
    return FOOTPRINT_LABELS[min(max(i, 0), len(FOOTPRINT_LABELS) - 1)]

def flat_share_for_activity(activity: str) -> float:
    # Flat-roof share for this activity, matched to its modelled footprint band. If no OSM sample, fall back to the nearest populated band.
    table = osm_flat_by_footprint()
    band  = _footprint_band(roof_area_m2_for_activity(activity))
    if (activity, band) in table:
        return table[(activity, band)]
    here = FOOTPRINT_LABELS.index(band)                          # nearest populated band fallback
    for offset in range(1, len(FOOTPRINT_LABELS)):
        for j in (here - offset, here + offset):
            if 0 <= j < len(FOOTPRINT_LABELS) and (activity, FOOTPRINT_LABELS[j]) in table:
                return table[(activity, FOOTPRINT_LABELS[j])]
    raise KeyError(f"No flat-share data for {activity!r} in any footprint band")

def roof_usable_frac_for_activity(activity: str) -> float:
    # Blended usable-area fraction of the footprint, combining the flat and pitched portions.
    # Both roof types carry the SAME per-activity usable-area fraction, differing only in the roof-type-specific derate:
    #   flat    → × inter-row spacing (self-shading clearance between rows of a tilted rack);
    #   pitched → × sec(pitch) slope-area gain × PITCHED_USABLE_SLOPE_FRAC. 
    #               Pitched modules mount flush to the slope, so there is no inter-row derate (BRE NSC 2016).
    flat          = flat_share_for_activity(activity)
    props         = ROOF_PROPERTIES[activity]
    flat_frac     = props["pv_usable_frac"] * props["pv_inter_row_frac"]
    pitch_factor  = ((1.0 / np.cos(np.radians(ROOF_PITCH_DEG)))
                     * PITCHED_USABLE_SLOPE_FRAC * props["pv_usable_frac"])
    return flat * flat_frac + (1.0 - flat) * pitch_factor

def n_pv_max_for_activity(activity: str) -> int:
    # Maximum installable modules — binding of area (flat+pitched split) and roof live-load weight.
    roof_area   = roof_area_m2_for_activity(activity)
    pv          = TECH_COSTS["pv"]
    usable_frac = roof_usable_frac_for_activity(activity)
    by_area   = (roof_area * usable_frac)        / pv["module_area_m2"]
    by_weight = (roof_area * ROOF_LOAD_KG_PER_M2) / pv["module_weight_kg"]
    return int(np.floor(min(by_area, by_weight)))


# 3 - Demand wrapper (electricity-only, lighting + plug + HVAC + EV)
def _wd_we_demand(activity: str, district: str, kw_per_m2) -> dict:
    # Shared scaffold for the non-heat and heat demand builders: month loop + WD/WE split.
    # kw_per_m2(month, parent, monthly_dd_m, dd_daily) -> 48-element kW/m² array for that month.
    # WD/WE split: base profile × wd_fac for WD and × wd_fac × f for WE, with wd_fac = 7/(5+2f) and f = WE_LOAD_FACTOR[activity]
    dd_daily   = dm.daily_hdd_by_district[district]
    monthly_dd = dm.monthly_dd_by_district[district]
    floor_area = dm.bees_floor_areas[activity]
    f          = dm.WE_LOAD_FACTOR[activity]
    wd_fac     = 7.0 / (5.0 + 2.0 * f)
    we_fac     = wd_fac * f

    out = {}
    for m in dm.MONTHS_ORDER:
        parent   = dm.MONTH_SEASON[m]                   # monthly DD goes into the parent-season slot
        base_kwh = kw_per_m2(m, parent, monthly_dd[m], dd_daily) * T_RES_H * floor_area
        out[(m, "WD")] = base_kwh * wd_fac
        out[(m, "WE")] = base_kwh * we_fac
    return out


def building_demand_kwh(activity: str, district: str) -> dict:
    # Pure non-heating electricity (lighting + small power + HVAC + EV) - kWh per half-hour for every (month, day_type) combo
    def kw_per_m2(m, parent, monthly_dd_m, dd_daily):
        return dm.half_hourly_kw_per_sqm(
            activity, "Gas Boiler", "Electricity", parent, {parent: monthly_dd_m}, dd_daily
        )
    return _wd_we_demand(activity, district, kw_per_m2)


# 3b - Heat demand + heat-pump COP
def building_heat_demand_kwh(activity: str, district: str) -> dict:
    # Useful thermal demand [kWh/half-hour] per (month, day_type) combo derived from the Gas Boiler breakdown
    # (space heating + hot water) × boiler efficiency = useful heat delivered. Same WD/WE scaling as electricity.
    def kw_per_m2(m, parent, monthly_dd_m, dd_daily):
        b = dm.half_hourly_kw_per_sqm_breakdown(
            activity, "Gas Boiler", parent, {parent: monthly_dd_m}, dd_daily
        )
        return (b["Space Heating"] + b["Hot Water & Process"]) * dm.ETA_BOILER  # gas × η = useful heat
    return _wd_we_demand(activity, district, kw_per_m2)

def cop_profile(heating: str, district: str, month: str, delta_t: float = 0.0) -> np.ndarray:
    # ASHP: varies with outdoor air temperature over the day;
    # GSHP: flat for vertical loops; for horizontal loops the COP follows the 1.5 m ground temperature 
    # delta_t (°C) shifts the driving temperature for future-climate warming
    system = dm.HEATING_SYSTEMS[heating]
    if not system["is_heat_pump"]:
        return None
    parent      = dm.MONTH_SEASON[month]
    monthly_hdd = dm.monthly_dd_by_district[district][month]["hdd_per_day"]
    loop_type   = system["loop_type"]
    if loop_type is None:                                   # ASHP — air source
        t_ext_mean = (dm.HDD_BASE - monthly_hdd) + delta_t
        t_hourly = dm.hourly_temp_profile(t_ext_mean, parent, district=district)
        cop24    = np.array([dm.cop_ashp(t) for t in t_hourly])
        return np.repeat(cop24, 2)
    stats       = dm.district_climate_stats[district]       # GSHP — ground source
    annual_mean = stats["annual_mean_T_C"] + delta_t
    first_doy   = dm.MONTH_START_DOY[month]
    days        = range(first_doy, first_doy + dm.MONTH_DAYS[month])
    cop_month   = float(np.mean([
        dm.cop_gshp(dm.brine_temperature(annual_mean, stats["surface_amplitude_C"],
                                         doy, loop_type))
        for doy in days]))
    return np.full(HH_PER_DAY, cop_month)

def peak_heat_kwth(activity: str, district: str) -> float:
    # Peak instantaneous useful-heat load [kW_th] across all representative slots (sizing reference)
    heat = building_heat_demand_kwh(activity, district)
    return max(float(arr.max()) for arr in heat.values()) / T_RES_H


# 4 - SOH curves and NPV factors
def soh_pv(year: int) -> float:
    return (1.0 - TECH_COSTS["pv"]["soh_decay_per_yr"]) ** year

def soh_inv(year: int) -> float:
    return (1.0 - TECH_COSTS["pv"]["soh_inv_decay_per_yr"]) ** year

def soh_batt(year: int) -> float:
    return (1.0 - TECH_COSTS["battery"]["soh_decay_per_yr"]) ** year

def discount_factor(year: int) -> float:
    return 1.0 / (1.0 + TECH_COSTS["discount_rate"]) ** year

def import_price(year: int, base: float) -> float:
    # base = year-0 import price (GBP/kWh) for the building's DESNZ size band (see select_elec_band)
    return base * (1.0 + TECH_COSTS["elec_price_growth"]) ** year

def export_price(year: int) -> float:
    return TECH_COSTS["elec_export_price"] * (1.0 + TECH_COSTS["elec_price_growth"]) ** year

def gas_price(year: int, base: float) -> float:
    # base = year-0 gas import price (GBP/kWh) for the building's DESNZ gas size band (see select_gas_band)
    return base * (1.0 + TECH_COSTS["gas_price_growth"]) ** year


# 5 - MILP build

def _prepare_data(district: str, activity: str, heating: str, horizon_years: int,
                  scenarios: list, *, demand_multiplier: float = 1.0,
                  import_limit_override_kw: float = None) -> dict:
    # Pre-compute non-heat demand, heat demand + COP, PV per-module arrays, and price/discount caches.
    Y = range(horizon_years)
    demand_kwh = building_demand_kwh(activity, district)
    heat_kwh   = building_heat_demand_kwh(activity, district)
    # Non-heat peak captured before demand_multiplier scaling — see the import-ceiling floor below.
    unscaled_peak_kwh = max(float(arr.max()) for arr in demand_kwh.values())
    if demand_multiplier != 1.0:
        demand_kwh = {k: v * demand_multiplier for k, v in demand_kwh.items()}
        heat_kwh   = {k: v * demand_multiplier for k, v in heat_kwh.items()}

    # Demand growth over the horizon.
    elec_growth = elec_growth_factors(horizon_years)
    heat_growth = heat_growth_factors(district, horizon_years)

    # Grid connection ceilings (kW) per district, from Scalars sheet.
    glim            = select_grid_limit(district)
    base_import_kw  = glim["import_kw"] if import_limit_override_kw is None else import_limit_override_kw
    # Floor the import ceiling at the final-horizon-year electricity so the site can serve its non-flexible load even after demand growth.
    # The floor is built from the UNSCALED non-heat peak: demand_multiplier must not raise the ceiling. 
    baseline_peak_kw = unscaled_peak_kwh / T_RES_H * max(elec_growth)
    import_limit_kw  = max(base_import_kw, baseline_peak_kw)
    export_limit_kw  = max(glim["export_kw"], 0.0)

    is_hp = dm.HEATING_SYSTEMS[heating]["is_heat_pump"]
    # COP improves over the horizon as the climate warms.
    if is_hp:
        air_source  = dm.HEATING_SYSTEMS[heating]["loop_type"] is None
        cop_delta   = cop_delta_factors(district, horizon_years, air_source=air_source)
        cop = {(y, m): cop_profile(heating, district, m, delta_t=cop_delta[(y, m)])
               for y in Y for m in dm.MONTHS_ORDER}
    else:
        cop = None
    peak  = peak_heat_kwth(activity, district) * demand_multiplier

    # DESNZ size-band import price.
    annual_nonheat_kwh = sum(demand_kwh[(m, d)].sum() * N_DAYS_OF[(m, d)] for (m, d) in S_KEYS)
    annual_hp_elec_kwh = (sum((heat_kwh[(m, d)] / cop[(0, m)]).sum() * N_DAYS_OF[(m, d)] for (m, d) in S_KEYS)
                          if is_hp else 0.0)
    elec_band = select_elec_band((annual_nonheat_kwh + annual_hp_elec_kwh) / 1000.0)
    im_base   = elec_band["price"]
    # Divide out any active import-price level multiplier; each scenario's multiplier applies on top of a clean central base.
    im_base_central = im_base / TECH_COSTS.get("elec_import_multiplier", 1.0)

    # DESNZ size-band gas price: select from gross annual gas burned (useful heat / boiler efficiency) in MWh/yr.
    annual_gas_kwh = (sum(heat_kwh[(m, d)].sum() * N_DAYS_OF[(m, d)] for (m, d) in S_KEYS) / dm.ETA_BOILER
                      if not is_hp else 0.0)
    gas_band = select_gas_band(annual_gas_kwh / 1000.0)
    gas_base = gas_band["price"]

    # Year-0 PV computed once per month; SOH multipliers applied per year (linear scaling)
    pv_y0_kwh_hh = {m: _pv_module_y0(district, m) * T_RES_H for m in dm.MONTHS_ORDER}   # kWh/HH/module
    pv_per_mod   = {(y, m): pv_y0_kwh_hh[m] * soh_pv(y) * soh_inv(y)
                    for y in Y for m in dm.MONTHS_ORDER}

    # Per-year coefficient caches.
    df_y       = [discount_factor(y) for y in Y]
    if USE_WHOLESALE_DUOS_BUILDUP:
        # Slot-dependent import price: year-0 central build-up per (day_type, slot), escalated by each scenario's path and discounted. 
        # Keyed (y, m, d, t) so the intra-day ToU + regional shape reaches the objective's per-slot import term.
        slot_c = import_price_slots_central(district, elec_band["name"])
        im_d = {w: {(y, m, d, t): slot_c[(m, d, t)] * float(scen.path[y]) * df_y[y]
                    for y in Y for (m, d) in S_KEYS for t in range(HH_PER_DAY)}
                for w, scen in enumerate(scenarios)}
        im_slot_dependent = True
        im_slot_central = slot_c   # year-0 central per-(m,d,t) price; reused by BAU + payback reporting
    else:
        im_d = {w: [scen.import_price(y, im_base_central) * df_y[y] for y in Y]
                for w, scen in enumerate(scenarios)}
        im_slot_dependent = False
        im_slot_central = None
    ex_d       = [export_price(y) * df_y[y] for y in Y]
    gas_d      = [gas_price(y, gas_base) * df_y[y] for y in Y]
    infl_d     = [(1.0 + TECH_COSTS["general_inflation"]) ** y * df_y[y] for y in Y]
    soh_batt_y = [soh_batt(y) for y in Y]
    # Carbon: grid emission-factor trajectory (kgCO2e/kWh) per horizon year, gas factor (constant),
    # and the appraisal carbon value (GBP/tCO2e) discounted for the monetised-carbon reporting figure.
    ef_elec_y  = [elec_emission_factor(y) for y in Y]
    cval_d     = [carbon_value(y) * df_y[y] for y in Y]

    return {
        "Y": Y, "S": S_KEYS, "T": range(HH_PER_DAY),
        "scenarios":   scenarios,
        "W":           range(len(scenarios)),
        "weights":     [s.weight for s in scenarios],
        "im_base_central": im_base_central,
        "heating":     heating,
        "is_hp":       is_hp,
        "demand_kwh":  demand_kwh,
        "heat_kwh":    heat_kwh,
        "elec_growth": elec_growth,
        "heat_growth": heat_growth,
        "cop":         cop,
        "peak_kwth":   peak,
        # Heat-plant capacity ceiling: 2x peak demand (headroom to charge the store), tightened for
        # GSHP (horizontal) by the collector land actually available. Expressed as a capacity BOUND
        # rather than a pre-solve yes/no so a land-limited site can still adopt a SMALLER horizontal
        # loop leaning harder on the thermal store; if even that cannot serve the load the LP returns
        # Infeasible, which is the physically meaningful answer.
        "q_heat_max":  min(peak * 2.0, land_limit_kwth(activity, heating)),
        "land_limit_kwth": land_limit_kwth(activity, heating),
        "th_max_kwh":  peak * 24.0,    # buffer up to ~a day of peak heat
        "pv_per_mod":  pv_per_mod,
        "n_pv_max":    n_pv_max_for_activity(activity),
        "import_limit_kw": import_limit_kw,
        "export_limit_kw": export_limit_kw,
        "df_y":        df_y,
        "im_d":        im_d,
        "im_slot_dependent": im_slot_dependent,
        "im_slot_central":   im_slot_central,
        "ex_d":        ex_d,
        "gas_d":       gas_d,
        "infl_d":      infl_d,
        "soh_batt_y":  soh_batt_y,
        "ef_elec_y":   ef_elec_y,
        "ef_gas":      GAS_EMISSION_FACTOR,
        "cval_d":      cval_d,
        "im_base":     im_base,
        "elec_band":   elec_band["name"],
        "gas_base":    gas_base,
        "gas_band":    gas_band["name"],
    }


def _declare_variables(data: dict, use_binary_mutex: bool) -> dict:
    # Stage-1 sizing variables (shared across scenarios) + stage-2 dispatch variables (per scenario w).
    Y, S, T, W = data["Y"], data["S"], data["T"], data["W"]

    V = {
        # STAGE 1 — here-and-now sizing/selection, shared across all scenarios.
        "n_pv":       pulp.LpVariable("n_pv",       lowBound=0, upBound=data["n_pv_max"],  cat="Continuous"),
        "e_batt":     pulp.LpVariable("e_batt",     lowBound=0, upBound=BATT_MAX_KWH,      cat="Continuous"),
        "o_batt":     pulp.LpVariable("o_batt",     lowBound=0, upBound=BATT_MAX_KW,       cat="Continuous"),
        # PV / thermal-store install gates (z_pv, z_th) are omitted: with no fixed install cost they carry
        # no objective pressure, so they optimise to 1 and their big-M constraints reduce to the n_pv_max /
        # th_max_kwh upper bounds already declared here. Reintroduce as cat="Binary" if a fixed cost is added.
        # Heat sizing: heating-plant thermal capacity (mandatory) + thermal store (optional)
        "q_heat_cap": pulp.LpVariable("q_heat_cap", lowBound=0, upBound=data["q_heat_max"], cat="Continuous"),
        "e_th":       pulp.LpVariable("e_th",       lowBound=0, upBound=data["th_max_kwh"], cat="Continuous"),
    }

    # STAGE 2 — recourse dispatch, replicated per scenario w.
    def vd(name, **kwargs):
        return {(w, y, m, d, t): pulp.LpVariable(f"{name}_{w}_{y}_{m[:3]}_{d}_{t}", **kwargs)
                for w in W for y in Y for (m, d) in S for t in T}

    V["e_im"]   = vd("e_im",   lowBound=0)
    V["e_ex"]   = vd("e_ex",   lowBound=0)
    V["e_chg"]  = vd("e_chg",  lowBound=0)
    V["e_disc"] = vd("e_disc", lowBound=0)
    V["e_lvl"]  = vd("e_lvl",  lowBound=0)

    # Heat dispatch: production + thermal-store charge/discharge/level (all techs)
    V["heat_prod"]  = vd("heat_prod",  lowBound=0)
    V["heat_chg"]   = vd("heat_chg",   lowBound=0)
    V["heat_dis"]   = vd("heat_dis",   lowBound=0)
    V["e_th_lvl"]   = vd("e_th_lvl",   lowBound=0)
    # Fuel: heat pumps draw electricity; boiler consumes gas
    if data["is_hp"]:
        V["elec_heat"] = vd("elec_heat", lowBound=0)
    else:
        V["gas_im"]    = vd("gas_im",    lowBound=0)

    if use_binary_mutex:
        V["beta_chg"]  = vd("b_chg",  cat="Binary")
        V["beta_disc"] = vd("b_disc", cat="Binary")
        V["beta_im"]   = vd("b_im",   cat="Binary")
        V["beta_ex"]   = vd("b_ex",   cat="Binary")

    return V


def _add_constraints(prob, V: dict, data: dict, use_binary_mutex: bool) -> None:
    bat       = TECH_COSTS["battery"]
    eta_chg   = bat["chg_eff"]
    eta_disc  = bat["disc_eff"]
    SOC_min   = bat["soc_min"]
    SOC_max   = bat["soc_max"]
    grid_im_max = data["import_limit_kw"] * T_RES_H   # kWh per half-hour (DNO demand headroom)
    grid_ex_max = data["export_limit_kw"] * T_RES_H   # kWh per half-hour (DNO generation headroom)

    n_pv,   e_batt, o_batt        = V["n_pv"], V["e_batt"], V["o_batt"]
    e_im,   e_ex                  = V["e_im"], V["e_ex"]
    e_chg,  e_disc, e_lvl         = V["e_chg"], V["e_disc"], V["e_lvl"]
    demand_kwh, pv_per_mod        = data["demand_kwh"], data["pv_per_mod"]
    elec_growth, heat_growth      = data["elec_growth"], data["heat_growth"]
    soh_batt_y                    = data["soh_batt_y"]

    # Heat side
    ts        = THERMAL_STORE
    eta_tchg  = ts["chg_eff"]
    eta_tdis  = ts["disc_eff"]
    th_loss   = ts["standing_loss_per_step"]
    TH_min    = ts["soc_min"]
    TH_max    = ts["soc_max"]
    is_hp     = data["is_hp"]
    cop       = data["cop"]
    heat_kwh  = data["heat_kwh"]
    q_heat_cap, e_th         = V["q_heat_cap"], V["e_th"]
    heat_prod, heat_chg      = V["heat_prod"], V["heat_chg"]
    heat_dis,  e_th_lvl      = V["heat_dis"],  V["e_th_lvl"]
    elec_heat = V.get("elec_heat")    # heat-pump techs only
    gas_im    = V.get("gas_im")       # gas boiler only

    # PV / thermal-store sizing caps are enforced by the variable upper bounds set in _declare_variables.

    # PCS sizing from C-rate
    prob += o_batt <= bat["c_rate_chg"]  * e_batt, "pcs_chg_c_rate"
    prob += o_batt <= bat["c_rate_disc"] * e_batt, "pcs_disc_c_rate"

    M_disp = (max(BATT_MAX_KW, data["import_limit_kw"], data["export_limit_kw"]) * T_RES_H
              if use_binary_mutex else None)

    for y in data["Y"]:
        soc_lo_y = SOC_min * e_batt * soh_batt_y[y]
        soc_hi_y = SOC_max * e_batt * soh_batt_y[y]

        for w in data["W"]:                                   # stage-2 recourse: one dispatch per scenario
          for (m, d) in data["S"]:
            tag = f"{w}_{y}_{m[:3]}_{d}"
            for t in data["T"]:
                dem_t = float(demand_kwh[(m, d)][t]) * elec_growth[y]   # non-heat elec grows (DESNZ)
                pv_t  = pv_per_mod[(y, m)][t]                           # PV is not uncertain 

                # Electricity balance: PV + batt discharge + import == non-heat demand + heat-pump electricity + batt charge + export
                elec_heat_t = elec_heat[(w,y,m,d,t)] if is_hp else 0
                prob += (n_pv * pv_t + e_disc[(w,y,m,d,t)] + e_im[(w,y,m,d,t)]
                         == dem_t + elec_heat_t + e_chg[(w,y,m,d,t)] + e_ex[(w,y,m,d,t)]), f"bal_{tag}_{t}"

                # SOC dynamics — day starts at SOC_min × residual capacity (daily-cycling assumption)
                if t == 0:
                    prob += (e_lvl[(w,y,m,d,t)]
                             == soc_lo_y
                              + e_chg[(w,y,m,d,t)]  * eta_chg
                              - e_disc[(w,y,m,d,t)] / eta_disc), f"soc_start_{tag}"
                else:
                    prob += (e_lvl[(w,y,m,d,t)]
                             == e_lvl[(w,y,m,d,t-1)]
                              + e_chg[(w,y,m,d,t)]  * eta_chg
                              - e_disc[(w,y,m,d,t)] / eta_disc), f"soc_{tag}_{t}"

                # SOC bounds
                prob += e_lvl[(w,y,m,d,t)] >= soc_lo_y, f"soc_lo_{tag}_{t}"
                prob += e_lvl[(w,y,m,d,t)] <= soc_hi_y, f"soc_hi_{tag}_{t}"

                # Charge/discharge rate limits
                prob += e_chg[(w,y,m,d,t)]  <= o_batt * T_RES_H, f"chg_rate_{tag}_{t}"
                prob += e_disc[(w,y,m,d,t)] <= o_batt * T_RES_H, f"disc_rate_{tag}_{t}"

                # Grid import/export limits (split: demand-headroom vs generation-headroom)
                prob += e_im[(w,y,m,d,t)] <= grid_im_max, f"im_lim_{tag}_{t}"
                prob += e_ex[(w,y,m,d,t)] <= grid_ex_max, f"ex_lim_{tag}_{t}"
                # Export must be GENERATED, not resold. Without this the balance alone permits
                # importing and exporting in the same slot: under the emissions objective the two
                # carry identical coefficients and cancel exactly, so the round-trip is free and the
                # LP is degenerate along it (simplex then returns an arbitrary vertex, pinned at the
                # grid limits); and whenever the export price exceeds the import price it becomes
                # strictly profitable, which is grid-to-grid arbitrage rather than a PV business
                # case. Capping export at own generation + storage discharge removes both, matches
                # SEG (which pays for exported generation only), and keeps the model an LP — the
                # binary import/export mutex would do the same at MILP cost.
                prob += (e_ex[(w,y,m,d,t)] <= n_pv * pv_t + e_disc[(w,y,m,d,t)]), f"ex_gen_{tag}_{t}"

                # Heat side
                heat_dem_t = float(heat_kwh[(m, d)][t]) * heat_growth[(y, m)]   # heat falls as climate warms
                # Heat balance: production + store discharge == demand + store charge
                prob += (heat_prod[(w,y,m,d,t)] + heat_dis[(w,y,m,d,t)]
                         == heat_dem_t + heat_chg[(w,y,m,d,t)]), f"heat_bal_{tag}_{t}"
                # Thermal store SOC — each rep day starts empty
                if t == 0:
                    prob += (e_th_lvl[(w,y,m,d,t)]
                             == heat_chg[(w,y,m,d,t)] * eta_tchg
                              - heat_dis[(w,y,m,d,t)] / eta_tdis), f"th_start_{tag}"
                else:
                    prob += (e_th_lvl[(w,y,m,d,t)]
                             == e_th_lvl[(w,y,m,d,t-1)] * (1.0 - th_loss)
                              + heat_chg[(w,y,m,d,t)] * eta_tchg
                              - heat_dis[(w,y,m,d,t)] / eta_tdis), f"th_soc_{tag}_{t}"
                # Store level bounds + charge/discharge rate
                prob += e_th_lvl[(w,y,m,d,t)] >= TH_min * e_th, f"th_lo_{tag}_{t}"
                prob += e_th_lvl[(w,y,m,d,t)] <= TH_max * e_th, f"th_hi_{tag}_{t}"
                prob += heat_chg[(w,y,m,d,t)] <= ts["c_rate"] * e_th * T_RES_H, f"th_chg_rate_{tag}_{t}"
                prob += heat_dis[(w,y,m,d,t)] <= ts["c_rate"] * e_th * T_RES_H, f"th_dis_rate_{tag}_{t}"
                # Heating-plant capacity limit
                prob += heat_prod[(w,y,m,d,t)] <= q_heat_cap * T_RES_H, f"heat_cap_{tag}_{t}"
                # Fuel conversion: heat pump → electricity (COP); boiler → gas (η_boiler)
                if is_hp:
                    prob += (elec_heat[(w,y,m,d,t)]
                             == heat_prod[(w,y,m,d,t)] / float(cop[(y, m)][t])), f"hp_elec_{tag}_{t}"
                else:
                    prob += (gas_im[(w,y,m,d,t)]
                             == heat_prod[(w,y,m,d,t)] / dm.ETA_BOILER), f"boiler_gas_{tag}_{t}"

                # Optional strict binaries (eq. 1.31, 1.54)
                if use_binary_mutex:
                    prob += e_chg[(w,y,m,d,t)]  <= M_disp * V["beta_chg"][(w,y,m,d,t)]
                    prob += e_disc[(w,y,m,d,t)] <= M_disp * V["beta_disc"][(w,y,m,d,t)]
                    prob += V["beta_chg"][(w,y,m,d,t)] + V["beta_disc"][(w,y,m,d,t)] <= 1
                    prob += e_im[(w,y,m,d,t)] <= M_disp * V["beta_im"][(w,y,m,d,t)]
                    prob += e_ex[(w,y,m,d,t)] <= M_disp * V["beta_ex"][(w,y,m,d,t)]
                    prob += V["beta_im"][(w,y,m,d,t)] + V["beta_ex"][(w,y,m,d,t)] <= 1

            # Daily SOC closure (per scenario): end-of-day level returns to the same floor it started at. 
            # Loss-weighted via e_lvl (not raw chg==disc) — since round-trip efficiency < 100%.
            prob += e_lvl[(w, y, m, d, HH_PER_DAY - 1)] == soc_lo_y, f"soc_closure_{tag}"


def _build_objective(prob, V: dict, data: dict,
                     objective: str = "cost", emissions_cap: float = None) -> None:
    # OF1 = 15-yr NPV cost; OF2 = lifetime operational carbon (kgCO2e).
    pv_c              = TECH_COSTS["pv"]
    bat               = TECH_COSTS["battery"]
    heat_c            = HEAT_COSTS[data["heating"]]
    ts                = THERMAL_STORE
    pv_kwp_per_module = pv_c["module_kwp"]
    is_hp             = data["is_hp"]

    def df_at(yr):   # discount factor, 0 if the replacement falls on/after the horizon end
        return data["df_y"][yr] if yr < len(data["df_y"]) else 0.0

    n_pv,   e_batt, o_batt = V["n_pv"], V["e_batt"], V["o_batt"]
    q_heat_cap, e_th       = V["q_heat_cap"], V["e_th"]

    # Equipment + total capex. 
    c_pv_modules = n_pv * pv_kwp_per_module * pv_c["capex_per_kwp"]
    c_batt_capex = (e_batt * bat["energy_capex_per_kwh"]   # BoS + commissioning already in the capex rates
                    + o_batt * bat["power_capex_per_kw"])
    c_heat_equip = q_heat_cap * heat_c["capex_per_kwth"]   # full equip+install of the chosen heating system
    # New-build: full heating capex (installation included), replaced at end of life. 
    # Retrofit:  gas-boiler scenario no capex / replacement; heat-pump scenario pays capex + one-off boiler decommissioning cost.
    if NEW_BUILD:
        c_heat_capex         = c_heat_equip
        c_heat_replace_basis = c_heat_equip
    elif is_hp:                                            # retrofit, heat pump
        # Decommissioning the old gas boiler = a one-off FIXED gas-service disconnection charge 
        # (per site, independent of boiler size, inclusive of strip-out/disposal).
        decommission_fixed   = HEAT_COSTS["Gas Boiler"].get("decommission_fixed", 0.0)
        c_heat_capex         = c_heat_equip + decommission_fixed
        c_heat_replace_basis = c_heat_equip               # the HP unit is replaced; decommissioning is one-off
    else:                                                 # retrofit, existing gas boiler 
        c_heat_capex         = 0.0
        c_heat_replace_basis = 0.0
    c_th_equip   = e_th * ts["energy_capex_per_kwh"]
    c_capex_y0   = c_pv_modules + c_batt_capex + c_heat_capex + c_th_equip

    W, weights = data["W"], data["weights"]
    ef_elec_y, ef_gas, cval_d = data["ef_elec_y"], data["ef_gas"], data["cval_d"]

    # STAGE 2 (recourse). Electricity opex uses that scenario's discounted import-price path; export and gas are deterministic. 
    # Emissions/monetised-carbon are price-independent but use each scenario's own dispatch, so they are kept per w too.
    c_opex_by_w, c_gas_by_w, emis_by_w, carbon_npv_by_w = {}, {}, {}, {}
    for w in W:
        # Import price coefficient is per-(year) flat by default, or per-(year, month, day, slot) when
        # the wholesale+DUoS build-up is active (intra-day ToU + regional shape). Export & gas unchanged.
        _slot_im = data["im_slot_dependent"]
        c_opex_by_w[w] = pulp.lpSum(
            (V["e_im"][(w,y,m,d,t)] * (data["im_d"][w][(y,m,d,t)] if _slot_im else data["im_d"][w][y])
             - V["e_ex"][(w,y,m,d,t)] * data["ex_d"][y])
            * N_DAYS_OF[(m, d)]
            for y in data["Y"] for (m, d) in data["S"] for t in data["T"]
        )
        c_gas_by_w[w] = 0 if is_hp else pulp.lpSum(
            V["gas_im"][(w,y,m,d,t)] * data["gas_d"][y] * N_DAYS_OF[(m, d)]
            for y in data["Y"] for (m, d) in data["S"] for t in data["T"]
        )
        # OF2 — lifetime operational emissions (kgCO2e), undiscounted. 
        # Grid import charged at the year's grid factor; exported PV at the long-run-marginal factor; gas at the constant factor.
        emis_elec_w = pulp.lpSum(
            (V["e_im"][(w,y,m,d,t)] - V["e_ex"][(w,y,m,d,t)]) * ef_elec_y[y] * N_DAYS_OF[(m, d)]
            for y in data["Y"] for (m, d) in data["S"] for t in data["T"]
        )
        emis_gas_w = 0 if is_hp else pulp.lpSum(
            V["gas_im"][(w,y,m,d,t)] * ef_gas * N_DAYS_OF[(m, d)]
            for y in data["Y"] for (m, d) in data["S"] for t in data["T"]
        )
        emis_by_w[w] = emis_elec_w + emis_gas_w
        carbon_npv_by_w[w] = pulp.lpSum(
            ((V["e_im"][(w,y,m,d,t)] - V["e_ex"][(w,y,m,d,t)]) * ef_elec_y[y]
             + (V["gas_im"][(w,y,m,d,t)] * ef_gas if not is_hp else 0))
            / 1000.0 * cval_d[y] * N_DAYS_OF[(m, d)]
            for y in data["Y"] for (m, d) in data["S"] for t in data["T"]
        )

    # STAGE 1 (deterministic) — capex, maintenance + insurance, and one-off replacements depend only on the shared sizing.
    # Annual maintenance + insurance, escalated by general inflation, discounted.
    c_maint_npv = pulp.lpSum(
        (n_pv * pv_kwp_per_module / 1000.0 * (pv_c["maint_per_mw_per_yr"]
                                              + pv_c["insurance_per_mw_per_yr"])
         + c_batt_capex * bat["maint_pct_capex"]
         + q_heat_cap   * heat_c["maint_per_kwth_per_yr"]   # £/kW_th-yr × installed kW_th
         + c_th_equip   * ts["maint_pct_capex"])
        * data["infl_d"][y]
        for y in data["Y"]
    )
    # One-off replacements (PV inverter; battery; heat-pump/boiler unit; thermal store).
    c_replace_npv = (
        c_batt_capex   * df_at(bat["replace_year"])
        + c_pv_modules * pv_c["inv_cost_share"]  * df_at(pv_c["inv_replace_year"])
        + c_heat_replace_basis * heat_c["replace_cost_share"] * df_at(heat_c["replace_year"])
        + c_th_equip   * df_at(ts["replace_year"])
    )
    c_stage1 = c_capex_y0 + c_maint_npv + c_replace_npv

    # Expected (probability-weighted) recourse cost / emissions over the scenario set.
    exp_opex_gas   = pulp.lpSum(weights[w] * (c_opex_by_w[w] + c_gas_by_w[w]) for w in W)
    exp_cost_expr  = c_stage1 + exp_opex_gas
    exp_emissions  = pulp.lpSum(weights[w] * emis_by_w[w] for w in W)

    # Objective selection
    if objective == "emissions":
        prob += exp_emissions, "exp_emissions"
    else:
        prob += exp_cost_expr, "exp_total_cost_npv"
    if emissions_cap is not None:
        prob += exp_emissions <= emissions_cap, "emissions_cap"   # epsilon-constraint on expected carbon

    # Stash for post-solve extraction
    V["objective"]        = objective
    V["c_stage1"]         = c_stage1
    V["exp_cost_expr"]    = exp_cost_expr
    V["exp_emissions"]    = exp_emissions
    V["c_opex_by_w"]      = c_opex_by_w
    V["c_gas_by_w"]       = c_gas_by_w
    V["emis_by_w"]        = emis_by_w
    V["carbon_npv_by_w"]  = carbon_npv_by_w
    V["c_pv_modules"]  = c_pv_modules
    V["c_batt_capex"]  = c_batt_capex
    V["c_heat_capex"]  = c_heat_capex
    V["c_heat_equip"]  = c_heat_equip
    V["c_heat_replace_basis"] = c_heat_replace_basis   
    V["c_th_equip"]    = c_th_equip
    V["c_capex_y0"]    = c_capex_y0
    V["c_maint_npv"]   = c_maint_npv
    V["c_replace_npv"] = c_replace_npv


def effective_import_limit_kw(district: str, activity: str, *,
                              horizon_years: int = HORIZON_YEARS,
                              import_limit_override_kw: float = None) -> float:
    # The grid-import ceiling the model actually applies to this cell: the district's 
    # DNO ceiling, floored at the building's own final-horizon-year non-heat peak.
    _ensure_dm_initialized()
    glim     = select_grid_limit(district)
    base_kw  = glim["import_kw"] if import_limit_override_kw is None else import_limit_override_kw
    peak_kwh = max(float(arr.max()) for arr in building_demand_kwh(activity, district).values())
    floor_kw = peak_kwh / T_RES_H * max(elec_growth_factors(horizon_years))
    return max(base_kw, floor_kw)


def build_milp(district: str, activity: str, heating: str, *,
               horizon_years: int = HORIZON_YEARS,
               use_binary_mutex: bool = False, # Simultaneous import/export is blocked by the ex_gen_* export-at-own-generation cap in _add_constraints, so the binary mutex is not required and the model stays an LP
               objective: str = "cost",
               emissions_cap: float = None,
               scenarios: list = None,
               demand_multiplier: float = 1.0,
               import_limit_override_kw: float = None):
    scenarios = _scenarios_for(objective, scenarios)
    prob = pulp.LpProblem(f"opt_{activity}_{district}_{heating}".replace(" ", "_").replace(":", ""),
                          pulp.LpMinimize)
    data = _prepare_data(district, activity, heating, horizon_years, scenarios,
                         demand_multiplier=demand_multiplier,
                         import_limit_override_kw=import_limit_override_kw)
    V    = _declare_variables(data, use_binary_mutex)
    _add_constraints(prob, V, data, use_binary_mutex)
    _build_objective(prob, V, data, objective=objective, emissions_cap=emissions_cap)
    # Pass demand/PV/heat arrays + scenario metadata through V so the extractor can compute
    # annual totals + BAU. All of these already live in `data` under the same key.
    V.update({k: data[k] for k in _V_FROM_DATA})
    V["district"] = district
    return prob, V


# Keys forwarded from the prepared-data dict into V for the results extractor.
_V_FROM_DATA = (
    "demand_kwh", "pv_per_mod", "heat_kwh", "elec_growth", "heat_growth", "is_hp",
    "heating", "peak_kwth", "land_limit_kwth", "im_base", "elec_band", "gas_base", "gas_band",
    "import_limit_kw", "export_limit_kw", "ef_elec_y", "ef_gas", "scenarios", "W",
    "weights", "im_base_central", "im_slot_dependent", "im_slot_central",
)


# 6 - SOLVER & RESULTS
def _make_solver(solver_msg: bool, time_limit_s: int, threads: int = None):
    base  = dict(msg=solver_msg, timeLimit=time_limit_s, gapRel=MIP_GAP_REL)
    tuned = dict(base)
    if threads is not None:
        tuned["threads"] = threads
    if LP_METHOD:
        tuned["solver"] = LP_METHOD            # HiGHS option name for the LP method
    for kwargs in (tuned, base):               
        try:
            s = pulp.HiGHS(**kwargs)
            if s.available():
                return s
        except Exception:
            pass
    return pulp.PULP_CBC_CMD(**base)


def compute_payback(incr_capex: float, bau_annual: list, scen_annual: list, horizon_years: int) -> float:
    # Greenfield payback on expected cash flows: cumulative (BAU − scenario annual cost) recovered
    # against upfront incremental capex (scenario − BAU gas-boiler capex). 
    # Both annual streams are probability-weighted over the scenario set and undiscounted.
    if incr_capex <= 0:
        return 0.0    # already cheaper than BAU at year 0, so payback is immediate
    cum  = -incr_capex
    for y in range(horizon_years):
        cf_y = bau_annual[y] - scen_annual[y]
        prev = cum
        cum += cf_y
        if cum >= 0:
            return round(y + (-prev / cf_y), 2) if cf_y > 0 else float(y + 1)
    return float("nan")    # not paid back within horizon


def _bau_base(demand_kwh: dict, heat_kwh: dict, peak_kwth: float, horizon_years: int,
              elec_growth: list, heat_growth: dict, district: str = None,
              slot_prices: bool = False) -> dict:
    # Scenario-invariant BAU baseline: gas boiler sized to peak heat, all electricity from the grid, all heat from gas. 
    # Capex, emissions, non-import annual cost (gas + maintenance + replacement), annual import kWh are all price-independent, 
    # so this is computed once per cell and the cheap per-scenario import-price overlay is applied by _bau_assemble.
    gb   = HEAT_COSTS["Gas Boiler"]
    infl = TECH_COSTS["general_inflation"]   # insurance is now PV-only, so BAU has none
    dem_annual  = sum(demand_kwh[(m, d)].sum() * N_DAYS_OF[(m, d)] for (m, d) in S_KEYS)   # non-heat elec, year 0
    # Year-0 useful heat per month (summed over WD/WE) so the monthly heat-growth factor can be applied per year.
    heat_by_month = {}
    for (m, d) in S_KEYS:
        heat_by_month[m] = heat_by_month.get(m, 0.0) + heat_kwh[(m, d)].sum() * N_DAYS_OF[(m, d)]
    gas_annual  = sum(heat_by_month.values()) / dm.ETA_BOILER                              # gas burned, year 0
    # Band on the BAU's OWN consumption — non-heat electricity only, since the counterfactual burns
    # gas for heat. Deliberately NOT the optimised design's band: a heat pump lifts site electricity
    # into a larger (cheaper) DESNZ band, and charging the gas-boiler counterfactual that cheaper
    # band credits the intervention's volume discount to the baseline it is measured against.
    bau_elec_band = select_elec_band(dem_annual / 1000.0)
    bau_base     = bau_elec_band["price"]
    bau_gas_base = select_gas_band(gas_annual / 1000.0)["price"]   # BAU is always a gas boiler
    # Per-scenario import price: central year-0 reference (active multiplier divided out) × scenario path.
    base_central = bau_base / TECH_COSTS.get("elec_import_multiplier", 1.0)
    # When the wholesale+DUoS build-up is active, BAU pays the same tariff STRUCTURE as the optimised
    # case (same wholesale shape, same DUoS bands) but keyed to its own size band's residual.
    if slot_prices:
        slot_c = import_price_slots_central(district, bau_elec_band["name"])
        num = sum(demand_kwh[(m, d)][t] * slot_c[(m, d, t)] * N_DAYS_OF[(m, d)]
                  for (m, d) in S_KEYS for t in range(HH_PER_DAY))
        den = sum(demand_kwh[(m, d)].sum() * N_DAYS_OF[(m, d)] for (m, d) in S_KEYS)
        base_central = num / den if den else base_central
    equip = peak_kwth * gb["capex_per_kwth"]   # capex_per_kwth already includes installation
    if NEW_BUILD:
        capex, repl_basis = equip, equip
    else:
        capex, repl_basis = 0.0, 0.0
    # Per-year price-independent streams: import kWh, the non-import annual cost, and operational carbon.
    dem_y, nonimport_y = [], []
    emissions_kg = 0.0   # BAU lifetime operational carbon: all elec from grid + all heat from gas
    for y in range(horizon_years):
        dy        = dem_annual * elec_growth[y]                                            # non-heat elec grows
        gas_y_kwh = sum(heat_by_month[m] * heat_growth[(y, m)] for m in heat_by_month) / dm.ETA_BOILER  # heat falls
        maint     = peak_kwth * gb["maint_per_kwth_per_yr"] * (1.0 + infl) ** y
        repl      = repl_basis * gb["replace_cost_share"] if y == gb["replace_year"] else 0.0
        dem_y.append(dy)
        nonimport_y.append(gas_y_kwh * gas_price(y, bau_gas_base) + maint + repl)
        emissions_kg += dy * elec_emission_factor(y) + gas_y_kwh * GAS_EMISSION_FACTOR
    return {"capex": capex, "dem_annual": dem_annual, "gas_annual": gas_annual,
            "emissions_kg": emissions_kg, "bau_base": bau_base, "base_central": base_central,
            "dem_y": dem_y, "nonimport_y": nonimport_y}


def _bau_assemble(base: dict, horizon_years: int, scenario=None) -> dict:
    # Apply one scenario's import-price path to the invariant BAU base.
    def im_price_y(y):
        return base["base_central"] * float(scenario.path[y]) if scenario is not None \
            else import_price(y, base["bau_base"])
    annual_undisc, npv = [], base["capex"]
    for y in range(horizon_years):
        annual = base["dem_y"][y] * im_price_y(y) + base["nonimport_y"][y]
        annual_undisc.append(annual)
        npv += annual * discount_factor(y)
    return {"capex": base["capex"], "npv": npv, "annual_undisc": annual_undisc,
            "dem_annual": base["dem_annual"], "gas_annual": base["gas_annual"],
            "emissions_kg": base["emissions_kg"]}


def _extract_results(prob, V: dict, status, district: str, activity: str, heating: str,
                     horizon_years: int) -> dict:
    if pulp.LpStatus[status] not in ("Optimal", "Not Solved"):
        # Carry the land ceiling onto the stub row too — otherwise a horizontal loop that is
        # Infeasible *because* the collector doesn't fit is indistinguishable from one that is
        # Infeasible on grid headroom or economics.
        _lim = float(V.get("land_limit_kwth", float("inf")))
        return {"status": pulp.LpStatus[status], "district": district,
                "activity": activity, "heating": heating,
                "land_limit_kwth": (None if not np.isfinite(_lim) else round(_lim, 1))}

    val = pulp.value
    # STAGE-1 sizing decisions (shared across scenarios = robust here-and-now design)
    n_pv_val = int(round(val(V["n_pv"]) or 0))
    n_pv_v   = float(val(V["n_pv"]) or 0.0)
    e_batt   = float(val(V["e_batt"]) or 0.0)
    o_batt   = float(val(V["o_batt"]) or 0.0)
    pv_kwp   = n_pv_val * TECH_COSTS["pv"]["module_kwp"]
    q_heat   = float(val(V["q_heat_cap"]) or 0.0)
    e_th     = float(val(V["e_th"]) or 0.0)
    _land_lim = float(V.get("land_limit_kwth", float("inf")))
    is_hp    = V["is_hp"]

    scenarios = V["scenarios"]; W = list(V["W"]); weights = V["weights"]
    H = horizon_years
    ww = np.array(weights, dtype=float)
    def _wmean(d):  return float(np.average([d[w] for w in W], weights=ww))
    def _wstd(d, mean=None):
        arr = np.array([d[w] for w in W], dtype=float)
        mu  = _wmean(d) if mean is None else mean
        return float(np.sqrt(np.average((arr - mu) ** 2, weights=ww)))

    def _slot_sum(name, w, y, m, d):   # Σ over the 48 half-hour slots of a solved dispatch variable
        return sum(float(val(V[name][(w, y, m, d, t)]) or 0.0) for t in range(HH_PER_DAY))

    # Deterministic per-year maintenance + replacement (depend only on stage-1 sizing - same every scenario)
    pv_c, bat = TECH_COSTS["pv"], TECH_COSTS["battery"]
    heat_c    = HEAT_COSTS[V["heating"]]; ts = THERMAL_STORE
    infl      = TECH_COSTS["general_inflation"]
    pv_mw_inst = n_pv_v * pv_c["module_kwp"] / 1000.0
    c_batt_v  = float(val(V["c_batt_capex"]) or 0.0)
    c_heat_eq = float(val(V["c_heat_replace_basis"]) or 0.0)   # retrofit-aware boiler/HP replacement basis
    c_th_eq   = float(val(V["c_th_equip"]) or 0.0)
    c_pv_mod  = float(val(V["c_pv_modules"]) or 0.0)
    maint_year, replace_year = [], []
    for y in range(H):
        maint_year.append((pv_mw_inst * (pv_c["maint_per_mw_per_yr"] + pv_c["insurance_per_mw_per_yr"])
                           + c_batt_v * bat["maint_pct_capex"]
                           + q_heat * heat_c["maint_per_kwth_per_yr"] + c_th_eq * ts["maint_pct_capex"]
                           ) * (1.0 + infl) ** y)
        r = 0.0
        if y == bat["replace_year"]:      r += c_batt_v
        if y == pv_c["inv_replace_year"]: r += c_pv_mod * pv_c["inv_cost_share"]
        if y == heat_c["replace_year"]:   r += c_heat_eq * heat_c["replace_cost_share"]
        if y == ts["replace_year"]:       r += c_th_eq
        replace_year.append(r)

    # Demand / heat / PV year-0 totals (shared across scenarios — not price-dependent)
    Y0 = 0
    total_demand = total_heat = total_pv = 0.0
    for (m, d) in S_KEYS:
        nd = N_DAYS_OF[(m, d)]
        total_demand += V["demand_kwh"][(m, d)].sum() * nd
        total_heat   += V["heat_kwh"][(m, d)].sum()   * nd
        total_pv     += n_pv_v * V["pv_per_mod"][(Y0, m)].sum() * nd

    # STAGE-2 per-scenario dispatch: year-0 energy totals + per-year undiscounted energy cost
    y0 = {w: dict(imp=0.0, exp=0.0, chg=0.0, disc=0.0, gas=0.0, eh=0.0) for w in W}
    scen_energy_year = {w: [0.0] * H for w in W}
    im_base_central, gas_base = V["im_base_central"], V["gas_base"]
    im_slot_dependent, im_slot_central = V.get("im_slot_dependent"), V.get("im_slot_central")
    for w in W:
        scen = scenarios[w]
        for (m, d) in S_KEYS:
            nd = N_DAYS_OF[(m, d)]
            for y in range(H):
                im_slots = [float(val(V["e_im"][(w, y, m, d, t)]) or 0.0) for t in range(HH_PER_DAY)]
                imp = sum(im_slots)
                exp = _slot_sum("e_ex", w, y, m, d)
                gas = 0.0 if is_hp else _slot_sum("gas_im", w, y, m, d)
                # Import cost: per-slot when the build-up is active (ToU), else flat band price
                if im_slot_dependent:
                    im_cost = float(scen.path[y]) * sum(v * im_slot_central[(m, d, t)]
                                                        for t, v in enumerate(im_slots))
                else:
                    im_cost = imp * im_base_central * float(scen.path[y])
                gp   = 0.0 if is_hp else gas_price(y, gas_base)
                scen_energy_year[w][y] += (im_cost - exp * export_price(y) + gas * gp) * nd
                if y == Y0:
                    y0[w]["imp"] += imp * nd; y0[w]["exp"] += exp * nd; y0[w]["gas"] += gas * nd
                    y0[w]["chg"]  += _slot_sum("e_chg",  w, y, m, d) * nd
                    y0[w]["disc"] += _slot_sum("e_disc", w, y, m, d) * nd
                    if is_hp:
                        y0[w]["eh"] += _slot_sum("elec_heat", w, y, m, d) * nd

    total_import    = _wmean({w: y0[w]["imp"]  for w in W})
    total_export    = _wmean({w: y0[w]["exp"]  for w in W})
    total_chg       = _wmean({w: y0[w]["chg"]  for w in W})
    total_disc      = _wmean({w: y0[w]["disc"] for w in W})
    total_gas       = _wmean({w: y0[w]["gas"]  for w in W})
    total_elec_heat = _wmean({w: y0[w]["eh"]   for w in W})
    self_consumption      = max(0.0, total_pv - total_export)
    self_consumption_rate = (self_consumption / total_pv) if total_pv > 0 else 0.0
    self_sufficiency_rate = (self_consumption / total_demand) if total_demand > 0 else 0.0

    # Valid design must serve the building's positive electricity and heat demand.
    served_elec = total_import + total_pv + total_disc
    served_heat = total_elec_heat if is_hp else total_gas
    if (total_demand > 0 and served_elec <= 1e-6) or (total_heat > 0 and served_heat <= 1e-6):
        return {"status": "No incumbent", "district": district,
                "activity": activity, "heating": heating}

    # Post-solve feasibility guard. 
    TOL = 1.02   # 2% slack 
    heat_cap_ok   = total_heat   <= q_heat * 8760.0 * TOL + 1e-6
    import_cap_ok = total_import <= V["import_limit_kw"] * 8760.0 * TOL + 1e-6
    if not (heat_cap_ok and import_cap_ok):
        return {"status": "Infeasible (solver tolerance)", "district": district,
                "activity": activity, "heating": heating,
                "_debug_total_heat": total_heat, "_debug_q_heat_cap": q_heat,
                "_debug_total_import": total_import, "_debug_import_limit_kw": V["import_limit_kw"]}

    # Stage-1 deterministic cost components
    c_capex      = float(val(V["c_capex_y0"]))
    c_maint      = float(val(V["c_maint_npv"]))
    c_replace    = float(val(V["c_replace_npv"]))
    c_heat_capex = float(val(V["c_heat_capex"]))
    c_stage1     = float(val(V["c_stage1"]))

    # Per-scenario recourse cost / emissions / BAU, then probability-weighted aggregates.
    bau_base = _bau_base(V["demand_kwh"], V["heat_kwh"], V["peak_kwth"], H,
                         V["elec_growth"], V["heat_growth"], district=district,
                         slot_prices=bool(V.get("im_slot_dependent")))
    scen_cost, scen_opex, scen_gas = {}, {}, {}
    scen_emis, scen_cnpv, scen_bau_npv, scen_sav = {}, {}, {}, {}
    bau_by_w = {}
    for w in W:
        opx = float(val(V["c_opex_by_w"][w]))
        gpx = 0.0 if is_hp else float(val(V["c_gas_by_w"][w]))
        scen_opex[w] = opx; scen_gas[w] = gpx
        scen_cost[w] = c_stage1 + opx + gpx
        scen_emis[w] = float(val(V["emis_by_w"][w]))
        scen_cnpv[w] = float(val(V["carbon_npv_by_w"][w]))
        bw = _bau_assemble(bau_base, H, scenario=scenarios[w])
        bau_by_w[w] = bw
        scen_bau_npv[w] = bw["npv"]
        scen_sav[w]     = bw["npv"] - scen_cost[w]

    exp_cost    = _wmean(scen_cost)
    exp_opex    = _wmean(scen_opex)
    exp_gas     = _wmean(scen_gas)
    exp_emis    = _wmean(scen_emis)
    exp_cnpv    = _wmean(scen_cnpv)
    exp_bau_npv = _wmean(scen_bau_npv)
    exp_sav     = _wmean(scen_sav)
    cost_min, cost_max = min(scen_cost.values()), max(scen_cost.values())
    sav_min,  sav_max  = min(scen_sav.values()),  max(scen_sav.values())

    roi = (exp_sav / c_capex) if c_capex > 0 else float("nan")
    # Payback on expected (probability-weighted) cash flows
    scen_annual = [_wmean({w: scen_energy_year[w][y] for w in W}) + maint_year[y] + replace_year[y]
                   for y in range(H)]
    bau_annual  = [_wmean({w: bau_by_w[w]["annual_undisc"][y] for w in W}) for y in range(H)]
    bau_capex   = bau_by_w[W[0]]["capex"]
    payback     = compute_payback(c_capex - bau_capex, bau_annual, scen_annual, H)

    # Emissions: BAU carbon is price-independent (same every scenario); savings vs expected design carbon.
    bau_emissions_kg = bau_by_w[W[0]]["emissions_kg"]
    emis_saving_kg   = bau_emissions_kg - exp_emis
    emis_saving_fraction  = (emis_saving_kg / bau_emissions_kg) if bau_emissions_kg > 0 else float("nan")
    mac_gbp_per_t    = ((exp_cost - exp_bau_npv) / (emis_saving_kg / 1000.0)
                        if abs(emis_saving_kg) > 1e-6 else float("nan"))

    return {
        "status":                pulp.LpStatus[status],
        "district":              district,
        "activity":              activity,
        "heating":               heating,
        "n_scenarios":           len(W),
        "elec_import_band":      V.get("elec_band"),
        "elec_import_price_central_GBP": round(im_base_central, 4),
        "gas_import_band":       None if V.get("is_hp") else V.get("gas_band"),
        "gas_import_price_GBP":  None if V.get("is_hp") else round(V.get("gas_base", float("nan")), 4),
        "grid_import_limit_kw":  round(V.get("import_limit_kw", float("nan")), 1),
        "grid_export_limit_kw":  round(V.get("export_limit_kw", float("nan")), 1),
        "n_pv":                  n_pv_val,
        "pv_kwp":                round(pv_kwp, 2),
        "e_batt_kwh":            round(e_batt, 1),
        "o_batt_kw":             round(o_batt, 1),
        "q_heat_cap_kwth":       round(q_heat, 1),
        # Horizontal-ground-loop land constraint. inf/None for every other heating type, so a blank
        # here means "surface area was never a limit", not "not checked".
        "land_limit_kwth":       (None if not np.isfinite(_land_lim) else round(_land_lim, 1)),
        "land_limit_binds":      (bool(np.isfinite(_land_lim) and q_heat >= _land_lim - 1e-6)),
        "e_th_kwh":              round(e_th, 1),
        "annual_demand_kwh":     round(total_demand, 0),
        "annual_heat_demand_kwh":round(total_heat, 0),
        "annual_pv_gen_kwh":     round(total_pv, 0),
        "annual_import_kwh":     round(total_import, 0),     # expected over scenarios (year 0)
        "annual_export_kwh":     round(total_export, 0),
        "annual_gas_kwh":        round(total_gas, 0),
        "annual_elec_heat_kwh":  round(total_elec_heat, 0),
        "annual_batt_chg_kwh":   round(total_chg, 0),
        "annual_batt_disc_kwh":  round(total_disc, 0),
        "self_consumption_rate": round(self_consumption_rate, 3),
        "self_sufficiency_rate": round(self_sufficiency_rate, 3),
        "capex_GBP":             round(c_capex, 0),
        "heat_capex_GBP":        round(c_heat_capex, 0),
        "opex_npv_GBP":          round(exp_opex, 0),         # expected over scenarios
        "gas_npv_GBP":           round(exp_gas, 0),
        "maint_npv_GBP":         round(c_maint, 0),
        "replace_npv_GBP":       round(c_replace, 0),
        # Headline figures = expected (probability-weighted) over the import-price scenario set
        "total_cost_npv_GBP":    round(exp_cost, 0),
        "bau_cost_npv_GBP":      round(exp_bau_npv, 0),
        "npv_savings_GBP":       round(exp_sav, 0),
        # Robustness spread of the single robust design across the scenario set
        "cost_npv_min_GBP":      round(cost_min, 0),
        "cost_npv_max_GBP":      round(cost_max, 0),
        "cost_npv_std_GBP":      round(_wstd(scen_cost, mean=exp_cost), 0),
        "npv_savings_min_GBP":   round(sav_min, 0),          # worst-case scenario savings
        "npv_savings_max_GBP":   round(sav_max, 0),          # best-case scenario savings
        "roi":                   round(roi, 3),
        "payback_years":         payback,
        "objective":             V.get("objective", "cost"),
        "lifetime_emissions_tco2e":     round(exp_emis / 1000.0, 1),
        "bau_emissions_tco2e":          round(bau_emissions_kg / 1000.0, 1),
        "emissions_saving_tco2e":       round(emis_saving_kg / 1000.0, 1),
        "emissions_saving_fraction":         round(emis_saving_fraction, 3),
        "carbon_value_npv_GBP":         round(exp_cnpv, 0),
        "mac_GBP_per_tco2e":            round(mac_gbp_per_t, 1),
    }


def solve_scenario(district: str, activity: str, heating: str = "Gas Boiler", *,
                   horizon_years: int = HORIZON_YEARS,
                   use_binary_mutex: bool = False,
                   objective: str = "cost",
                   emissions_cap: float = None,
                   scenarios: list = None,
                   solver=None,
                   solver_msg: bool = False,
                   time_limit_s: int = DEFAULT_TIME_LIMIT_S,
                   threads: int = None,
                   demand_multiplier: float = 1.0,
                   import_limit_override_kw: float = None) -> dict:
    # Build, solve, and extract metrics for one (district, activity, heating) TSSP instance.
    _ensure_dm_initialized()
    # Short-circuit: a horizontal loop with zero available soft ground cannot produce any heat, so the
    # LP is trivially infeasible. Skipping the build saves ~35 s/cell. The verdict is the same
    # "Infeasible" a solver-proven cell gets; land_limit_kwth on the row carries the cause.
    _land_lim = land_limit_kwth(activity, heating)
    if _land_lim <= 0.0:
        return {"status": "Infeasible", "district": district, "activity": activity,
                "heating": heating, "land_limit_kwth": 0.0}
    prob, V = build_milp(district, activity, heating,
                         horizon_years=horizon_years,
                         use_binary_mutex=use_binary_mutex,
                         objective=objective, emissions_cap=emissions_cap,
                         scenarios=scenarios,
                         demand_multiplier=demand_multiplier,
                         import_limit_override_kw=import_limit_override_kw)
    if solver is None:
        solver = _make_solver(solver_msg, time_limit_s, threads=threads)
    status = prob.solve(solver)
    return _extract_results(prob, V, status, district, activity, heating, horizon_years)


# 6b - PARETO FRONT
def pareto_front(district: str, activity: str, heating: str = "Gas Boiler", *,
                 n_points: int = 6,
                 scenarios: list = None,
                 horizon_years: int = HORIZON_YEARS,
                 time_limit_s: int = DEFAULT_TIME_LIMIT_S,
                 solver_msg: bool = False,
                 verbose: bool = True) -> pd.DataFrame:
    # Trace the OF1(cost)/OF2(carbon) trade-off for one (district, activity, heating) by the epsilon-constraint method: 
    # anchor on min-cost and min-emissions solutions, then minimise cost subject to a sweep of emission caps between the two anchors. 
    # Returns one row per point.
    _ensure_dm_initialized()
    common = dict(horizon_years=horizon_years, time_limit_s=time_limit_s, solver_msg=solver_msg)
    cost_opt = solve_scenario(district, activity, heating, objective="cost", scenarios=scenarios, **common)
    # The min-emissions anchor takes the same scenario set as the min-cost anchor and the eps sweep
    # below. Without it the anchor fell back to central prices while every other point on the front
    # carried the round's own set, so its cost could land BELOW the min-cost anchor — impossible on
    # a real front, and enough to flip the non-dominated flags.
    emis_opt = solve_scenario(district, activity, heating, objective="emissions",
                              scenarios=scenarios, **common)

    # Only sweep when both anchors are Optimal. A non-Optimal anchor returns a stub row that carries
    # no metrics at all, so the emissions bounds must not be read until anchors_ok is known good —
    # otherwise an infeasible heating (e.g. a land-capped horizontal loop) raises KeyError here and
    # takes the whole featured-cell front down with it.
    anchors_ok = cost_opt.get("status") == "Optimal" and emis_opt.get("status") == "Optimal"
    e_lo = e_hi = None
    if anchors_ok:
        e_cost = cost_opt["lifetime_emissions_tco2e"]
        e_emis = emis_opt["lifetime_emissions_tco2e"]
        e_lo, e_hi = min(e_cost, e_emis), max(e_cost, e_emis)
    elif verbose:
        print(f"  WARNING: anchor not Optimal (cost={cost_opt.get('status')}, "
              f"emissions={emis_opt.get('status')}) — interior sweep skipped. Raise time_limit_s.")

    rows = [{**cost_opt, "pareto_point": "min-cost",      "emissions_cap_tco2e": None}]
    if anchors_ok and e_hi - e_lo > 1e-3 and n_points > 2:
        for i, cap_t in enumerate(np.linspace(e_lo, e_hi, n_points)[1:-1], 1):
            r = solve_scenario(district, activity, heating, objective="cost",
                               emissions_cap=cap_t * 1000.0, scenarios=scenarios, **common)   # cap in kg
            rows.append({**r, "pareto_point": f"eps-{i}", "emissions_cap_tco2e": round(cap_t, 1)})
    rows.append({**emis_opt, "pareto_point": "min-emissions", "emissions_cap_tco2e": None})

    # When every row is a stub the metric columns are absent entirely — return unsorted rather than
    # raise; featured_cell_front drops the non-Optimal rows immediately after.
    df = pd.DataFrame(rows)
    if "lifetime_emissions_tco2e" in df.columns:
        df = df.sort_values("lifetime_emissions_tco2e")
    df = df.reset_index(drop=True)
    if verbose:
        print(f"\nCost/carbon Pareto front — {activity} / {heating} / {district}")
        cols = ["pareto_point", "total_cost_npv_GBP", "lifetime_emissions_tco2e",
                "emissions_saving_fraction", "pv_kwp", "e_batt_kwh", "mac_GBP_per_tco2e"]
        print(df[[c for c in cols if c in df.columns]].to_string(index=False))
    return df


# 6c - KNEE SELECTION
def _nondominated(cost, emis) -> np.ndarray:
    c = np.asarray(cost, dtype=float)
    e = np.asarray(emis, dtype=float)
    mask = np.ones(len(c), dtype=bool)
    for i in range(len(c)):
        dominated = (c <= c[i]) & (e <= e[i]) & ((c < c[i]) | (e < e[i]))
        mask[i] = not dominated.any()
    return mask

def _flag_pareto(df: pd.DataFrame) -> np.ndarray:
    # Non-dominated mask over a frontier DataFrame's (cost NPV, emissions) columns.
    return _nondominated(df["total_cost_npv_GBP"].to_numpy(float),
                         df["lifetime_emissions_tco2e"].to_numpy(float))


def featured_cell_front(district: str, activity: str, *, heatings: list = None,
                        n_points: int = 5, scenarios: list = None,
                        horizon_years: int = HORIZON_YEARS,
                        time_limit_s: int = DEFAULT_TIME_LIMIT_S,
                        verbose: bool = True) -> pd.DataFrame:
    # Trace the cost/carbon trade-off for ONE (district, activity) cell ACROSS its heating types:
    # run the epsilon-constraint pareto_front per heating and pool the points. 
    # Flags the cell-wide non-dominated set (pareto_optimal) and the single knee (best-compromise) design (is_knee).
    heatings = heatings or list(dm.HEATING_OPTIONS)
    frames = []
    for h in heatings:
        if verbose:
            print(f"  · front: {activity} / {h} / {district}")
        frames.append(pareto_front(district, activity, h, n_points=n_points, scenarios=scenarios,
                                   horizon_years=horizon_years, time_limit_s=time_limit_s,
                                   verbose=False))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["status"] == "Optimal"].copy().reset_index(drop=True)
    if df.empty:
        return df
    df["pareto_optimal"] = _flag_pareto(df)
    df["is_knee"] = False
    pf = df[df["pareto_optimal"]]
    if len(pf) == 1:
        df.loc[pf.index, "is_knee"] = True
    elif len(pf) > 1:
        c, e = pf["total_cost_npv_GBP"], pf["lifetime_emissions_tco2e"]
        cn = (c - c.min()) / (c.max() - c.min()) if c.max() > c.min() else c * 0.0
        en = (e - e.min()) / (e.max() - e.min()) if e.max() > e.min() else e * 0.0
        df.loc[np.hypot(cn, en).idxmin(), "is_knee"] = True
    return df.sort_values("lifetime_emissions_tco2e").reset_index(drop=True)


def knee_design(df_front: pd.DataFrame) -> dict:
    # The knee row as a dict, plus the (_objective, _cap_kg) needed to reproduce its exact dispatch.
    if df_front.empty or "is_knee" not in df_front or not df_front["is_knee"].any():
        return {}
    row = df_front[df_front["is_knee"]].iloc[0].to_dict()
    pp = row.get("pareto_point")
    if pp == "min-emissions":
        row["_objective"], row["_cap_kg"] = "emissions", None
    elif pp == "min-cost":
        row["_objective"], row["_cap_kg"] = "cost", None
    else:                                                   # interior eps point: cost-min under a cap
        cap_t = row.get("emissions_cap_tco2e")
        row["_objective"] = "cost"
        row["_cap_kg"] = (float(cap_t) * 1000.0) if cap_t is not None else None
    return row



# 7 - Rank all 9 × 4 × 4 combinations (district × activity × heating)
def _solve_one_scenario(task: tuple) -> dict:
    district, activity, heating, horizon_years, use_binary_mutex, time_limit_s, objective, threads, scenarios = task
    _ensure_dm_initialized()
    return solve_scenario(district, activity, heating,
                          horizon_years=horizon_years, use_binary_mutex=use_binary_mutex,
                          objective=objective, scenarios=scenarios, time_limit_s=time_limit_s, threads=threads)

def _solve_tagged_scenario(tagged_task: tuple) -> dict:
    tag, task = tagged_task
    row = _solve_one_scenario(task)
    row["_phase"] = tag
    return row

def _print_scenario_row(i: int, n_total: int, row: dict, tag: str = None) -> None:
    prefix = f"[{tag}] " if tag else ""
    head = f"{prefix}[{i:>3}/{n_total}] {row.get('activity','?')!r:27s} {row.get('heating','?')!r:20s} {row.get('district','?')!r:28s}"
    if row.get("status") == "Optimal":
        print(f"{head} NPV £{row['npv_savings_GBP']:>13,.0f}  ROI {row['roi']:>5.2f}  "
              f"PV {row['pv_kwp']:>6.1f}kWp  Batt {row['e_batt_kwh']:>6.0f}kWh  "
              f"Heat {row['q_heat_cap_kwth']:>5.0f}kWth  Store {row['e_th_kwh']:>5.0f}kWh")
    else:
        print(f"{head} {row.get('status', 'ERROR')}")

def rank_all_combinations(*, horizon_years: int = HORIZON_YEARS,
                          use_binary_mutex: bool = False,
                          objective: str = "cost",
                          activities: list = None,
                          districts: list = None,
                          heatings: list = None,
                          verbose: bool = True,
                          time_limit_s: int = DEFAULT_TIME_LIMIT_S,
                          n_jobs: int = PARALLEL_JOBS,
                          scenarios: list = None) -> pd.DataFrame:
    # Solve every (district, activity, heating) scenario under one objective and rank vs the common BAU.
    # scenarios = explicit import-price scenario set passed to every cell.
    # The cells are independent, so they are solved in parallel across n_jobs processes
    _ensure_dm_initialized()
    activities = activities or list(ROOF_PROPERTIES.keys())
    districts  = districts  or list(dm.DISTRICT_STATIONS.keys())
    heatings   = heatings   or list(dm.HEATING_OPTIONS)
    n_total = len(activities) * len(districts) * len(heatings)
    logical = os.cpu_count() or 2
    # Default to HALF of physical cores (≈ logical/4 on SMT CPUs).
    if n_jobs is None:
        n_jobs = max(1, logical // 4)
    n_jobs = max(1, min(n_jobs, n_total))
    threads_per_worker = SOLVER_THREADS or max(1, logical // n_jobs)
    tasks = [(d, a, h, horizon_years, use_binary_mutex, time_limit_s, objective, threads_per_worker, scenarios)
             for a in activities for d in districts for h in heatings]

    rows = []
    if n_jobs == 1:
        solver = _make_solver(solver_msg=False, time_limit_s=time_limit_s)
        for i, (d, a, h, *_rest) in enumerate(tasks, 1):
            row = solve_scenario(d, a, h, horizon_years=horizon_years,
                                 use_binary_mutex=use_binary_mutex, objective=objective,
                                 scenarios=scenarios, solver=solver)
            rows.append(row)
            if verbose:
                _print_scenario_row(i, n_total, row)
    else:
        if verbose:
            print(f"Solving {n_total} scenarios ({objective}) across {n_jobs} processes "
                  f"× {threads_per_worker} solver threads (of {logical} logical cores) …")
        with mp.Pool(processes=n_jobs) as pool:
            for i, row in enumerate(pool.imap_unordered(_solve_one_scenario, tasks), 1):
                rows.append(row)
                if verbose:
                    _print_scenario_row(i, n_total, row)

    sort_col = "emissions_saving_tco2e" if objective == "emissions" else "npv_savings_GBP"
    df = pd.DataFrame(rows)
    if sort_col in df.columns:   # absent only if every cell in this sweep failed to solve
        df = df.sort_values(sort_col, ascending=False).reset_index(drop=True)
    return df


def _run_merged_sweep(specs: list, *, activities: list = None, districts: list = None,
                      heatings: list = None, horizon_years: int = HORIZON_YEARS,
                      use_binary_mutex: bool = False, verbose: bool = True,
                      time_limit_s: int = DEFAULT_TIME_LIMIT_S, n_jobs: int = PARALLEL_JOBS) -> dict:
    # Solve several independent sweeps through ONE shared worker pool instead of one rank_all_combinations() pool per sweep. 
    # Each spec is {"tag": str, "objective": "cost"|"emissions", "scenarios": list|None, "sort_col": str}.
    _ensure_dm_initialized()
    activities = activities or list(ROOF_PROPERTIES.keys())
    districts  = districts  or list(dm.DISTRICT_STATIONS.keys())
    heatings   = heatings   or list(dm.HEATING_OPTIONS)
    n_per_spec = len(activities) * len(districts) * len(heatings)
    n_total    = n_per_spec * len(specs)
    logical    = os.cpu_count() or 2
    # Half of physical cores, not the full physical-core count — see rank_all_combinations() above.
    if n_jobs is None:
        n_jobs = max(1, logical // 4)
    n_jobs = max(1, min(n_jobs, n_total))
    threads_per_worker = SOLVER_THREADS or max(1, logical // n_jobs)

    tagged_tasks = []
    for spec in specs:
        tasks = [(d, a, h, horizon_years, use_binary_mutex, time_limit_s, spec["objective"],
                  threads_per_worker, spec["scenarios"])
                 for a in activities for d in districts for h in heatings]
        tagged_tasks.extend((spec["tag"], t) for t in tasks)

    if verbose:
        tags = ", ".join(f"{spec['tag']} ({spec['objective']})" for spec in specs)
        print(f"Solving {n_total} scenarios across {len(specs)} phase(s) [{tags}] in one pool "
              f"of {n_jobs} processes × {threads_per_worker} solver threads (of {logical} logical cores) …")

    buckets = {spec["tag"]: [] for spec in specs}
    if n_jobs == 1:
        for i, (tag, task) in enumerate(tagged_tasks, 1):
            row = _solve_one_scenario(task)
            buckets[tag].append(row)
            if verbose:
                _print_scenario_row(i, n_total, row, tag=tag)
    else:
        with mp.Pool(processes=n_jobs) as pool:
            for i, row in enumerate(pool.imap_unordered(_solve_tagged_scenario, tagged_tasks), 1):
                tag = row.pop("_phase")
                buckets[tag].append(row)
                if verbose:
                    _print_scenario_row(i, n_total, row, tag=tag)

    out = {}
    for spec in specs:
        df = pd.DataFrame(buckets[spec["tag"]])
        if spec["sort_col"] in df.columns:   # absent only if every cell in this phase failed to solve
            df = df.sort_values(spec["sort_col"], ascending=False).reset_index(drop=True)
        out[spec["tag"]] = df
    return out


def assemble_pareto(df_npv: pd.DataFrame, df_carbon: pd.DataFrame) -> pd.DataFrame:
    # Combine the cost-min and emissions-min sweeps into the cost/carbon trade-off set. 
    # A design is dominated if another has <= NPV cost AND <= emissions, with at least one strictly better. 
    # Both objective sweeps already carry full cost + emissions columns.
    a = df_npv.assign(source_objective="cost")
    b = df_carbon.assign(source_objective="emissions")
    df = pd.concat([a, b], ignore_index=True)
    df = df[df["status"] == "Optimal"].copy()
    # Drop duplicate designs (same cell + identical cost/emissions point)
    df = df.drop_duplicates(
        subset=["district", "activity", "heating", "total_cost_npv_GBP", "lifetime_emissions_tco2e"]
    ).reset_index(drop=True)

    # Non-dominated flag computed within each (district, activity) cell.
    df["pareto_optimal"] = False
    for _, idx in df.groupby(["district", "activity"]).groups.items():
        sub = df.loc[idx]
        df.loc[idx, "pareto_optimal"] = _nondominated(
            sub["total_cost_npv_GBP"].to_numpy(float),
            sub["lifetime_emissions_tco2e"].to_numpy(float))
    # Order: non-dominated first, then by cost
    return df.sort_values(["pareto_optimal", "total_cost_npv_GBP"],
                          ascending=[False, True]).reset_index(drop=True)


# 8 - REPORT
def write_results_workbook(dfs: dict, out_path: str, run_meta: dict = None,
                           scenarios: list = None, grid_sensitivity: pd.DataFrame = None) -> None:
    import optimisation_report
    import optimisation_plots          # pulls in matplotlib only when charts are drawn
    df_npv = dfs["NPV"]
    scen = scenarios if scenarios is not None else price_scenarios()
    meta = {
        "n_scenarios":   len(df_npv),
        # Cells the solver returned a design for. The complement is Infeasible — physically
        # impossible cells (e.g. a horizontal ground loop with no room for the collector), not
        # solver failures — so the cover reports this as "feasible", not "optimal".
        "n_feasible":    int((df_npv["status"] == "Optimal").sum()) if "status" in df_npv else len(df_npv),
        "horizon_years": HORIZON_YEARS,
        "discount_rate": TECH_COSTS["discount_rate"],
        "solver":        "HiGHS (CBC fallback)",
        # Import-price scenario set (id, weight, level, growth) for the Cover sheet — this round's set
        "price_scenarios": [(s.id, s.weight, s.level, s.growth) for s in scen],
        **(run_meta or {}),
    }
    charts_dir = os.path.join(os.path.dirname(out_path), "charts")
    os.makedirs(charts_dir, exist_ok=True)
    # Both cost rounds share one charts/ folder, so tag every PNG with the round it came from.
    _round = (run_meta or {}).get("round")
    chart_prefix = f"{_round}_" if _round else ""

    front_png = dispatch_png = demand_png = None
    opt = df_npv[df_npv["status"] == "Optimal"].dropna(subset=["npv_savings_GBP"]) \
        if "status" in df_npv else df_npv.dropna(subset=["npv_savings_GBP"])
    if not opt.empty:
        win = opt.loc[opt["npv_savings_GBP"].idxmax()]
        d, a = win["district"], win["activity"]
        try:
            print(f"\nFeatured cell (top NPV): {a} · {d} — tracing cost/carbon frontier …")
            front = featured_cell_front(d, a, scenarios=scenarios)
            knee = knee_design(front)
            # Frontier + MAC chart
            fpath = os.path.join(charts_dir, chart_prefix + "cell_front.png")
            optimisation_plots.plot_cell_front(
                front, out_path=fpath,
                title=f"Cost / carbon frontier — {a}{optimisation_plots.activity_area_suffix(a)} · {d}")
            front_png = fpath
            dfs["Pareto front"] = front                     # data sheet for the featured-cell front
            # Knee dispatch — reproduce the exact knee design (objective + emissions cap)
            if knee:
                print(f"  knee: {knee['heating']} ({knee.get('pareto_point')}) — "
                      f"NPV £{knee.get('total_cost_npv_GBP', 0):,.0f}, "
                      f"{knee.get('lifetime_emissions_tco2e', 0):,.0f} tCO2e")
                dpath = os.path.join(charts_dir, chart_prefix + "dispatch_knee.png")
                optimisation_plots.plot_dispatch(d, a, knee["heating"], months=("January", "July"), day_type="WD",
                                                 objective=knee["_objective"], emissions_cap=knee["_cap_kg"],
                                                 scenarios=scenarios, out_path=dpath)
                dispatch_png = dpath
            # Underlying demand (heat vs non-heat elec) for the same cell/months — input profiles.
            mpath = os.path.join(charts_dir, chart_prefix + "demand_profile.png")
            optimisation_plots.plot_demand_profile(d, a, months=("January", "July"), day_type="WD", out_path=mpath)
            demand_png = mpath
        except Exception as e:                              # never let a figure abort the run
            print(f"  ! featured-cell front / dispatch skipped ({a} · {d}): {e}")

    pngs = optimisation_report.write_report(dfs, out_path,
                                            run_meta=meta, charts_dir=charts_dir,
                                            dispatch_png=dispatch_png, demand_png=demand_png,
                                            front_png=front_png, grid_sensitivity=grid_sensitivity,
                                            chart_prefix=chart_prefix)
    print(f"Saved: {out_path}")
    print(f"  + {len(pngs)} charts in {charts_dir}")


# 9 - main()
def _osm_survey_info() -> str:
    # One-line provenance for the OSM roof data the model consumes, plus a refresh prompt.
    if not os.path.exists(OSM_STOREYS_XLSX):
        return (f"OSM roof survey not found at {OSM_STOREYS_XLSX}.\n"
                f"  Run api_osm_storeys.py to generate the storey + roof-shape data before optimising.")
    surveyed = datetime.fromtimestamp(os.path.getmtime(OSM_STOREYS_XLSX))
    return (f"Roof data from OSM survey conducted {surveyed:%Y-%m-%d %H:%M} "
            f"({OSM_STOREYS_XLSX}).\n"
            f"  To refresh the storey / roof-shape data, run: python api_osm_storeys.py")

