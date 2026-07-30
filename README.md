# energy-optimisation

Optimal sizing and dispatch of on-site electricity and heating systems for UK non-domestic buildings, across nine UK districts and four building activity classes.

The model builds half-hourly building demand profiles from published benchmarks, then solves a linear program that sizes rooftop PV, battery storage, thermal storage and a heating system over a 15-year horizon — ranking each configuration against a gas-boiler business-as-usual baseline on both net present value and cumulative carbon.

---

## Methodology

**1. Demand modelling.** Half-hourly electricity and heat profiles per (district, activity), assembled from CIBSE annual benchmarks, NCM occupancy schedules, TM46 baseload/HDD splits, Met Office degree-days and sunshine hours, and ERA5 hourly temperatures.

**2. Optimisation.** For each (district, activity, heating) cell, a linear program sizes PV, battery, thermal store and heating capacity to minimise discounted total cost, subject to energy balance, roof area, grid import/export limits and land availability.

**3. Uncertainty.** A two-stage stochastic program over reduced electricity import-price scenarios, run alongside a deterministic central-price round for comparison.

**4. Downstream analysis.** Carbon objective and epsilon-constraint Pareto fronts, grid-headroom sensitivity, back-calculated capital-cost rebates needed to make heat pumps viable per region, and two technology sensitivities (a cheaper low-efficiency PV panel, a lower-COP air-source heat pump).

Scope: 9 districts × 4 activities × 4 heating systems.

| Districts | Activities | Heating systems |
|---|---|---|
| East Anglia | Health: Health centre | Gas Boiler |
| England E and NE | Health: Hospital | ASHP |
| England NW and N Wales | Office: A/C standard | GSHP (vertical) |
| England SE and Central S | Retail: Department store | GSHP (horizontal) |
| England SW and S Wales | | |
| Midlands | | |
| Scotland E / N / W | | |

---

## Installation

Requires Python ≥ 3.12. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra optim
```

---

## Running

### Full pipeline

```bash
python optimisation_model.py
```

Writes everything to a timestamped `outputs/Optimisation (YYYYMMDD, HHMM)/` directory: the results workbook, standalone chart PNGs, policy recommendations, and both sensitivity workbooks.

⚠️ **This takes around 31 hours.** The stochastic round is roughly 13× the cost of the deterministic one. To skip it:

```bash
python optimisation_model.py --skip-stochastic
```

That brings it down to about 9–10 hours.

Useful flags:

| Flag | Effect |
|---|---|
| `--skip-stochastic` | Deterministic central-price round only |
| `--skip-policy` | Skip the capital-rebate back-calculation |
| `--skip-grid-sensitivity` | Skip the grid-headroom bisection |
| `--skip-ashp-sensitivity` | Skip the Aerona3 heat-pump sensitivity |
| `--skip-pv-sensitivity` | Skip the Panel-1 PV sensitivity |
| `--jobs N` | Parallel worker processes |
| `--time-limit S` | Per-solve time limit, seconds |

### Demand profiles alone

```bash
python demand_profile_model.py
```

Generates profiles and plots without touching the solver. Also importable — this is how the optimiser consumes demand.

---

## Module map

| Module | Role |
|---|---|
| `optimisation_model.py` | Entry point — thin CLI over the engine |
| `optimisation_engine.py` | LP construction, solve, sweep and pool mechanics |
| `optimisation_config.py` | Solver and horizon constants |
| `optimisation_report.py` | Results workbook assembly |
| `optimisation_plots.py` | Chart generation |
| `demand_profile_model.py` | Half-hourly demand profiles per district and activity |
| `demand_report.py` | Demand workbook output |
| `model_params.py` | Technology costs and parameters, read from `data/model_parameters.xlsx` |
| `districts.py` | District table — ICAO stations, UKCP regions, coordinates |
| `datasets.py` | Input file loading |
| `pricing.py` | Wholesale + DUoS + CCL import price build-up |
| `projections.py` | Climate and electricity demand projections |
| `growth.py` | Demand growth and COP improvement over the horizon |
| `uncertainty.py` | Price-scenario generation and reduction |
| `grid_sensitivity.py` | Grid headroom / demand-margin bisection |
| `policy_recommendations.py` | Back-calculated regional capital rebates |
| `pv_panel1_sensitivity.py` | Alternative PV panel sensitivity |
| `ashp_grant_aerona3_sensitivity.py` | Alternative ASHP sensitivity |
| `api_osm_storeys.py` | OpenStreetMap building storey survey |
| `api_temperature_profiles.py` | ERA5 hourly temperature via Open-Meteo |
| `api_wholesale_prices.py` | Wholesale electricity price series |

Modules prefixed `api_` pull live external data and write their output into `data/`. They are imported at runtime by other modules but only need re-running when refreshing the underlying data.

---

## Data

| Directory | Contents | Source |
|---|---|---|
| `data/hdd/` | Daily heating degree-days, base 15.5 °C | Met Office |
| `data/sunlighthours/` | Monthly sunshine hours | Met Office |
| `data/climateprojections/` | UKCP18 RCP8.5 temperature anomalies | Met Office / CEDA |
| `data/headroom/` | Substation headroom and network development data | UKPN, NPG, ENWL, SP, NGED, SSEN |
| `data/duoscharges/` | 2025 DUoS Schedules of Charges | All UK DNOs |
| `data/inputs.xlsx` | CIBSE, NCM, TM46, BEES benchmarks | Various |
| `data/model_parameters.xlsx` | Technology costs and model parameters | Various |
| `data/00 readme.xlsx` | Directory tab indexing every source | — |
| `cache/` | OpenStreetMap survey checkpoints | OpenStreetMap |

`cache/` is committed deliberately: it pins the building-stock survey to the data the model was actually run against, rather than whatever OpenStreetMap returns today.

### Attribution

The datasets in `data/` are redistributed under open licences that permit reuse with attribution:

- **UKCP18 climate projections** — Open Government Licence v3.0. Requires the CEDA citation: *Met Office Hadley Centre (2018): UKCP18 [dataset]. Centre for Environmental Data Analysis.*
- **UK Power Networks** — CC BY 4.0
- **SSEN** — CC BY 4.0
- **National Grid Electricity Distribution** — NGED Open Data Licence ("Supported by NGED Open Data")
- **Northern Powergrid** — Northern Powergrid Open Data Licence v1.0
- **DUoS Schedules of Charges** — published by DNOs under their licence conditions

Full per-file attribution is being consolidated into the Directory tab of `data/00 readme.xlsx`.

---

## Notes

`outputs/` is not tracked. Every writer creates its own directory on first run.

The model runs as a pure linear program by default. The relaxation was verified lossless against the mixed-integer formulation, and solves roughly 3× faster.
