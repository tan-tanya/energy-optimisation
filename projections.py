"""
Run only via import, by demand_profile_model.py, to regenerate climate and electricity projection outputs:

ELECTRICITY — non-heating electricity demand growth factors from DESNZ's Reference scenario.
    Inputs
    1. data/model_parameters.xlsx; Scalars 'ELEC_DEMAND_TWH [YYYY]' rows
        - DESNZ Reference non-heating electricity demand (TWh) by year, transcribed from DESNZ Annex F
          ("Total excluding international aviation" / Electricity).
    Outputs
    1. data/electricity_projection_output.csv; one row per year (2026-2050)
        - electricity_twh    (DESNZ Reference projection, TWh)
        - growth_factor      (year TWh / 2025 TWh; applied to baseload electricity in downstream demand)
    Pipeline: _load_electricity_series() -> run_electricity_projection()

CLIMATE — monthly temperature distributions fitted from 2025 HDDs, projected forward under UKCP18 RCP8.5.
    Inputs
    1. data/hdd/{ICAO}_HDD_15.5C.csv; daily heating degree-days (T_base = 15.5 C) per district (Met Office)
    2. data/climateprojections/UKCP_{region}.csv; UKCP18 RCP8.5 monthly mean-temperature anomalies, 2026-2080
    Outputs
    1. data/climate_projection_output.csv; one row per (district, year, month)
        - delta_T_mean             (UKCP18 ensemble-mean monthly anomaly, C)
        - baseline_hdd_per_day     (E[HDD] under the 2025-fitted distribution)
        - projected_hdd_per_day    (E[HDD] after shifting mu by delta_T_mean)
        - hdd_reduction            (baseline - projected, HDD/day)
    Method:
        (a) Recover daily mean temperatures from HDDs: T = T_base - HDD on uncensored days (HDD > 0).
            Days with HDD = 0 are right-censored at T_base as no heating is required.
        (b) Fit T ~ N(mu, sigma) per district per month:
                - standard MLE when <5% of days are censored
                - censored MLE otherwise (adds the censored-mass term n_cen . log P(T >= T_base))
        (c) Closed-form expected HDD under a normal:
                E[HDD] = (T_base - mu) . Phi(z) + sigma . phi(z),   z = (T_base - mu) / sigma
        (d) Project future HDDs by shifting mu by the UKCP18 monthly ensemble-mean delta-T, holding sigma fixed.
    Pipeline: run_sigma_fit() -> run_hdd_projection()

Run directly (python projections.py) to regenerate both CSVs.
"""

import os
from calendar import month_name

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import minimize

import datasets
from districts import DISTRICT_STATIONS, UKCP_TO_DISTRICT
from model_params import HDD_BASE

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


# ELECTRICITY PROJECTION 
ELEC_OUTPUT_PATH   = os.path.join(DATA_DIR, "electricity_projection_output.csv")
ELEC_BASELINE_YEAR = 2025
ELEC_START_YEAR    = 2026
ELEC_END_YEAR      = 2050


def _load_electricity_series():
    """Return TWh by year (pd.Series) from the DESNZ Reference demand series in model_parameters.xlsx
    (Scalars 'ELEC_DEMAND_TWH [YYYY]' rows; originally DESNZ Annex F)."""
    from model_params import ELEC_DEMAND_TWH
    return pd.Series(ELEC_DEMAND_TWH, dtype=float).sort_index()


def run_electricity_projection():
    series = _load_electricity_series()
    for required in (ELEC_BASELINE_YEAR, ELEC_START_YEAR, ELEC_END_YEAR):
        if required not in series.index:
            raise ValueError(f"Year {required} not in DESNZ projection (sheet may have shifted)")
    baseline = series[ELEC_BASELINE_YEAR]

    years = list(range(ELEC_START_YEAR, ELEC_END_YEAR + 1))
    twh   = series.loc[years]
    out   = pd.DataFrame({
        "year":            years,
        "electricity_twh": twh.round(3).values,
        "growth_factor":   (twh / baseline).round(6).values,
    })

    print(f"\n--- Electricity demand growth factors (DESNZ Reference, base {ELEC_BASELINE_YEAR}) ---")
    print(f"\n{'Year':>6}  {'TWh':>10}  {'Growth factor':>14}  {'% change':>10}")
    print("-" * 48)
    for _, r in out.iterrows():
        print(f"{int(r['year']):>6}  {r['electricity_twh']:>10.1f}  {r['growth_factor']:>14.4f}"
              f"  {(r['growth_factor']-1)*100:>+9.1f}%")

    print(f"\nBaseline ({ELEC_BASELINE_YEAR}): {baseline:.1f} TWh")
    print(f"End year ({ELEC_END_YEAR}):    {series[ELEC_END_YEAR]:.1f} TWh  "
          f"({(series[ELEC_END_YEAR]/baseline - 1)*100:+.1f}%)")

    out.to_csv(ELEC_OUTPUT_PATH, index=False)
    print(f"\nSaved: {ELEC_OUTPUT_PATH}")
    return out


# CLIMATE PROJECTION
CLIMATE_OUTPUT_PATH        = os.path.join(DATA_DIR, "climate_projection_output.csv")
CENSOR_THRESHOLD           = HDD_BASE  # 15.5 C; days with HDD = 0 imply T >= this
CENSORED_MLE_PCT_THRESHOLD = 5.0       # switch to censored MLE when censored fraction exceeds this %
UKCP_PROJECTION_START_YEAR = 2026

MONTH_NAMES = dict(enumerate(month_name))  

# UKCP_TO_DISTRICT comes from the districts registry (imported above).

# Expected HDD when daily-mean T ~ N(mu, sigma); vectorised over arrays
def expected_hdd(mu, sigma, t_base=HDD_BASE):
    z = (t_base - mu) / sigma
    return (t_base - mu) * norm.cdf(z) + sigma * norm.pdf(z)

# Expected HDD after shifting the temperature distribution by delta_T (sigma held fixed)
def _project_hdd(mu, sigma, delta_T, t_base=HDD_BASE):
    return expected_hdd(mu + delta_T, sigma, t_base)

# Standard MLE for a normal distribution, ignoring censored days
def _fit_normal_standard(temps):
    mu, sigma = norm.fit(temps)
    return float(mu), float(sigma)

# Censored MLE accounting for right-censored mass at T >= CENSOR_THRESHOLD
def _fit_normal_censored(temps_obs, n_censored):
    def neg_log_likelihood(params):
        mu, log_sigma = params
        sigma  = np.exp(log_sigma) # Ensure sigma > 0
        ll_obs = norm.logpdf(temps_obs, loc=mu, scale=sigma).sum() # Log-likelihood of observed (uncensored) data
        ll_cen = n_censored * norm.logsf(CENSOR_THRESHOLD, loc=mu, scale=sigma) # Log-likelihood of censored data
        return -(ll_obs + ll_cen)

    mu0    = temps_obs.mean()
    sigma0 = max(temps_obs.std(), 0.5)
    result = minimize(neg_log_likelihood, [mu0, np.log(sigma0)], method="Nelder-Mead",
                      options={"xatol": 1e-6, "fatol": 1e-6, "maxiter": 10_000})
    return float(result.x[0]), float(np.exp(result.x[1]))

# Raw degree-day / UKCP file reads live in datasets.get_degree_days / datasets.get_ukcp.

# Fit (mu, sigma) per month for one district, returning per-month diagnostic rows.
# Uses the last datasets.HDD_BASELINE_YEARS years of HDD data (pooled per calendar month) for the fit.
def _fit_district(district, icao):
    df = datasets.get_degree_days(icao, last_years=datasets.HDD_BASELINE_YEARS).copy()
    df["month"] = df["date"].dt.month

    rows = []
    for month, sub in df.groupby("month"):
        n_total      = len(sub)
        n_censored   = int((sub["hdd"] == 0).sum())
        pct_cen      = 100.0 * n_censored / n_total if n_total > 0 else 0.0
        hdd_pos      = sub.loc[sub["hdd"] > 0, "hdd"].values
        temps        = CENSOR_THRESHOLD - hdd_pos
        obs_mean_hdd = sub["hdd"].mean()

        if n_censored == 0 or pct_cen < CENSORED_MLE_PCT_THRESHOLD:
            mu, sigma = _fit_normal_standard(temps)
            method = "MLE"
        else:
            mu, sigma = _fit_normal_censored(temps, n_censored)
            method = "censored MLE"

        pred_hdd = expected_hdd(mu, sigma)

        rows.append({
            "district":      district,
            "month":         int(month),
            "month_name":    MONTH_NAMES[int(month)],
            "mu_C":          round(mu, 3),
            "sigma_C":       round(sigma, 3),
            "n_days":        n_total,
            "n_censored":    n_censored,
            "pct_censored":  round(pct_cen, 1),
            "obs_hdd_mean":  round(obs_mean_hdd, 3),
            "pred_hdd_mean": round(pred_hdd, 3),
            "hdd_error":     round(pred_hdd - obs_mean_hdd, 3),
            "method":        method,
        })
    return rows

def run_sigma_fit():
    # Fit N(mu, sigma) per district per month from 2025 HDD data
    all_rows = []
    for district, icao in DISTRICT_STATIONS.items():
        all_rows.extend(_fit_district(district, icao))
    df = pd.DataFrame(all_rows)

    print(f"\n--- Stage 1: Temperature distribution fit (last {datasets.HDD_BASELINE_YEARS}-yr HDD data) ---")
    print(f"\n{'District':<36} {'Month':<12} {'mu(C)':>7} {'sigma(C)':>8} "
          f"{'Days':>5} {'Cens%':>6} {'ObsHDD':>7} {'PredHDD':>8} {'Err':>6} {'Method'}")
    print("-" * 112)

    for i, (_, sub) in enumerate(df.groupby("district", sort=False)):
        if i > 0:
            print()
        for _, r in sub.iterrows():
            print(f"{r['district']:<36} {r['month_name']:<12} {r['mu_C']:>7.2f} {r['sigma_C']:>7.2f} "
                  f"{r['n_days']:>5} {r['pct_censored']:>5.1f}% {r['obs_hdd_mean']:>7.3f} "
                  f"{r['pred_hdd_mean']:>8.3f} {r['hdd_error']:>6.3f}  {r['method']}")

    print(f"\nMax absolute HDD validation error: {df['hdd_error'].abs().max():.4f} HDD/day")
    print()
    return df

# Aggregate per-day HDDs to annual totals per (district, year) over a given column
def _annual_hdd(df, col):
    days = pd.to_datetime(dict(year=df["year"], month=df["month"], day=1)).dt.days_in_month
    return (df[col] * days).groupby([df["district"], df["year"]]).sum()

# Project HDDs using UKCP18 RCP8.5 monthly temperature anomalies and fitted sigma values
def run_hdd_projection(sigma_df):
    """Stage 2: project monthly HDDs using UKCP18 anomalies and fitted sigma."""
    ukcp_frames = []
    for ukcp_name, district in UKCP_TO_DISTRICT.items():
        u = datasets.get_ukcp(ukcp_name)
        u = u[u["year"] >= UKCP_PROJECTION_START_YEAR].copy()
        u["district"] = district
        ukcp_frames.append(u)
    ukcp = pd.concat(ukcp_frames, ignore_index=True)

    df = ukcp.merge(
        sigma_df[["district", "month", "mu_C", "sigma_C"]],
        on=["district", "month"], how="left",
    )
    mu, sigma, dT = df["mu_C"].values, df["sigma_C"].values, df["delta_T_mean"].values
    baseline_hdd  = expected_hdd(mu, sigma)
    projected_hdd = _project_hdd(mu, sigma, dT)
    df["baseline_hdd_per_day"]  = baseline_hdd.round(4)
    df["projected_hdd_per_day"] = projected_hdd.round(4)
    df["hdd_reduction"]         = (baseline_hdd - projected_hdd).round(4)
    df["delta_T_mean"]          = df["delta_T_mean"].round(4)
    df["month_name"]            = df["month"].map(MONTH_NAMES)

    df = df[["district", "year", "month", "month_name",
             "delta_T_mean", "baseline_hdd_per_day", "projected_hdd_per_day", "hdd_reduction"]] \
            .sort_values(["district", "year", "month"]).reset_index(drop=True)

    yr_first, yr_last = int(df["year"].min()), int(df["year"].max())

    print(f"--- Stage 2: HDD projection (UKCP18 RCP8.5, {yr_first}-{yr_last}) ---")
    print(f"\n{'District':<36} {f'{yr_first} Ann.HDD':>12} {f'{yr_last} Ann.HDD':>12} {'Reduction':>10} {'Change%':>8}")
    print("-" * 82)

    proj_annual = _annual_hdd(df, "projected_hdd_per_day")
    base_annual = _annual_hdd(df, "baseline_hdd_per_day")

    for district in sorted(df["district"].unique()):
        hdd_first = proj_annual.loc[(district, yr_first)]
        hdd_last  = proj_annual.loc[(district, yr_last)]
        baseline  = base_annual.loc[(district, yr_first)]
        reduction = hdd_first - hdd_last
        pct       = 100 * reduction / baseline if baseline > 0 else 0
        print(f"{district:<36} {hdd_first:>12.1f} {hdd_last:>12.1f} {reduction:>10.1f} {pct:>7.1f}%")

    df.to_csv(CLIMATE_OUTPUT_PATH, index=False)
    print(f"\nSaved: {CLIMATE_OUTPUT_PATH}")
    print(f"Rows: {len(df)}  ({df['district'].nunique()} districts x {df['year'].nunique()} years x 12 months)\n")
    return df


# Run both projection pipelines.
def main():
    sigma_df = run_sigma_fit()
    run_hdd_projection(sigma_df)
    run_electricity_projection()

if __name__ == "__main__":
    main()
