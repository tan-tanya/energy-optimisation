"""
Grid electricity import-price uncertainty for the TSSP.

Generates a weighted set of representative 15-year electricity IMPORT-price scenarios that drive the
TSSP second stage (recourse dispatch). Uncertainty is sampled over two facets:
  1. price level multiplier — IMPORT_PRICE_SCENARIOS (Low/Central/High), relative to Central = 1.0
  2. annual escalation rate — ENERGY_GROWTH_SCENARIOS["elec_price_growth"] (Low/Central/High)
Each draw maps to a 15-year multiplier path
    path[y] = level * (1 + growth) ** y
applied on top of the year-0 size-band CENTRAL import price the deterministic model already selects.

Method (literature: Latin Hypercube Sampling -> scenario reduction by clustering):
  - 2-D LHS (scipy.stats.qmc.LatinHypercube) with TRIANGULAR marginals (each band's min/mode/max).
  - k-medoid reduction of sampled paths to k weighted representatives.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import qmc, triang

from model_params import IMPORT_PRICE_SCENARIOS, ENERGY_GROWTH_SCENARIOS

HORIZON_YEARS = 15

# Default sampling knobs — N draws reduced to k weighted scenarios.
DEFAULT_N_SAMPLES = 1000
DEFAULT_N_REDUCED = 3      # reduced scenario count (k); 3 keeps the spread while ~40% smaller than 5
DEFAULT_SEED      = 0


# 1 - DEFINITION 
def _triangular_params(low: float, mode: float, high: float) -> dict:
    # scipy.stats.triang is parameterised by c=(mode-loc)/scale, loc=min, scale=(max-min).
    span = high - low
    if span <= 0:
        raise ValueError(f"degenerate triangular band: low={low}, mode={mode}, high={high}")
    return {"c": (mode - low) / span, "loc": low, "scale": span}

def level_band() -> tuple:
    # (low, central, high) electricity import-price LEVEL multipliers, relative to Central.
    s = IMPORT_PRICE_SCENARIOS
    return (s["Low"]["elec_import_multiplier"],
            s["Central"]["elec_import_multiplier"],
            s["High"]["elec_import_multiplier"])

def growth_band() -> tuple:
    # (low, central, high) annual real escalation rates for electricity import price.
    g = ENERGY_GROWTH_SCENARIOS
    return (g["Low"]["elec_price_growth"],
            g["Central"]["elec_price_growth"],
            g["High"]["elec_price_growth"])


def price_multiplier_path(level: float, growth: float, horizon: int = HORIZON_YEARS) -> np.ndarray:
    # 15-year import-price multiplier (relative to the year-0 Central band price); y=0 -> level.
    y = np.arange(horizon)
    return level * (1.0 + growth) ** y


# 2 - LHS of (level, growth)
def lhs_samples(n: int = DEFAULT_N_SAMPLES, seed: int = DEFAULT_SEED) -> np.ndarray:
    # n x 2 array of (level, growth) drawn by 2-D LHS through the triangular inverse-CDFs.
    lo_l, mo_l, hi_l = level_band()
    lo_g, mo_g, hi_g = growth_band()
    pl = _triangular_params(lo_l, mo_l, hi_l)
    pg = _triangular_params(lo_g, mo_g, hi_g)

    u = qmc.LatinHypercube(d=2, seed=seed).random(n)        
    level  = triang.ppf(u[:, 0], **pl)
    growth = triang.ppf(u[:, 1], **pg)
    return np.column_stack([level, growth])

def mc_samples(n: int = DEFAULT_N_SAMPLES, seed: int = DEFAULT_SEED) -> np.ndarray:
    # Plain Monte Carlo draw of (level, growth) — for the LHS-vs-MC coverage comparison only.
    rng = np.random.default_rng(seed)
    lo_l, mo_l, hi_l = level_band()
    lo_g, mo_g, hi_g = growth_band()
    pl = _triangular_params(lo_l, mo_l, hi_l)
    pg = _triangular_params(lo_g, mo_g, hi_g)
    level  = triang.ppf(rng.random(n), **pl)
    growth = triang.ppf(rng.random(n), **pg)
    return np.column_stack([level, growth])


# 3 - SCENARIO REDUCTION
def reduce_scenarios(paths: np.ndarray, k: int, seed: int = DEFAULT_SEED) -> tuple:
    # Cluster the N x horizon path matrix into <= k groups (Euclidean k-means on the path vectors),
    # then pick each cluster's MEDOID (sample nearest its centroid) as the representative.
    # Returns (medoid_row_indices, weights) with weights = cluster share (summing to 1).
    from scipy.cluster.vq import kmeans2

    n = paths.shape[0]
    if k >= n:
        return np.arange(n), np.full(n, 1.0 / n)

    centroids, labels = kmeans2(paths, k, seed=seed, minit="++", missing="warn")
    medoid_idx, weights = [], []
    for c in range(len(centroids)):
        members = np.where(labels == c)[0]
        if members.size == 0:
            continue                                        # empty cluster — drop, renormalise below
        d = np.linalg.norm(paths[members] - centroids[c], axis=1)
        medoid_idx.append(int(members[np.argmin(d)]))
        weights.append(members.size / n)
    weights = np.asarray(weights)
    return np.asarray(medoid_idx), weights / weights.sum()


# 4 - PUBLIC INTERFACE
@dataclass(frozen=True)
class Scenario:
    # One representative electricity import-price realisation for the TSSP second stage.
    id: str
    weight: float                 # probability mass (reduced scenarios sum to 1)
    level: float                  # year-0 multiplier relative to Central
    growth: float                 # annual real escalation rate
    path: np.ndarray = field(repr=False)   # length-horizon multiplier path, path[y] = level*(1+growth)**y

    def import_price(self, y: int, base_central: float) -> float:
        # Scenario import price in year y, GBP/kWh. base_central = year-0 CENTRAL size-band price
        # (active level multiplier divided out by the caller), escalation supplied by this scenario.
        return base_central * float(self.path[y])


def generate_price_scenarios(n_samples: int = DEFAULT_N_SAMPLES,
                             n_reduced: int = DEFAULT_N_REDUCED,
                             horizon: int = HORIZON_YEARS,
                             seed: int = DEFAULT_SEED) -> list:
    # Full pipeline: LHS-sample (level, growth) -> build paths -> k-medoid reduce -> weighted Scenarios.
    draws = lhs_samples(n_samples, seed=seed)
    paths = np.array([price_multiplier_path(l, g, horizon) for l, g in draws])
    idx, weights = reduce_scenarios(paths, n_reduced, seed=seed)

    order = np.argsort([paths[i][-1] for i in idx])          # order low -> high final-year price
    scenarios = []
    for rank, j in enumerate(order):
        i = int(idx[j])
        scenarios.append(Scenario(id=f"S{rank+1}", weight=float(weights[j]),
                                   level=float(draws[i, 0]), growth=float(draws[i, 1]),
                                   path=paths[i].copy()))
    return scenarios


def scenario_table(scenarios: list) -> pd.DataFrame:
    # Tidy summary table of the reduced scenario set (for the report / assumptions).
    return pd.DataFrame([{
        "scenario":        s.id,
        "weight":          round(s.weight, 4),
        "level_mult":      round(s.level, 4),
        "growth_rate":     round(s.growth, 5),
        "yr0_mult":        round(float(s.path[0]), 4),
        "yr14_mult":       round(float(s.path[-1]), 4),
    } for s in scenarios])


def validate(scenarios: list, n_samples: int = DEFAULT_N_SAMPLES,
             horizon: int = HORIZON_YEARS, seed: int = DEFAULT_SEED) -> dict:
    # Sanity checks: weights sum to 1, and the weighted-mean reduced path tracks the full-sample mean.
    draws = lhs_samples(n_samples, seed=seed)
    full  = np.array([price_multiplier_path(l, g, horizon) for l, g in draws]).mean(axis=0)
    red   = np.average(np.array([s.path for s in scenarios]),
                       axis=0, weights=[s.weight for s in scenarios])
    return {
        "weight_sum":          float(sum(s.weight for s in scenarios)),
        "n_scenarios":         len(scenarios),
        "full_mean_yr0":       float(full[0]),
        "reduced_mean_yr0":    float(red[0]),
        "full_mean_yr14":      float(full[-1]),
        "reduced_mean_yr14":   float(red[-1]),
        "max_abs_path_err":    float(np.max(np.abs(full - red))),
    }


# 5 - DIAGNOSTICS 
def _coverage_figure(ax_path: str, n: int = 200, seed: int = DEFAULT_SEED) -> None:
    import matplotlib.pyplot as plt
    lhs = lhs_samples(n, seed=seed)
    mc  = mc_samples(n, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharex=True, sharey=True)
    for ax, pts, title in ((axes[0], mc, f"Monte Carlo (n={n})"),
                           (axes[1], lhs, f"Latin Hypercube (n={n})")):
        ax.scatter(pts[:, 0], pts[:, 1], s=10, alpha=0.5, edgecolors="none")
        ax.set_title(title)
        ax.set_xlabel("level multiplier")
    axes[0].set_ylabel("annual escalation rate")
    fig.suptitle("Sampling coverage of the electricity import-price space")
    fig.tight_layout()
    fig.savefig(ax_path, dpi=150)
    plt.close(fig)

def _fan_chart(out_path: str, scenarios: list, n: int = DEFAULT_N_SAMPLES,
               horizon: int = HORIZON_YEARS, seed: int = DEFAULT_SEED) -> None:
    import matplotlib.pyplot as plt
    draws = lhs_samples(n, seed=seed)
    paths = np.array([price_multiplier_path(l, g, horizon) for l, g in draws])
    yrs = np.arange(horizon)
    fig, ax = plt.subplots(figsize=(8, 5))
    for p in (5, 25, 50, 75, 95):
        ax.plot(yrs, np.percentile(paths, p, axis=0), color="0.6", lw=0.8)
    ax.fill_between(yrs, np.percentile(paths, 5, axis=0), np.percentile(paths, 95, axis=0),
                    color="0.85", label="LHS 5-95th pct")
    for s in scenarios:
        ax.plot(yrs, s.path, lw=2, label=f"{s.id} (w={s.weight:.2f})")
    ax.set_xlabel("horizon year"); ax.set_ylabel("import-price multiplier vs Central yr-0")
    ax.set_title("Electricity import-price scenarios (LHS envelope + reduced set)")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout(); fig.savefig(out_path, dpi=150); plt.close(fig)

def main():
    import os
    out_dir = os.path.join("outputs", "uncertainty")
    os.makedirs(out_dir, exist_ok=True)
    scen = generate_price_scenarios()
    tbl  = scenario_table(scen)
    print("Reduced electricity import-price scenarios:")
    print(tbl.to_string(index=False))
    print("\nValidation:", validate(scen))
    tbl.to_csv(os.path.join(out_dir, "price_scenarios.csv"), index=False)
    _coverage_figure(os.path.join(out_dir, "lhs_vs_mc_coverage.png"))
    _fan_chart(os.path.join(out_dir, "price_scenario_fan.png"), scen)
    print(f"\nWrote scenario table + figures to {out_dir}")


if __name__ == "__main__":
    main()
