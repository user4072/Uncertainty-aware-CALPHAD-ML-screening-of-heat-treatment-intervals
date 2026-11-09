"""Sensitivity grid of the design-gain probability to the discrepancy model.

The manuscript reports P(gain > 0) = 98.1% for the window gain of the rank-1
candidate over Rene 77.  The review asks that this be reported as a
conditional probability and that the effect of alternative discrepancy
distributions and covariance assumptions be shown.

This script propagates the three uncertainty sources through a Monte Carlo
over a grid of assumptions:

* solvus differential prior sigma_solv in {0, 10, 20, 30, 40, 50, 60} K
  (the 30 K production value sits inside the 7.5-82.5 K band spanned by the
  independent Thermo-Calc comparison and the calorimetric dissolution onset);
* correlation rho in {0, 0.25, 0.5, 0.75, 1.0} between the thermodynamic
  discrepancies of the designed alloy and of Rene 77 (rho = 1 is the
  common-mode limit, rho = 0 the fully independent limit);
* distribution family for each discrepancy term: normal, uniform with the
  same variance, and Student-t with 4 degrees of freedom scaled to unit
  variance.

The solidus term is the measured standard deviation of the four reference
residuals; the surrogate term is the calibrated predictive standard
deviation of W for the rank-1 candidate.  The gain uncertainty is the
difference of two independently drawn window errors; the correlation acts on
the thermodynamic components only, which is the case the review raises.

Outputs
-------
``results/uncertainty_grid.csv``
``results/uncertainty_grid.json``  boundary-vs-difference decomposition.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
N_DRAWS = 400_000
SEED = 20261031

SIGMA_SOLV_GRID = (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0)
RHO_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
FAMILIES = ("normal", "uniform", "t4")


def measured_sigma_solidus() -> float:
    budget = json.loads((RESULTS / "dsc_error_budget.json").read_text(encoding="utf-8"))
    return float(budget["targets"]["T_solidus"]["sd_residual"])


def draw(family: str, sigma: np.ndarray, rng) -> np.ndarray:
    """Unit-variance-preserving draws of the requested family."""
    if family == "normal":
        return rng.standard_normal(len(sigma)) * sigma
    if family == "uniform":
        return (rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), len(sigma))) * sigma
    if family == "t4":
        # Student-t(4) scaled to unit variance: var = nu/(nu-2) = 2.
        z = rng.standard_t(4, len(sigma)) / np.sqrt(2.0)
        return z * sigma
    raise ValueError(family)


def main() -> None:
    rng = np.random.default_rng(SEED)
    design = pd.read_csv(RESULTS / "design_validated.csv")
    prod = design.loc[design["W_calphad_K"].idxmax()]
    robust = json.loads((RESULTS / "design_robustness.json").read_text(encoding="utf-8"))
    rene = robust["rene77"]
    w_design = float(prod["W_calphad_K"])
    w_rene = float(rene["W_K"])
    gain = w_design - w_rene

    sigma_sol = measured_sigma_solidus()
    sigma_gp = float(prod["W_surrogate_std_K"])
    n = N_DRAWS

    rows = []
    for family in FAMILIES:
        for rho in RHO_GRID:
            # Common thermodynamic discrepancy components per window; the
            # surrogate term is always independent between the two windows.
            for sig_solv in SIGMA_SOLV_GRID:
                sig_c = np.sqrt(sigma_sol ** 2 + sig_solv ** 2)
                z_c1 = draw(family, np.full(n, sig_c), rng)
                z_c2 = rho * z_c1 + np.sqrt(max(0.0, 1.0 - rho ** 2)) * draw(family, np.full(n, sig_c), rng)
                z_gp1 = draw(family, np.full(n, sigma_gp), rng)
                z_gp2 = draw(family, np.full(n, sigma_gp), rng)
                err_design = z_c1 + z_gp1
                err_rene = z_c2 + z_gp2
                gain_draws = (w_design + err_design) - (w_rene + err_rene)
                rows.append({
                    "family": family, "rho": rho, "sigma_solv_K": sig_solv,
                    "P_gain_positive": float((gain_draws > 0).mean()),
                    "gain_p05_K": float(np.percentile(gain_draws, 5)),
                    "gain_p50_K": float(np.percentile(gain_draws, 50)),
                    "gain_sd_K": float(gain_draws.std(ddof=1)),
                    "sigma_window_K": float(sig_c),
                })
    grid = pd.DataFrame(rows)
    grid.to_csv(RESULTS / "uncertainty_grid.csv", index=False)

    # Boundary vs difference decomposition at the production assumptions.
    decomp = {
        "W_design_K": w_design,
        "W_rene77_K": w_rene,
        "gain_K": gain,
        "sigma_solidus_measured_K": sigma_sol,
        "sigma_solvus_prior_K": 30.0,
        "sigma_surrogate_W_K": sigma_gp,
        "sigma_one_window_K": float(np.sqrt(sigma_sol ** 2 + 30.0 ** 2 + sigma_gp ** 2)),
        "sigma_gain_independent_K": float(np.sqrt(2.0) * np.sqrt(sigma_sol ** 2 + 30.0 ** 2 + sigma_gp ** 2)),
        "sigma_gain_common_mode_K": float(np.sqrt((2 * sigma_gp) ** 2)),
        "note": ("Common-mode limit leaves the thermodynamic discrepancy in one window only "
                 "from the difference; the reported 99.8% figure corresponds to applying the "
                 "single-window sigma to the gain."),
    }
    # P(gain>0) under the production assumptions for each family (rho = 0).
    prod_rows = grid[(grid["rho"] == 0.0) & (grid["sigma_solv_K"] == 30.0)]
    decomp["P_gain_positive_by_family"] = {
        r["family"]: r["P_gain_positive"] for _, r in prod_rows.iterrows()
    }
    # The range across the whole evaluated grid.
    decomp["P_gain_positive_range"] = {
        "min": float(grid["P_gain_positive"].min()),
        "max": float(grid["P_gain_positive"].max()),
        "min_at": grid.loc[grid["P_gain_positive"].idxmin(),
                           ["family", "rho", "sigma_solv_K"]].to_dict(),
    }
    (RESULTS / "uncertainty_grid.json").write_text(json.dumps(decomp, indent=2))
    print(json.dumps(decomp, indent=2))


if __name__ == "__main__":
    main()
