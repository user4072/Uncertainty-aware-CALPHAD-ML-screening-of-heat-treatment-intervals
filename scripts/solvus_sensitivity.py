"""Sensitivity of the design window to the unanchored gamma-prime solvus.

The DSC anchors constrain solidus, liquidus and fusion enthalpy, not the
equilibrium gamma-prime solvus.  This script separates two cases:

* a common-mode solvus bias (the same shift for the designed alloy and for
  Rene 77), which cancels from the window gain but changes the absolute
  window;
* a differential solvus bias, which changes the gain.

It also propagates the measured solidus discrepancy and an assumed solvus
discrepancy through a Monte Carlo to report the probability that the gain
remains positive.  The solvus sigma is a sensitivity prior, not a measured
uncertainty, and is labelled as such.

The solidus sigma is the standard deviation of the signed residuals over the
reference anchors, not their mean absolute error: the mean absolute error of a
two- or three-point set is not a standard deviation, and using it understates
the scatter.  It is read from the error-budget output so that the two analyses
cannot drift apart.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SIGMA_SOLIDUS_FALLBACK = 16.633135568285223  # K, legacy three-anchor MAE
SIGMA_GRID = [0.0, 10.0, 20.0, 30.0, 40.0]
N_DRAWS = 200_000
SEED = 20261030


def measured_sigma_solidus() -> tuple[float, float, int]:
    """Solidus residual mean, standard deviation and anchor count."""
    budget = json.loads((RESULTS / "dsc_error_budget.json").read_text(encoding="utf-8"))
    tgt = budget["targets"]["T_solidus"]
    n = int(tgt["n_anchors"])
    sd = float(tgt["sd_residual"])
    if not np.isfinite(sd) or n < 2:
        return 0.0, SIGMA_SOLIDUS_FALLBACK, n
    return float(tgt["mean_residual"]), sd, n


def main() -> None:
    design = pd.read_csv(RESULTS / "design_validated.csv")
    prod = design.loc[design["W_calphad_K"].idxmax()]
    robust = json.loads((RESULTS / "design_robustness.json").read_text(encoding="utf-8"))
    rene = robust["rene77"]

    tsol_d = float(prod["T_solidus_calphad"])
    tgp_d = float(prod["T_gp_solvus_calphad"])
    tsol_r = float(rene["T_solidus"])
    tgp_r = float(rene["T_gp_solvus"])
    w_d = tsol_d - tgp_d
    w_r = tsol_r - tgp_r
    gain = w_d - w_r

    # Common-mode shift: both solvi move by the same amount.
    common = []
    for delta in (-80.0, -40.0, -20.0, 0.0, 20.0, 40.0, 80.0):
        common.append({
            "solvus_shift_K": delta,
            "W_design_K": w_d - delta,
            "W_rene77_K": w_r - delta,
            "gain_K": gain,
        })

    # Differential shift: designed solvus bias minus Rene 77 solvus bias.
    diff_rows = []
    for b_d in (-50.0, -25.0, 0.0, 25.0, 50.0):
        for b_r in (-50.0, -25.0, 0.0, 25.0, 50.0):
            diff_rows.append({
                "bias_design_K": b_d,
                "bias_rene77_K": b_r,
                "gain_K": gain + b_r - b_d,
            })

    # Monte Carlo: independent solidus errors and solvus errors per alloy.
    bias_sol, sigma_sol, n_anchors = measured_sigma_solidus()
    sigma_gp_W = 4.5  # K, held-out prediction error of the surrogate on W itself
    rng = np.random.default_rng(SEED)
    mc = []
    for sigma_gp in SIGMA_GRID:
        es_d = rng.normal(0.0, sigma_sol, N_DRAWS)
        es_r = rng.normal(0.0, sigma_sol, N_DRAWS)
        eg_d = rng.normal(0.0, sigma_gp, N_DRAWS) if sigma_gp else 0.0
        eg_r = rng.normal(0.0, sigma_gp, N_DRAWS) if sigma_gp else 0.0
        draws = gain + (es_d - es_r) - (eg_d - eg_r)
        mc.append({
            "solvus_sigma_K": sigma_gp,
            "gain_mean_K": float(np.mean(draws)),
            "gain_p05_K": float(np.percentile(draws, 5)),
            "gain_p50_K": float(np.percentile(draws, 50)),
            "gain_p95_K": float(np.percentile(draws, 95)),
            "prob_gain_positive": float(np.mean(draws > 0.0)),
        })

    # Absolute uncertainty of a single window: the same measured solidus
    # scatter, the assumed solvus prior and the surrogate's own error on W.
    sigma_abs = float(np.sqrt(sigma_sol**2 + 30.0**2 + sigma_gp_W**2))
    abs_rows = []
    rng_abs = np.random.default_rng(SEED + 1)
    for sigma_gp in SIGMA_GRID:
        draws = (
            w_d
            + bias_sol
            + rng_abs.normal(0.0, sigma_sol, N_DRAWS)
            + rng_abs.normal(0.0, sigma_gp_W, N_DRAWS)
            + (rng_abs.normal(0.0, sigma_gp, N_DRAWS) if sigma_gp else 0.0)
        )
        abs_rows.append({
            "solvus_sigma_K": sigma_gp,
            "W_mean_K": float(np.mean(draws)),
            "W_p05_K": float(np.percentile(draws, 5)),
            "W_p95_K": float(np.percentile(draws, 95)),
        })

    # Independent cross-check of the equilibrium solvus for Rene 77.
    # This work: 6-phase CALPHAD at the Binczyk composition.
    # Thermo-Calc TCNI10 equilibrium and DSC secondary-gamma' solvus from
    # Grudzien-Rakoczy et al., Sci. Rep. 15 (2025) 18237.
    solvus = {
        "our_calphad_K": tgp_r,
        "our_calphad_C": tgp_r - 273.15,
        "thermocalc_tcni10_C": 1139.0,
        "dsc_secondary_C": 1229.0,
        "our_minus_tcni10_K": (tgp_r - 273.15) - 1139.0,
        "dsc_minus_tcni10_K": 1229.0 - 1139.0,
        "dsc_minus_our_K": 1229.0 - (tgp_r - 273.15),
    }

    summary = {
        "design_alloy": {
            "rank": int(prod["rank"]),
            "T_solidus_K": tsol_d,
            "T_gp_solvus_K": tgp_d,
            "W_K": w_d,
        },
        "rene77": {
            "T_solidus_K": tsol_r,
            "T_gp_solvus_K": tgp_r,
            "W_K": w_r,
        },
        "gain_K": gain,
        "common_mode": common,
        "differential": diff_rows,
        "monte_carlo": mc,
        "absolute": abs_rows,
        "solvus_definitions": solvus,
        "sigma_solidus_bias_K": bias_sol,
        "sigma_solidus_K": sigma_sol,
        "sigma_solidus_n": n_anchors,
        "sigma_W_absolute_K": sigma_abs,
        "sigma_gp_W_K": sigma_gp_W,
        "sigma_solvus_grid_K": SIGMA_GRID,
        "n_draws": N_DRAWS,
        "seed": SEED,
        "note": "Solvus sigma is a sensitivity prior, not a measured uncertainty; "
                "the solidus sigma is the measured residual standard deviation "
                "over the reference anchors.",
    }

    pd.DataFrame(common).to_csv(RESULTS / "solvus_sensitivity_common.csv", index=False)
    pd.DataFrame(diff_rows).to_csv(RESULTS / "solvus_sensitivity_differential.csv", index=False)
    pd.DataFrame(mc).to_csv(RESULTS / "solvus_sensitivity_monte_carlo.csv", index=False)
    pd.DataFrame(abs_rows).to_csv(RESULTS / "solvus_sensitivity_absolute.csv", index=False)
    with open(RESULTS / "solvus_sensitivity.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print("design", prod["rank"], "W", round(w_d, 2), "Rene", round(w_r, 2), "gain", round(gain, 2))
    print(pd.DataFrame(mc).to_string(index=False))
    print("solvus", {k: round(v, 2) if isinstance(v, float) else v for k, v in solvus.items()})


if __name__ == "__main__":
    main()
