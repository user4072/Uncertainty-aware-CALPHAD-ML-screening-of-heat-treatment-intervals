"""Predictive-interval calibration at several nominal levels.

The manuscript reports a single coverage number (PICP95) per target.  A
reviewer correctly noted that high coverage can be achieved by very wide
intervals and that a single nominal level does not demonstrate calibration.
This script reports, for every scalar target and for the derived heat-treatment
window,

* empirical coverage at nominal levels 50, 68, 80, 90, 95 and 99 %,
* a Wilson 95 % confidence interval for each coverage value (the test set is
  small, so this is reported explicitly),
* the mean and median width of the 95 % interval,
* the width relative to the spread (standard deviation) of the test targets.

Only held-out predictions are used.  No calibration factor is fitted here; the
per-fold factor used by the surrogate is described in the manuscript.

Output: ``results/interval_calibration.csv`` and ``results/interval_calibration.json``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]

TARGETS = {
    "T_gp_solvus": "T_solvus (K)",
    "T_solidus": "T_solidus (K)",
    "T_liquidus": "T_liquidus (K)",
    "dH_fusion_J_per_g": "dH_fusion (J/g)",
    "gp_frac_900C": "f_gamma' (900 C)",
}

LEVELS = (0.50, 0.68, 0.80, 0.90, 0.95, 0.99)


def wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / denom
    return (centre - half, centre + half)


def calibrate(y: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> dict:
    ok = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd) & (sd > 0)
    y, mu, sd = y[ok], mu[ok], sd[ok]
    z = (y - mu) / sd
    n = int(len(z))
    rows = {}
    for level in LEVELS:
        q = norm.ppf(0.5 + level / 2.0)
        k = int(np.sum(np.abs(z) <= q))
        lo, hi = wilson(k, n)
        rows[level] = {
            "nominal": level,
            "empirical": k / n if n else float("nan"),
            "n_inside": k,
            "n": n,
            "wilson_lo": lo,
            "wilson_hi": hi,
            "mean_width": float(2.0 * q * np.mean(sd)) if n else float("nan"),
        }
    q95 = norm.ppf(0.975)
    return {
        "n": n,
        "levels": rows,
        "z_mean": float(np.mean(z)) if n else float("nan"),
        "z_std": float(np.std(z, ddof=1)) if n > 1 else float("nan"),
        "width95_mean": float(2.0 * q95 * np.mean(sd)) if n else float("nan"),
        "width95_median": float(2.0 * q95 * np.median(sd)) if n else float("nan"),
        "target_sd": float(np.std(y, ddof=1)) if n > 1 else float("nan"),
    }


def main() -> None:
    pred = pd.read_csv(ROOT / "results" / "surrogate_test_predictions.csv")
    out = {}

    table_rows = []
    for target, label in TARGETS.items():
        y = pred[f"{target}_dsc"].to_numpy(float)
        mu = pred[f"{target}_gp"].to_numpy(float)
        sd = pred[f"{target}_gp_std"].to_numpy(float)
        res = calibrate(y, mu, sd)
        out[target] = res
        for level, r in res["levels"].items():
            table_rows.append({
                "target": target,
                "label": label,
                "nominal": level,
                "empirical": r["empirical"],
                "wilson_lo": r["wilson_lo"],
                "wilson_hi": r["wilson_hi"],
                "mean_width": r["mean_width"],
                "n": r["n"],
            })
        print(f"{label:22s} n={res['n']:3d}  z={res['z_mean']:+.2f}  "
              f"sd_z={res['z_std']:.2f}  PICP95={res['levels'][0.95]['empirical']:.3f}  "
              f"width95={res['width95_mean']:.3g}  target sd={res['target_sd']:.3g}")

    # Derived window: W = T_solidus - T_solvus, with the measured residual
    # correlation rho = 0.098 already used in the manuscript.
    rho = 0.098
    s = pred["T_solidus_gp"].to_numpy(float)
    v = pred["T_gp_solvus_gp"].to_numpy(float)
    ss = pred["T_solidus_gp_std"].to_numpy(float)
    sv = pred["T_gp_solvus_gp_std"].to_numpy(float)
    w_mu = s - v
    w_sd = np.sqrt(ss ** 2 + sv ** 2 - 2.0 * rho * ss * sv)
    w_y = pred["T_solidus_dsc"].to_numpy(float) - pred["T_gp_solvus_dsc"].to_numpy(float)
    res = calibrate(w_y, w_mu, w_sd)
    out["window"] = res
    for level, r in res["levels"].items():
        table_rows.append({
            "target": "window",
            "label": "W = Tsol - Tsolvus (K)",
            "nominal": level,
            "empirical": r["empirical"],
            "wilson_lo": r["wilson_lo"],
            "wilson_hi": r["wilson_hi"],
            "mean_width": r["mean_width"],
            "n": r["n"],
        })
    print(f"{'W = Tsol - Tsolvus':22s} n={res['n']:3d}  z={res['z_mean']:+.2f}  "
          f"sd_z={res['z_std']:.2f}  PICP95={res['levels'][0.95]['empirical']:.3f}  "
          f"width95={res['width95_mean']:.3g}  target sd={res['target_sd']:.3g}")

    pd.DataFrame(table_rows).to_csv(ROOT / "results" / "interval_calibration.csv", index=False)
    (ROOT / "results" / "interval_calibration.json").write_text(json.dumps(out, indent=2))
    print("\nWrote results/interval_calibration.{csv,json}")


if __name__ == "__main__":
    main()
