"""Error budget and bias identifiability for the reference melting anchors.

The four reference anchors are the two DSC measurements (Rene 77, IN100) and the
two manufacturer melting-range determinations (Nimonic 105, Nimonic 115).  Only
the two temperatures have four values; the fusion enthalpy has two, because the
manufacturer bulletins report a melting range only.

Separates three quantities for each target:

1. surrogate vs CALPHAD at the anchor compositions (regression error),
2. CALPHAD vs the reported value (database + melting-model discrepancy),
3. surrogate vs the reported value (the sum actually observed if the surrogate
   is used).

It also fits a constant bias model to the residuals and reports a 95% t-interval
for the mean residual.  A correction is only defensible if that interval
excludes zero and the residual is roughly constant across anchors.  The
leave-one-alloy-out (LOO) corrected errors are reproduced for continuity with
the manuscript, but they are undefined below three anchors and are a sensitivity
check rather than a calibration.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"

TARGETS = [
    ("T_solidus", "Solidus", "K"),
    ("T_liquidus", "Liquidus", "K"),
    ("dH_fusion_J_per_g", "Fusion enthalpy", "J/g"),
]


def loo_corrected_mae(obs: np.ndarray, pred: np.ndarray) -> float:
    """Leave-one-out constant-bias correction, as used in the manuscript."""
    resid = pred - obs
    errs = []
    for i in range(len(resid)):
        bias = np.delete(resid, i).mean()
        errs.append(abs((pred[i] - bias) - obs[i]))
    return float(np.mean(errs))


def main() -> None:
    dsc = pd.read_csv(RESULTS / "dsc_validation.csv")
    # Calibration and validation tiers together form the in-domain set; the
    # off-space projections are excluded from every statistic.
    indomain = dsc[dsc["role"].isin(("calibration", "validation"))].reset_index(drop=True)

    summary = {"n_anchors": int(len(indomain)), "targets": {}}
    rows = []
    for key, label, unit in TARGETS:
        # The manufacturer anchors report no fusion enthalpy, so the sample
        # size is target-specific.
        ok = (np.isfinite(indomain[f"{key}_dsc"].to_numpy(float))
              & np.isfinite(indomain[f"{key}_calphad"].to_numpy(float)))
        sub = indomain.loc[ok].reset_index(drop=True)
        n = len(sub)
        obs = sub[f"{key}_dsc"].to_numpy(float)
        calphad = sub[f"{key}_calphad"].to_numpy(float)
        gp = sub[f"{key}_gp"].to_numpy(float)
        resid = calphad - obs

        mean = float(resid.mean())
        sd = float(resid.std(ddof=1))
        se = sd / np.sqrt(n)
        tcrit = float(stats.t.ppf(0.975, n - 1))
        ci = (mean - tcrit * se, mean + tcrit * se)

        mae_calphad = float(np.mean(np.abs(resid)))
        mae_gp = float(np.mean(np.abs(gp - obs)))
        mae_gp_calphad = float(np.mean(np.abs(gp - calphad)))
        # A leave-one-out bias correction is only defined for n >= 3; with two
        # anchors it would replace each residual by the other one.
        loo = loo_corrected_mae(obs, calphad) if n >= 3 else float("nan")
        supported = bool(ci[0] > 0.0 or ci[1] < 0.0)
        share = 100.0 * mae_gp_calphad / mae_calphad if mae_calphad else float("nan")

        summary["targets"][key] = {
            "label": label,
            "unit": unit,
            "n_anchors": int(n),
            "alloys": sub["alloy"].tolist(),
            "observed": obs.tolist(),
            "calphad": calphad.tolist(),
            "gp": gp.tolist(),
            "residual_calphad_minus_dsc": resid.tolist(),
            "mae_calphad_vs_dsc": mae_calphad,
            "mae_gp_vs_dsc": mae_gp,
            "mae_gp_vs_calphad": mae_gp_calphad,
            "surrogate_share_pct": share,
            "mean_residual": mean,
            "sd_residual": sd,
            "se_residual": float(se),
            "ci95_mean_residual": [float(ci[0]), float(ci[1])],
            "constant_bias_supported": supported,
            "loo_corrected_mae": loo,
            "loo_changes_mae": float(loo - mae_calphad),
        }
        rows.append({
            "target": key,
            "unit": unit,
            "n_anchors": int(n),
            "mae_gp_vs_calphad": mae_gp_calphad,
            "mae_calphad_vs_dsc": mae_calphad,
            "mae_gp_vs_dsc": mae_gp,
            "surrogate_share_pct": share,
            "mean_residual": mean,
            "ci95_low": float(ci[0]),
            "ci95_high": float(ci[1]),
            "constant_bias_supported": supported,
            "loo_corrected_mae": loo,
            "loo_changes_mae": float(loo - mae_calphad),
        })

    pd.DataFrame(rows).to_csv(RESULTS / "dsc_error_budget.csv", index=False)
    with open(RESULTS / "dsc_error_budget.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)

    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
