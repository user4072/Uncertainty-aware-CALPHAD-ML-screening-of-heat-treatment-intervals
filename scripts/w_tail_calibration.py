"""Calibration of the predicted heat-treatment window on held-out alloys.

The design score is built from the difference of two predicted quantities, so
its error is not the error of either target.  This script forms
W = T_solidus - T_gp_solvus for every test alloy on which both targets are
available, compares it with the CALPHAD window, and reports the overall error,
the bias, the 95% coverage from the (diagonal) predictive standard deviation
and the same statistics restricted to the upper quartile and decile of the
window distribution---the tail in which the design operates.

Output: ``results/W_tail_calibration.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    pred = pd.read_csv(ROOT / "results" / "surrogate_test_predictions.csv")
    need = ["T_solidus_dsc", "T_solidus_gp", "T_solidus_gp_std",
            "T_gp_solvus_dsc", "T_gp_solvus_gp", "T_gp_solvus_gp_std"]
    d = pred.dropna(subset=need).copy()
    w_true = d["T_solidus_dsc"] - d["T_gp_solvus_dsc"]
    w_pred = d["T_solidus_gp"] - d["T_gp_solvus_gp"]
    sd = np.hypot(d["T_solidus_gp_std"], d["T_gp_solvus_gp_std"])
    err = w_pred - w_true
    cover = (err.abs() <= 1.959964 * sd)

    out = {
        "n": int(len(d)),
        "W_mae": float(err.abs().mean()),
        "W_bias": float(err.mean()),
        "W_rmse": float(np.sqrt((err ** 2).mean())),
        "W_sd": float(err.std(ddof=1)) if len(err) > 2 else float("nan"),
        "cov95_all": float(cover.mean()),
        "W_quantiles": {str(q): float(w_true.quantile(q)) for q in (0.5, 0.75, 0.9)},
    }
    for tag, q in (("top25", 0.75), ("top10", 0.90)):
        thr = w_true.quantile(q)
        m = w_true >= thr
        out[tag] = {
            "n": int(m.sum()),
            "threshold_K": float(thr),
            "mae": float(err[m].abs().mean()),
            "bias": float(err[m].mean()),
            "cov95": float(cover[m].mean()),
        }

    path = ROOT / "results" / "W_tail_calibration.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
