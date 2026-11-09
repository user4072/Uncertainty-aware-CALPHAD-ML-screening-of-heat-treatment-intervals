"""Side-by-side numerical resolution and regression error for every target.

The review asks that numerical resolution and regression error be reported
side by side for every thermodynamic target, because the liquidus surrogate
error (3.31 K) is smaller than the median movement of its own label under
grid refinement (5.9 K) and the interpretation should not hide that.

Sources, all previously computed:
* melt grid 10 K -> 2 K: ``results/melt_grid_audit.csv`` (23 alloys);
* solvus coarse 25 K -> full-bracket 2 K: ``results/solvus_refine_check.csv``
  (stored-vs-corrected shift for the 410 uncensored labels);
* minimiser density 150 -> 400: ``results/numerical_audit.json``;
* phase list 6 -> 12: ``results/phase_pruning_solvus_expanded.json``;
* GP held-out RMSE: ``results/surrogate_metrics.csv``;
* GP independent-test RMSE: ``results/independent_metrics.csv``.

Output: ``results/resolution_vs_error.json``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def main() -> None:
    melt = pd.read_csv(RESULTS / "melt_grid_audit.csv")
    refine = json.loads((RESULTS / "solvus_correction_audit.json").read_text(encoding="utf-8"))
    audit = json.loads((RESULTS / "numerical_audit.json").read_text(encoding="utf-8"))
    pruning = json.loads((RESULTS / "phase_pruning_solvus_expanded.json").read_text(encoding="utf-8"))
    metrics = pd.read_csv(RESULTS / "surrogate_metrics.csv").set_index("target")
    indep = pd.read_csv(RESULTS / "independent_metrics.csv").set_index("target")

    shift = None

    rows = [
        {
            "target": "T_gp_solvus",
            "refinement": "solid-state grid 25 K -> full-bracket 2 K",
            "n": int(refine["n_compared"]),
            "median_movement_K": float(abs(refine["median_shift_K"])),
            "p90_abs_movement_K": float(-refine["min_shift_K"]),
            "frac_gt_10K": float(refine["n_gt10"]) / float(refine["n_compared"]),
            "gp_rmse_heldout_K": float(metrics.loc["T_gp_solvus", "gp_rmse"]),
            "gp_rmse_independent_K": float(indep.loc["T_gp_solvus", "rmse_gp"]),
        },
        {
            "target": "T_solidus",
            "refinement": "melting grid 10 K -> 2 K",
            "n": int(len(melt)),
            "median_movement_K": float(melt["d_solidus_K"].abs().median()),
            "p90_abs_movement_K": float(melt["d_solidus_K"].abs().quantile(0.9)),
            "frac_gt_10K": float((melt["d_solidus_K"].abs() > 10).mean()),
            "gp_rmse_heldout_K": float(metrics.loc["T_solidus", "gp_rmse"]),
            "gp_rmse_independent_K": float(indep.loc["T_solidus", "rmse_gp"]),
        },
        {
            "target": "T_liquidus",
            "refinement": "melting grid 10 K -> 2 K",
            "n": int(len(melt)),
            "median_movement_K": float(melt["d_liquidus_K"].abs().median()),
            "p90_abs_movement_K": float(melt["d_liquidus_K"].abs().quantile(0.9)),
            "frac_gt_10K": float((melt["d_liquidus_K"].abs() > 10).mean()),
            "gp_rmse_heldout_K": float(metrics.loc["T_liquidus", "gp_rmse"]),
            "gp_rmse_independent_K": float(indep.loc["T_liquidus", "rmse_gp"]),
        },
        {
            "target": "dH_fusion_J_per_g",
            "refinement": "melting grid 10 K -> 2 K",
            "n": int(len(melt)),
            "median_movement_K": float(melt["d_dH"].abs().median()),
            "p90_abs_movement_K": float(melt["d_dH"].abs().quantile(0.9)),
            "frac_gt_10K": float((melt["d_dH"].abs() > 10).mean()),
            "gp_rmse_heldout_K": float(metrics.loc["dH_fusion_J_per_g", "gp_rmse"]),
            "gp_rmse_independent_K": float(indep.loc["dH_fusion_J_per_g", "rmse_gp"]),
        },
        {
            "target": "gp_frac_900C",
            "refinement": "minimiser density 150 -> 400",
            "n": int(audit["label_changes_150_to_400"]["gp_frac_900C"]["n"]),
            "median_movement_K": float(audit["label_changes_150_to_400"]["gp_frac_900C"]["median"]),
            "p90_abs_movement_K": float(np.nanpercentile(
                np.abs(audit["label_changes_150_to_400"]["gp_frac_900C"]["p95"]), 0)
                if False else np.nan),
            "frac_gt_10K": float(audit["label_changes_150_to_400"]["gp_frac_900C"]["frac_abs_gt"]["0.1"]),
            "gp_rmse_heldout_K": float(metrics.loc["gp_frac_900C", "gp_rmse"]),
            "gp_rmse_independent_K": float(indep.loc["gp_frac_900C", "rmse_gp"]),
        },
    ]
    # Solvus density sensitivity and phase-list sensitivity, for the text.
    extra = {
        "solvus_pdens_150_to_400": audit["label_changes_150_to_400"]["T_gp_solvus"],
        "solvus_phase_list_6_to_12": pruning,
    }
    out = {"targets": rows, "extra": extra}
    (RESULTS / "resolution_vs_error.json").write_text(json.dumps(out, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
