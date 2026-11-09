"""Quantify the effect of the corrected solvus-refinement window.

Compares the pre-correction dataset (``dataset_signatures.csv.prebracket.bak``)
with the regenerated one and writes ``results/solvus_correction_audit.csv``
plus a JSON summary that feeds the manuscript macros.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SIG = ROOT / "data" / "processed" / "dataset_signatures.csv"
BAK = ROOT / "data" / "processed" / "dataset_signatures.csv.prebracket.bak"


def main() -> None:
    new = pd.read_csv(SIG)
    old = pd.read_csv(BAK)
    cols = ["id", "T_gp_solvus", "T_solidus", "censored_gp_high", "solvus_refined",
            "gp_frac_900C", "solid_valid", "melt_valid", "Co", "Cr", "Al", "Mo",
            "Ti", "C", "Ni"]
    d = old[cols].merge(new[cols], on="id", suffixes=("_old", "_new"))
    both = (d["censored_gp_high_old"] == 0) & (d["censored_gp_high_new"] == 0) \
        & d["T_gp_solvus_old"].notna() & d["T_gp_solvus_new"].notna()
    d["d_solvus_K"] = d["T_gp_solvus_new"] - d["T_gp_solvus_old"]
    d["d_W_K"] = -d["d_solvus_K"]
    d.to_csv(ROOT / "results" / "solvus_correction_audit.csv", index=False)

    sub = d.loc[both]
    summary = {
        "n_total": int(len(d)),
        "n_compared": int(len(sub)),
        "n_unrefined_before": int(((d["solvus_refined_old"] == 0)
                                   & (d["censored_gp_high_old"] == 0)).sum()),
        "n_unrefined_after": int(((d["solvus_refined_new"] == 0)
                                  & (d["censored_gp_high_new"] == 0)).sum()),
        "n_censored_before": int((d["censored_gp_high_old"].fillna(1) == 1).sum()),
        "n_censored_after": int((d["censored_gp_high_new"].fillna(1) == 1).sum()),
        "median_shift_K": float(sub["d_solvus_K"].median()),
        "mean_shift_K": float(sub["d_solvus_K"].mean()),
        "min_shift_K": float(sub["d_solvus_K"].min()),
        "max_shift_K": float(sub["d_solvus_K"].max()),
        "pct_change_gt5": float((sub["d_solvus_K"].abs() > 5.0).mean() * 100.0),
        "n_gt1": int((sub["d_solvus_K"].abs() > 1.0).sum()),
        "n_gt5": int((sub["d_solvus_K"].abs() > 5.0).sum()),
        "n_gt10": int((sub["d_solvus_K"].abs() > 10.0).sum()),
        "n_gt15": int((sub["d_solvus_K"].abs() > 15.0).sum()),
        "n_gt20": int((sub["d_solvus_K"].abs() > 20.0).sum()),
        "W_median_increase_K": float(sub["d_W_K"].median()),
        "W_max_increase_K": float(sub["d_W_K"].max()),
        "frac_refined_after": float((d.loc[d["censored_gp_high_new"] == 0,
                                           "solvus_refined_new"] == 1).mean()),
    }
    with open(ROOT / "results" / "solvus_correction_audit.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
