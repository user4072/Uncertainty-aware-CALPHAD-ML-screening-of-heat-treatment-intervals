"""Applicability-domain audit for the four literature DSC alloys.

The surrogate is defined on Ni-Co-Cr-Al-Mo-Ti-C.  An alloy is chemically
outside that space if it contains solutes the database does not represent
(V, Nb, Ta, W, ...).  It can also be compositionally remote inside the six
represented solutes.  This script reports both: the unrepresented solute mass
and the nearest-neighbour distance to the training set in the six-solute
space.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SOLUTES = ["Co", "Cr", "Al", "Mo", "Ti", "C"]
# Grain-boundary elements (B, Zr) are outside the seven-component model but act at the
# 0.01-0.1 wt% level and do not move the melting range.  A minor refractory
# addition (V in IN100, 0.8 wt%) is treated as represented-by-projection; a
# deliberate 2-5 wt% addition of Nb, Ta or W is not.
TRACE_WT = 0.1
MINOR_WT = 1.0


def main() -> None:
    sig = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    train = sig[sig["split"] == "train"].dropna(subset=SOLUTES).reset_index(drop=True)
    X = train[SOLUTES].to_numpy(float)
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    Z = (X - mu) / sd

    def knn_dist(row: pd.Series) -> float:
        z = (row[SOLUTES].to_numpy(float) - mu) / sd
        return float(np.min(np.linalg.norm(Z - z, axis=1)))

    # Training self-distances define the 99th-percentile radius.
    self_d = np.array([np.min(np.linalg.norm(Z - z, axis=1) + 1e9 * (np.arange(len(Z)) == i))
                       for i, z in enumerate(Z)])
    radius = float(np.percentile(self_d, 99.0))

    omitted = pd.read_csv(ROOT / "data" / "dsc_omitted_elements.csv")
    lit = pd.read_csv(ROOT / "data" / "dsc_literature.csv")
    dsc = pd.read_csv(RESULTS / "dsc_validation.csv")
    merged = (dsc.merge(lit[["alloy"] + SOLUTES], on="alloy", how="left")
                 .merge(omitted, on="alloy", how="left"))
    rows = []
    for _, r in merged.iterrows():
        dist = knn_dist(r)
        pct = float((self_d < dist).mean() * 100.0)
        omitted = float(r.get("total_wt", 0.0))
        if omitted == 0.0:
            omitted_class = "none"
        elif omitted <= TRACE_WT:
            omitted_class = "trace"
        elif omitted <= MINOR_WT:
            omitted_class = "minor"
        else:
            omitted_class = "projection"
        rows.append({
            "alloy": r["alloy"],
            "role": r.get("role", "calibration"),
            "omitted_wt": omitted,
            "omitted_note": r.get("note", ""),
            "omitted_class": omitted_class,
            "knn_dist_6d": dist,
            "knn_percentile": pct,
            "inside_knn_radius": bool(dist <= radius),
            "chemically_in_domain": bool(omitted <= MINOR_WT),
            "trace_only": bool(0.0 < omitted <= TRACE_WT),
            "solidus_error_K": float(r["T_solidus_calphad"] - r["T_solidus_dsc"]),
            "liquidus_error_K": float(r["T_liquidus_calphad"] - r["T_liquidus_dsc"]),
            "dH_error_J_per_g": (float(r["dH_fusion_J_per_g_calphad"] - r["dH_fusion_J_per_g_dsc"])
                                 if pd.notna(r.get("dH_fusion_J_per_g_dsc")) else float("nan")),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "dsc_domain.csv", index=False)
    summary = {
        "solutes": SOLUTES,
        "n_train": int(len(train)),
        "knn_radius_p99": radius,
        "trace_threshold_wt": TRACE_WT,
        "minor_threshold_wt": MINOR_WT,
        "alloys": out.to_dict(orient="records"),
        "note": ("chemically_in_domain requires the unrepresented mass to stay at or below "
                 f"{MINOR_WT} wt%; additions up to {TRACE_WT} wt% are classed as trace. "
                 "V, Nb, Ta and W are counted; trace B and Zr are reported but do not "
                 "remove an alloy from the space."),
    }
    with open(RESULTS / "dsc_domain.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(out.to_string(index=False))
    print("radius p99", round(radius, 3))


if __name__ == "__main__":
    main()
