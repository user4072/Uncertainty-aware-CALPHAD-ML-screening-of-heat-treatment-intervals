"""Expanded phase-pruning audit for the gamma-prime solvus.

The production solid-state calculation keeps six phases.  The earlier audit
tested the solvus on five random alloys; this script repeats the six-phase
versus twelve-phase comparison on a larger random sample plus a deliberate
high-C/high-Mo corner sample, where sigma and carbide phases are most likely
to appear and to displace gamma-prime.

Output: ``results/phase_pruning_solvus_expanded.csv`` and ``.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import compute_alloy  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")
FULL_PHASES = [
    "FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "SIGMA", "BCC_A2",
    "LIQUID", "M6C", "M7C3", "HCP_A3", "MU_PHASE", "LAVES",
]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    sig = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    sig = sig.loc[(sig["melt_valid"] == 1) & (sig["solid_valid"] == 1)].reset_index(drop=True)

    rng = np.random.default_rng(3)
    random_ids = set(sig.iloc[rng.choice(len(sig), size=10, replace=False)]["id"])
    corner = sig.assign(score=sig["C"] / sig["C"].max() + sig["Mo"] / sig["Mo"].max())
    corner = corner.sort_values("score", ascending=False).head(4)
    corner_ids = set(corner["id"])
    # The original five-alloy audit reported a 18.9 K solvus difference for id
    # 35.  Those four surviving ids are repeated here so that the discrepancy
    # is resolved with the current refinement protocol rather than left as an
    # unexplained bound.
    former_ids = {7, 35, 82, 122}
    ids = list(random_ids | corner_ids | former_ids)
    sample = sig.loc[sig["id"].isin(ids)].copy()
    sample["group"] = np.where(sample["id"].isin(corner_ids), "corner",
                               np.where(sample["id"].isin(former_ids), "former", "random"))
    print(f"Auditing {len(sample)} alloys ({len(corner_ids)} corner, "
          f"{len(former_ids)} former, "
          f"{len(sample) - len(corner_ids) - len(former_ids)} random)")

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    T_solid = np.arange(cfg["temperature"]["solid_min"],
                        cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    T_melt = np.arange(cfg["temperature"]["melt_min"],
                       cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    rows = []
    for _, row in sample.iterrows():
        comp = {f: float(row[f]) for f in FEATURES}
        comp["Ni"] = float(row["Ni"])
        t0 = time.time()
        rec = compute_alloy(
            db, comp, cfg["elements"], FULL_PHASES, T_melt, T_solid,
            pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
            refine=cfg.get("refine_solvus", True),
        )
        rows.append({
            "id": int(row["id"]), "group": row["group"], **comp,
            "C_plus_Mo": float(row["C"] + row["Mo"]),
            "T_solvus_six": float(row["T_gp_solvus"]),
            "T_solvus_full": float(rec["T_gp_solvus"]),
            "d_solvus_K": float(rec["T_gp_solvus"] - row["T_gp_solvus"]),
            "f900_six": float(row["gp_frac_900C"]),
            "f900_full": float(rec["gp_frac_900C"]),
            "d_f900": float(rec["gp_frac_900C"] - row["gp_frac_900C"]),
            "solid_valid_full": float(rec["solid_valid"]),
            "seconds": round(time.time() - t0, 1),
        })
        q = rows[-1]
        print(f"{q['group']:>7} id={q['id']:>4} C+Mo={q['C_plus_Mo']:.2f}: "
              f"solvus {q['T_solvus_six']:.1f} -> {q['T_solvus_full']:.1f} "
              f"({q['d_solvus_K']:+.1f} K), f900 {q['d_f900']:+.3f} "
              f"({q['seconds']:.0f} s)")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "phase_pruning_solvus_expanded.csv", index=False)
    summary = {
        "n": int(len(out)),
        "n_corner": int((out["group"] == "corner").sum()),
        "n_former": int((out["group"] == "former").sum()),
        "max_abs_d_solvus_K": float(out["d_solvus_K"].abs().max()),
        "max_abs_d_f900": float(out["d_f900"].abs().max()),
        "n_solvus_gt_5K": int((out["d_solvus_K"].abs() > 5.0).sum()),
        "n_solvus_gt_1K": int((out["d_solvus_K"].abs() > 1.0).sum()),
        "corner_max_abs_d_solvus_K": float(
            out.loc[out["group"] == "corner", "d_solvus_K"].abs().max()),
        "former_max_abs_d_solvus_K": float(
            out.loc[out["group"] == "former", "d_solvus_K"].abs().max())
        if (out["group"] == "former").any() else None,
        "worst": out.loc[out["d_solvus_K"].abs().idxmax()].to_dict(),
    }
    with open(ROOT / "results" / "phase_pruning_solvus_expanded.json", "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))


if __name__ == "__main__":
    main()
