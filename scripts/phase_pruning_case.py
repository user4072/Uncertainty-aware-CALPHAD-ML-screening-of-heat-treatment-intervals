"""Diagnose the single large phase-pruning solvus shift (dataset id 35).

The expanded audit found six- versus twelve-phase agreement to numerical
precision for fourteen alloys, including the four highest C+Mo compositions.
One earlier case (id 35, C 0.109, Mo 2.41, Al 6.36, Ti 5.47) had shown an
18.9 K reduction of the gamma-prime solvus when the excluded phases were
added.  This script repeats that comparison with the production protocol and,
if the shift reproduces, identifies which excluded phase is stable across the
two solvi.

Outputs ``results/phase_pruning_case.csv`` (phase fractions against
temperature for 1420--1490 K) and prints the six- and twelve-phase solvi.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import compute_alloy, equilibrium_curve  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")
SIX = ["FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "SIGMA", "BCC_A2"]
FULL = SIX + ["LIQUID", "M6C", "M7C3", "HCP_A3", "MU_PHASE", "LAVES"]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    sig = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    row = sig.loc[sig["id"] == 35].iloc[0]
    comp = {f: float(row[f]) for f in FEATURES}
    comp["Ni"] = float(row["Ni"])
    print("composition:", {k: round(v, 3) for k, v in comp.items()})

    T_solid = np.arange(cfg["temperature"]["solid_min"],
                        cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    T_melt = np.arange(cfg["temperature"]["melt_min"],
                       cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    for label, phases in (("six", SIX), ("full", FULL)):
        rec = compute_alloy(
            db, comp, cfg["elements"], phases, T_melt, T_solid,
            pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
            refine=cfg.get("refine_solvus", True),
        )
        print(f"{label:4s}: solvus {rec['T_gp_solvus']:.3f} K, "
              f"f900 {rec['gp_frac_900C']:.4f}, valid {rec['solid_valid']}")
        if label == "six":
            solvus_six = float(rec["T_gp_solvus"])
        else:
            solvus_full = float(rec["T_gp_solvus"])

    print(f"shift: {solvus_full - solvus_six:+.2f} K")

    # Which phases are present between the two solvi?
    T = np.arange(1420.0, 1490.0 + 1e-9, 5.0)
    curve = equilibrium_curve(db, comp, T, cfg["elements"], FULL,
                              pdens=cfg["pdens_solid"], output_hm=False)
    Tc_grid = np.asarray(curve["T"], dtype=float)
    rows = []
    for i, Tc in enumerate(Tc_grid):
        for phase, frac in curve["frac"].items():
            f = float(np.asarray(frac)[i])
            if f > 1e-4:
                rows.append({"T_K": float(Tc), "phase": phase, "fraction": f})
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "phase_pruning_case.csv", index=False)
    wide = out.pivot_table(index="T_K", columns="phase", values="fraction",
                           aggfunc="sum").fillna(0.0)
    print(wide.round(4).to_string())


if __name__ == "__main__":
    main()
