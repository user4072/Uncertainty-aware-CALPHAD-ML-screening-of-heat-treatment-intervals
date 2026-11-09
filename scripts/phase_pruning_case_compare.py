"""Second-level diagnostic of dataset id 35 (the 18.9 K pruning shift).

The shift reproduces with the production protocol, but a 5 K scan of the
twelve-phase list shows no excluded phase above 1e-4 between 1420 and
1490 K.  This script therefore compares the six- and twelve-phase gamma-prime
fractions and Gibbs energies on the same 5 K grid, and prints every phase
down to 1e-10, to decide whether the shift is caused by an excluded phase
present at trace level or by the minimiser selecting different branches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import equilibrium_curve  # noqa: E402

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

    T = np.arange(1435.0, 1475.0 + 1e-9, 5.0)
    out = {"T_K": T}
    for label, phases in (("six", SIX), ("full", FULL)):
        curve = equilibrium_curve(db, comp, T, cfg["elements"], phases,
                                  pdens=cfg["pdens_solid"], output_hm=True)
        Tc = np.asarray(curve["T"], dtype=float)
        gp = np.asarray(curve["frac"].get("GAMMA_PRIME", np.zeros_like(Tc)))
        fcc = np.asarray(curve["frac"].get("FCC_A1", np.zeros_like(Tc)))
        G = np.asarray(curve.get("G", np.zeros_like(Tc)))
        out[f"{label}_gp"] = gp
        out[f"{label}_fcc"] = fcc
        out[f"{label}_G"] = G
        print(f"--- {label} ---")
        for i, t in enumerate(Tc):
            others = {p: float(np.asarray(f)[i]) for p, f in curve["frac"].items()
                      if float(np.asarray(f)[i]) > 1e-10}
            print(f" {t:7.1f} K  G={G[i]:.4f}  {others}")

    df = pd.DataFrame(out)
    df.to_csv(ROOT / "results" / "phase_pruning_case_compare.csv", index=False)
    print(df.round(5).to_string(index=False))


if __name__ == "__main__":
    main()
