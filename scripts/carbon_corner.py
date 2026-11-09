"""Carbon-corner audit of the best designed alloy.

The design objective is free to drive C to its lower bound.  Carbon is an
interstitial that depresses the matrix solidus in the CALPHAD database, so a
carbon-free corner can buy window at the edge of the composition box.  This
script scans the carbon content of the best train-only design (A1) and of the
best production candidate at fixed base composition, and reports the matrix
solidus, liquidus and fusion enthalpy.  It is a sensitivity check, not a
design run: the other five solutes are held fixed.

Output: ``results/carbon_corner.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycalphad import Database  # noqa: E402

from calphad_engine import equilibrium_curve, extract_signatures  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    db = Database(ROOT / cfg["database"])
    robust = pd.read_csv(ROOT / "results" / "design_robustness.csv")
    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    a1 = robust.sort_values("W_calphad_K", ascending=False).iloc[0]
    p1 = prod.sort_values("W_calphad_K", ascending=False).iloc[0]

    T = np.arange(1540.0, 1660.0 + 1e-9, 2.0)
    carbon = (0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.15)
    rows = []
    for tag, ref in (("A1", a1), ("P1", p1)):
        base = {f: float(ref[f]) for f in FEATURES}
        for c in carbon:
            comp = dict(base)
            comp["C"] = c
            comp["Ni"] = 100.0 - sum(comp[f] for f in FEATURES)
            curve = equilibrium_curve(
                db, comp, T, cfg["elements"], ("LIQUID", "FCC_A1"),
                pdens=cfg["pdens_melt"], output_hm=True,
            )
            sig = extract_signatures(curve)
            rows.append({
                "alloy": tag,
                "C_wt": c,
                "T_solidus": float(sig.get("T_solidus", np.nan)),
                "T_liquidus": float(sig.get("T_liquidus", np.nan)),
                "dH_fusion_J_per_g": float(sig.get("dH_fusion_J_per_g", np.nan)),
                "valid": bool(sig.get("valid", False)),
                "W_calphad_K": float(sig.get("T_solidus", np.nan)
                                     - float(ref["T_gp_solvus_calphad"])),
            })
            print(f"{tag} C={c:.3f}: Tsol={rows[-1]['T_solidus']:.1f} "
                  f"Tliq={rows[-1]['T_liquidus']:.1f} "
                  f"dH={rows[-1]['dH_fusion_J_per_g']:.1f} "
                  f"W={rows[-1]['W_calphad_K']:.1f}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "carbon_corner.csv", index=False)
    for tag in ("A1", "P1"):
        sub = df.loc[df["alloy"] == tag].sort_values("C_wt")
        drop = float(sub.iloc[0]["T_solidus"] - sub.loc[sub["C_wt"] == 0.02, "T_solidus"].iloc[0])
        print(f"{tag}: solidus drop from C=0 to C=0.02 wt%: {drop:.1f} K")


if __name__ == "__main__":
    main()
