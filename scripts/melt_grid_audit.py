"""Convergence of the matrix-only melting labels with respect to grid and density.

The production melting calculation uses a 10 K grid and pdens=150.  This
script recomputes the same matrix-only curve on a 2 K grid at pdens=400 for a
stratified sample and reports the change in solidus, liquidus and fusion
enthalpy.  These are the three labels that the design interval inherits;
their numerical tolerance is otherwise unmeasured.

Output: ``results/melt_grid_audit.csv``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import equilibrium_curve, extract_signatures  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    d = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    d = d.loc[(d["melt_valid"] == 1) & d["T_solidus"].notna()]
    sample = d.sample(12, random_state=11).assign(group="random")
    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    short = prod.assign(group="shortlist")
    dsc = pd.read_csv(ROOT / "data" / "dsc_literature.csv").assign(group="dsc")
    train = pd.read_csv(ROOT / "results" / "design_robustness.csv").iloc[:1]
    a1 = train.assign(group="A1")

    cols = [*FEATURES, "Ni", "group"]
    sample = sample.assign(id=sample.index)[["id", *cols]]
    short = short.assign(id=-np.arange(1, len(short) + 1))[["id", *cols]]
    dsc = dsc.assign(id=-100 - np.arange(len(dsc)))[["id", *cols]]
    a1 = a1.assign(id=-200)[["id", *cols]]
    # Recompute the production labels for every pick with the production
    # settings (10 K, pdens=150) so that the comparison is self-consistent.
    work = pd.concat([sample, short, dsc, a1], ignore_index=True)

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    T_prod = np.arange(cfg["temperature"]["melt_min"],
                       cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_fine = np.arange(cfg["temperature"]["melt_min"],
                       cfg["temperature"]["melt_max"] + 1e-9, 2.0)
    rows = []
    for _, r in work.iterrows():
        comp = {f: float(r[f]) for f in FEATURES}
        comp["Ni"] = float(r["Ni"])
        t0 = time.time()
        prod_curve = equilibrium_curve(
            db, comp, T_prod, cfg["elements"], ["LIQUID", "FCC_A1"],
            pdens=cfg["pdens_melt"])
        prod_sig = extract_signatures(prod_curve)
        fine_curve = equilibrium_curve(
            db, comp, T_fine, cfg["elements"], ["LIQUID", "FCC_A1"], pdens=400)
        fine_sig = extract_signatures(fine_curve)
        fine_liq = np.asarray(fine_curve["frac"].get("LIQUID", []), dtype=float)
        rows.append({
            "id": int(r["id"]), "group": r["group"], **comp,
            "T_solidus_10K": float(prod_sig["T_solidus"]),
            "T_liquidus_10K": float(prod_sig["T_liquidus"]),
            "dH_10K": float(prod_sig["dH_fusion_J_per_g"]),
            "T_solidus_2K": float(fine_sig["T_solidus"]),
            "T_liquidus_2K": float(fine_sig["T_liquidus"]),
            "dH_2K": float(fine_sig["dH_fusion_J_per_g"]),
            "d_solidus_K": float(fine_sig["T_solidus"] - prod_sig["T_solidus"]),
            "d_liquidus_K": float(fine_sig["T_liquidus"] - prod_sig["T_liquidus"]),
            "d_dH": float(fine_sig["dH_fusion_J_per_g"] - prod_sig["dH_fusion_J_per_g"]),
            # Convergence diagnostics on the fine curve: the largest one-step
            # change in liquid fraction, and the number of isolated
            # single-phase-liquid points caught by the repair pass.  A smooth
            # melting curve has a small maximum jump; a solver branch failure
            # produces a jump of order one.
            "max_dliq": float(np.max(np.abs(np.diff(fine_liq)))) if len(fine_liq) > 1 else np.nan,
            "liquid_repairs": int(fine_curve.get("liquid_branch_repaired", 0) or 0),
            "seconds": round(time.time() - t0, 1),
        })
        q = rows[-1]
        print(f"{q['group']:>9} id={q['id']:>4}: dTsol {q['d_solidus_K']:+.2f} K, "
              f"dTliq {q['d_liquidus_K']:+.2f} K, dH {q['d_dH']:+.2f} J/g "
              f"({q['seconds']:.0f} s)")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "melt_grid_audit.csv", index=False)
    for name, sub in out.groupby("group"):
        print(f"{name}: n={len(sub)} max|dTsol|={sub.d_solidus_K.abs().max():.2f} K, "
              f"max|dTliq|={sub.d_liquidus_K.abs().max():.2f} K, "
              f"max|dH|={sub.d_dH.abs().max():.1f} J/g, "
              f"max step dliq={sub.max_dliq.max():.3f}, "
              f"repairs={int(sub.liquid_repairs.sum())}")
    print("Wrote results/melt_grid_audit.csv")


if __name__ == "__main__":
    main()
