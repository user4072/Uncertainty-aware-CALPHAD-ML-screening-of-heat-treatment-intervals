"""Production-versus-higher-density audit of the solid-state labels.

The production dataset uses pdens=400 for the six-phase solid calculation.
This script recomputes a stratified sample of alloys (repaired and unrepaired)
and the six shortlisted design alloys at pdens=800 and reports the change in
the gamma-prime solvus and in the 900 C fraction.

Output: ``results/density800_audit.csv``.
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

from calphad_engine import compute_alloy  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    d = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    base = (d["melt_valid"] == 1) & (d["solid_valid"] == 1) & d["T_gp_solvus"].notna()
    rep = d.loc[base & (d["solvus_refined"].fillna(0) == 1)]
    unrep = d.loc[base & (d["solvus_refined"].fillna(0) != 1)]
    rng = np.random.default_rng(5)
    picks = pd.concat([
        rep.sample(min(12, len(rep)), random_state=5).assign(group="repaired"),
        unrep.sample(min(12, len(unrep)), random_state=6).assign(group="unrepaired"),
    ])

    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    short = prod[[*FEATURES, "Ni", "T_gp_solvus_calphad", "gp900_calphad"]].copy()
    short = short.rename(columns={"T_gp_solvus_calphad": "T_gp_solvus",
                                  "gp900_calphad": "gp_frac_900C"})
    short["id"] = -np.arange(1, len(short) + 1)
    short["group"] = "shortlist"
    cols = ["id", "group", *FEATURES, "Ni", "T_gp_solvus", "gp_frac_900C"]
    work = pd.concat([picks[cols], short[cols]], ignore_index=True)

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    rows = []
    for _, r in work.iterrows():
        comp = {f: float(r[f]) for f in FEATURES}
        comp["Ni"] = float(r["Ni"])
        t0 = time.time()
        rec = compute_alloy(
            db, comp, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
            pdens_melt=cfg["pdens_melt"], pdens_solid=800,
            refine=cfg.get("refine_solvus", True),
        )
        rows.append({
            "id": int(r["id"]), "group": r["group"], **comp,
            "T_gp_solvus_400": float(r["T_gp_solvus"]),
            "T_gp_solvus_800": float(rec["T_gp_solvus"]),
            "d_solvus_K": float(rec["T_gp_solvus"] - r["T_gp_solvus"]),
            "f900_400": float(r["gp_frac_900C"]),
            "f900_800": float(rec["gp_frac_900C"]),
            "d_f900": float(rec["gp_frac_900C"] - r["gp_frac_900C"]),
            "seconds": round(time.time() - t0, 1),
        })
        print(f"{r['group']:>10} id={int(r['id']):>4}: solvus "
              f"{rows[-1]['d_solvus_K']:+.2f} K, f900 {rows[-1]['d_f900']:+.4f} "
              f"({rows[-1]['seconds']:.0f} s)")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "density800_audit.csv", index=False)
    for name, sub in out.groupby("group"):
        print(f"{name}: n={len(sub)} max|dsolvus|={sub.d_solvus_K.abs().max():.2f} K, "
              f"max|df|={sub.d_f900.abs().max():.4f}")
    print("Wrote results/density800_audit.csv")


if __name__ == "__main__":
    main()
