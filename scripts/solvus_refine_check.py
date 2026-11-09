"""Quantify the solvus-refinement failure found on dataset id 35.

The production refinement places a 2 K grid at ``t_coarse +/- 10 K``, where
``t_coarse`` is the linear interpolation of the 25 K coarse bracket.  When the
gamma-prime fraction at the last coarse point above the threshold is small,
the interpolation overshoots the true dissolution temperature by more than
10 K, the fine grid contains no gamma-prime at all, and the code silently
keeps the coarse value (``solvus_refined = 0``).  This is not a phase-pruning
effect: id 35 reproduces the 18.9 K difference because the twelve-phase list
returns exactly zero gamma-prime at the 1450 K node, which moves the
interpolation one node down.

This script recomputes the solvus for a sample of alloys with a corrected
grid that spans the *whole* coarse bracket plus a 2 K margin, and compares it
with the stored label.  Output: ``results/solvus_refine_check.csv``.
"""

from __future__ import annotations

import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import equilibrium_curve, solvus_from_fractions  # noqa: E402

SIX = ["FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "SIGMA", "BCC_A2"]
FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")


def check_one(payload):
    comp, T_solid, elements, db_path, pdens = payload
    from pycalphad import Database

    db = Database(db_path)
    solid = equilibrium_curve(db, comp, T_solid, elements, SIX,
                              pdens=pdens, output_hm=False)
    T = np.asarray(solid["T"], dtype=float)
    gp = np.asarray(solid["frac"].get("GAMMA_PRIME", np.zeros(len(T))), dtype=float)
    fcc = np.asarray(solid["frac"].get("FCC_A1", np.ones(len(T))), dtype=float)
    t_coarse, cens = solvus_from_fractions(T, gp, fcc)
    if cens or not np.isfinite(t_coarse):
        return dict(t_coarse=t_coarse, t_fine=np.nan, gp_last=np.nan,
                    bracket_lo=np.nan, bracket_hi=np.nan, cens=cens)
    above = (gp > 1e-3) & (fcc >= 0.5)
    i = int(np.where(above)[0][-1])
    lo = T[i]
    hi = T[i + 1] if i + 1 < len(T) else T[-1]
    fine_T = np.arange(max(T[0], lo - 2.0), min(T[-1], hi + 2.0) + 0.5, 2.0)
    fine = equilibrium_curve(db, comp, fine_T, elements, SIX,
                             pdens=pdens, output_hm=False)
    fT = np.asarray(fine["T"], dtype=float)
    fgp = np.asarray(fine["frac"].get("GAMMA_PRIME", np.zeros(len(fT))), dtype=float)
    ffcc = np.asarray(fine["frac"].get("FCC_A1", np.ones(len(fT))), dtype=float)
    t_fine, cens_fine = solvus_from_fractions(fT, fgp, ffcc)
    return dict(t_coarse=t_coarse, t_fine=t_fine, gp_last=float(gp[i]),
                bracket_lo=float(lo), bracket_hi=float(hi), cens=cens_fine)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    sig = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    usable = sig.loc[(sig["melt_valid"] == 1) & (sig["solid_valid"] == 1)
                     & (sig["censored_gp_high"] == 0)].reset_index(drop=True)
    rng = np.random.default_rng(11)
    unref = usable.loc[usable["solvus_refined"] == 0]
    ref = usable.loc[usable["solvus_refined"] == 1]
    ids = list(rng.choice(unref["id"], size=12, replace=False)) + \
          list(rng.choice(ref["id"], size=4, replace=False)) + [35]
    sample = usable.loc[usable["id"].isin(ids)].copy()
    print(f"checking {len(sample)} alloys ({len(ids)-1} flagged unrefined + 4 refined + id 35)")

    T_solid = np.arange(cfg["temperature"]["solid_min"],
                        cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    db_path = str(ROOT / cfg["database"])
    payloads = []
    for _, row in sample.iterrows():
        comp = {f: float(row[f]) for f in FEATURES}
        comp["Ni"] = float(row["Ni"])
        payloads.append((comp, T_solid, cfg["elements"], db_path,
                         int(cfg["pdens_solid"])))

    with ProcessPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(check_one, payloads))

    rows = []
    for (_, row), res in zip(sample.iterrows(), results):
        stored = float(row["T_gp_solvus"])
        t_fine = res["t_fine"] if np.isfinite(res["t_fine"]) else res["t_coarse"]
        rows.append({
            "id": int(row["id"]),
            "solvus_refined": float(row["solvus_refined"]),
            "T_stored": stored,
            "T_coarse": res["t_coarse"],
            "T_corrected": t_fine,
            "shift_K": t_fine - stored,
            "gp_at_last_node": res["gp_last"],
            "bracket_lo": res["bracket_lo"],
            "bracket_hi": res["bracket_hi"],
            "censored_fine": res["cens"],
        })
        print(f" id {int(row['id']):3d} refined={int(row['solvus_refined'])} "
              f"stored {stored:7.2f} coarse {res['t_coarse']:7.2f} "
              f"corrected {t_fine:7.2f} shift {t_fine-stored:+6.2f} K "
              f"gp(last node) {res['gp_last']:.4f}")
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "solvus_refine_check.csv", index=False)
    print(out[["solvus_refined", "shift_K"]].groupby("solvus_refined")
          .describe().round(2).to_string())


if __name__ == "__main__":
    main()
