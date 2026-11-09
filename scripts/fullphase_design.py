"""Full-equilibrium melting for selected design candidates.

The production screen uses the matrix-only melting calculation.  A reviewer
asked how the melting definition affects the selected alloys and the reported
window gain, not only the literature DSC anchors.  This script repeats the
12-phase equilibrium melting calculation for four shortlisted alloys:

* production ranks 1, 3 and 6 (a low, the best and a high window),
* row A1, the carbon-free train-only candidate.

The solidus enters the window directly, so the full-equilibrium solidus gives
a full-equilibrium window while the solvus is held at its six-phase value
(the solvus is unaffected by the melting phase list).  The script reports the
largest one-step rise of the liquid fraction, because a near-isothermal jump
is the numerical signature of an unresolved multi-phase invariant rather than
a physical melting range.

Output: ``results/fullphase_design.csv``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import equilibrium_curve, extract_signatures  # noqa: E402

FULL_PHASES = (
    "LIQUID", "FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "M6C",
    "M7C3", "SIGMA", "BCC_A2", "HCP_A3", "MU_PHASE", "LAVES",
)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    STEP_K = float(os.environ.get("FULLPHASE_STEP_K", 2.0))
    PDENS = int(os.environ.get("FULLPHASE_PDENS", 2000))
    T = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9, STEP_K)

    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    train = pd.read_csv(ROOT / "results" / "design_robustness.csv")
    a1 = train.iloc[0].astype(object).to_dict()
    a1["rank"] = "A1"
    a1["W_surrogate_K"] = a1["W_train_K"]

    picks = pd.concat([prod.loc[prod["rank"].isin([1, 3, 6])], pd.DataFrame([a1])], ignore_index=True)

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    rows = []
    for _, cand in picks.iterrows():
        comp = {k: float(cand[k]) for k in ("Co", "Cr", "Al", "Mo", "Ti", "C", "Ni")}
        curve = equilibrium_curve(db, comp, T, cfg["elements"], FULL_PHASES, pdens=PDENS)
        sig = extract_signatures(curve)
        liquid = np.asarray(curve.get("frac", {}).get("LIQUID", []), dtype=float)
        jumps = np.diff(liquid) if len(liquid) else np.array([np.nan])
        tsol_full = float(sig.get("T_solidus", np.nan))
        solvus = float(cand["T_gp_solvus_calphad"])
        w_full = tsol_full - solvus
        rows.append({
            "rank": cand["rank"],
            "step_K": STEP_K,
            "pdens": PDENS,
            "C_wt": comp["C"],
            "T_solidus_matrix_K": float(cand["T_solidus_calphad"]),
            "T_solidus_full_K": tsol_full,
            "T_liquidus_full_K": float(sig.get("T_liquidus", np.nan)),
            "T_gp_solvus_K": solvus,
            "W_matrix_K": float(cand["W_calphad_K"]),
            "W_full_K": w_full,
            "dW_full_minus_matrix_K": w_full - float(cand["W_calphad_K"]),
            "liquid_max": float(np.nanmax(liquid)) if len(liquid) else np.nan,
            "max_liquid_jump_per_step": float(np.nanmax(jumps)) if len(jumps) else np.nan,
            "valid": sig.get("valid", 0.0),
            "error": sig.get("error", ""),
        })
        r = rows[-1]
        print(f"rank {str(r['rank']):>2s}: matrix {r['T_solidus_matrix_K']:.1f} -> "
              f"full {r['T_solidus_full_K']:.1f} K, W {r['W_matrix_K']:.1f} -> "
              f"{r['W_full_K']:.1f} K (dW {r['dW_full_minus_matrix_K']:+.1f}), "
              f"max jump {r['max_liquid_jump_per_step']:.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "fullphase_design.csv", index=False)
    print("Wrote results/fullphase_design.csv")


if __name__ == "__main__":
    main()
