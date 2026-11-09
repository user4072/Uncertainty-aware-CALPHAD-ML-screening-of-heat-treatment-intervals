"""Convergence diagnostics for the full-equilibrium melting calculation.

The 12-phase melting calculation produces a near-isothermal liquid-fraction
jump for some shortlisted alloys.  This script repeats the calculation at
finer temperature steps and higher point densities and records whether the
jump survives, where it is located, and which phases change across it.  A jump
that survives refinement and coincides with the disappearance of a solid
phase is an invariant of the database; a jump that moves or vanishes is a grid
artefact.

Output: ``results/fullphase_convergence.csv``.
"""

from __future__ import annotations

import argparse
import sys
import time
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-fine", action="store_true",
                        help="skip the 1 K / pdens 3000 refinement")
    args = parser.parse_args()
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    train = pd.read_csv(ROOT / "results" / "design_robustness.csv")
    a1 = train.iloc[0].astype(object).to_dict()
    a1["rank"] = "A1"
    picks = pd.concat([prod.loc[prod["rank"].isin([1, 3])], pd.DataFrame([a1])],
                      ignore_index=True)

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    settings = [("all", 5.0, 1000), ("all", 2.0, 2000)]
    fine = [] if args.no_fine else [("rank3", 1.0, 3000), ("A1", 1.0, 3000)]

    rows = []
    for label, step, pdens in settings + fine:
        for _, cand in picks.iterrows():
            if label != "all" and str(cand["rank"]) != label.replace("rank", ""):
                continue
            comp = {k: float(cand[k]) for k in ("Co", "Cr", "Al", "Mo", "Ti", "C", "Ni")}
            T = np.arange(cfg["temperature"]["melt_min"],
                          cfg["temperature"]["melt_max"] + 1e-9, step)
            t0 = time.time()
            curve = equilibrium_curve(db, comp, T, cfg["elements"], FULL_PHASES, pdens=pdens)
            sig = extract_signatures(curve)
            liquid = np.asarray(curve.get("frac", {}).get("LIQUID", []), dtype=float)
            jumps = np.diff(liquid) if len(liquid) else np.array([np.nan])
            if len(jumps) and np.isfinite(jumps).any():
                k = int(np.nanargmax(jumps))
            else:
                k = -1
            t_below = float(T[k]) if k >= 0 else np.nan
            t_above = float(T[k + 1]) if k >= 0 else np.nan
            phases_below = {}
            phases_above = {}
            if k >= 0:
                for ph, frac in curve.get("frac", {}).items():
                    f = np.asarray(frac, dtype=float)
                    if f[k] > 0.01:
                        phases_below[ph] = round(float(f[k]), 3)
                    if f[k + 1] > 0.01:
                        phases_above[ph] = round(float(f[k + 1]), 3)
            rows.append({
                "rank": cand["rank"],
                "step_K": step,
                "pdens": pdens,
                "T_solidus": float(sig.get("T_solidus", np.nan)),
                "T_liquidus": float(sig.get("T_liquidus", np.nan)),
                "max_jump": float(np.nanmax(jumps)) if len(jumps) else np.nan,
                "jump_below_K": t_below,
                "jump_above_K": t_above,
                "phases_below": phases_below,
                "phases_above": phases_above,
                "valid": sig.get("valid", 0.0),
                "seconds": round(time.time() - t0, 1),
            })
            r = rows[-1]
            print(f"{r['rank']:>2} step {step:g}K pdens {pdens}: Tsol={r['T_solidus']:.1f} "
                  f"Tliq={r['T_liquidus']:.1f} jump={r['max_jump']:.2f} at {t_below:.0f} K "
                  f"({r['seconds']:.0f} s)")
            print(f"   below: {phases_below}")
            print(f"   above: {phases_above}")
            # Write after every alloy so that a failure in a later setting does
            # not discard the rows already computed.
            pd.DataFrame(rows).to_csv(
                ROOT / "results" / "fullphase_convergence.csv", index=False)

    pd.DataFrame(rows).to_csv(ROOT / "results" / "fullphase_convergence.csv", index=False)
    print("Wrote results/fullphase_convergence.csv")


if __name__ == "__main__":
    main()
