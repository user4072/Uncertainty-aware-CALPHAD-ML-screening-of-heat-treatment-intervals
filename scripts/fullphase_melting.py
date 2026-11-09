"""Full-phase melting calculation for the literature DSC alloys.

The production melting calculation restricts the phase list to
``LIQUID + FCC_A1``.  This script performs the complementary full-equilibrium
calculation for the four DSC reference alloys, allowing every phase of the
12-phase list, so that the difference between the restricted ("matrix") solidus
and the full-equilibrium solidus can be quantified and reported.

The full-phase calculation is slower and produces near-isothermal liquid
fractions in this database, but it is the correct thermodynamic reference and
is needed to separate three quantities that the manuscript must distinguish:

* matrix-only (restricted) solidus used by the surrogate;
* full-equilibrium solidus;
* experimentally observed incipient melting.

Output: ``results/fullphase_melting.csv``.

The published audit uses ``--step 2 --pdens 2000``; the defaults (5 K, 1,000
points) are a cheaper setting for smoke tests, and at that setting Rene 77 and
IN100 collapse to a single liquid-fraction step, so they must not be used for
the reported solidus.  ``--alloys`` and ``--merge`` allow a single alloy to be
recomputed at the refined setting without repeating the whole table.
"""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--pdens", type=int, default=1000)
    parser.add_argument("--step", type=float, default=5.0)
    parser.add_argument("--alloys", default="",
                        help="comma-separated subset of alloys (default: all rows)")
    parser.add_argument("--merge", action="store_true",
                        help="merge the new rows into an existing output table")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    lit = pd.read_csv(ROOT / "data" / "dsc_literature.csv")
    if args.alloys:
        wanted = [a.strip() for a in args.alloys.split(",") if a.strip()]
        lit = lit[lit["alloy"].isin(wanted)].reset_index(drop=True)
    T = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9, args.step)

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    rows = []
    for _, alloy in lit.iterrows():
        comp = {k: float(alloy[k]) for k in ("Co", "Cr", "Al", "Mo", "Ti", "C", "Ni")}
        curve = equilibrium_curve(
            db, comp, T, cfg["elements"], FULL_PHASES, pdens=args.pdens
        )
        sig = extract_signatures(curve)
        liquid = np.asarray(curve.get("frac", {}).get("LIQUID", []), dtype=float)
        # Largest one-step rise of the liquid fraction: a large jump means that
        # the full-phase calculation melts almost isothermally, which is the
        # numerical signature of an unresolved multi-phase invariant.
        jumps = np.diff(liquid) if len(liquid) else np.array([np.nan])
        rows.append({
            "alloy": alloy["alloy"],
            "role": alloy.get("role", "calibration"),
            "T_solidus_full": sig.get("T_solidus", np.nan),
            "T_liquidus_full": sig.get("T_liquidus", np.nan),
            "dH_fusion_full_J_per_g": sig.get("dH_fusion_J_per_g", np.nan),
            "liquid_max": float(np.nanmax(liquid)) if len(liquid) else np.nan,
            "max_liquid_jump_per_step": float(np.nanmax(jumps)) if len(jumps) else np.nan,
            "valid": sig.get("valid", 0.0),
            "error": sig.get("error", ""),
        })
        print(f"{alloy['alloy']:8s} full: Tsol={rows[-1]['T_solidus_full']:7.1f} K "
              f"Tliq={rows[-1]['T_liquidus_full']:7.1f} K "
              f"dH={rows[-1]['dH_fusion_full_J_per_g']:7.1f} J/g "
              f"max jump={rows[-1]['max_liquid_jump_per_step']:.2f}")

    out = pd.DataFrame(rows)
    target = Path(args.out) / "fullphase_melting.csv"
    if args.merge and target.exists():
        old = pd.read_csv(target)
        out = pd.concat([old[~old["alloy"].isin(out["alloy"])], out], ignore_index=True)
        # Keep the literature order so the table columns do not move.
        order = pd.read_csv(ROOT / "data" / "dsc_literature.csv")["alloy"].tolist()
        out["_o"] = out["alloy"].map({a: i for i, a in enumerate(order)})
        out = out.sort_values("_o").drop(columns="_o").reset_index(drop=True)
    out.to_csv(target, index=False)
    print(f"\nWrote {target}")


if __name__ == "__main__":
    main()
