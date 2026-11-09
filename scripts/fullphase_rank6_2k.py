"""Full-equilibrium melting of shortlist rank 6 on the refined grid.

The convergence audit covers ranks 1, 3 and A1.  Rank 6 is added here at the
same refined setting (2 K, pdens 2000) so that Table 12 uses one grid for all
four alloys.

Output: ``results/fullphase_rank6_2k.json``.
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

from calphad_engine import equilibrium_curve, extract_signatures  # noqa: E402

FULL_PHASES = (
    "LIQUID", "FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "M6C",
    "M7C3", "SIGMA", "BCC_A2", "HCP_A3", "MU_PHASE", "LAVES",
)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    cand = prod.loc[prod["rank"] == 6].iloc[0]
    comp = {k: float(cand[k]) for k in ("Co", "Cr", "Al", "Mo", "Ti", "C", "Ni")}

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    T = np.arange(cfg["temperature"]["melt_min"],
                  cfg["temperature"]["melt_max"] + 1e-9, 2.0)
    t0 = time.time()
    curve = equilibrium_curve(db, comp, T, cfg["elements"], FULL_PHASES, pdens=2000)
    sig = extract_signatures(curve)
    liquid = np.asarray(curve["frac"]["LIQUID"], dtype=float)
    jumps = np.diff(liquid)
    k = int(np.nanargmax(jumps))
    rec = {
        "rank": 6,
        "step_K": 2.0,
        "pdens": 2000,
        "T_solidus": float(sig["T_solidus"]),
        "T_liquidus": float(sig["T_liquidus"]),
        "max_jump": float(np.nanmax(jumps)),
        "jump_below_K": float(T[k]),
        "jump_above_K": float(T[k + 1]),
        "phases_below": {ph: float(np.asarray(f, dtype=float)[k])
                         for ph, f in curve["frac"].items()
                         if float(np.asarray(f, dtype=float)[k]) > 0.01},
        "phases_above": {ph: float(np.asarray(f, dtype=float)[k + 1])
                         for ph, f in curve["frac"].items()
                         if float(np.asarray(f, dtype=float)[k + 1]) > 0.01},
        "seconds": round(time.time() - t0, 1),
    }
    with open(ROOT / "results" / "fullphase_rank6_2k.json", "w", encoding="utf-8") as fh:
        json.dump(rec, fh, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
