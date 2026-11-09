"""Full-phase melting audit of the current shortlist at two resolutions.

Runs the 12-phase melting calculation for production ranks 1, 3 and 6 and for
the train-only candidate A1, at the original 5 K / pdens 1000 setting and at
2 K / pdens 2000.  Writes ``results/fullphase_convergence.csv`` (ranks 1, 3 and
A1, both settings) and ``results/fullphase_rank6_2k.json`` (rank 6 at 2 K) so
that the existing table builders can be reused.

The compositions are read from ``results/design_validated.csv`` (ranks 1, 3, 6)
and ``results/design_robustness.csv`` (row 1 = A1).
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import equilibrium_curve, extract_signatures  # noqa: E402

FULL = ["LIQUID", "FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "M6C",
        "M7C3", "SIGMA", "BCC_A2", "HCP_A3", "MU_PHASE", "LAVES"]

_W = {}


def _init(db_path, elements):
    from pycalphad import Database
    _W["db"] = Database(db_path)
    _W["elements"] = elements


def _run(job):
    label, comp, step, pdens, Tmin, Tmax = job
    T = np.arange(Tmin, Tmax + 1e-9, step)
    t0 = time.time()
    curve = equilibrium_curve(_W["db"], comp, T, _W["elements"], FULL,
                              pdens=pdens, output_hm=True)
    sig = extract_signatures(curve)
    liquid = np.asarray(curve["frac"].get("LIQUID", []), dtype=float)
    jumps = np.diff(liquid) if len(liquid) > 1 else np.array([np.nan])
    if len(jumps) and np.isfinite(jumps).any():
        k = int(np.nanargmax(jumps))
        jb, ja = float(T[k]), float(T[k + 1])
        below = {p: round(float(np.asarray(f)[k]), 3)
                 for p, f in curve["frac"].items() if float(np.asarray(f)[k]) > 1e-3}
        above = {p: round(float(np.asarray(f)[k + 1]), 3)
                 for p, f in curve["frac"].items() if float(np.asarray(f)[k + 1]) > 1e-3}
    else:
        k, jb, ja, below, above = -1, np.nan, np.nan, {}, {}
    return {
        "rank": label, "step_K": step, "pdens": pdens,
        "T_solidus": float(sig.get("T_solidus", np.nan)),
        "T_liquidus": float(sig.get("T_liquidus", np.nan)),
        "dH": float(sig.get("dH_fusion_J_per_g", np.nan)),
        "max_jump": float(np.nanmax(jumps)) if len(jumps) else np.nan,
        "jump_below_K": jb, "jump_above_K": ja,
        "phases_below": str(below), "phases_above": str(above),
        "valid": float(sig.get("valid", 0.0)),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    a1 = pd.read_csv(ROOT / "results" / "design_robustness.csv").iloc[0]

    jobs, labels = [], []
    for rank in (1, 3, 6):
        row = prod.loc[prod["rank"] == rank].iloc[0]
        comp = {k: float(row[k]) for k in ("Co", "Cr", "Al", "Mo", "Ti", "C", "Ni")}
        for step, pdens in ((5.0, 1000), (2.0, 2000)):
            jobs.append((str(rank), comp, step, pdens,
                         cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"]))
            labels.append(str(rank))
    comp = {k: float(a1[k]) for k in ("Co", "Cr", "Al", "Mo", "Ti", "C", "Ni")}
    for step, pdens in ((5.0, 1000), (2.0, 2000)):
        jobs.append(("A1", comp, step, pdens,
                     cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"]))

    with ProcessPoolExecutor(max_workers=4, initializer=_init,
                             initargs=(str(ROOT / cfg["database"]), cfg["elements"])) as pool:
        results = list(pool.map(_run, jobs))
    for r in results:
        print(f" {r['rank']:2s} {r['step_K']:.0f}K/{r['pdens']}: Tsol={r['T_solidus']:.2f} "
              f"Tliq={r['T_liquidus']:.2f} dH={r['dH']:.1f} jump={r['max_jump']:.2f} "
              f"({r['seconds']:.0f}s)")

    conv = pd.DataFrame([r for r in results if r["rank"] != "6"])
    conv.to_csv(ROOT / "results" / "fullphase_convergence.csv", index=False)
    r6 = [r for r in results if r["rank"] == "6" and r["step_K"] == 2.0][0]
    with open(ROOT / "results" / "fullphase_rank6_2k.json", "w", encoding="utf-8") as fh:
        json.dump(r6, fh, indent=2)
    print("wrote fullphase_convergence.csv and fullphase_rank6_2k.json")


if __name__ == "__main__":
    main()
