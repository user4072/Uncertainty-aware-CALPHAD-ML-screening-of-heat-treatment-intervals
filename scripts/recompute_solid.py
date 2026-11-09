"""Recompute the solid-state part of the dataset at a denser point density.

The original dataset used pdens=150.  A phase-fraction audit found that, for a
large fraction of the alloys, the grid minimiser then returned a metastable
branch in which the ordered GAMMA_PRIME phase alone is selected at high
temperature (gamma-prime fraction near one, FCC_A1 fraction zero).  The branch
shifts the gamma-prime solvus and the phase fractions at 750/900 C, sometimes
by hundreds of kelvin.  Recalculating at pdens=400 removes it; 400 and 800
agree in tests, so 400 is used for cost.

The melting calculation is unchanged (it uses only LIQUID + FCC_A1 and is
insensitive to this branch), so only the solid arrays and the solid-state
signatures are replaced.  The old files are kept as *_pdens150 backups.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import (  # noqa: E402
    equilibrium_curve,
    extract_signatures,
    refinement_grid,
    solvus_from_fractions,
)

_WORKER = {}


def _init(db_path, elements, phases, T_solid, pdens):
    from pycalphad import Database
    _WORKER["db"] = Database(db_path)
    _WORKER.update(elements=elements, phases=phases, T_solid=T_solid, pdens=pdens)


def _run(comp):
    db = _WORKER["db"]
    T = _WORKER["T_solid"]
    pdens = _WORKER["pdens"]
    solid = equilibrium_curve(db, comp, T, _WORKER["elements"], _WORKER["phases"],
                              pdens=pdens, output_hm=True)
    sig = extract_signatures(solid)
    refined = 0.0
    if solid.get("valid", False):
        Tc = np.asarray(solid["T"], dtype=float)
        gp = solid["frac"].get("GAMMA_PRIME")
        fcc = solid["frac"].get("FCC_A1")
        if gp is not None and fcc is not None:
            t0, cens0 = solvus_from_fractions(Tc, gp, fcc)
            if cens0 == 0.0 and np.isfinite(t0):
                # Refine over the whole coarse bracket; see compute_alloy.
                above = (np.asarray(gp) > 1.0e-3) & (np.asarray(fcc) >= 0.5)
                idx = np.where(above)[0]
                if len(idx):
                    i_above = int(idx[-1])
                    lo = float(Tc[i_above])
                    hi = float(Tc[i_above + 1]) if i_above + 1 < len(Tc) else float(Tc[-1])
                    fine_T = np.arange(max(float(Tc[0]), lo - 2.0),
                                       min(float(Tc[-1]), hi + 2.0) + 0.5, 2.0)
                else:
                    fine_T = refinement_grid(t0, Tc, half_width=10.0, step=2.0)
                fine = equilibrium_curve(db, comp, fine_T, _WORKER["elements"],
                                         _WORKER["phases"], pdens=pdens, output_hm=False)
                if fine.get("valid", False):
                    tf, cf = solvus_from_fractions(
                        np.asarray(fine["T"], dtype=float),
                        fine["frac"].get("GAMMA_PRIME", np.zeros(len(fine["T"]))),
                        fine["frac"].get("FCC_A1", np.ones(len(fine["T"]))),
                    )
                    if cf == 0.0 and np.isfinite(tf):
                        sig["T_gp_solvus"] = tf
                        refined = 1.0
    return {
        "valid": float(sig.get("valid", 0.0)),
        "T_gp_solvus": float(sig.get("T_gp_solvus", np.nan)),
        "censored_gp_high": float(sig.get("censored_gp_high", 1.0)),
        "solvus_refined": refined,
        "gp_frac_750C": float(sig.get("gp_frac_750C", np.nan)),
        "gp_frac_900C": float(sig.get("gp_frac_900C", np.nan)),
        "fcc_frac_750C": float(sig.get("fcc_frac_750C", np.nan)),
        "fcc_frac_900C": float(sig.get("fcc_frac_900C", np.nan)),
        "T": np.asarray(solid.get("T", T), dtype=float),
        "H": np.asarray(solid.get("H", np.full(len(T), np.nan)), dtype=float),
        "gp": np.asarray(solid.get("frac", {}).get("GAMMA_PRIME", np.zeros(len(T))), dtype=float),
        "fcc": np.asarray(solid.get("frac", {}).get("FCC_A1", np.zeros(len(T))), dtype=float),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--prefix", default="dataset")
    parser.add_argument("--pdens", type=int, default=400)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sig_path = ROOT / "data" / "processed" / f"{args.prefix}_signatures.csv"
    npz_path = ROOT / "data" / "processed" / f"{args.prefix}_curves.npz"
    sig = pd.read_csv(sig_path)
    curves = np.load(npz_path)
    ids = [int(i) for i in curves["ids"]]
    order = {pid: k for k, pid in enumerate(ids)}

    if not (sig_path.with_suffix(".csv.pdens150.bak")).exists():
        shutil.copy2(sig_path, sig_path.with_suffix(".csv.pdens150.bak"))
        shutil.copy2(npz_path, npz_path.with_suffix(".npz.pdens150.bak"))

    T_solid = np.arange(cfg["temperature"]["solid_min"],
                        cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    comp_cols = ["Co", "Cr", "Al", "Mo", "Ti", "C", "Ni"]

    results = {}
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init,
        initargs=(str(ROOT / cfg["database"]), cfg["elements"], cfg["active_phases"],
                  T_solid, args.pdens),
    ) as pool:
        futures = {}
        for _, row in sig.iterrows():
            comp = {k: float(row[k]) for k in comp_cols}
            futures[pool.submit(_run, comp)] = int(row["id"])
        for fut in tqdm(as_completed(futures), total=len(futures), desc="solid pdens=%d" % args.pdens):
            pid = futures[fut]
            try:
                results[pid] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  {pid}: {type(exc).__name__}: {exc}")
                results[pid] = None

    # Replace solid-state columns; keep the melting columns untouched.
    for key in ("T_gp_solvus", "censored_gp_high", "solvus_refined", "gp_frac_750C",
                "gp_frac_900C", "fcc_frac_750C", "fcc_frac_900C", "solid_valid"):
        if key not in sig:
            sig[key] = np.nan
    H = np.full((len(ids), len(curves["T_solid"])), np.nan)
    GP = np.full_like(H, np.nan)
    FCC = np.full_like(H, np.nan)
    for i, pid in enumerate(ids):
        rec = results.get(pid)
        if rec is None:
            sig.loc[sig["id"] == pid, "solid_valid"] = 0.0
            continue
        row = sig.index[sig["id"] == pid][0]
        for key in ("T_gp_solvus", "censored_gp_high", "solvus_refined", "gp_frac_750C",
                    "gp_frac_900C", "fcc_frac_750C", "fcc_frac_900C"):
            sig.loc[row, key] = rec[key]
        sig.loc[row, "solid_valid"] = rec["valid"]
        if np.array_equal(rec["T"], curves["T_solid"]):
            H[i], GP[i], FCC[i] = rec["H"], rec["gp"], rec["fcc"]
        else:
            print(f"  {pid}: temperature grid mismatch, curve not stored")

    # Gamma-prime cannot persist above the solidus, and a solvus below a
    # measured 900 C fraction is inconsistent with the stored fractions.
    inconsistent = (sig["T_gp_solvus"] >= sig["T_solidus"] - 5.0) | (
        (sig["gp_frac_900C"] > 0.5) & (sig["T_gp_solvus"] < 1173.15))
    sig.loc[inconsistent, "censored_gp_high"] = 1.0

    sig.to_csv(sig_path, index=False)
    np.savez_compressed(
        npz_path,
        ids=curves["ids"], T_melt=curves["T_melt"], H_melt=curves["H_melt"],
        Cp_melt=curves["Cp_melt"], liquid_melt=curves["liquid_melt"],
        T_solid=curves["T_solid"], H_solid=H, Cp_solid=curves["Cp_solid"],
        gp_solid=GP, fcc_solid=FCC,
    )
    print(f"Finished in {(time.time()-t0)/60:.1f} min")
    print(f"solid valid: {int(sig['solid_valid'].sum())}/{len(sig)}")
    print(f"censored solvus: {int(sig['censored_gp_high'].fillna(1).sum())}")
    print(f"solvus refined: {int(sig['solvus_refined'].fillna(0).sum())}")


if __name__ == "__main__":
    main()
