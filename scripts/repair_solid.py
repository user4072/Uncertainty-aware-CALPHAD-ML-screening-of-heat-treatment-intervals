"""Repair isolated ordered-branch points in the stored solid-state curves.

At a small number of high-Co/Cr/Mo, low-Al+Ti compositions the grid minimiser
returns a metastable single-phase GAMMA_PRIME state at individual
temperatures (``f_gamma' > 0.85``, ``f_fcc < 0.15``) even at pdens 400.  A
denser grid recovers the stable gamma + gamma' mixture, with a lower Gibbs
energy, and shifts the gamma-prime fraction at 900 C by up to 0.5.

The stored curves do not contain the Gibbs energy, so the repair accepts a
denser solution when it restores an FCC matrix fraction.  After the repair the
solvus, the 750/900 C fractions and the consistency flags are recomputed with
the production extractor, and the 2 K solvus refinement is repeated where the
coarse solvus moved.
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
    repair_ordered_points,
    solvus_from_fractions,
)

_WORKER = {}


def _init(db_path, elements, phases, pdens_retry, pdens_solid):
    from pycalphad import Database
    _WORKER["db"] = Database(db_path)
    _WORKER.update(elements=elements, phases=phases, pdens_retry=pdens_retry,
                   pdens_solid=pdens_solid)


def _run(payload):
    pid, comp, T, gp, fcc, H = payload
    db = _WORKER["db"]
    frac = {"GAMMA_PRIME": gp.copy(), "FCC_A1": fcc.copy()}
    frac2, H2, _, n_rep = repair_ordered_points(
        db, comp, _WORKER["elements"], _WORKER["phases"], T, frac, H,
        G=None, pdens_retry=_WORKER["pdens_retry"],
    )
    out = {"repaired": int(n_rep), "gp": gp, "fcc": fcc, "H": H}
    if not n_rep:
        return out

    out.update(gp=frac2["GAMMA_PRIME"], fcc=frac2["FCC_A1"], H=H2)
    curve = {
        "valid": True, "T": T, "frac": frac2, "H": H2,
        "Cp": np.gradient(H2, T), "molar_mass": None,
    }
    from calphad_engine import mean_molar_mass
    curve["molar_mass"] = mean_molar_mass(comp)
    sig = extract_signatures(curve)
    t_gp = float(sig.get("T_gp_solvus", np.nan))
    cens = float(sig.get("censored_gp_high", 1.0))

    # Repeat the local refinement around the repaired solvus.
    if cens == 0.0 and np.isfinite(t_gp):
        gp2 = np.asarray(frac2["GAMMA_PRIME"], dtype=float)
        fcc2 = np.asarray(frac2["FCC_A1"], dtype=float)
        above = (gp2 > 1.0e-3) & (fcc2 >= 0.5)
        idx = np.where(above)[0]
        if len(idx):
            i_above = int(idx[-1])
            lo = float(T[i_above])
            hi = float(T[i_above + 1]) if i_above + 1 < len(T) else float(T[-1])
            fine_T = np.arange(max(float(T[0]), lo - 2.0),
                               min(float(T[-1]), hi + 2.0) + 0.5, 2.0)
        else:
            fine_T = refinement_grid(t_gp, T, half_width=10.0, step=2.0)
        fine = equilibrium_curve(
            db, comp, fine_T, _WORKER["elements"], _WORKER["phases"],
            pdens=_WORKER["pdens_solid"], output_hm=False,
        )
        if fine.get("valid", False):
            tf, cf = solvus_from_fractions(
                np.asarray(fine["T"], dtype=float),
                fine["frac"].get("GAMMA_PRIME", np.zeros(len(fine["T"]))),
                fine["frac"].get("FCC_A1", np.ones(len(fine["T"]))),
            )
            if cf == 0.0 and np.isfinite(tf):
                t_gp = float(tf)

    out.update({
        "T_gp_solvus": t_gp,
        "censored_gp_high": cens,
        "gp_frac_750C": float(sig.get("gp_frac_750C", np.nan)),
        "gp_frac_900C": float(sig.get("gp_frac_900C", np.nan)),
        "fcc_frac_750C": float(sig.get("fcc_frac_750C", np.nan)),
        "fcc_frac_900C": float(sig.get("fcc_frac_900C", np.nan)),
    })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--prefix", default="dataset")
    parser.add_argument("--pdens-retry", type=int, default=4000)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    sig_path = ROOT / "data" / "processed" / f"{args.prefix}_signatures.csv"
    npz_path = ROOT / "data" / "processed" / f"{args.prefix}_curves.npz"
    sig = pd.read_csv(sig_path)
    curves = np.load(npz_path)
    ids = [int(i) for i in curves["ids"]]
    T = np.asarray(curves["T_solid"], dtype=float)
    H = np.asarray(curves["H_solid"], dtype=float)
    GP = np.asarray(curves["gp_solid"], dtype=float)
    FCC = np.asarray(curves["fcc_solid"], dtype=float)

    if not (sig_path.with_suffix(".csv.pdens400.bak")).exists():
        shutil.copy2(sig_path, sig_path.with_suffix(".csv.pdens400.bak"))
        shutil.copy2(npz_path, npz_path.with_suffix(".npz.pdens400.bak"))

    comp_cols = ["Co", "Cr", "Al", "Mo", "Ti", "C", "Ni"]
    row_of = {int(r["id"]): r for _, r in sig.iterrows()}
    tasks = []
    for i, pid in enumerate(ids):
        row = row_of[pid]
        comp = {k: float(row[k]) for k in comp_cols}
        tasks.append((pid, comp, T, GP[i], FCC[i], H[i]))

    t0 = time.time()
    results = {}
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init,
        initargs=(str(ROOT / cfg["database"]), cfg["elements"], cfg["active_phases"],
                  args.pdens_retry, cfg["pdens_solid"]),
    ) as pool:
        futures = {pool.submit(_run, t): t[0] for t in tasks}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="repair"):
            pid = futures[fut]
            try:
                results[pid] = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  {pid}: {type(exc).__name__}: {exc}")
                results[pid] = None

    n_rep = 0
    for i, pid in enumerate(ids):
        rec = results.get(pid)
        if not rec or not rec["repaired"]:
            continue
        n_rep += 1
        GP[i], FCC[i], H[i] = rec["gp"], rec["fcc"], rec["H"]
        j = sig.index[sig["id"] == pid][0]
        for key in ("T_gp_solvus", "censored_gp_high", "gp_frac_750C",
                    "gp_frac_900C", "fcc_frac_750C", "fcc_frac_900C"):
            sig.loc[j, key] = rec[key]

    # Consistency: gamma-prime cannot persist above the solidus.
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
    print(f"Repaired alloys: {n_rep}/{len(ids)} in {(time.time()-t0)/60:.1f} min")
    print(f"censored solvus: {int(sig['censored_gp_high'].fillna(1).sum())}")


if __name__ == "__main__":
    main()
