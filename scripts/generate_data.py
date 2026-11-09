"""Generate the CALPHAD dataset for the René 77-like composition space."""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# pycalphad spends most of its time in NumPy/SymEngine kernels that default to
# multi-threaded BLAS.  With one process per core this oversubscribes the CPU
# and slows the whole batch by an order of magnitude, so each worker is pinned
# to a single thread before NumPy is imported.
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
    MOLAR_MASS,
    compute_alloy,
    sample_compositions,
    wt_to_mol,
)

_WORKER_DB = None
_WORKER_CFG = None


def _init_worker(db_path: str, elements, phases, T_melt, T_solid,
                 pdens_melt: int, pdens_solid: int, refine: bool):
    from pycalphad import Database

    global _WORKER_DB, _WORKER_CFG
    _WORKER_DB = Database(db_path)
    _WORKER_CFG = {
        "elements": elements,
        "phases": phases,
        "T_melt": T_melt,
        "T_solid": T_solid,
        "pdens_melt": pdens_melt,
        "pdens_solid": pdens_solid,
        "refine": refine,
    }


def _run_one(comp: dict) -> dict:
    cfg = _WORKER_CFG
    return compute_alloy(
        _WORKER_DB,
        comp,
        cfg["elements"],
        cfg["phases"],
        cfg["T_melt"],
        cfg["T_solid"],
        pdens_melt=cfg["pdens_melt"],
        pdens_solid=cfg["pdens_solid"],
        refine=cfg["refine"],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--n-train", type=int, default=None)
    parser.add_argument("--n-test", type=int, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--out-prefix", default="dataset")
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    n_train = args.n_train or cfg["sampling"]["n_train"]
    n_test = args.n_test or cfg["sampling"]["n_test"]
    workers = args.workers or cfg["workers"]

    elements = cfg["elements"]
    phases = cfg["active_phases"]
    T_melt = np.arange(
        cfg["temperature"]["melt_min"],
        cfg["temperature"]["melt_max"] + 1e-9,
        cfg["temperature"]["melt_step"],
    )
    T_solid = np.arange(
        cfg["temperature"]["solid_min"],
        cfg["temperature"]["solid_max"] + 1e-9,
        cfg["temperature"]["solid_step"],
    )

    comps = sample_compositions(
        n_train + n_test,
        cfg["composition_space_wt"],
        seed=cfg["sampling"]["seed"],
        ni_min_wt=cfg["composition_space_wt"]["Ni_min"],
        min_al_ti_wt=cfg["composition_space_wt"].get("min_Al_Ti_sum", 0.0),
    )
    # sample_compositions returns Co, Cr, Al, Mo, Ti, C, Ni columns.
    comps["split"] = ["train"] * n_train + ["test"] * n_test
    comps.insert(0, "id", np.arange(len(comps)))

    out_dir = ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(comps)} alloys with {workers} workers")
    print(f"  melt grid : {T_melt[0]:.0f}-{T_melt[-1]:.0f} K, {len(T_melt)} points")
    print(f"  solid grid: {T_solid[0]:.0f}-{T_solid[-1]:.0f} K, {len(T_solid)} points")

    t0 = time.time()
    records = [None] * len(comps)
    failures = []
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_init_worker,
        initargs=(
            str(ROOT / cfg["database"]), elements, phases, T_melt, T_solid,
            cfg["pdens_melt"], cfg["pdens_solid"], cfg.get("refine_solvus", True),
        ),
    ) as pool:
        comp_cols = ["Co", "Cr", "Al", "Mo", "Ti", "C", "Ni"]
        futures = {
            pool.submit(_run_one, {k: float(row[k]) for k in comp_cols}): i
            for i, row in comps.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="CALPHAD"):
            idx = futures[fut]
            try:
                records[idx] = fut.result()
            except Exception as exc:  # noqa: BLE001
                failures.append({"id": int(comps.iloc[idx]["id"]), "error": f"{type(exc).__name__}: {exc}"})
                records[idx] = None

    elapsed = time.time() - t0
    print(f"Finished in {elapsed/60:.1f} min")

    rows = []
    curve_rows = []
    for i, rec in enumerate(records):
        base = comps.iloc[i][["id", "split", "Co", "Cr", "Al", "Mo", "Ti", "C", "Ni"]].to_dict()
        if rec is None:
            row = dict(base)
            row["melt_valid"] = 0.0
            row["solid_valid"] = 0.0
            rows.append(row)
            curve_rows.append(None)
            continue
        row = dict(base)
        for key in (
            "T_solidus", "T_liquidus", "dH_fusion_J_per_mol", "dH_fusion_J_per_g",
            "censored_solidus_low", "censored_liquidus_high",
            "T_gp_solvus", "censored_gp_high", "solvus_refined",
            "gp_frac_750C", "gp_frac_900C", "fcc_frac_750C", "fcc_frac_900C",
            "melt_valid", "solid_valid",
        ):
            row[key] = rec.get(key, np.nan)
        for el in ("Co", "Cr", "Al", "Mo", "Ti", "C"):
            row[f"x_{el}"] = wt_to_mol(rec["comp_wt"]).get(el, 0.0)
        rows.append(row)
        rec["id"] = int(comps.iloc[i]["id"])
        curve_rows.append(rec)

    sig = pd.DataFrame(rows)
    sig.to_csv(out_dir / f"{args.out_prefix}_signatures.csv", index=False)

    # Curves are stored on the common temperature grids; missing rows are NaN.
    good = [r for r in curve_rows if r is not None]
    if good:
        np.savez_compressed(
            out_dir / f"{args.out_prefix}_curves.npz",
            ids=np.array([r["id"] for r in good], dtype=int),
            T_melt=np.array(good[0]["melt_T"]),
            H_melt=np.vstack([r["melt_H"] for r in good]),
            Cp_melt=np.vstack([r["melt_Cp"] for r in good]),
            liquid_melt=np.vstack([r["melt_liquid"] for r in good]),
            T_solid=np.array(good[0]["solid_T"]),
            H_solid=np.vstack([r["solid_H"] for r in good]),
            Cp_solid=np.vstack([r["solid_Cp"] for r in good]),
            gp_solid=np.vstack([r["solid_gp"] for r in good]),
            fcc_solid=np.vstack([r["solid_fcc"] for r in good]),
        )

    if failures:
        pd.DataFrame(failures).to_csv(out_dir / f"{args.out_prefix}_failures.csv", index=False)
    print(f"Wrote {out_dir / f'{args.out_prefix}_signatures.csv'}")
    print(f"Valid melt calculations: {int(sig['melt_valid'].sum())}/{len(sig)}")
    print(f"Valid solid calculations: {int(sig['solid_valid'].sum())}/{len(sig)}")


if __name__ == "__main__":
    main()
