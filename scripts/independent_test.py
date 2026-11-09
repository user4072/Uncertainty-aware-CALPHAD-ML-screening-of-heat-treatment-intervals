"""Generate and evaluate an independent CALPHAD test set.

The 48 compositions are drawn with a seed that was never used in the design,
sampling, training, calibration or model selection, and their labels are
computed with the production settings.  The archived models are applied without
refitting.  Nothing in this script selects compositions by any predicted or
calculated property.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
             "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import joblib
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from calphad_engine import sample_compositions, wt_to_mol  # noqa: E402
from generate_data import _init_worker, _run_one  # noqa: E402
from surrogate import FEATURES, TARGETS  # noqa: E402

SIG_KEYS = (
    "T_solidus", "T_liquidus", "dH_fusion_J_per_mol", "dH_fusion_J_per_g",
    "censored_solidus_low", "censored_liquidus_high",
    "T_gp_solvus", "censored_gp_high", "solvus_refined",
    "gp_frac_750C", "gp_frac_900C", "fcc_frac_750C", "fcc_frac_900C",
    "melt_valid", "solid_valid",
)


def valid_masks(sig: pd.DataFrame) -> dict:
    """Validity rules, copied from surrogate.load_dataset."""
    base = ((sig["melt_valid"] == 1) & (sig["solid_valid"] == 1)).to_numpy()
    flag = {}
    for name in ("censored_gp_high", "censored_solidus_low", "censored_liquidus_high"):
        flag[name] = (sig[name].fillna(0.0).to_numpy() == 1.0) if name in sig else np.zeros(len(sig), bool)
    t_sol = sig["T_solidus"].to_numpy(dtype=float)
    y = {t: sig[t].to_numpy(dtype=float) for t in list(TARGETS) + ["gp_frac_900C"] if t in sig}
    return {
        "T_solidus": base & ~flag["censored_solidus_low"] & np.isfinite(t_sol),
        "T_liquidus": base & ~flag["censored_liquidus_high"] & np.isfinite(y.get("T_liquidus", np.nan * t_sol)),
        "T_gp_solvus": (base & ~flag["censored_gp_high"] & np.isfinite(y.get("T_gp_solvus", np.nan * t_sol))
                        & (y.get("T_gp_solvus", 0.0 * t_sol) < t_sol - 5.0)),
        "dH_fusion_J_per_g": (base & ~flag["censored_solidus_low"] & ~flag["censored_liquidus_high"]
                              & np.isfinite(y.get("dH_fusion_J_per_g", np.nan * t_sol))
                              & (y.get("dH_fusion_J_per_g", 0.0 * t_sol) > 0.0)),
        "gp_frac_900C": base & np.isfinite(y.get("gp_frac_900C", np.nan * t_sol)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--n", type=int, default=48)
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    elements = cfg["elements"]
    phases = cfg["active_phases"]
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])

    comps = sample_compositions(
        args.n,
        cfg["composition_space_wt"],
        seed=args.seed,
        ni_min_wt=cfg["composition_space_wt"]["Ni_min"],
        min_al_ti_wt=cfg["composition_space_wt"].get("min_Al_Ti_sum", 0.0),
    )
    comps.insert(0, "id", np.arange(len(comps)))

    # Confirm that no new composition duplicates a training composition.
    old = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    dmin = np.sqrt(((comps[list(FEATURES)].to_numpy()[:, None, :]
                     - old[list(FEATURES)].to_numpy()[None, :, :]) ** 2).sum(-1)).min()
    print(f"Nearest training composition to the new set: {dmin:.3f} wt% (Euclidean)")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    partial = out_dir / "independent_partial.csv"

    print(f"Generating {len(comps)} independent alloys with {args.workers} workers")
    t0 = time.time()
    records = [None] * len(comps)
    rows = [None] * len(comps)
    with ProcessPoolExecutor(
        max_workers=args.workers,
        initializer=_init_worker,
        initargs=(str(ROOT / cfg["database"]), elements, phases, T_melt, T_solid,
                  cfg["pdens_melt"], cfg["pdens_solid"], cfg.get("refine_solvus", True)),
    ) as pool:
        futures = {
            pool.submit(_run_one, {k: float(row[k]) for k in list(FEATURES) + ["Ni"]}): i
            for i, row in comps.iterrows()
        }
        for fut in tqdm(as_completed(futures), total=len(futures), desc="CALPHAD"):
            i = futures[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # noqa: BLE001
                print(f"  failed id {i}: {type(exc).__name__}: {exc}")
                records[i] = None
                base = comps.iloc[i][["id", "Co", "Cr", "Al", "Mo", "Ti", "C", "Ni"]].to_dict()
                base.update({"melt_valid": 0.0, "solid_valid": 0.0})
                rows[i] = base
                continue
            base = comps.iloc[i][["id", "Co", "Cr", "Al", "Mo", "Ti", "C", "Ni"]].to_dict()
            for key in SIG_KEYS:
                base[key] = rec.get(key, np.nan)
            for el in ("Co", "Cr", "Al", "Mo", "Ti", "C"):
                base[f"x_{el}"] = wt_to_mol(rec["comp_wt"]).get(el, 0.0)
            rows[i] = base
            records[i] = rec
            done = [r for r in rows if r is not None]
            pd.DataFrame(done).to_csv(partial, index=False)

    sig = pd.DataFrame([r for r in rows if r is not None])
    sig = sig.sort_values("id").reset_index(drop=True)
    sig.to_csv(out_dir / "independent_signatures.csv", index=False)
    print(f"Finished in {(time.time() - t0) / 60:.1f} min; "
          f"valid melt {int(sig['melt_valid'].sum())}/{len(sig)}, "
          f"valid solid {int(sig['solid_valid'].sum())}/{len(sig)}")

    # Curves for completeness.
    good = [(int(sig.loc[k, "id"]), r) for k, r in enumerate(records) if r is not None]
    if good:
        np.savez_compressed(
            out_dir / "independent_curves.npz",
            ids=np.array([g[0] for g in good], dtype=int),
            T_melt=np.array(good[0][1]["melt_T"]),
            H_melt=np.vstack([r["melt_H"] for _, r in good]),
            Cp_melt=np.vstack([r["melt_Cp"] for _, r in good]),
            liquid_melt=np.vstack([r["melt_liquid"] for _, r in good]),
            T_solid=np.array(good[0][1]["solid_T"]),
            H_solid=np.vstack([r["solid_H"] for _, r in good]),
            Cp_solid=np.vstack([r["solid_Cp"] for _, r in good]),
            gp_solid=np.vstack([r["solid_gp"] for _, r in good]),
            fcc_solid=np.vstack([r["solid_fcc"] for _, r in good]),
        )

    # Evaluation with the archived models, no refitting.
    X = sig[list(FEATURES)].to_numpy(dtype=float)
    masks = valid_masks(sig)
    targets = list(TARGETS) + ["gp_frac_900C"]
    preds = sig[["id"] + list(FEATURES)].copy()
    metrics = []
    for target in targets:
        m = masks[target]
        y = sig.loc[m, target].to_numpy(dtype=float)
        Xv = X[m]
        out = {"target": target, "n": int(m.sum())}
        for fam in ("gp", "pce", "nn"):
            path = out_dir / f"model_{fam}_{target}.joblib"
            if not path.exists():
                continue
            model = joblib.load(path)["model"]
            mu = model.predict(Xv)
            if fam == "gp":
                _, sd = model.predict(Xv, return_std=True)
                out["coverage95"] = float(np.mean(np.abs(y - mu) <= 1.96 * sd))
                out["mean_sd"] = float(np.mean(sd))
                out["rmse_std"] = float(np.sqrt(np.mean(((y - mu) / sd) ** 2)))
            out[f"rmse_{fam}"] = float(np.sqrt(np.mean((y - mu) ** 2)))
            out[f"mae_{fam}"] = float(np.mean(np.abs(y - mu)))
            out[f"bias_{fam}"] = float(np.mean(mu - y))
            out[f"max_abs_{fam}"] = float(np.max(np.abs(y - mu)))
            out[f"r2_{fam}"] = float(1.0 - np.sum((y - mu) ** 2) / np.sum((y - y.mean()) ** 2))
            preds.loc[m, f"{target}_{fam}"] = mu
            if fam == "gp":
                preds.loc[m, f"{target}_gp_std"] = sd
        preds.loc[m, f"{target}_obs"] = y
        metrics.append(out)
        with open(out_dir / "independent_metrics.json", "w", encoding="utf-8") as fh:
            json.dump({"seed": args.seed, "n": len(sig),
                       "min_distance_to_training_wt": float(dmin),
                       "metrics": metrics}, fh, indent=1)
        print(json.dumps(out))

    pd.DataFrame(metrics).to_csv(out_dir / "independent_metrics.csv", index=False)
    preds.to_csv(out_dir / "independent_predictions.csv", index=False)


if __name__ == "__main__":
    main()
