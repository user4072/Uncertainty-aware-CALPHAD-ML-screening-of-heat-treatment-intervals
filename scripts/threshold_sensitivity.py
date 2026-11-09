"""Sensitivity of the shortlist to the gamma-prime fraction constraint.

The production screen requires f_gamma'(900 C) >= 0.40.  The second review
asked whether that threshold is doing the work of the design and how the
shortlist changes at stricter floors.  This script rebuilds the same Sobol
cloud (same seed), applies the same trust region and all other constraints,
raises only the gamma-prime floor, ranks by the same lower confidence bound,
applies the same diversity filter, and re-validates the top six with CALPHAD.

Output: ``results/threshold_sensitivity.csv`` and ``.json``.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import compute_alloy  # noqa: E402
from surrogate import FEATURES, load_dataset  # noqa: E402

DESIGN_TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")
KNN_K = 5
TRUST_PERCENTILE = 95.0
MAX_LIQ = 1650.0
FLOORS = (0.40, 0.45, 0.50, 0.55)


def load_models() -> dict:
    return {t: joblib.load(ROOT / "results" / f"model_gp_{t}.joblib")["model"]
            for t in DESIGN_TARGETS}


def predict(models: dict, X: np.ndarray) -> dict:
    out = {}
    for t, m in models.items():
        mu, sd = m.predict(X, return_std=True)
        out[t] = mu
        out[f"{t}_std"] = sd
    return out


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    space = cfg["composition_space_wt"]
    lo = np.array([space[f][0] for f in FEATURES], dtype=float)
    hi = np.array([space[f][1] for f in FEATURES], dtype=float)
    models = load_models()

    sampler = qmc.Sobol(d=len(FEATURES), scramble=True, seed=7)
    X = qmc.scale(sampler.random(120000), lo, hi)
    al_i, ti_i, c_i = FEATURES.index("Al"), FEATURES.index("Ti"), FEATURES.index("C")
    ni = 100.0 - X.sum(axis=1)

    ds = load_dataset("dataset", root=str(ROOT))
    X_train = ds.comp[ds.valid["curve"]]
    mu_x, sd_x = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    nn = NearestNeighbors(n_neighbors=KNN_K).fit((X_train - mu_x) / sd_x)
    d_train, _ = nn.kneighbors((X_train - mu_x) / sd_x)
    radius = float(np.percentile(d_train[:, -1], TRUST_PERCENTILE))
    d_cand, _ = nn.kneighbors((X - mu_x) / sd_x)
    inside = d_cand[:, -1] <= radius

    pred = predict(models, X)
    W = pred["T_solidus"] - pred["T_gp_solvus"]
    W_std = np.sqrt(pred["T_solidus_std"] ** 2 + pred["T_gp_solvus_std"] ** 2)
    score = W - W_std

    base = (
        inside
        & (ni >= space["Ni_min"])
        & (X[:, al_i] + X[:, ti_i] >= space.get("min_Al_Ti_sum", 0.0))
        & (pred["dH_fusion_J_per_g"] >= 190.0)
        & (pred["dH_fusion_J_per_g"] <= 290.0)
        & (pred["T_solidus"] >= 1500.0)
        & (pred["T_liquidus"] <= MAX_LIQ)
        & (W > 0.0)
        & (X[:, c_i] <= 0.15)
    )

    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    prod_comp = prod[[*FEATURES]].to_numpy(float)
    prod_w = float(prod["W_calphad_K"].max())

    db = None
    from pycalphad import Database
    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])

    rows = []
    summary = {}
    for floor in FLOORS:
        feas = base & (pred["gp_frac_900C"] >= floor)
        n_feas = int(feas.sum())
        if n_feas < 6:
            summary[f"{floor:.2f}"] = {"n_feasible": n_feas, "selected": 0}
            print(f"f>={floor:.2f}: n_feasible={n_feas} -- too few candidates")
            continue
        idx = np.where(feas)[0]
        order = idx[np.argsort(score[idx])[::-1]]
        mu_c, sd_c = X[idx].mean(axis=0), X[idx].std(axis=0) + 1e-12
        chosen, min_dist = [], 0.55
        for j in order:
            z = (X[j] - mu_c) / sd_c
            if all(np.linalg.norm(z - (X[k] - mu_c) / sd_c) > min_dist for k in chosen):
                chosen.append(int(j))
            if len(chosen) >= 6:
                break
        print(f"f>={floor:.2f}: n_feasible={n_feas}, selected={len(chosen)}, "
              f"best predicted={W[chosen[0]]:.1f} K")
        for rank, j in enumerate(chosen, start=1):
            comp = {f: float(X[j, k]) for k, f in enumerate(FEATURES)}
            comp["Ni"] = float(ni[j])
            t0 = time.time()
            rec = compute_alloy(
                db, comp, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
                pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
                refine=cfg.get("refine_solvus", True),
            )
            w_cal = float(rec["T_solidus"] - rec["T_gp_solvus"])
            ov = int(np.min(np.abs(prod_comp - X[j]).sum(axis=1)) < 1e-6)
            rows.append({
                "floor": floor, "rank": rank, **comp,
                "f_gp_pred": float(pred["gp_frac_900C"][j]),
                "W_pred": float(W[j]), "W_pred_std": float(W_std[j]),
                "T_solidus_calphad": float(rec["T_solidus"]),
                "T_gp_solvus_calphad": float(rec["T_gp_solvus"]),
                "W_calphad": w_cal,
                "dW": w_cal - float(W[j]),
                "in_production_six": ov,
                "knn_dist": float(d_cand[j, -1]),
                "valid": float(rec["melt_valid"] * rec["solid_valid"]),
            })
            print(f"  rank {rank}: pred {W[j]:.1f}, calphad {w_cal:.1f}, "
                  f"in prod six {ov}, ({time.time()-t0:.0f} s)")
        sub = pd.DataFrame(rows[-len(chosen):])
        summary[f"{floor:.2f}"] = {
            "n_feasible": n_feas,
            "selected": len(chosen),
            "best_predicted_K": float(W[chosen[0]]),
            "best_calphad_K": float(sub["W_calphad"].max()),
            "mean_calphad_K": float(sub["W_calphad"].mean()),
            "bias_K": float((sub["W_calphad"] - sub["W_pred"]).mean()),
            "mae_K": float((sub["W_calphad"] - sub["W_pred"]).abs().mean()),
            "n_in_production_six": int(sub["in_production_six"].sum()),
            "n_above_240": int((sub["W_calphad"] > 240.0).sum()),
            "min_selected_f": float(sub["f_gp_pred"].min()),
        }

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "threshold_sensitivity.csv", index=False)
    summary["production_reference"] = {"best_calphad_K": prod_w}
    with open(ROOT / "results" / "threshold_sensitivity.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print("Wrote results/threshold_sensitivity.csv/.json")


if __name__ == "__main__":
    main()
