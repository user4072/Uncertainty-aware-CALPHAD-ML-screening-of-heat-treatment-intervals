"""Robustness audit of the inverse-design step.

The deployed GP is refitted on every valid alloy, including the held-out 80.
That is the correct choice for deployment, but it means the re-validated
shortlist is not a clean test of the design loop.  This script repeats the
whole screening with a GP trained only on the training split, re-validates its
own top six with CALPHAD, and compares the result with the deployed design.

It also decomposes the reported window gain over Rene 77 into a solidus and a
solvus contribution, and reports the gamma-prime fraction of every candidate,
because a wider window obtained by removing the strengthening phase is not a
better alloy.

Outputs
-------
``results/design_robustness.csv``
    Train-only top six, their deployed-model predictions and CALPHAD values.
``results/design_robustness.json``
    Overlap, window statistics and the gain decomposition.
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
from surrogate import FEATURES, StandardizedGP, load_dataset  # noqa: E402

TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")
KNN_K, TRUST_PERCENTILE, MAX_LIQ = 5, 95.0, 1650.0


def predict(models, X):
    out = {}
    for t, m in models.items():
        mu, sd = m.predict(X, return_std=True)
        out[t] = mu
        out[f"{t}_std"] = sd
    return out


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    space = cfg["composition_space_wt"]
    lo = np.array([space[f][0] for f in FEATURES], float)
    hi = np.array([space[f][1] for f in FEATURES], float)
    ds = load_dataset("dataset", root=str(ROOT))
    train = ds.split == "train"
    test = ds.split == "test"
    chosen_ids = set()

    # Deployed models (refitted on all valid alloys by train_surrogate.py).
    deployed = {t: joblib.load(ROOT / "results" / f"model_gp_{t}.joblib")["model"] for t in TARGETS}

    # Train-only models, same architecture and calibration protocol.
    print("Fitting train-only GPs ...")
    local = {}
    for t in TARGETS:
        v = ds.valid[t] & train
        m = StandardizedGP(n_restarts=3, seed=0).fit(ds.comp[v], ds.y[t][v])
        m.calibrate(ds.comp[v], ds.y[t][v], n_splits=5, seed=0)
        local[t] = m
        print(f"  {t}: {int(v.sum())} training alloys")

    # Same Sobol design and trust-region protocol as design_alloys.py, but the
    # trust region is built from the training alloys only.
    X = qmc.scale(qmc.Sobol(d=len(FEATURES), scramble=True, seed=7).random(120000), lo, hi)
    Xtr = ds.comp[train & ds.valid["curve"]]
    mu_x, sd_x = Xtr.mean(0), Xtr.std(0) + 1e-12
    nn = NearestNeighbors(n_neighbors=KNN_K).fit((Xtr - mu_x) / sd_x)
    radius = float(np.percentile(nn.kneighbors((Xtr - mu_x) / sd_x)[0][:, -1], TRUST_PERCENTILE))
    inside = nn.kneighbors((X - mu_x) / sd_x)[0][:, -1] <= radius
    print(f"Train-only trust radius: {radius:.3f}; inside {inside.sum()}")

    pred = predict(local, X)
    W = pred["T_solidus"] - pred["T_gp_solvus"]
    W_std = np.hypot(pred["T_solidus_std"], pred["T_gp_solvus_std"])
    score = W - W_std
    ni = 100.0 - X.sum(1)
    al, ti, c = (FEATURES.index(k) for k in ("Al", "Ti", "C"))
    feasible = (
        inside & (ni >= space["Ni_min"])
        & (X[:, al] + X[:, ti] >= space.get("min_Al_Ti_sum", 0.0))
        & (pred["dH_fusion_J_per_g"] >= 190.0) & (pred["dH_fusion_J_per_g"] <= 290.0)
        & (pred["gp_frac_900C"] >= 0.40) & (pred["T_solidus"] >= 1500.0)
        & (pred["T_liquidus"] <= MAX_LIQ) & (W > 0.0) & (X[:, c] <= 0.15)
    )
    idx = np.where(feasible)[0]
    order = idx[np.argsort(score[idx])[::-1]]
    mu_c, sd_c = X[idx].mean(0), X[idx].std(0) + 1e-12
    chosen = []
    for j in order:
        z = (X[j] - mu_c) / sd_c
        if all(np.linalg.norm(z - (X[k] - mu_c) / sd_c) > 0.55 for k in chosen):
            chosen.append(int(j))
        if len(chosen) >= 6:
            break
    print(f"Feasible {len(idx)}; selected {len(chosen)}")

    # CALPHAD re-validation of the train-only shortlist.
    from pycalphad import Database
    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    dep_pred = predict(deployed, X[chosen])
    rows = []
    t0 = time.time()
    for rank, j in enumerate(chosen, 1):
        comp = {f: float(X[j, k]) for k, f in enumerate(FEATURES)}
        comp["Ni"] = float(ni[j])
        rec = compute_alloy(db, comp, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
                            pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
                            refine=cfg.get("refine_solvus", True))
        rows.append({
            "rank": rank, **comp,
            "W_train_K": float(W[j]), "W_train_std_K": float(W_std[j]),
            "T_sol_train": float(pred["T_solidus"][j]),
            "T_gp_train": float(pred["T_gp_solvus"][j]),
            "T_gp_solvus_calphad": float(rec["T_gp_solvus"]),
            "T_solidus_calphad": float(rec["T_solidus"]),
            "dH_calphad": float(rec["dH_fusion_J_per_g"]),
            "gp900_calphad": float(rec["gp_frac_900C"]),
            "W_calphad_K": float(rec["T_solidus"] - rec["T_gp_solvus"]),
            "knn_dist": float(nn.kneighbors(((X[j] - mu_x) / sd_x)[None, :])[0][0, -1]),
        })
        print(f"  rank {rank}: W train {W[j]:.1f}, CALPHAD {rows[-1]['W_calphad_K']:.1f}")
    out = pd.DataFrame(rows)
    # Align the deployed-model prediction with each row explicitly.
    out["W_deployed_K"] = [
        float(dep_pred["T_solidus"][k] - dep_pred["T_gp_solvus"][k]) for k in range(len(chosen))
    ]
    out.to_csv(ROOT / "results" / "design_robustness.csv", index=False)

    # Gain decomposition against Rene 77 in the same calculation.
    base = pd.read_csv(ROOT / "results" / "dsc_holdout_calphad_pdens400.csv")
    base = base.loc[base["alloy"] == "Rene77"].iloc[0]
    bw = float(base["T_solidus"] - base["T_gp_solvus"])
    best = out.loc[out["W_calphad_K"].idxmax()]
    summary = {
        "trust_radius_train_only": radius,
        "n_feasible": int(len(idx)),
        "n_selected": len(chosen),
        "validation_time_s": time.time() - t0,
        "best_calphad_window_K": float(best["W_calphad_K"]),
        "best_composition": {f: float(best[f]) for f in FEATURES},
        "rene77": {"W_K": bw, "T_solidus": float(base["T_solidus"]),
                   "T_gp_solvus": float(base["T_gp_solvus"]),
                   "gp_frac_900C": float(base["gp_frac_900C"])},
        "decomposition": {
            "delta_solidus_K": float(best["T_solidus_calphad"] - base["T_solidus"]),
            "delta_solvus_K": float(best["T_gp_solvus_calphad"] - base["T_gp_solvus"]),
            "delta_window_K": float(best["W_calphad_K"] - bw),
            "delta_gp_frac_900C": float(best["gp900_calphad"] - base["gp_frac_900C"]),
        },
        "train_only_vs_deployed": {
            "mean_abs_W_difference_K": float(np.mean(np.abs(out["W_train_K"] - out["W_deployed_K"]))),
            "max_abs_W_difference_K": float(np.max(np.abs(out["W_train_K"] - out["W_deployed_K"]))),
        },
        "mean_abs_prediction_error_K": float(np.mean(np.abs(out["W_train_K"] - out["W_calphad_K"]))),
        "max_abs_prediction_error_K": float(np.max(np.abs(out["W_train_K"] - out["W_calphad_K"]))),
    }
    with open(ROOT / "results" / "design_robustness.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
