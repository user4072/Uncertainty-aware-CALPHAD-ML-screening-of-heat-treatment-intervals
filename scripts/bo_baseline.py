"""Sequential Bayesian-optimisation baseline for the heat-treatment screen.

The one-shot screen ranks a fixed Sobol cloud with a single GP and spends its
six CALPHAD evaluations on the six diverse top candidates.  A reviewer can
reasonably ask whether an adaptive strategy, which updates the surrogate after
every expensive evaluation, does better with the same budget.

This script answers that question directly.  Starting from a GP fitted on the
training split only, it repeats six times:

1. rank the same 120 000-point Sobol cloud by the lower confidence bound
   ``W - sigma_W`` under the production constraints (including the 0.03 wt% C
   floor used for the reported design);
2. compute the CALPHAD label of the best not-yet-evaluated candidate;
3. add that label to the training set and refit the five scalar GPs.

The best validated window after six evaluations is compared with the one-shot
screen.  The comparison is at matched budget in the sense that matters---six
CALPHAD evaluations---and both procedures see the same candidate cloud and the
same constraints.

Outputs
-------
``results/bo_baseline.csv``   one row per sequential evaluation.
``results/bo_baseline.json``  summary, overlap with the production shortlist.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

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
MIN_C = 0.03
N_STEPS = 6


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

    X = qmc.scale(qmc.Sobol(d=len(FEATURES), scramble=True, seed=7).random(120000), lo, hi)
    al, ti, c = (FEATURES.index(k) for k in ("Al", "Ti", "C"))
    ni = 100.0 - X.sum(1)

    # Training pool, grown by one CALPHAD label per step.  Each target keeps
    # its own validity mask.
    Xtr = {t: ds.comp[train & ds.valid[t]].copy() for t in TARGETS}
    ytr = {t: ds.y[t][train & ds.valid[t]].copy() for t in TARGETS}

    print("Initial train-only GP fit ...", flush=True)
    models = {}
    for t in TARGETS:
        m = StandardizedGP(n_restarts=3, seed=0).fit(Xtr[t], ytr[t])
        m.calibrate(Xtr[t], ytr[t], n_splits=5, seed=0)
        models[t] = m
        print(f"  {t}: {len(ytr[t])} labels", flush=True)

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])

    evaluated = np.zeros(len(X), dtype=bool)
    rows = []
    best = -np.inf
    t_start = time.time()

    for step in range(1, N_STEPS + 1):
        # Trust region from the current training compositions.
        Xall = np.vstack([Xtr[t] for t in TARGETS])
        mu_x, sd_x = Xall.mean(0), Xall.std(0) + 1e-12
        nn = NearestNeighbors(n_neighbors=KNN_K).fit((Xall - mu_x) / sd_x)
        radius = float(np.percentile(nn.kneighbors((Xall - mu_x) / sd_x)[0][:, -1],
                                     TRUST_PERCENTILE))
        inside = nn.kneighbors((X - mu_x) / sd_x)[0][:, -1] <= radius

        pred = predict(models, X)
        W = pred["T_solidus"] - pred["T_gp_solvus"]
        W_std = np.hypot(pred["T_solidus_std"], pred["T_gp_solvus_std"])
        score = W - W_std
        feasible = (
            inside & ~evaluated & (ni >= space["Ni_min"])
            & (X[:, al] + X[:, ti] >= space.get("min_Al_Ti_sum", 0.0))
            & (pred["dH_fusion_J_per_g"] >= 190.0) & (pred["dH_fusion_J_per_g"] <= 290.0)
            & (pred["gp_frac_900C"] >= 0.40) & (pred["T_solidus"] >= 1500.0)
            & (pred["T_liquidus"] <= MAX_LIQ) & (W > 0.0)
            & (X[:, c] <= 0.15) & (X[:, c] >= MIN_C)
        )
        if not feasible.any():
            print("No feasible candidate left", flush=True)
            break
        idx = np.where(feasible)[0]
        j = int(idx[np.argmax(score[idx])])

        comp = {f: float(X[j, k]) for k, f in enumerate(FEATURES)}
        comp["Ni"] = float(ni[j])
        t0 = time.time()
        rec = compute_alloy(
            db, comp, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
            pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
            refine=cfg.get("refine_solvus", True),
        )
        elapsed = time.time() - t0
        w_cal = float(rec["T_solidus"] - rec["T_gp_solvus"])
        best = max(best, w_cal)
        rows.append({
            "step": step,
            **{f: comp[f] for f in FEATURES},
            "Ni": comp["Ni"],
            "W_predicted_K": float(W[j]),
            "W_std_K": float(W_std[j]),
            "W_calphad_K": w_cal,
            "T_solidus_calphad": float(rec["T_solidus"]),
            "T_gp_solvus_calphad": float(rec["T_gp_solvus"]),
            "dH_calphad": float(rec["dH_fusion_J_per_g"]),
            "gp900_calphad": float(rec["gp_frac_900C"]),
            "knn_dist": float(nn.kneighbors(((X[j] - mu_x) / sd_x).reshape(1, -1))[0][0, -1]),
            "best_so_far_K": best,
            "calphad_s": elapsed,
        })
        print(f"step {step}: predicted {W[j]:.1f} +/- {W_std[j]:.1f} K, "
              f"CALPHAD {w_cal:.1f} K (best {best:.1f}), {elapsed:.0f}s", flush=True)

        evaluated[j] = True
        Xtr["T_gp_solvus"] = np.vstack([Xtr["T_gp_solvus"], X[j]])
        ytr["T_gp_solvus"] = np.append(ytr["T_gp_solvus"], rec["T_gp_solvus"])
        Xtr["T_solidus"] = np.vstack([Xtr["T_solidus"], X[j]])
        ytr["T_solidus"] = np.append(ytr["T_solidus"], rec["T_solidus"])
        if rec["melt_valid"]:
            for t, key in (("T_liquidus", "T_liquidus"), ("dH_fusion_J_per_g", "dH_fusion_J_per_g")):
                Xtr[t] = np.vstack([Xtr[t], X[j]])
                ytr[t] = np.append(ytr[t], rec[key])
        if rec["solid_valid"]:
            Xtr["gp_frac_900C"] = np.vstack([Xtr["gp_frac_900C"], X[j]])
            ytr["gp_frac_900C"] = np.append(ytr["gp_frac_900C"], rec["gp_frac_900C"])

        print(f"  refitting GPs ...", flush=True)
        for t in TARGETS:
            m = StandardizedGP(n_restarts=1, seed=0).fit(Xtr[t], ytr[t])
            m.sd_scale = models[t].sd_scale
            models[t] = m

    table = pd.DataFrame(rows)
    table.to_csv(ROOT / "results" / "bo_baseline.csv", index=False)

    # Overlap with the production shortlist: same cloud, so compare compositions.
    overlap = 0
    prod_path = ROOT / "results" / "design_validated.csv"
    if prod_path.exists():
        prod = pd.read_csv(prod_path)
        for _, r in table.iterrows():
            d = np.sqrt(sum((r[f] - prod[f]) ** 2 for f in FEATURES))
            if (d < 1e-6).any():
                overlap += 1

    summary = {
        "n_steps": int(len(table)),
        "constraint_C_min_wt": MIN_C,
        "best_calphad_K": float(table["best_so_far_K"].iloc[-1]) if len(table) else None,
        "mean_calphad_K": float(table["W_calphad_K"].mean()) if len(table) else None,
        "overlap_with_production_shortlist": int(overlap),
        "total_calphad_s": float(table["calphad_s"].sum()) if len(table) else 0.0,
        "wall_s": time.time() - t_start,
    }
    (ROOT / "results" / "bo_baseline.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
