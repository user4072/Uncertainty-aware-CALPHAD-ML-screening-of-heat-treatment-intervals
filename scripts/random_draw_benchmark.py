"""Random-draw and diversity benchmark for the design screen.

Rebuilds the production Sobol cloud, predicts with the production GP ensemble,
and compares the selected six-alloy shortlist with many random six-alloy draws
from the same feasible cloud:

* distribution of the best and mean predicted window over the draws;
* the probability that a random draw reaches the shortlist's best window;
* mean pairwise distance in standardised composition space (diversity);
* the fraction of random points in the composition box that satisfy all
  constraints (constraint satisfaction).

Output: ``results/random_draw_benchmark.json``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import FEATURES, load_dataset  # noqa: E402

TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")
KNN_K = 5
TRUST_PERCENTILE = 95.0
MAX_LIQ = 1650.0
NDRAW = 2000
DRAW = 6


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    space = cfg["composition_space_wt"]
    lo = np.array([space[f][0] for f in FEATURES], dtype=float)
    hi = np.array([space[f][1] for f in FEATURES], dtype=float)
    models = {t: joblib.load(ROOT / "results" / f"model_gp_{t}.joblib")["model"] for t in TARGETS}

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

    pred = {}
    for t, m in models.items():
        mu, sd = m.predict(X, return_std=True)
        pred[t], pred[f"{t}_std"] = mu, sd
    W = pred["T_solidus"] - pred["T_gp_solvus"]

    base = (
        inside
        & (ni >= space["Ni_min"])
        & (X[:, al_i] + X[:, ti_i] >= space.get("min_Al_Ti_sum", 0.0))
        & (pred["dH_fusion_J_per_g"] >= 190.0)
        & (pred["dH_fusion_J_per_g"] <= 290.0)
        & (pred["gp_frac_900C"] >= 0.40)
        & (pred["T_solidus"] >= 1500.0)
        & (pred["T_liquidus"] <= MAX_LIQ)
        & (W > 0.0)
        & (X[:, c_i] <= 0.15)
    )
    idx = np.where(base)[0]
    Wf = W[idx]
    Xf = X[idx]
    Zf = (Xf - mu_x) / sd_x

    rng = np.random.default_rng(2024)
    best = np.empty(NDRAW)
    mean = np.empty(NDRAW)
    diversity = np.empty(NDRAW)
    for k in range(NDRAW):
        pick = rng.choice(len(idx), size=DRAW, replace=False)
        w = Wf[pick]
        best[k] = w.max()
        mean[k] = w.mean()
        z = Zf[pick]
        pd_ = [np.linalg.norm(z[a] - z[b]) for a in range(DRAW) for b in range(a + 1, DRAW)]
        diversity[k] = float(np.mean(pd_))

    pop = pd.read_csv(ROOT / "results" / "design_validated.csv")
    Zpop = (pop[[*FEATURES]].to_numpy(float) - mu_x) / sd_x
    zd = [np.linalg.norm(Zpop[a] - Zpop[b]) for a in range(len(Zpop)) for b in range(a + 1, len(Zpop))]

    # Constraint satisfaction of random points drawn uniformly from the box.
    n_box = 20000
    Xb = rng.uniform(lo, hi, size=(n_box, len(FEATURES)))
    nib = 100.0 - Xb.sum(axis=1)
    pb = {}
    for t, m in models.items():
        mu, sd = m.predict(Xb, return_std=True)
        pb[t] = mu
    Wb = pb["T_solidus"] - pb["T_gp_solvus"]
    db, _ = nn.kneighbors((Xb - mu_x) / sd_x)
    ok = (
        (db[:, -1] <= radius)
        & (nib >= space["Ni_min"])
        & (Xb[:, al_i] + Xb[:, ti_i] >= space.get("min_Al_Ti_sum", 0.0))
        & (pb["dH_fusion_J_per_g"] >= 190.0) & (pb["dH_fusion_J_per_g"] <= 290.0)
        & (pb["gp_frac_900C"] >= 0.40) & (pb["T_solidus"] >= 1500.0)
        & (pb["T_liquidus"] <= MAX_LIQ) & (Wb > 0.0) & (Xb[:, c_i] <= 0.15)
    )

    out = {
        "n_feasible": int(len(idx)),
        "ndraw": NDRAW,
        "draw_size": DRAW,
        "best_predicted_K": {
            "median": float(np.median(best)),
            "p025": float(np.percentile(best, 2.5)),
            "p975": float(np.percentile(best, 97.5)),
            "max": float(best.max()),
        },
        "mean_predicted_K": {
            "median": float(np.median(mean)),
            "p025": float(np.percentile(mean, 2.5)),
            "p975": float(np.percentile(mean, 97.5)),
        },
        "prob_best_above_240": float((best > 240.0).mean()),
        "prob_best_above_250": float((best > 250.0).mean()),
        "prob_best_above_254_9": float((best >= 254.9).mean()),
        "diversity_shortlist_mean_pairwise_z": float(np.mean(zd)),
        "diversity_random_median": float(np.median(diversity)),
        "diversity_random_p025": float(np.percentile(diversity, 2.5)),
        "diversity_random_p975": float(np.percentile(diversity, 97.5)),
        "box_constraint_satisfaction": float(ok.mean()),
        "n_box": n_box,
    }
    (ROOT / "results" / "random_draw_benchmark.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
