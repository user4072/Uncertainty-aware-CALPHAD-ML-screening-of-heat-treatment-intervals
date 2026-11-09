"""Sensitivity of the shortlist to the trust-region threshold.

The production screen restricts candidates to the 95th percentile of the
five-nearest-neighbour distance of the training alloys.  That threshold is a
heuristic, as a reviewer noted: composition-space distance is not
thermodynamic similarity.  This script repeats the ranking at several radii
and with no restriction, and reports how the shortlist changes.  No CALPHAD
calculations are run: the purpose is to show how much of the reported window is
controlled by the threshold, and which radius would have to be adopted before
an extrapolated candidate outranks the production shortlist.

Output: ``results/trust_region_sensitivity.csv``.
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
RADII = (1.2, 1.4, 1.6, None, 1.8, 2.0, 2.4, np.inf)  # None = production radius
N_SELECT = 6


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    space = cfg["composition_space_wt"]
    lo = np.array([space[f][0] for f in FEATURES], dtype=float)
    hi = np.array([space[f][1] for f in FEATURES], dtype=float)
    models = {t: joblib.load(ROOT / "results" / f"model_gp_{t}.joblib")["model"] for t in TARGETS}

    X = qmc.scale(qmc.Sobol(d=len(FEATURES), scramble=True, seed=7).random(120000), lo, hi)
    al_i, ti_i, c_i = FEATURES.index("Al"), FEATURES.index("Ti"), FEATURES.index("C")

    ds = load_dataset("dataset", root=str(ROOT))
    X_train = ds.comp[ds.valid["curve"]]
    mu_x, sd_x = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    nn = NearestNeighbors(n_neighbors=5).fit((X_train - mu_x) / sd_x)
    d_train, _ = nn.kneighbors((X_train - mu_x) / sd_x)
    radius = float(np.percentile(d_train[:, -1], 95.0))
    d_cand, _ = nn.kneighbors((X - mu_x) / sd_x)
    dist = d_cand[:, -1]

    pred = {}
    for t, model in models.items():
        mu, sd = model.predict(X, return_std=True)
        pred[t] = mu
        pred[f"{t}_std"] = sd
    W = pred["T_solidus"] - pred["T_gp_solvus"]
    W_std = np.sqrt(pred["T_solidus_std"] ** 2 + pred["T_gp_solvus_std"] ** 2)
    score = W - W_std

    ni = 100.0 - X.sum(axis=1)
    base = (
        (ni >= space["Ni_min"])
        & (X[:, al_i] + X[:, ti_i] >= space.get("min_Al_Ti_sum", 0.0))
        & (pred["dH_fusion_J_per_g"] >= 190.0)
        & (pred["dH_fusion_J_per_g"] <= 290.0)
        & (pred["gp_frac_900C"] >= 0.40)
        & (pred["T_solidus"] >= 1500.0)
        & (pred["T_liquidus"] <= 1650.0)
        & (W > 0.0)
        & (X[:, c_i] <= 0.15)
        & (X[:, c_i] >= 0.0)
    )

    prod = pd.read_csv(ROOT / "results" / "design_candidates.csv").head(N_SELECT)
    prod_comp = prod[list(FEATURES)].to_numpy(float)

    rows = []
    for r in RADII:
        r = radius if r is None else r
        mask = base & (dist <= r)
        idx = np.where(mask)[0]
        if len(idx) == 0:
            continue
        mu_c, sd_c = X[idx].mean(axis=0), X[idx].std(axis=0) + 1e-12
        chosen = []
        for j in idx[np.argsort(score[idx])[::-1]]:
            z = (X[j] - mu_c) / sd_c
            if all(np.linalg.norm(z - (X[k] - mu_c) / sd_c) > 0.55 for k in chosen):
                chosen.append(int(j))
            if len(chosen) >= 12:  # production filter, then the top six
                break
        chosen = chosen[:N_SELECT]
        comp = X[chosen]
        overlap = sum(
            bool((np.abs(prod_comp - c) < 0.05).all(axis=1).any()) for c in comp
        )
        rows.append({
            "radius": (float(r) if np.isfinite(r) else np.nan),
            "unrestricted": (not np.isfinite(r)),
            "n_feasible": int(len(idx)),
            "median_knn": float(np.median(dist[idx])),
            "best_predicted_W_K": float(W[chosen[0]]),
            "mean_predicted_W_K": float(np.mean(W[chosen])),
            "min_predicted_W_K": float(np.min(W[chosen])),
            "mean_std_K": float(np.mean(W_std[chosen])),
            "overlap_with_production": int(overlap),
            "max_knn_selected": float(np.max(dist[chosen])),
        })
        print(f"r={'inf' if not np.isfinite(r) else f'{r:.3f}'}: n={len(idx):6d}  "
              f"best pred {W[chosen[0]]:.1f} +/- {W_std[chosen[0]]:.1f}  "
              f"overlap {overlap}/{N_SELECT}  max knn {np.max(dist[chosen]):.2f}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "trust_region_sensitivity.csv", index=False)
    (ROOT / "results" / "trust_region_sensitivity.json").write_text(
        json.dumps({"production_radius": radius, "rows": rows}, indent=2))
    print("Wrote results/trust_region_sensitivity.csv")


if __name__ == "__main__":
    main()
