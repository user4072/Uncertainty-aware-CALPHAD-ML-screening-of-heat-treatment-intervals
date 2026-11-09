"""Pareto trade-off analysis of the feasible design cloud.

The review asks the single-optimum presentation to be replaced by a trade-off
perspective: the 114.3 K gain is conditional on the gamma-prime floor, and a
Pareto-based design analysis makes the price of each constraint explicit.

This script works on the validated feasible cloud (``design_cloud.csv``,
surrogate predictions for the 56,471 feasible candidates) plus every
CALPHAD-validated point available in ``results/``:

* two-objective Pareto front: maximise the matrix-model interval W and
  maximise the gamma-prime fraction f(900 C);
* front in the (W, T_solidus) plane, i.e. how much solidus is preserved as
  the solvus is depressed;
* floor-price curve: best predicted W as a function of the gamma-prime floor,
  with the CALPHAD-validated anchors at 0.40/0.45/0.50/0.55;
* trade-off with the trust-region distance and the predictive sigma.

Outputs
-------
``results/pareto_front.csv``      non-dominated points of the 2-D front.
``results/pareto_summary.json``   numbers used by the manuscript.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import FEATURES, load_dataset  # noqa: E402

RESULTS = ROOT / "results"


def pareto_front_2d(W: np.ndarray, f: np.ndarray) -> np.ndarray:
    """Indices of the non-dominated points (maximise both coordinates)."""
    order = np.lexsort((-f, -W))  # descending W, then descending f
    idx = order
    keep = []
    best_f = -np.inf
    for j in idx:
        if f[j] > best_f:
            keep.append(j)
            best_f = f[j]
    return np.array(keep, dtype=int)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    space = cfg["composition_space_wt"]

    cloud = pd.read_csv(RESULTS / "design_cloud.csv")
    print(f"cloud: {len(cloud)} feasible candidates")

    ds = load_dataset("dataset", root=str(ROOT))
    X_train = ds.comp[ds.valid["curve"]]
    mu_x, sd_x = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    nn = NearestNeighbors(n_neighbors=5).fit((X_train - mu_x) / sd_x)
    d_cand = nn.kneighbors((cloud[list(FEATURES)].to_numpy(float) - mu_x) / sd_x)[0][:, -1]
    cloud = cloud.assign(knn_dist=d_cand)

    # Predictive sigma of W from the deployed models.
    import joblib

    targets = ("T_gp_solvus", "T_solidus")
    sds = []
    for t in targets:
        m = joblib.load(RESULTS / f"model_gp_{t}.joblib")["model"]
        sds.append(m.predict(cloud[list(FEATURES)].to_numpy(float), return_std=True)[1])
    cloud = cloud.assign(W_std=np.hypot(sds[0], sds[1]))

    W = cloud["W_K"].to_numpy(float)
    f = cloud["gp_frac_900C"].to_numpy(float)
    front_idx = pareto_front_2d(W, f)
    front = cloud.iloc[front_idx].copy()
    front.to_csv(RESULTS / "pareto_front.csv", index=False)

    # Floor-price curve from the cloud, with the validated anchors.
    thresholds = pd.read_csv(RESULTS / "threshold_sensitivity.csv")
    anchors = []
    for floor in (0.40, 0.45, 0.50, 0.55):
        sub = thresholds.loc[np.isclose(thresholds["floor"], floor)]
        val = sub.loc[sub["valid"] > 0, "W_calphad"]
        anchors.append({
            "floor": floor,
            "best_predicted_W_K": float(np.max(W[f >= floor - 1e-9])) if (f >= floor - 1e-9).any() else None,
            "n_cloud_above_floor": int((f >= floor - 1e-9).sum()),
            "best_validated_W_K": float(val.max()) if len(val) else None,
            "n_validated": int(len(val)),
        })

    # Correlation structure of the feasible cloud.
    corr = {
        "W_vs_gp_frac": float(np.corrcoef(W, f)[0, 1]),
        "W_vs_T_gp_solvus": float(np.corrcoef(W, cloud["T_gp_solvus"])[0, 1]),
        "W_vs_T_solidus": float(np.corrcoef(W, cloud["T_solidus"])[0, 1]),
        "W_vs_knn_dist": float(np.corrcoef(W, cloud["knn_dist"])[0, 1]),
        "W_vs_W_std": float(np.corrcoef(W, cloud["W_std"])[0, 1]),
    }

    # How much W is available at a preserved solidus?  Best W among candidates
    # whose predicted solidus is at least that of Rene 77 (1570.2 K).
    rene_sol = 1570.2405355028714
    for dtsol in (0.0, -10.0, -25.0):
        m = cloud["T_solidus"] >= rene_sol + dtsol
        corr[f"best_W_among_Tsol_ge_Rene77_minus{abs(dtsol):.0f}K"] = (
            float(np.max(W[m])) if m.any() else None)
        corr[f"n_among_Tsol_ge_Rene77_minus{abs(dtsol):.0f}K"] = int(m.sum())

    # Fraction of the cloud dominated by the production rank-1 point.
    rank1 = cloud.loc[cloud["W_K"].idxmax()]
    dominated = int(((cloud["W_K"] <= rank1["W_K"]) & (cloud["gp_frac_900C"] <= rank1["gp_frac_900C"])).sum())
    # Points that beat rank-1 on BOTH objectives:
    strictly_better = int(((cloud["W_K"] > rank1["W_K"]) & (cloud["gp_frac_900C"] > rank1["gp_frac_900C"])).sum())

    # Trade-off with distance and uncertainty on the front.
    front_stats = {
        "n_front": int(len(front)),
        "knn_dist_front_p10_p90": [float(front["knn_dist"].quantile(0.1)),
                                   float(front["knn_dist"].quantile(0.9))],
        "W_std_front_max_K": float(front["W_std"].max()),
        "front_max_frac": float(front["gp_frac_900C"].max()),
        "front_W_at_max_frac": float(front.loc[front["gp_frac_900C"].idxmax(), "W_K"]),
    }

    summary = {
        "n_cloud": int(len(cloud)),
        "n_front": int(len(front)),
        "floor_anchors": anchors,
        "correlations": corr,
        "rank1_dominated_count": dominated,
        "rank1_strictly_better_count": strictly_better,
        "front_stats": front_stats,
    }
    (RESULTS / "pareto_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
