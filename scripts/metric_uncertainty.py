"""Bootstrap uncertainty of the held-out accuracy metrics.

A single 80-alloy test set gives point estimates of RMSE and MAE without a
stated precision.  This script resamples the held-out alloys with replacement
and reports percentile confidence intervals for every model, plus a paired
bootstrap of the difference between the GP and the strongest non-GP baseline.

The inputs are the predictions written by ``train_surrogate.py`` and
``baseline_models.py``; no model is retrained here except the three cheap
baselines, which are refitted on the same split to obtain per-alloy
predictions.

Outputs
-------
``results/metric_uncertainty.csv``
    Point estimate and 95% CI for RMSE and MAE of every model and target.
``results/metric_uncertainty.json``
    Paired GP-minus-baseline differences and the fraction of bootstrap
    resamples in which the GP is better.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

from surrogate import TARGETS, load_dataset  # noqa: E402

N_BOOT = 4000
SEED = 20261030


def baseline_models() -> dict:
    return {
        "linear": Pipeline([("scale", StandardScaler()), ("fit", LinearRegression())]),
        "quadratic": Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("fit", RidgeCV(alphas=np.logspace(-6, 3, 30))),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1
        ),
    }


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a - b)))


def ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main() -> None:
    ds = load_dataset("dataset", root=str(ROOT))
    pred = pd.read_csv(ROOT / "results" / "surrogate_test_predictions.csv")
    train, test = ds.split == "train", ds.split == "test"
    rng = np.random.default_rng(SEED)

    rows, paired = [], {}
    for target in list(TARGETS) + (["gp_frac_900C"] if "gp_frac_900C" in ds.y else []):
        col = f"{target}_dsc"
        if col not in pred:
            continue
        sub = pred.loc[pred[col].notna()].reset_index(drop=True)
        y = sub[col].to_numpy(float)
        ids = sub["id"].to_numpy(int)
        n = len(y)

        # Refit the cheap baselines on the identical split and target mask.
        v = ds.valid[target]
        tr, te = train & v, test & v
        Xtr, ytr = ds.comp[tr], ds.y[target][tr]
        id_to_row = {int(i): k for k, i in enumerate(ds.ids)}
        Xte = ds.comp[[id_to_row[int(i)] for i in ids]]
        models = {"gp": sub[f"{target}_gp"].to_numpy(float),
                  "pce": sub[f"{target}_pce"].to_numpy(float),
                  "nn": sub[f"{target}_nn"].to_numpy(float)}
        for name, model in baseline_models().items():
            model.fit(Xtr, ytr)
            models[name] = model.predict(Xte)

        boot = rng.integers(0, n, size=(N_BOOT, n))
        boot_rmse = {m: np.array([rmse(y[b], p[b]) for b in boot]) for m, p in models.items()}
        boot_mae = {m: np.array([mae(y[b], p[b]) for b in boot]) for m, p in models.items()}

        for name, p in models.items():
            lo_r, hi_r = ci(boot_rmse[name])
            lo_m, hi_m = ci(boot_mae[name])
            rows.append({
                "target": target, "n_test": n, "model": name,
                "rmse": rmse(y, p), "rmse_lo": lo_r, "rmse_hi": hi_r,
                "mae": mae(y, p), "mae_lo": lo_m, "mae_hi": hi_m,
            })

        # Paired comparison against the best non-GP model by point RMSE.
        others = {m: rmse(y, p) for m, p in models.items() if m != "gp"}
        best_other = min(others, key=others.get)
        diff = boot_rmse[best_other] - boot_rmse["gp"]  # positive = GP better
        lo_d, hi_d = ci(diff)
        paired[target] = {
            "compared_with": best_other,
            "rmse_gp": rmse(y, models["gp"]),
            "rmse_other": others[best_other],
            "improvement_pct": 100.0 * (others[best_other] - rmse(y, models["gp"])) / others[best_other],
            "diff_lo": lo_d, "diff_hi": hi_d,
            "prob_gp_better": float(np.mean(diff > 0)),
        }

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "metric_uncertainty.csv", index=False)
    with open(ROOT / "results" / "metric_uncertainty.json", "w", encoding="utf-8") as fh:
        json.dump({"n_boot": N_BOOT, "seed": SEED, "paired_gp_vs_best_baseline": paired},
                  fh, indent=2)
    print(out.round(4).to_string(index=False))
    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
