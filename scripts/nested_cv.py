"""Nested cross-validation of the GP pipeline inside the training split.

The held-out test alloys were reserved before any calculation, but earlier
inspection of the same test split informed the choice of model family and the
constraint set.  This script removes the hyperparameter and calibration part of
that dependence: an outer five-fold split of the 400 training alloys is used,
and within each outer training fold the GP length scales, noise and interval
scale are estimated without any outer-test label.  Quadratic and cubic ridge
regressions are carried through the same outer folds as comparators.

The result is an internal estimate of generalisation for a fixed model family.
It cannot undo the effect of the test alloys on the family choice, which is
stated in the paper rather than corrected here.

Output: ``results/nested_cv.csv``.
"""

from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import TARGETS, StandardizedGP, load_dataset  # noqa: E402


def poly(degree: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("fit", RidgeCV(alphas=np.logspace(-8, 2, 30))),
    ])


def metrics(y: np.ndarray, mu: np.ndarray) -> dict:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y, mu))),
        "mae": float(mean_absolute_error(y, mu)),
        "r2": float(r2_score(y, mu)),
    }


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    ds = load_dataset("dataset", root=str(ROOT))
    train = ds.split == "train"
    targets = list(TARGETS) + (["gp_frac_900C"] if "gp_frac_900C" in ds.y else [])
    rows = []
    per_point = []

    for target in targets:
        valid = ds.valid[target]
        idx = np.where(train & valid)[0]
        X, y = ds.comp[idx], ds.y[target][idx]
        print(f"{target}: {len(idx)} training alloys", flush=True)

        kf = KFold(n_splits=5, shuffle=True, random_state=0)
        mu_gp = np.empty(len(idx))
        sd_gp = np.empty(len(idx))
        mu_q = np.empty(len(idx))
        mu_c = np.empty(len(idx))
        t0 = time.time()

        for fold, (tr, te) in enumerate(kf.split(X), start=1):
            gp = StandardizedGP(n_restarts=3, seed=0).fit(X[tr], y[tr])
            gp.calibrate(X[tr], y[tr], n_splits=5, seed=0)
            m, s = gp.predict(X[te], return_std=True)
            mu_gp[te], sd_gp[te] = m, s

            q = poly(2).fit(X[tr], y[tr])
            mu_q[te] = q.predict(X[te])
            c = poly(3).fit(X[tr], y[tr])
            mu_c[te] = c.predict(X[te])
            print(f"  outer fold {fold}/5 done", flush=True)

        row = {"target": target, "n": int(len(idx))}
        for name, mu in (("gp", mu_gp), ("quadratic", mu_q), ("cubic", mu_c)):
            for key, value in metrics(y, mu).items():
                row[f"{name}_{key}"] = value
        z = (y - mu_gp) / np.maximum(sd_gp, 1e-12)
        row["gp_picp95"] = float(np.mean(np.abs(z) <= 1.96))
        row["gp_sd_scale_median"] = float(np.median(gp.sd_scale))
        row["elapsed_s"] = float(time.time() - t0)
        rows.append(row)

        per_point.append(pd.DataFrame({
            "id": ds.ids[idx],
            "target": target,
            "y": y,
            "gp": mu_gp,
            "gp_sd": sd_gp,
            "quadratic": mu_q,
            "cubic": mu_c,
        }))

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "nested_cv.csv", index=False)
    pd.concat(per_point).to_csv(ROOT / "results" / "nested_cv_predictions.csv", index=False)
    (ROOT / "results" / "nested_cv.json").write_text(
        json.dumps(out.to_dict(orient="records"), indent=2), encoding="utf-8"
    )
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
