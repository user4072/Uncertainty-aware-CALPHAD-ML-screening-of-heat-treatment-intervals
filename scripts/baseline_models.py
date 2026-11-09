"""Simpler baseline regressors for the scalar thermal signatures.

The manuscript compares three surrogate families.  This script adds the
baselines a reviewer would reasonably ask for, using exactly the same split and
per-target validity masks as ``train_surrogate.py``:

* ordinary least squares on the six solute concentrations (linear),
* ridge regression on all quadratic monomials (degree-2 polynomial),
* a random forest.

Output: ``results/baseline_metrics.csv``.  The baselines are diagnostics only;
they are not used by the design stage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sklearn.ensemble import RandomForestRegressor  # noqa: E402
from sklearn.linear_model import LinearRegression, RidgeCV  # noqa: E402
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import PolynomialFeatures, StandardScaler  # noqa: E402

from surrogate import TARGETS, load_dataset  # noqa: E402


def models() -> dict:
    return {
        "linear": Pipeline([
            ("scale", StandardScaler()),
            ("fit", LinearRegression()),
        ]),
        "quadratic": Pipeline([
            ("scale", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, include_bias=False)),
            ("fit", RidgeCV(alphas=np.logspace(-6, 3, 30))),
        ]),
        "random_forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=2, random_state=0, n_jobs=1
        ),
    }


def main() -> None:
    ds = load_dataset("dataset", root=str(ROOT))
    train = ds.split == "train"
    test = ds.split == "test"
    rows = []
    for target in list(TARGETS) + (["gp_frac_900C"] if "gp_frac_900C" in ds.y else []):
        v = ds.valid[target]
        tr, te = train & v, test & v
        Xtr, Xte = ds.comp[tr], ds.comp[te]
        ytr, yte = ds.y[target][tr], ds.y[target][te]
        keep = np.isfinite(yte)
        Xte, yte = Xte[keep], yte[keep]
        for name, model in models().items():
            model.fit(Xtr, ytr)
            pred = model.predict(Xte)
            rows.append({
                "target": target,
                "model": name,
                "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                "mae": float(mean_absolute_error(yte, pred)),
                "r2": float(r2_score(yte, pred)),
                "n_train": int(tr.sum()),
                "n_test": int(len(yte)),
            })
        line = "  ".join(
            f"{r['model']}={r['r2']:.4f}" for r in rows[-len(models()):]
        )
        print(f"{target:20s} {line}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "baseline_metrics.csv", index=False)
    print(f"Wrote {ROOT / 'results' / 'baseline_metrics.csv'}")


if __name__ == "__main__":
    main()
