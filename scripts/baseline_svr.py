"""Add an RBF support-vector-regression baseline to the comparison.

Uses exactly the split, features and per-target masks of ``train_surrogate.py``.
The penalty and kernel width are selected by five-fold grid search on the
training split alone, so the test alloys are not used for tuning.

Output: ``results/baseline_svr.csv`` (appended to the baseline table by
``make_tables.py``).
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import TARGETS, load_dataset  # noqa: E402


def main() -> None:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

    ds = load_dataset("dataset", root=str(ROOT))
    train, test = ds.split == "train", ds.split == "test"
    targets = list(TARGETS) + (["gp_frac_900C"] if "gp_frac_900C" in ds.y else [])
    rows = []

    for target in targets:
        v = ds.valid[target]
        tr, te = train & v, test & v
        Xtr, Xte = ds.comp[tr], ds.comp[te]
        ytr, yte = ds.y[target][tr], ds.y[target][te]
        keep = np.isfinite(yte)
        Xte, yte = Xte[keep], yte[keep]

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("svr", SVR(kernel="rbf", epsilon=0.01)),
        ])
        grid = {
            "svr__C": np.logspace(-1, 3, 5),
            "svr__gamma": np.logspace(-3, 0, 4),
        }
        search = GridSearchCV(pipe, grid, cv=5, scoring="neg_mean_squared_error", n_jobs=1)
        search.fit(Xtr, ytr)
        pred = search.predict(Xte)
        rows.append({
            "target": target,
            "model": "svr",
            "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "mae": float(mean_absolute_error(yte, pred)),
            "r2": float(r2_score(yte, pred)),
            "n_train": int(tr.sum()),
            "n_test": int(len(yte)),
            "C": float(search.best_params_["svr__C"]),
            "gamma": float(search.best_params_["svr__gamma"]),
        })
        print(f"{target:20s} SVR RMSE={rows[-1]['rmse']:.4g} "
              f"R2={rows[-1]['r2']:.4f} C={rows[-1]['C']:.3g} "
              f"gamma={rows[-1]['gamma']:.3g}", flush=True)

    pd.DataFrame(rows).to_csv(ROOT / "results" / "baseline_svr.csv", index=False)


if __name__ == "__main__":
    main()
