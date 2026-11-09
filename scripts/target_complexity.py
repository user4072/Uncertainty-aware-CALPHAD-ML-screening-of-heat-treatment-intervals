"""How complex is the CALPHAD label surface, and how much does the GP add?

The manuscript claims that the surrogate is accurate; a reviewer may counter
that the targets are smooth functions of six solutes, so almost any smooth
regressor would be accurate.  This script tests that claim with the same split
and per-target validity masks as ``train_surrogate.py``:

* linear OLS on the six solutes,
* ridge on all quadratic monomials (28 terms),
* ridge on all cubic monomials (84 terms),
* additive cubic splines (one 1-D spline per solute, no interactions),
* the production ARD Matern-5/2 GP.

It reports per-target errors, paired bootstrap comparisons of the GP against
the quadratic, the fraction of test points at which the polynomial wins, where
the GP advantage is localised, and learning curves for the polynomial models.

Outputs: results/target_complexity.csv, results/target_complexity_learning.csv,
results/target_complexity.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import FEATURES, TARGETS, load_dataset, make_kernel  # noqa: E402

ALPHAS = np.logspace(-8, 2, 30)


def poly_model(degree: int) -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("fit", RidgeCV(alphas=ALPHAS)),
    ])


def additive_spline(Xtr: np.ndarray, Xte: np.ndarray, ytr: np.ndarray):
    """One cubic spline per solute, summed; no interaction terms."""
    blocks_tr, blocks_te = [], []
    for j in range(Xtr.shape[1]):
        st = SplineTransformer(n_knots=5, degree=3, include_bias=False,
                               extrapolation="linear")
        blocks_tr.append(st.fit_transform(Xtr[:, [j]]))
        blocks_te.append(st.transform(Xte[:, [j]]))
    Atr = np.hstack(blocks_tr)
    Ate = np.hstack(blocks_te)
    model = Pipeline([("scale", StandardScaler()), ("fit", RidgeCV(alphas=ALPHAS))])
    model.fit(Atr, ytr)
    return model.predict(Ate)


def fit_gp(Xtr: np.ndarray, ytr: np.ndarray, restarts: int = 3, seed: int = 0):
    xs = StandardScaler().fit(Xtr)
    ym, ys = float(np.mean(ytr)), float(np.std(ytr)) or 1.0
    gp = GaussianProcessRegressor(
        kernel=make_kernel(Xtr.shape[1]), normalize_y=False,
        n_restarts_optimizer=restarts, random_state=seed,
    ).fit(xs.transform(Xtr), (ytr - ym) / ys)
    return lambda X: gp.predict(xs.transform(X)) * ys + ym


def main() -> None:
    ds = load_dataset("dataset", root=str(ROOT))
    train, test = ds.split == "train", ds.split == "test"
    rows, boot_rows, learn_rows = [], [], []
    summary = {}

    for target in list(TARGETS) + ["gp_frac_900C"]:
        v = ds.valid[target]
        tr, te = train & v, test & v
        Xtr, Xte = ds.comp[tr], ds.comp[te]
        ytr, yte = ds.y[target][tr], ds.y[target][te]
        keep = np.isfinite(yte)
        Xte, yte = Xte[keep], yte[keep]

        preds = {}
        preds["linear"] = Pipeline([
            ("scale", StandardScaler()), ("fit", LinearRegression())
        ]).fit(Xtr, ytr).predict(Xte)
        preds["quadratic"] = poly_model(2).fit(Xtr, ytr).predict(Xte)
        preds["cubic"] = poly_model(3).fit(Xtr, ytr).predict(Xte)
        preds["additive_spline"] = additive_spline(Xtr, Xte, ytr)
        preds["gp"] = fit_gp(Xtr, ytr)(Xte)

        err = {k: yte - p for k, p in preds.items()}
        for name, p in preds.items():
            rows.append({
                "target": target, "model": name,
                "rmse": float(np.sqrt(mean_squared_error(yte, p))),
                "mae": float(mean_absolute_error(yte, p)),
                "r2": float(r2_score(yte, p)),
                "n_train": int(tr.sum()), "n_test": int(len(yte)),
            })

        # Paired bootstrap: GP minus quadratic, RMSE.
        rng = np.random.default_rng(0)
        n = len(yte)
        d = np.empty(4000)
        for b in range(4000):
            idx = rng.integers(0, n, n)
            d[b] = (np.sqrt(np.mean(err["gp"][idx] ** 2))
                    - np.sqrt(np.mean(err["quadratic"][idx] ** 2)))
        boot_rows.append({
            "target": target,
            "delta_rmse_gp_minus_quad": float(np.sqrt(np.mean(err["gp"] ** 2))
                                              - np.sqrt(np.mean(err["quadratic"] ** 2))),
            "ci_lo": float(np.percentile(d, 2.5)),
            "ci_hi": float(np.percentile(d, 97.5)),
            "frac_quad_better": float(np.mean(np.abs(err["quadratic"]) < np.abs(err["gp"]))),
            "ratio_gp_quad_rmse": float(np.sqrt(np.mean(err["gp"] ** 2))
                                        / np.sqrt(np.mean(err["quadratic"] ** 2))),
        })

        # Where does the GP win?  Positive dloc = quadratic worse than GP.
        dloc = np.abs(err["quadratic"]) - np.abs(err["gp"])
        corr = {f: float(np.corrcoef(Xte[:, j], dloc)[0, 1])
                for j, f in enumerate(FEATURES)}
        carbon = Xte[:, FEATURES.index("C")]
        lo, hi = np.quantile(carbon, [1 / 3, 2 / 3])
        summary[target] = {
            "corr_gp_advantage": corr,
            "median_gp_advantage_lowC": float(np.median(dloc[carbon <= lo])),
            "median_gp_advantage_midC": float(np.median(dloc[(carbon > lo) & (carbon <= hi)])),
            "median_gp_advantage_highC": float(np.median(dloc[carbon > hi])),
        }

        # Learning curves for the polynomial models and the GP (3 seeds).
        full = np.where(tr)[0]
        sizes = [25, 46, 86, 161, 250, len(full)]
        sizes = sorted(set(s for s in sizes if s <= len(full)))
        for size in sizes:
            for seed in range(3):
                rng = np.random.default_rng(seed)
                sub = rng.choice(full, size=size, replace=False)
                Xs, ys = ds.comp[sub], ds.y[target][sub]
                for name, model in (("quadratic", poly_model(2)),
                                    ("cubic", poly_model(3)),
                                    ("gp", None)):
                    if name == "gp":
                        pred = fit_gp(Xs, ys, restarts=1, seed=seed)(Xte)
                    else:
                        pred = model.fit(Xs, ys).predict(Xte)
                    learn_rows.append({
                        "target": target, "model": name, "n_train": int(size),
                        "seed": seed,
                        "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
                    })
        print(f"done {target}")

    pd.DataFrame(rows).to_csv(ROOT / "results" / "target_complexity.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(ROOT / "results" / "target_complexity_bootstrap.csv",
                                   index=False)
    pd.DataFrame(learn_rows).to_csv(ROOT / "results" / "target_complexity_learning.csv",
                                    index=False)
    (ROOT / "results" / "target_complexity.json").write_text(json.dumps(summary, indent=2))

    df = pd.DataFrame(rows)
    piv = df.pivot(index="target", columns="model", values="rmse")
    print("\nRMSE by model (same split)")
    print(piv.round(3).to_string())
    print("\nGP/quadratic RMSE ratio")
    print((piv["gp"] / piv["quadratic"]).round(3).to_string())
    b = pd.DataFrame(boot_rows)
    print("\nPaired bootstrap, GP minus quadratic RMSE")
    print(b.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
