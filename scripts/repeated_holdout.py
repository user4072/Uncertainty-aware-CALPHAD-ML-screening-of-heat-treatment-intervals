"""Repeated hold-out evaluation of the GP and PCE surrogates.

The manuscript's headline comparison uses one stratified 400/80 split drawn
before any CALPHAD calculation.  A reviewer asked for repeated, prespecified
splits so that the ranking is not a property of that single draw.  This script
repeats the comparison on five 80/20 splits of the usable alloys, with the same
per-target validity masks, and reports the mean and standard deviation of the
test RMSE together with the number of splits in which the GP is better.

The MLP is excluded to keep the cost bounded (it is the slowest model and it is
never the best in the single-split comparison); GP and PCE are the two models
whose ranking is actually in question for the solvus and liquidus.

Output: ``results/repeated_holdout.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import (  # noqa: E402
    PolynomialChaos,
    StandardizedGP,
    load_dataset,
    regression_metrics,
)

TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")
N_REPEATS = 5


def main() -> None:
    ds = load_dataset("dataset", root=str(ROOT))
    rows = []
    for target in TARGETS:
        v = ds.valid.get(target, np.ones(len(ds.ids), dtype=bool))
        X, y = ds.comp[v], ds.y[target][v]
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
        gp_rmse, pce_rmse = [], []
        for rep in range(N_REPEATS):
            rng = np.random.default_rng(1000 + rep)
            idx = rng.permutation(len(y))
            n_test = max(12, int(round(0.2 * len(y))))
            te, tr = idx[:n_test], idx[n_test:]
            gp = StandardizedGP(n_restarts=2, seed=rep).fit(X[tr], y[tr])
            pce = PolynomialChaos(degree=3).fit(X[tr], y[tr])
            gp_m = regression_metrics(y[te], gp.predict(X[te]))
            pce_m = regression_metrics(y[te], pce.predict(X[te]))
            gp_rmse.append(gp_m["rmse"])
            pce_rmse.append(pce_m["rmse"])
            rows.append({
                "target": target,
                "repeat": rep,
                "n_train": len(tr),
                "n_test": len(te),
                "gp_rmse": gp_m["rmse"],
                "gp_r2": gp_m["r2"],
                "pce_rmse": pce_m["rmse"],
                "pce_r2": pce_m["r2"],
            })
        gp_rmse = np.array(gp_rmse)
        pce_rmse = np.array(pce_rmse)
        print(f"{target:20s} GP {gp_rmse.mean():.3g} +/- {gp_rmse.std(ddof=1):.2g} | "
              f"PCE {pce_rmse.mean():.3g} +/- {pce_rmse.std(ddof=1):.2g} | "
              f"GP better {int((gp_rmse < pce_rmse).sum())}/{N_REPEATS}")

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "repeated_holdout.csv", index=False)
    print("Wrote results/repeated_holdout.csv")


if __name__ == "__main__":
    main()
