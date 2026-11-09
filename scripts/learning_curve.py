"""Data-efficiency study for the scalar surrogates.

For each target, the GP and polynomial-chaos surrogates are trained on
subsets of the training split and evaluated on the fixed test split.  The
result quantifies how many CALPHAD alloys are needed before the surrogate
saturates, which is the practical cost driver of the workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import (  # noqa: E402
    TARGETS,
    PolynomialChaos,
    StandardizedGP,
    load_dataset,
    regression_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="dataset")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--sizes", type=int, nargs="*", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = load_dataset(args.prefix, root=str(ROOT))
    train = ds.split == "train"
    test = ds.split == "test"

    sizes_arg = args.sizes
    rng = np.random.default_rng(args.seed)
    rows = []
    for target in TARGETS:
        v = ds.valid.get(target, np.ones(len(ds.ids), dtype=bool))
        tr, te = train & v, test & v
        Xtr, Xte = ds.comp[tr], ds.comp[te]
        ytr = ds.y[target][tr]
        yte = ds.y[target][te]
        n_max = len(Xtr)
        if sizes_arg:
            sizes = sorted({s for s in sizes_arg if 10 <= s <= n_max})
        else:
            sizes = sorted({int(s) for s in np.geomspace(25, n_max, 5)} | {n_max})
        print(f"{target}: train pool {n_max}, test {int(te.sum())}, sizes {sizes}")
        for n in sizes:
            for rep in range(args.repeats):
                if n >= n_max:
                    idx = np.arange(n_max)
                else:
                    idx = rng.choice(n_max, size=n, replace=False)
                gp = StandardizedGP(n_restarts=1, seed=args.seed + rep).fit(Xtr[idx], ytr[idx])
                pce = PolynomialChaos(degree=3).fit(Xtr[idx], ytr[idx])
                for name, pred in (("gp", gp.predict(Xte)), ("pce", pce.predict(Xte))):
                    m = regression_metrics(yte, pred)
                    rows.append({
                        "target": target, "model": name, "n_train": int(n),
                        "repeat": rep, "rmse": m["rmse"], "r2": m["r2"],
                    })
            print(f"  {target:20s} n={n:3d} done")

    data = pd.DataFrame(rows)
    data.to_csv(out_dir / "learning_curve.csv", index=False)
    summary = (data.groupby(["target", "model", "n_train"])
               .agg(rmse_mean=("rmse", "mean"), rmse_std=("rmse", "std"),
                    r2_mean=("r2", "mean"))
               .reset_index())
    summary.to_csv(out_dir / "learning_curve_summary.csv", index=False)
    print(f"Wrote {out_dir / 'learning_curve_summary.csv'}")


if __name__ == "__main__":
    main()
