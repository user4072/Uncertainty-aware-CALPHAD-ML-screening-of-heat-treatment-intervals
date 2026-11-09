"""Train and benchmark the surrogate models on the CALPHAD dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import (  # noqa: E402
    FEATURES,
    TARGETS,
    TARGET_LABELS,
    CurveSurrogate,
    NeuralNetworkSurrogate,
    PolynomialChaos,
    StandardizedGP,
    load_dataset,
    regression_metrics,
    uncertainty_metrics,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prefix", default="dataset")
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    parser.add_argument("--curve-components", type=int, default=8)
    parser.add_argument("--cv", type=int, default=0, help="k-fold CV on the training split (0 = off)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.prefix, root=str(ROOT))
    targets = list(TARGETS) + (["gp_frac_900C"] if "gp_frac_900C" in ds.y else [])
    train = ds.split == "train"
    test = ds.split == "test"
    print(f"Dataset: {len(ds.comp)} alloys ({train.sum()} train / {test.sum()} test)")
    print(f"Targets: {targets}")

    rows = []
    predictions = pd.DataFrame({"id": ds.ids[test]})
    for target in targets:
        # Each target has its own validity mask: a censored solvus does not
        # remove the solidus or the fusion enthalpy of the same alloy.
        v = ds.valid.get(target, np.ones(len(ds.ids), dtype=bool))
        tr, te = train & v, test & v
        Xtr, Xte = ds.comp[tr], ds.comp[te]
        ytr, yte = ds.y[target][tr], ds.y[target][te]
        keep = np.isfinite(yte)
        yte_eval, Xte_eval = yte[keep], Xte[keep]
        print(f"  {target}: {int(tr.sum())} train / {int(te.sum())} test")

        t0 = time.time()
        gp = StandardizedGP(n_restarts=3, seed=args.seed).fit(Xtr, ytr)
        gp.calibrate(Xtr, ytr, n_splits=5, seed=args.seed)
        mu, sd = gp.predict(Xte_eval, return_std=True)
        gp_m = regression_metrics(yte_eval, mu)
        gp_u = uncertainty_metrics(yte_eval, mu, sd)

        pce = PolynomialChaos(degree=3).fit(Xtr, ytr)
        pce_m = regression_metrics(yte_eval, pce.predict(Xte_eval))

        nn = NeuralNetworkSurrogate(seed=args.seed).fit(Xtr, ytr)
        nn_m = regression_metrics(yte_eval, nn.predict(Xte_eval))

        elapsed = time.time() - t0
        rows.append({
            "target": target,
            "gp_rmse": gp_m["rmse"], "gp_mae": gp_m["mae"], "gp_r2": gp_m["r2"],
            **{f"gp_{k}": v for k, v in gp_u.items()},
            "pce_rmse": pce_m["rmse"], "pce_mae": pce_m["mae"], "pce_r2": pce_m["r2"],
            "nn_rmse": nn_m["rmse"], "nn_mae": nn_m["mae"], "nn_r2": nn_m["r2"],
            "train_time_s": elapsed,
            "n_train": int(tr.sum()), "n_test": int(len(yte_eval)),
        })
        insert = np.zeros(int(test.sum()), dtype=bool)
        insert[te[test]] = keep
        predictions.loc[insert, f"{target}_dsc"] = yte_eval
        predictions.loc[insert, f"{target}_gp"] = mu
        predictions.loc[insert, f"{target}_gp_std"] = sd
        predictions.loc[insert, f"{target}_pce"] = pce.predict(Xte_eval)
        predictions.loc[insert, f"{target}_nn"] = nn.predict(Xte_eval)
        print(f"  {target:20s} GP R2={gp_m['r2']:.4f} RMSE={gp_m['rmse']:.3g} "
              f"PICP95={gp_u['picp95']:.2f} | PCE R2={pce_m['r2']:.4f} | NN R2={nn_m['r2']:.4f} "
              f"({elapsed:.1f}s)")

        # Refit GP on every valid alloy for deployment.  The variance
        # calibration is transferred from the train split so that the deployed
        # interval keeps the same coverage behaviour.
        gp_full = StandardizedGP(n_restarts=3, seed=args.seed).fit(ds.comp[v], ds.y[target][v])
        gp_full.sd_scale = gp.sd_scale
        joblib.dump({
            "model": gp_full,
            "model_family": "gp",
            "target": target,
            "features": list(FEATURES),
            "n_train": int(v.sum()),
        }, out_dir / f"model_gp_{target}.joblib")
        joblib.dump({"model": pce, "model_family": "pce", "target": target,
                     "features": list(FEATURES)}, out_dir / f"model_pce_{target}.joblib")
        joblib.dump({"model": nn, "model_family": "nn", "target": target,
                     "features": list(FEATURES)}, out_dir / f"model_nn_{target}.joblib")

    metrics = pd.DataFrame(rows)
    metrics.to_csv(out_dir / "surrogate_metrics.csv", index=False)
    predictions.to_csv(out_dir / "surrogate_test_predictions.csv", index=False)

    # ------------------------------------------------------------------
    # Functional H(T) / Cp(T) surrogate
    # ------------------------------------------------------------------
    if args.curve_components > 0:
        t0 = time.time()
        # The solid-state calculation is a metastable extrapolation above the
        # solidus, so the functional model is trained only on the physically
        # meaningful heat-treatment window.
        curve_mask = ds.T_solid <= 1450.0
        T_curve = ds.T_solid[curve_mask]
        H_curve = ds.H_solid[:, curve_mask]
        vc = ds.valid.get("curve", np.ones(len(ds.ids), dtype=bool))
        trc, tec = train & vc, test & vc
        Xtrc, Xtec = ds.comp[trc], ds.comp[tec]
        curve = CurveSurrogate(n_components=args.curve_components)
        curve.fit(Xtrc, H_curve[trc], T_curve, ds.molar_mass[trc])
        H_true = (H_curve[tec] - H_curve[tec][:, :1]) / ds.molar_mass[tec][:, None]
        H_pred = curve.predict(Xtec)
        Hm = regression_metrics(H_true.ravel(), H_pred.ravel())
        Ht_dev = H_true - H_true.mean(axis=1, keepdims=True)
        Hp_dev = H_pred - H_pred.mean(axis=1, keepdims=True)
        H_shape = regression_metrics(Ht_dev.ravel(), Hp_dev.ravel())
        Cp_true = np.gradient(H_true, T_curve, axis=1)
        Cp_pred = curve.predict_cp(Xte)
        Cpm = regression_metrics(Cp_true.ravel(), Cp_pred.ravel())
        Ct_dev = Cp_true - Cp_true.mean(axis=1, keepdims=True)
        Cp_dev = Cp_pred - Cp_pred.mean(axis=1, keepdims=True)
        Cp_shape = regression_metrics(Ct_dev.ravel(), Cp_dev.ravel())
        var = float(np.sum(curve.pca.explained_variance_ratio_))
        print(f"  H(T) PCA-GP: {args.curve_components} PCs, {100*var:.2f}% variance | "
              f"H RMSE={Hm['rmse']:.2f} J/g R2={Hm['r2']:.5f} "
              f"shape-R2={H_shape['r2']:.4f} | "
              f"Cp RMSE={Cpm['rmse']:.3f} J/g/K R2={Cpm['r2']:.4f} "
              f"shape-R2={Cp_shape['r2']:.4f} ({time.time()-t0:.1f}s)")

        curve.fit(ds.comp[vc], H_curve[vc], T_curve, ds.molar_mass[vc])
        joblib.dump({
            "model": curve, "model_family": "pca-gp",
            "T": T_curve, "n_components": args.curve_components,
            "features": list(FEATURES),
        }, out_dir / "model_curve.joblib")

        curve_metrics = {
            "n_components": args.curve_components,
            "variance_explained": var,
            "H_rmse_J_per_g": Hm["rmse"], "H_r2": Hm["r2"],
            "H_shape_r2": H_shape["r2"],
            "Cp_rmse_J_per_g_K": Cpm["rmse"], "Cp_r2": Cpm["r2"],
            "Cp_shape_r2": Cp_shape["r2"],
            "eval_T_max": 1450.0,
        }
        with open(out_dir / "curve_metrics.json", "w", encoding="utf-8") as fh:
            json.dump(curve_metrics, fh, indent=2)

        np.savez_compressed(
            out_dir / "curve_test_predictions.npz",
            ids=ds.ids[tec], T=T_curve, comp=Xtec,
            H_true=H_true, H_pred=H_pred, Cp_true=Cp_true, Cp_pred=Cp_pred,
        )

    summary = {
        "n_train": int(train.sum()),
        "n_test": int(test.sum()),
        "targets": targets,
        "metrics": metrics.to_dict(orient="records"),
    }
    with open(out_dir / "surrogate_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(f"Wrote {out_dir / 'surrogate_metrics.csv'}")


if __name__ == "__main__":
    main()
