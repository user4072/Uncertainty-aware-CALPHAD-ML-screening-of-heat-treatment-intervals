"""Benchmark of screening strategies at a matched CALPHAD re-validation budget.

A reviewer asked whether the uncertainty penalty in the design score actually
buys better candidates than simpler alternatives.  This script repeats the
Sobol screen with the deployed GP and selects a six-alloy shortlist under three
strategies that share the same constraint set, trust region and diversity
filter:

* ``lcb1``   -- lower confidence bound, W - 1 * sigma_W (the production score);
* ``mean``   -- the deterministic surrogate mean W (no uncertainty penalty);
* ``lcb2``   -- a stronger penalty, W - 2 * sigma_W;
* ``random`` -- six feasible compositions drawn at random, i.e. a direct
  CALPHAD campaign with the same number of evaluations as one shortlist.

The first three are re-validated with the production CALPHAD settings.  The
``random`` arm uses six fresh compositions.  The ``lcb1`` arm re-uses the
already validated production shortlist (``results/design_validated.csv``), so
no duplicate calculations are run.

A fourth reference is read from the training design: the best window among the
480 LHS alloys is the outcome of a direct CALPHAD campaign with 480
evaluations, i.e. the budget that built the surrogate.  This is the fair
"direct CALPHAD" comparator, not a 6-evaluation random draw.

Outputs: ``results/screening_benchmark.csv`` and
``results/screening_benchmark.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import compute_alloy  # noqa: E402
from surrogate import FEATURES, load_dataset  # noqa: E402

DESIGN_TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")
KNN_K = 5
TRUST_PERCENTILE = 95.0
MAX_PREDICTED_LIQUIDUS = 1650.0
MIN_DIST = 0.55


def load_models(models_dir: Path) -> dict:
    return {t: joblib.load(models_dir / f"model_gp_{t}.joblib")["model"] for t in DESIGN_TARGETS}


def predict(models: dict, X: np.ndarray) -> dict:
    out = {}
    for t, model in models.items():
        mu, sd = model.predict(X, return_std=True)
        out[t] = mu
        out[f"{t}_std"] = sd
    return out


def diversity_filter(order, X, mu_c, sd_c, n_select: int) -> list[int]:
    chosen: list[int] = []
    for j in order:
        z = (X[j] - mu_c) / sd_c
        if all(np.linalg.norm(z - (X[k] - mu_c) / sd_c) > MIN_DIST for k in chosen):
            chosen.append(int(j))
        if len(chosen) >= n_select:
            break
    return chosen


def validate(db, cfg, X, idx, W, W_std, d_cand, labels) -> pd.DataFrame:
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    rows = []
    for rank, j in enumerate(idx, start=1):
        comp = {f: float(X[j, k]) for k, f in enumerate(FEATURES)}
        comp["Ni"] = float(100.0 - X[j].sum())
        rec = compute_alloy(
            db, comp, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
            pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
            refine=cfg.get("refine_solvus", True),
        )
        rows.append({
            "rank": rank,
            "strategy": labels,
            **{f: comp[f] for f in FEATURES},
            "Ni": comp["Ni"],
            "W_surrogate_K": float(W[j]),
            "W_surrogate_std_K": float(W_std[j]),
            "T_gp_solvus_calphad": rec["T_gp_solvus"],
            "T_solidus_calphad": rec["T_solidus"],
            "T_liquidus_calphad": rec["T_liquidus"],
            "dH_calphad": rec["dH_fusion_J_per_g"],
            "gp900_calphad": rec["gp_frac_900C"],
            "W_calphad_K": rec["T_solidus"] - rec["T_gp_solvus"],
            "knn_dist": float(d_cand[j, -1]),
            "valid": float(rec["melt_valid"] * rec["solid_valid"]),
        })
        print(f"  [{labels}] rank {rank}: pred {W[j]:.1f} +/- {W_std[j]:.1f} -> "
              f"CALPHAD {rows[-1]['W_calphad_K']:.1f} K")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--models", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--n-candidates", type=int, default=120000)
    parser.add_argument("--n-select", type=int, default=12)
    parser.add_argument("--n-validate", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    space = cfg["composition_space_wt"]
    lo = np.array([space[f][0] for f in FEATURES], dtype=float)
    hi = np.array([space[f][1] for f in FEATURES], dtype=float)
    models = load_models(Path(args.models))

    X = qmc.scale(qmc.Sobol(d=len(FEATURES), scramble=True, seed=args.seed).random(args.n_candidates), lo, hi)
    al_i, ti_i, c_i = FEATURES.index("Al"), FEATURES.index("Ti"), FEATURES.index("C")

    ds = load_dataset("dataset", root=str(ROOT))
    X_train = ds.comp[ds.valid["curve"]]
    mu_x, sd_x = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    nn = NearestNeighbors(n_neighbors=KNN_K).fit((X_train - mu_x) / sd_x)
    d_train, _ = nn.kneighbors((X_train - mu_x) / sd_x)
    radius = float(np.percentile(d_train[:, -1], TRUST_PERCENTILE))
    d_cand, _ = nn.kneighbors((X - mu_x) / sd_x)
    inside = d_cand[:, -1] <= radius

    pred = predict(models, X)
    W = pred["T_solidus"] - pred["T_gp_solvus"]
    W_std = np.sqrt(pred["T_solidus_std"] ** 2 + pred["T_gp_solvus_std"] ** 2)

    ni = 100.0 - X.sum(axis=1)
    feasible = (
        inside
        & (ni >= space["Ni_min"])
        & (X[:, al_i] + X[:, ti_i] >= space.get("min_Al_Ti_sum", 0.0))
        & (pred["dH_fusion_J_per_g"] >= 190.0)
        & (pred["dH_fusion_J_per_g"] <= 290.0)
        & (pred["gp_frac_900C"] >= 0.40)
        & (pred["T_solidus"] >= 1500.0)
        & (pred["T_liquidus"] <= MAX_PREDICTED_LIQUIDUS)
        & (W > 0.0)
        & (X[:, c_i] <= 0.15)
        & (X[:, c_i] >= 0.0)
    )
    idx = np.where(feasible)[0]
    print(f"Feasible: {len(idx)} of {len(X)} (trust radius {radius:.3f})")

    mu_c, sd_c = X[idx].mean(axis=0), X[idx].std(axis=0) + 1e-12

    strategies = {
        "mean": W,
        "lcb1": W - 1.0 * W_std,
        "lcb2": W - 2.0 * W_std,
    }
    selections = {}
    for name, score in strategies.items():
        order = idx[np.argsort(score[idx])[::-1]]
        chosen = diversity_filter(order, X, mu_c, sd_c, args.n_select)
        selections[name] = chosen[: args.n_validate]
        print(f"{name:5s} top-6 predicted W: " +
              ", ".join(f"{W[j]:.1f}" for j in selections[name]))

    rng = np.random.default_rng(args.seed + 1)
    selections["random"] = list(rng.choice(idx, size=args.n_validate, replace=False))
    print("random top-6 predicted W: " +
          ", ".join(f"{W[j]:.1f}" for j in selections["random"]))

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    frames = []

    # Production arm: reuse the validated shortlist instead of recomputing it.
    prod_path = Path(args.out) / "design_validated.csv"
    if prod_path.exists():
        prod = pd.read_csv(prod_path)
        prod["strategy"] = "lcb1"
        prod["W_calphad_K"] = prod["T_solidus_calphad"] - prod["T_gp_solvus_calphad"]
        frames.append(prod[[
            "rank", "strategy", *FEATURES, "Ni", "W_surrogate_K", "W_surrogate_std_K",
            "T_gp_solvus_calphad", "T_solidus_calphad", "T_liquidus_calphad",
            "dH_calphad", "gp900_calphad", "W_calphad_K", "knn_dist", "valid"]])
        # The production screen kept ranks 1-6 of a twelve-alloy list; rebuild
        # their rank numbering so that the comparison uses six rows.
        frames[-1] = frames[-1].head(args.n_validate).copy()
        frames[-1]["rank"] = range(1, len(frames[-1]) + 1)

    t0 = time.time()
    for name in ("mean", "lcb2", "random"):
        frames.append(validate(db, cfg, X, selections[name], W, W_std, d_cand, name))
    elapsed = time.time() - t0

    bench = pd.concat(frames, ignore_index=True)
    bench.to_csv(Path(args.out) / "screening_benchmark.csv", index=False)

    # Direct-CALPHAD reference: the 480-alloy training design is the outcome of
    # a direct campaign with the budget that built the surrogate.  The raw
    # maximum is not a fair comparator because it can violate the design
    # constraints (a low gamma-prime fraction widens the window trivially), so
    # the same constraints are applied to the true CALPHAD labels.
    data_path = ROOT / "data" / "processed" / "dataset_signatures.csv"
    data = pd.read_csv(data_path) if data_path.exists() else None
    direct = {}
    if data is not None and {"T_solidus", "T_gp_solvus"} <= set(data.columns):
        ok = (data.get("melt_valid", 1) == 1) & (data.get("solid_valid", 1) == 1)
        for col in ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C"):
            ok &= np.isfinite(data[col])
        sub = data.loc[ok].copy()
        sub["Ni"] = 100.0 - sub[list(FEATURES)].sum(axis=1)
        sub["W"] = sub["T_solidus"] - sub["T_gp_solvus"]
        feas = (
            (sub["Ni"] >= space["Ni_min"])
            & (sub[["Al", "Ti"]].sum(axis=1) >= space.get("min_Al_Ti_sum", 0.0))
            & sub["dH_fusion_J_per_g"].between(190.0, 290.0)
            & (sub["gp_frac_900C"] >= 0.40)
            & (sub["T_solidus"] >= 1500.0)
            & (sub["T_liquidus"] <= 1650.0)
            & (sub["C"] <= 0.15)
            & (sub["W"] > 0.0)
        )
        direct = {
            "n": int(len(sub)),
            "n_feasible": int(feas.sum()),
            "best": float(sub["W"].max()),
            "best_feasible": float(sub.loc[feas, "W"].max()),
            "median_feasible": float(sub.loc[feas, "W"].median()),
            "n_above_240": int((sub.loc[feas, "W"] > 240.0).sum()),
            "n_above_200": int((sub.loc[feas, "W"] > 200.0).sum()),
        }

    summary = {
        "n_feasible": int(len(idx)),
        "trust_radius": radius,
        "n_validate": args.n_validate,
        "elapsed_s": elapsed,
        "direct_calphad_480": direct,
        "strategies": {},
    }
    rene = 150.1
    for name, grp in bench.groupby("strategy"):
        summary["strategies"][name] = {
            "best_W_calphad_K": float(grp["W_calphad_K"].max()),
            "mean_W_calphad_K": float(grp["W_calphad_K"].mean()),
            "min_W_calphad_K": float(grp["W_calphad_K"].min()),
            "n_above_rene": int((grp["W_calphad_K"] > rene).sum()),
            "bias_calphad_minus_surrogate_K": float((grp["W_calphad_K"] - grp["W_surrogate_K"]).mean()),
            "mae_K": float(np.abs(grp["W_calphad_K"] - grp["W_surrogate_K"]).mean()),
            "max_overprediction_K": float((grp["W_surrogate_K"] - grp["W_calphad_K"]).max()),
            "n_validated": int(len(grp)),
            "all_valid": bool(grp["valid"].min() > 0),
        }

    ref = bench.loc[bench["strategy"] == "lcb1", list(FEATURES)].to_numpy(float)
    summary["overlap_with_lcb1"] = {}
    for name, grp in bench.groupby("strategy"):
        comp = grp[list(FEATURES)].to_numpy(float)
        summary["overlap_with_lcb1"][name] = int(sum(
            bool((np.abs(ref - c) < 0.05).all(axis=1).any()) for c in comp))
    (Path(args.out) / "screening_benchmark.json").write_text(json.dumps(summary, indent=2))

    print("\nSummary")
    for name, s in summary["strategies"].items():
        print(f"  {name:6s} best {s['best_W_calphad_K']:6.1f}  mean {s['mean_W_calphad_K']:6.1f}  "
              f"bias {s['bias_calphad_minus_surrogate_K']:+6.1f}  MAE {s['mae_K']:5.1f}  "
              f"overlap {summary['overlap_with_lcb1'][name]}/6")
    print(f"  direct 480-alloy CALPHAD: best feasible {direct.get('best_feasible', float('nan')):.1f} K "
          f"of {direct.get('n_feasible', 0)} feasible, {direct.get('n_above_240', 0)} above 240 K")
    print(f"Wrote results/screening_benchmark.{{csv,json}} in {elapsed:.0f} s")


if __name__ == "__main__":
    main()
