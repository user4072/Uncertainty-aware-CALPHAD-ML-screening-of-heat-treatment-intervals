"""Inverse design of Ni superalloys with a widened heat-treatment window.

The GP surrogate is evaluated on a Sobol design; candidates are ranked by a
lower-confidence-bound heat-treatment window

    W = T_solidus - T_gamma'

subject to constraints on fusion enthalpy and the gamma-prime fraction at
900 C.  The best candidates are re-checked with the CALPHAD engine.
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

# Candidates must stay inside the region covered by the training design.
# The GP reverts to its prior (and its variance collapses) far from the data,
# so ranking the raw Sobol cloud produces alloys whose predicted window is
# several hundred kelvin too large.  The trust region is the 95th percentile
# of the 5-nearest-neighbour distance of the training alloys in standardised
# composition space.
KNN_K = 5
TRUST_PERCENTILE = 95.0
MAX_PREDICTED_LIQUIDUS = 1650.0  # keeps the fusion integral inside the window


def load_models(models_dir: Path) -> dict:
    models = {}
    for t in DESIGN_TARGETS:
        path = models_dir / f"model_gp_{t}.joblib"
        if not path.exists():
            raise FileNotFoundError(f"Missing {path}; run train_surrogate.py first")
        models[t] = joblib.load(path)["model"]
    return models


def predict(models: dict, X: np.ndarray) -> dict:
    out = {}
    for t, model in models.items():
        mu, sd = model.predict(X, return_std=True)
        out[t] = mu
        out[f"{t}_std"] = sd
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--models", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    parser.add_argument("--n-candidates", type=int, default=120000)
    parser.add_argument("--n-select", type=int, default=12)
    parser.add_argument("--n-validate", type=int, default=6)
    parser.add_argument("--beta", type=float, default=1.0, help="uncertainty penalty on W")
    parser.add_argument("--min-C", type=float, default=0.0,
                        help="lower bound on C in wt% (carbon-floor audit)")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    space = cfg["composition_space_wt"]
    lo = np.array([space[f][0] for f in FEATURES], dtype=float)
    hi = np.array([space[f][1] for f in FEATURES], dtype=float)
    models = load_models(Path(args.models))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sampler = qmc.Sobol(d=len(FEATURES), scramble=True, seed=args.seed)
    U = sampler.random(args.n_candidates)
    X = qmc.scale(U, lo, hi)
    al_i, ti_i, c_i = FEATURES.index("Al"), FEATURES.index("Ti"), FEATURES.index("C")

    # Trust region: standardise with the training statistics and measure the
    # distance to the k-th nearest training alloy.
    ds = load_dataset("dataset", root=str(ROOT))
    X_train = ds.comp[ds.valid["curve"]]
    mu_x, sd_x = X_train.mean(axis=0), X_train.std(axis=0) + 1e-12
    nn = NearestNeighbors(n_neighbors=KNN_K).fit((X_train - mu_x) / sd_x)
    d_train, _ = nn.kneighbors((X_train - mu_x) / sd_x)
    radius = float(np.percentile(d_train[:, -1], TRUST_PERCENTILE))
    d_cand, _ = nn.kneighbors((X - mu_x) / sd_x)
    inside = d_cand[:, -1] <= radius
    print(f"Trust region: k={KNN_K}, radius={radius:.3f} (p{TRUST_PERCENTILE:g} of training), "
          f"inside={int(inside.sum())}/{len(X)}")

    t0_screen = time.time()
    pred = predict(models, X)
    screen_s = time.time() - t0_screen
    W = pred["T_solidus"] - pred["T_gp_solvus"]
    W_std = np.sqrt(pred["T_solidus_std"] ** 2 + pred["T_gp_solvus_std"] ** 2)
    score = W - args.beta * W_std

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
        & (X[:, c_i] >= args.min_C)
    )
    print(f"Candidates: {len(X)}, feasible: {int(feasible.sum())}")
    if feasible.sum() == 0:
        raise RuntimeError("No feasible design found; relax the constraints.")

    idx = np.where(feasible)[0]
    order = idx[np.argsort(score[idx])[::-1]]

    # Keep a subsample of the feasible cloud for the design map figure.
    rng = np.random.default_rng(args.seed)
    cloud_idx = rng.choice(idx, size=min(20000, len(idx)), replace=False)
    cloud = pd.DataFrame({
        "W_K": W[cloud_idx],
        "gp_frac_900C": pred["gp_frac_900C"][cloud_idx],
        "dH_fusion_J_per_g": pred["dH_fusion_J_per_g"][cloud_idx],
        "T_solidus": pred["T_solidus"][cloud_idx],
        "T_gp_solvus": pred["T_gp_solvus"][cloud_idx],
        **{f: X[cloud_idx, k] for k, f in enumerate(FEATURES)},
    })
    cloud.to_csv(out_dir / "design_cloud.csv", index=False)

    # Greedy diversity filter in standardised composition space.
    mu_c, sd_c = X[idx].mean(axis=0), X[idx].std(axis=0) + 1e-12
    chosen = []
    min_dist = 0.55
    for j in order:
        z = (X[j] - mu_c) / sd_c
        if all(np.linalg.norm(z - (X[k] - mu_c) / sd_c) > min_dist for k in chosen):
            chosen.append(int(j))
        if len(chosen) >= args.n_select:
            break

    rows = []
    for rank, j in enumerate(chosen, start=1):
        row = {"rank": rank}
        row.update({f: float(X[j, k]) for k, f in enumerate(FEATURES)})
        row["Ni"] = float(ni[j])
        for t in DESIGN_TARGETS:
            row[t] = float(pred[t][j])
            row[f"{t}_std"] = float(pred[f"{t}_std"][j])
        row["W_K"] = float(W[j])
        row["W_std_K"] = float(W_std[j])
        row["score"] = float(score[j])
        row["knn_dist"] = float(d_cand[j, -1])
        rows.append(row)
    design = pd.DataFrame(rows)
    design.to_csv(out_dir / "design_candidates.csv", index=False)
    print(design[["rank", *FEATURES, "Ni", "W_K", "W_std_K", "T_solidus", "T_gp_solvus",
                  "dH_fusion_J_per_g", "gp_frac_900C"]].round(3).to_string(index=False))

    # ------------------------------------------------------------------
    # CALPHAD re-validation of the top candidates
    # ------------------------------------------------------------------
    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    val_rows = []
    t0_val = time.time()
    for _, cand in design.head(args.n_validate).iterrows():
        comp = {f: float(cand[f]) for f in FEATURES}
        comp["Ni"] = float(cand["Ni"])
        rec = compute_alloy(
            db, comp, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
            pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
            refine=cfg.get("refine_solvus", True),
        )
        val_rows.append({
            "rank": int(cand["rank"]),
            **{f: comp[f] for f in FEATURES},
            "Ni": comp["Ni"],
            "W_surrogate_K": float(cand["W_K"]),
            "W_surrogate_std_K": float(cand["W_std_K"]),
            "T_gp_solvus_calphad": rec["T_gp_solvus"],
            "T_solidus_calphad": rec["T_solidus"],
            "T_liquidus_calphad": rec["T_liquidus"],
            "dH_calphad": rec["dH_fusion_J_per_g"],
            "gp900_calphad": rec["gp_frac_900C"],
            "W_calphad_K": rec["T_solidus"] - rec["T_gp_solvus"],
            "knn_dist": float(cand["knn_dist"]),
            "valid": float(rec["melt_valid"] * rec["solid_valid"]),
        })
    validated = pd.DataFrame(val_rows)
    validated.to_csv(out_dir / "design_validated.csv", index=False)
    val_s = time.time() - t0_val
    print("\nCALPHAD validation")
    print(validated[["rank", "W_surrogate_K", "W_calphad_K", "W_surrogate_std_K",
                     "T_solidus_calphad", "T_gp_solvus_calphad", "T_liquidus_calphad",
                     "dH_calphad", "gp900_calphad", "knn_dist", "valid"]].round(2).to_string(index=False))

    # Reference: Rene 77 baseline from the DSC holdout table when available.
    baseline = None
    baseline_path = out_dir / f"dsc_holdout_calphad_pdens{cfg['pdens_solid']}.csv"
    if not baseline_path.exists():
        baseline_path = out_dir / "dsc_holdout_calphad.csv"
    if baseline_path.exists():
        hold = pd.read_csv(baseline_path)
        ref = hold.loc[hold["alloy"] == "Rene77"]
        if len(ref):
            ref = ref.iloc[0]
            baseline = {
                "alloy": "Rene77",
                "T_solidus": float(ref["T_solidus"]),
                "T_gp_solvus": float(ref["T_gp_solvus"]),
                "dH_fusion_J_per_g": float(ref["dH_fusion_J_per_g"]),
                "gp_frac_900C": float(ref["gp_frac_900C"]),
                "W_K": float(ref["T_solidus"] - ref["T_gp_solvus"]),
            }
    summary = {
        "n_candidates": int(len(X)),
        "n_feasible": int(feasible.sum()),
        "constraints": {
            "dH_range_J_per_g": [190.0, 290.0],
            "gp_frac_900C_min": 0.40,
            "T_solidus_min_K": 1500.0,
            "T_liquidus_max_predicted_K": MAX_PREDICTED_LIQUIDUS,
            "Ni_min_wt": space["Ni_min"],
            "Al_Ti_sum_min_wt": space.get("min_Al_Ti_sum", 0.0),
            "C_min_wt": float(args.min_C),
            "trust_region_knn": {"k": KNN_K, "percentile": TRUST_PERCENTILE,
                                  "radius": radius},
        },
        "top_surrogate": design.head(args.n_validate)[
            ["rank", "W_K", "W_std_K", "T_solidus", "T_gp_solvus", "dH_fusion_J_per_g", "gp_frac_900C"]
        ].to_dict(orient="records"),
        "top_calphad": validated.to_dict(orient="records"),
        "baseline_rene77": baseline,
        "timing": {
            "surrogate_screen_s": float(screen_s),
            "calphad_validation_s": float(val_s),
            "calphad_s_per_alloy": float(val_s / max(len(val_rows), 1)),
        },
    }
    with open(out_dir / "design_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)


if __name__ == "__main__":
    main()
