"""Censoring and local-reliability diagnostics for the design screen.

The review notes that label censoring is composition-dependent (high-Al/high-Ti
solvus censoring, high-C/high-Mo invalid curves) and asks for evidence that
the optimisation does not benefit from uneven label reliability:

1. target-specific training sample counts;
2. the feasible-candidate distribution relative to retained training
   compositions (distance to the nearest composition with a *valid* label for
   each target);
3. prediction errors by local composition region (Al+Ti terciles of the
   held-out set);
4. the distribution of distances from selected candidates to training
   samples, per target validity mask;
5. a sensitivity test that removes candidates near the censoring boundary
   from the feasible cloud and re-ranks.

Also included (review 7.2): prediction-interval calibration by composition
region, distance tercile and predicted-window quartile, with bin counts.

Outputs
-------
``results/censoring_diagnostics.json``
``results/censoring_local_errors.csv``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import FEATURES, load_dataset  # noqa: E402

RESULTS = ROOT / "results"
TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")


def main() -> None:
    ds = load_dataset("dataset", root=str(ROOT))
    train = ds.split == "train"
    test = ds.split == "test"

    # 1. target-specific counts.
    counts = {}
    for t in TARGETS:
        counts[t] = {
            "train_valid": int((train & ds.valid[t]).sum()),
            "test_valid": int((test & ds.valid[t]).sum()),
            "total_valid": int(ds.valid[t].sum()),
        }

    comp = ds.comp
    comp_tr = comp[train & ds.valid["curve"]]
    mu_x, sd_x = comp_tr.mean(axis=0), comp_tr.std(axis=0) + 1e-12
    Z_tr = (comp_tr - mu_x) / sd_x

    # 2./4. distance to the nearest composition with a valid label per target.
    dist_to_valid = {}
    for t in TARGETS:
        nn = NearestNeighbors(n_neighbors=1).fit((comp[train & ds.valid[t]] - mu_x) / sd_x)
        dist_to_valid[t] = nn

    sel = pd.read_csv(RESULTS / "design_validated.csv")
    rows = []
    for _, cand in sel.iterrows():
        z = ((cand[list(FEATURES)].to_numpy(float) - mu_x) / sd_x).reshape(1, -1)
        entry = {"rank": int(cand["rank"])}
        for t in TARGETS:
            entry[f"d_valid_{t}"] = float(dist_to_valid[t].kneighbors(z)[0][0, 0])
        rows.append(entry)
    dsel = pd.DataFrame(rows)

    # Cloud-wide comparison: is the top of the objective unusually far from
    # valid solvus labels?
    cloud = pd.read_csv(RESULTS / "design_cloud.csv")
    z_cloud = (cloud[list(FEATURES)].to_numpy(float) - mu_x) / sd_x
    d_solvus_cloud = dist_to_valid["T_gp_solvus"].kneighbors(z_cloud)[0][:, 0]
    W_cloud = cloud["W_K"].to_numpy(float)
    top = W_cloud >= np.quantile(W_cloud, 0.99)
    cloud_stats = {
        "d_solvus_valid_cloud_median": float(np.median(d_solvus_cloud)),
        "d_solvus_valid_top1pct_median": float(np.median(d_solvus_cloud[top])),
        "d_solvus_valid_top1pct_p90": float(np.percentile(d_solvus_cloud[top], 90)),
        "corr_W_vs_d_solvus_valid": float(np.corrcoef(W_cloud, d_solvus_cloud)[0, 1]),
    }

    # 5. sensitivity test: drop candidates in the censoring-adjacent corner.
    # Corner definition: the solvus-censored training alloys' mean Al/Ti plus
    # one standard deviation, i.e. the region the labels left.
    sig = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    cens = sig[(sig["censored_gp_high"] == 1) & (sig["split"] == "train")]
    keep = sig[(sig["censored_gp_high"] == 0) & (sig["solid_valid"] == 1) & (sig["split"] == "train")]
    corner = {"Al_mean": float(cens["Al"].mean()), "Al_sd": float(cens["Al"].std() or 1.0),
              "Ti_mean": float(cens["Ti"].mean()), "Ti_sd": float(cens["Ti"].std() or 1.0)}
    corner["Al_thr"] = corner["Al_mean"] + 0.5 * corner["Al_sd"]
    corner["Ti_thr"] = corner["Ti_mean"] + 0.5 * corner["Ti_sd"]
    in_corner = (cloud["Al"] >= corner["Al_thr"]) & (cloud["Ti"] >= corner["Ti_thr"])
    W_kept = W_cloud[~in_corner]
    f_kept = cloud["gp_frac_900C"].to_numpy(float)[~in_corner]
    rank1_kept = float(np.max(W_kept))
    # Does the rank-1 composition change?  Compare best-of-kept with the
    # overall best and locate the best kept candidate.
    best_kept_i = np.where(~in_corner, W_cloud, -np.inf).argmax()
    best_all_i = int(np.argmax(W_cloud))
    same = bool(np.allclose(cloud.iloc[best_kept_i][list(FEATURES)],
                            cloud.iloc[best_all_i][list(FEATURES)]))
    # Also require solvus-label distance within the training self-coverage.
    d_tr_solvus = NearestNeighbors(n_neighbors=5).fit(
        (comp[train & ds.valid["T_gp_solvus"]] - mu_x) / sd_x)
    radius_solvus = float(np.percentile(
        d_tr_solvus.kneighbors((comp[train & ds.valid["T_gp_solvus"]] - mu_x) / sd_x)[0][:, -1], 95))
    near_label = d_solvus_cloud <= radius_solvus
    best_near = float(np.max(W_cloud[near_label & ~in_corner])) if (near_label & ~in_corner).any() else None

    # 3. local held-out errors (Al+Ti terciles) and 7.2 calibration by region.
    pred = pd.read_csv(RESULTS / "surrogate_test_predictions.csv")
    test_comp = comp[test]
    al_i, ti_i = FEATURES.index("Al"), FEATURES.index("Ti")
    at = test_comp[:, al_i] + test_comp[:, ti_i]
    terc = pd.qcut(at, 3, labels=["low", "mid", "high"])
    local = []
    z_test_d = NearestNeighbors(n_neighbors=5).fit(Z_tr).kneighbors(
        (test_comp - mu_x) / sd_x)[0][:, -1]
    dterc = pd.qcut(z_test_d, 3, labels=["near", "mid", "far"])
    wq = pd.qcut((pred["T_solidus_gp"] - pred["T_gp_solvus_gp"]).clip(lower=0), 4,
                 labels=["q1", "q2", "q3", "q4"], duplicates="drop")
    for t in ("T_gp_solvus", "T_solidus", "T_liquidus"):
        err = pred[f"{t}_gp"] - pred[f"{t}_dsc"]
        sd = pred[f"{t}_gp_std"]
        cover = np.abs(err) <= 1.959964 * sd
        for axis, bins in (("AlTi", terc), ("dist", dterc), ("Wq", wq)):
            bins_str = np.array([str(b) for b in np.asarray(bins, dtype=object)])
            for lev in sorted(set(bins_str)):
                m = bins_str == lev
                m &= np.isfinite(err.to_numpy())
                local.append({
                    "target": t, "axis": axis, "level": str(lev), "n": int(m.sum()),
                    "rmse_K": float(np.sqrt(np.mean(err.to_numpy()[m] ** 2))) if m.sum() else np.nan,
                    "mae_K": float(np.mean(np.abs(err.to_numpy()[m]))) if m.sum() else np.nan,
                    "cov95": float(np.mean(cover.to_numpy()[m])) if m.sum() else np.nan,
                })
    local_df = pd.DataFrame(local)
    local_df.to_csv(RESULTS / "censoring_local_errors.csv", index=False)

    out = {
        "target_counts": counts,
        "selected_candidate_distances_to_valid_labels": dsel.to_dict(orient="records"),
        "cloud_stats": cloud_stats,
        "corner_definition": corner,
        "corner_screen_sensitivity": {
            "n_cloud": int(len(cloud)),
            "n_in_corner": int(in_corner.sum()),
            "best_W_all_K": float(np.max(W_cloud)),
            "best_W_excl_corner_K": rank1_kept,
            "best_W_excl_corner_and_near_valid_labels_K": best_near,
            "rank1_unchanged": same,
            "solvus_self_coverage_radius": radius_solvus,
        },
    }
    (RESULTS / "censoring_diagnostics.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
