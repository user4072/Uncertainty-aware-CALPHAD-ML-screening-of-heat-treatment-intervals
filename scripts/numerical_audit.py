"""Numerical reliability audits requested by the second review.

Three label operations are separated so that they are not double counted:

1. Point density: the 150-point backup against the 400-point snapshot taken
   with the *same* (pre-correction) refinement window.  This is the part of
   the label change caused by the grid minimiser.
2. Refinement window: the 400-point pre-correction snapshot against the
   production labels, in which the 2 K refinement spans the whole coarse
   bracket instead of a +/-10 K window around the interpolated crossing.
3. Ordered-branch repair: stored curves before and after the repair pass.

Grouping of refined against unrefined alloys and the target masks are also
reported.

Outputs: ``results/numerical_audit.json`` and ``results/numerical_audit.csv``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ("T_gp_solvus", "gp_frac_900C", "T_solidus", "T_liquidus", "dH_fusion_J_per_g")


def distribution(d: pd.Series) -> dict:
    rec = {
        "n": int(len(d)),
        "median": float(d.median()),
        "mean": float(d.mean()),
        "min": float(d.min()) if len(d) else float("nan"),
        "max": float(d.max()) if len(d) else float("nan"),
        "p05": float(d.quantile(0.05)),
        "p95": float(d.quantile(0.95)),
        "p01": float(d.quantile(0.01)),
        "p99": float(d.quantile(0.99)),
        "frac_abs_gt": {},
    }
    name = d.name or ""
    if name.startswith("gp_frac"):
        for thr in (0.02, 0.05, 0.10):
            rec["frac_abs_gt"][str(thr)] = float((d.abs() > thr).mean())
    elif name.startswith("T_"):
        for thr in (5.0, 10.0, 20.0):
            rec["frac_abs_gt"][str(thr)] = float((d.abs() > thr).mean())
    elif name.startswith("dH"):
        for thr in (2.0, 5.0, 10.0):
            rec["frac_abs_gt"][str(thr)] = float((d.abs() > thr).mean())
    return rec


def main() -> None:
    proc = ROOT / "data" / "processed"
    low = pd.read_csv(proc / "dataset_signatures.csv.pdens150.bak")
    mid = pd.read_csv(proc / "dataset_signatures.csv.pdens400.bak")
    cur = pd.read_csv(proc / "dataset_signatures.csv")
    # Density comparison at fixed refinement protocol.
    m = low.merge(mid, on="id", suffixes=("_150", "_400"))
    # Refinement comparison at fixed density.
    r = mid.merge(cur, on="id", suffixes=("_pre", "_cur"))
    comps = ["Co", "Cr", "Al", "Mo", "Ti", "C"]

    changes = {}
    for t in TARGETS:
        if t == "T_gp_solvus":
            # Compare only labels that are resolved and uncensored on both
            # sides; a censored label is a boundary flag, not a measurement.
            use = ((m["censored_gp_high_150"] == 0) & (m["censored_gp_high_400"] == 0)
                   & m[f"{t}_150"].notna() & m[f"{t}_400"].notna())
        else:
            use = pd.Series(True, index=m.index)
        d = (m[f"{t}_400"] - m[f"{t}_150"])[use].dropna()
        d.name = t
        changes[t] = distribution(d)

    # Refinement-window correction (solvus only, since the fine grid is used
    # only for the gamma-prime transition).
    use = ((r["censored_gp_high_pre"] == 0) & (r["censored_gp_high_cur"] == 0)
           & r["T_gp_solvus_pre"].notna() & r["T_gp_solvus_cur"].notna())
    dref = (r["T_gp_solvus_cur"] - r["T_gp_solvus_pre"])[use].dropna()
    dref.name = "T_gp_solvus"
    refinement = distribution(dref)
    refinement["n_unrefined_before"] = int(
        (mid["solvus_refined"].fillna(0).astype(int) == 0).sum())
    refinement["n_unrefined_after"] = int(
        (cur["solvus_refined"].fillna(0).astype(int) == 0).sum())

    # Ordered-branch repair: compare the stored curves before and after the
    # repair pass (the arrays are untouched by the refinement correction).
    rep_curves = {}
    try:
        a = np.load(proc / "dataset_curves.npz.prebracket.bak")
        b = np.load(proc / "dataset_curves.npz")
        dgp = np.abs(np.nan_to_num(b["gp_solid"] - a["gp_solid"]))
        per_alloy = dgp.max(axis=1)
        tier = a["T_solid"]
        j900 = int(np.argmin(np.abs(tier - 1173.15)))
        changed = np.where(per_alloy > 1e-6)[0]
        # Effect on the solvus label, holding the extraction rule fixed: both
        # archives are reduced with the same rule, so the difference isolates
        # the repair from the refinement.
        sys.path.insert(0, str(ROOT / "src"))
        from calphad_engine import solvus_from_fractions  # noqa: E402

        d_sol = []
        for i in changed:
            ta, _ = solvus_from_fractions(tier, a["gp_solid"][i], a["fcc_solid"][i])
            tb, _ = solvus_from_fractions(tier, b["gp_solid"][i], b["fcc_solid"][i])
            if np.isfinite(ta) and np.isfinite(tb):
                d_sol.append(tb - ta)
        rep_ids = set(a["ids"][changed].tolist())
        grp_rep = cur.loc[cur["id"].isin(rep_ids)]
        grp_unr = cur.loc[~cur["id"].isin(rep_ids)]
        rep_curves = {
            "n_curves_changed": int(len(changed)),
            "max_abs_dgp": float(per_alloy.max()),
            "max_abs_dgp_900C": float(np.abs(b["gp_solid"][:, j900]
                                             - a["gp_solid"][:, j900]).max()),
            "solvus_max_abs_diff_K": float(np.max(np.abs(d_sol))) if d_sol else None,
            "n_solvus_moved_gt_0p1K": int(np.sum(np.abs(d_sol) > 0.1)),
            "group": {
                "n_repaired": int(len(grp_rep)),
                "n_unrepaired": int(len(grp_unr)),
                "mean_wt_repaired": {c: float(grp_rep[c].mean()) for c in comps},
                "mean_wt_unrepaired": {c: float(grp_unr[c].mean()) for c in comps},
                "n_test_repaired": int((grp_rep["split"] == "test").sum()),
                "n_test_unrepaired": int((grp_unr["split"] == "test").sum()),
            },
        }
    except Exception as exc:  # noqa: BLE001
        rep_curves = {"error": str(exc)}

    # Refined versus unrefined alloys.  Note that solvus_refined records the
    # refinement, not the ordered-branch repair (those are counted separately).
    refined = cur["solvus_refined"].fillna(0).astype(int) == 1
    groups = {}
    for name, mask in (("refined", refined), ("unrefined", ~refined)):
        sub = cur.loc[mask]
        groups[name] = {
            "n": int(mask.sum()),
            "mean_wt": {c: float(sub[c].mean()) for c in comps},
            "frac_censored_gp": float(sub["censored_gp_high"].fillna(0).mean()),
            "frac_solid_invalid": float((sub["solid_valid"] == 0).mean()),
            "frac_test": float((sub["split"] == "test").mean()),
        }

    # Test errors by refinement status.
    pred = pd.read_csv(ROOT / "results" / "surrogate_test_predictions.csv")
    pred = pred.merge(cur[["id", "solvus_refined", "C", "Mo"]], on="id", how="left")
    pred["refined"] = pred["solvus_refined"].fillna(0).astype(int) == 1
    errs = {}
    for t in TARGETS:
        col = f"{t}_gp"
        if col not in pred:
            continue
        e = (pred[col] - pred[f"{t}_dsc"]).abs()
        errs[t] = {}
        for name, mask in (("refined", pred["refined"]), ("unrefined", ~pred["refined"])):
            v = e[mask].dropna()
            errs[t][name] = {"n": int(len(v)), "mae": float(v.mean()) if len(v) else None}
        allv = e.dropna()
        errs[t]["all"] = {"n": int(len(allv)), "mae": float(allv.mean()) if len(allv) else None}

    # Mask / censoring by composition.
    masks = {}
    for target in ("melt_valid", "solid_valid", "censored_gp_high",
                   "censored_solidus_low", "censored_liquidus_high"):
        flag = cur[target].fillna(0).astype(float)
        if target.endswith("valid"):
            bad = flag == 0
        else:
            bad = flag == 1
        good = ~bad & cur[target].notna()
        if bad.sum() == 0 or good.sum() == 0:
            masks[target] = {"n_bad": int(bad.sum())}
            continue
        smd = {}
        for c in ("C", "Mo", "Cr", "Al", "Ti", "Co"):
            sd = float(cur[c].std())
            smd[c] = float((cur.loc[bad, c].mean() - cur.loc[good, c].mean()) / sd) if sd else 0.0
        masks[target] = {
            "n_bad": int(bad.sum()),
            "frac_bad": float(bad.mean() * cur[target].notna().mean()),
            "mean_wt_bad": {c: float(cur.loc[bad, c].mean()) for c in ("C", "Mo")},
            "mean_wt_good": {c: float(cur.loc[good, c].mean()) for c in ("C", "Mo")},
            "smd": smd,
        }

    # Complete-case count and per-target training counts.
    complete = cur[list(TARGETS)].notna().all(axis=1) & (cur["melt_valid"] == 1) & (cur["solid_valid"] == 1)
    counts = {t: int((cur[t].notna() & (cur["melt_valid"] == 1) & (cur["solid_valid"] == 1)).sum())
              for t in TARGETS}
    out = {
        "label_changes_150_to_400": changes,
        "refinement_correction": refinement,
        "repair_curves": rep_curves,
        "refinement_groups": groups,
        "test_errors_by_refinement": errs,
        "masks": masks,
        "complete_case": int(complete.sum()),
        "target_counts": counts,
    }
    (ROOT / "results" / "numerical_audit.json").write_text(json.dumps(out, indent=2), encoding="utf-8")

    rows = []
    for t, rec in changes.items():
        rows.append({"quantity": t, **{k: v for k, v in rec.items() if k != "frac_abs_gt"},
                     **{f"frac_gt_{k}": v for k, v in rec["frac_abs_gt"].items()}})
    pd.DataFrame(rows).to_csv(ROOT / "results" / "numerical_audit.csv", index=False)
    print(json.dumps(out, indent=2)[:4000])
    print("\nWrote results/numerical_audit.json/.csv")


if __name__ == "__main__":
    main()
