"""Assemble the screening-benchmark summary from the stored validation rows.

``screening_benchmark.py`` runs six CALPHAD validations per strategy (about
eleven minutes) and writes ``results/screening_benchmark.csv``.  The summary
JSON is a pure post-processing of that CSV plus the training labels, so it can
be rebuilt without repeating the CALPHAD work.  This script is used when only
the summary schema changes; it reuses exactly the rows the benchmark produced
and does not alter them.

Output: ``results/screening_benchmark.json`` with the same schema as
``screening_benchmark.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["Co", "Cr", "Al", "Mo", "Ti", "C"]
RENE_W_K = 150.1


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    space = cfg["composition_space_wt"]

    bench = pd.read_csv(ROOT / "results" / "screening_benchmark.csv")

    # The feasible-population size and the trust radius are fixed by the seed-7
    # Sobol cloud and the training kNN radii; they are recorded by the
    # trust-region sweep, which evaluates the same cloud.
    n_feasible, radius = np.nan, np.nan
    trust = ROOT / "results" / "trust_region_sensitivity.csv"
    if trust.exists():
        tr = pd.read_csv(trust)
        row = tr.loc[np.isclose(tr["radius"].astype(float), 1.704)]
        if len(row):
            n_feasible = int(row.iloc[0]["n_feasible"])
            radius = float(row.iloc[0]["radius"])

    direct = {}
    data_path = ROOT / "data" / "processed" / "dataset_signatures.csv"
    if data_path.exists():
        data = pd.read_csv(data_path)
        ok = (data.get("melt_valid", 1) == 1) & (data.get("solid_valid", 1) == 1)
        for col in ("T_gp_solvus", "T_solidus", "T_liquidus",
                    "dH_fusion_J_per_g", "gp_frac_900C"):
            ok &= np.isfinite(data[col])
        sub = data.loc[ok].copy()
        sub["Ni"] = 100.0 - sub[FEATURES].sum(axis=1)
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
        "n_feasible": n_feasible,
        "trust_radius": radius,
        "n_validate": int(bench.groupby("strategy").size().max()),
        "direct_calphad_480": direct,
        "strategies": {},
    }
    for name, grp in bench.groupby("strategy"):
        summary["strategies"][name] = {
            "best_W_calphad_K": float(grp["W_calphad_K"].max()),
            "mean_W_calphad_K": float(grp["W_calphad_K"].mean()),
            "min_W_calphad_K": float(grp["W_calphad_K"].min()),
            "n_above_rene": int((grp["W_calphad_K"] > RENE_W_K).sum()),
            "bias_calphad_minus_surrogate_K": float(
                (grp["W_calphad_K"] - grp["W_surrogate_K"]).mean()),
            "mae_K": float(np.abs(grp["W_calphad_K"] - grp["W_surrogate_K"]).mean()),
            "max_overprediction_K": float(
                (grp["W_surrogate_K"] - grp["W_calphad_K"]).max()),
            "n_validated": int(len(grp)),
            "all_valid": bool(grp["valid"].min() > 0),
        }

    ref = bench.loc[bench["strategy"] == "lcb1", FEATURES].to_numpy(float)
    summary["overlap_with_lcb1"] = {}
    for name, grp in bench.groupby("strategy"):
        comp = grp[FEATURES].to_numpy(float)
        summary["overlap_with_lcb1"][name] = int(sum(
            bool((np.abs(ref - c) < 0.05).all(axis=1).any()) for c in comp))

    (ROOT / "results" / "screening_benchmark.json").write_text(
        json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
