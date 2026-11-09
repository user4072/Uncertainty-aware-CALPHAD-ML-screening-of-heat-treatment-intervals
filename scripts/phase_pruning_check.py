"""Reproducible audit of the six-phase production list.

Two questions are answered for a random subset of the CALPHAD dataset:

1. At 750 and 900 C, does any phase outside the six-phase production list
   become stable with a non-negligible fraction?
2. Does the six-phase list reproduce the gamma-prime solvus obtained with the
   full twelve-phase list?

The first check is run on every sampled alloy at both temperatures.  The
second is more expensive (a full temperature sweep plus the 2 K refinement)
and is run on ``--solvus-n`` alloys.

Outputs
-------
``results/phase_pruning_check.csv``
    Long table: id, T_C, phase_list, phase, fraction.
``results/phase_pruning_summary.json``
    Maximum excluded-phase fraction, solvus comparison and pass/fail flags.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import yaml  # noqa: E402
from pycalphad import Database  # noqa: E402

from calphad_engine import compute_alloy, equilibrium_curve  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")
FULL_PHASES = [
    "FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "SIGMA", "BCC_A2",
    "LIQUID", "M6C", "M7C3", "HCP_A3", "MU_PHASE", "LAVES",
]


def composition(row: pd.Series) -> dict:
    comp = {f: float(row[f]) for f in FEATURES}
    comp["Ni"] = float(row["Ni"])
    return comp


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-alloys", type=int, default=20)
    parser.add_argument("--solvus-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pdens", type=int, default=400)
    parser.add_argument("--out-dir", default=str(ROOT / "results"))
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    db = Database(ROOT / cfg["database"])
    elements = cfg["elements"]
    production = list(cfg["active_phases"])
    excluded = [p for p in FULL_PHASES if p not in production]

    sig = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    sig = sig.loc[(sig["melt_valid"] == 1) & (sig["solid_valid"] == 1)].reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    sample = sig.iloc[np.sort(rng.choice(len(sig), size=args.n_alloys, replace=False))]
    print(f"Sampled {len(sample)} alloys; excluded phases: {excluded}")

    rows = []
    t0 = time.time()
    for k, (_, row) in enumerate(sample.iterrows(), start=1):
        comp = composition(row)
        for T_C in (750.0, 900.0):
            T = 273.15 + T_C
            for label, phases in (("full", FULL_PHASES), ("six", production)):
                res = equilibrium_curve(
                    db, comp, [T, T + 1.0], elements, phases,
                    pdens=args.pdens, output_hm=False,
                )
                if not res.get("valid", False):
                    rows.append({"id": int(row["id"]), "T_C": T_C, "phase_list": label,
                                 "phase": "<invalid>", "fraction": np.nan})
                    continue
                for phase, frac in res["frac"].items():
                    f = float(np.asarray(frac)[0])
                    if f > 1e-6:
                        rows.append({"id": int(row["id"]), "T_C": T_C,
                                     "phase_list": label, "phase": phase, "fraction": f})
        print(f"  [{k}/{len(sample)}] id={int(row['id'])} ({time.time()-t0:.0f}s)")

    long = pd.DataFrame(rows)
    long.to_csv(Path(args.out_dir) / "phase_pruning_check.csv", index=False)

    # A calculation that returned no phase above 1e-6 is recorded as
    # '<invalid>' and must not be read as a phase-fraction difference.
    invalid = long.loc[long["phase"] == "<invalid>", ["id", "T_C", "phase_list"]]
    valid = long.loc[long["phase"] != "<invalid>"]

    # Maximum fraction of any excluded phase, per alloy and temperature.
    ex = valid.loc[(valid["phase_list"] == "full") & valid["phase"].isin(excluded)]
    max_excluded = float(ex["fraction"].max()) if len(ex) else 0.0
    offending = (
        ex.groupby(["id", "phase"])["fraction"].max().reset_index()
        .loc[lambda d: d["fraction"] > 1e-4]
    )

    # Six-phase vs full-phase fractions at the two audit temperatures.
    wide = valid.pivot_table(index=["id", "T_C", "phase_list"], columns="phase",
                             values="fraction", aggfunc="sum").fillna(0.0)
    diff = []
    for (aid, T_C), grp in wide.groupby(level=["id", "T_C"]):
        if set(grp.index.get_level_values("phase_list")) != {"full", "six"}:
            continue
        full = grp.loc[(aid, T_C, "full")]
        six = grp.loc[(aid, T_C, "six")]
        for phase in sorted(set(full.index) | set(six.index)):
            d = abs(float(full.get(phase, 0.0)) - float(six.get(phase, 0.0)))
            diff.append({"id": int(aid), "T_C": float(T_C), "phase": phase, "abs_diff": d})
    diff = pd.DataFrame(diff)
    max_diff = float(diff["abs_diff"].max()) if len(diff) else float("nan")

    # Gamma-prime solvus with the full phase list on a smaller subset.  The
    # melting calculation is unchanged, so only the solid-state result matters.
    T_solid = np.arange(cfg["temperature"]["solid_min"],
                        cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    T_melt = np.arange(cfg["temperature"]["melt_min"],
                       cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    solvus_rows = []
    for _, row in sample.head(args.solvus_n).iterrows():
        rec = compute_alloy(
            db, composition(row), elements, FULL_PHASES, T_melt, T_solid,
            pdens_melt=cfg["pdens_melt"], pdens_solid=args.pdens,
            refine=cfg.get("refine_solvus", True),
        )
        ref = sig.loc[sig["id"] == row["id"]].iloc[0]
        solvus_rows.append({
            "id": int(row["id"]),
            "T_gp_solvus_six": float(ref["T_gp_solvus"]),
            "T_gp_solvus_full": float(rec["T_gp_solvus"]),
            "delta_solvus_K": float(rec["T_gp_solvus"] - ref["T_gp_solvus"]),
            "gp_frac_900C_six": float(ref["gp_frac_900C"]),
            "gp_frac_900C_full": float(rec["gp_frac_900C"]),
            "delta_frac_900C": float(rec["gp_frac_900C"] - ref["gp_frac_900C"]),
        })
        print(f"  solvus id={int(row['id'])}: six {ref['T_gp_solvus']:.1f} K, "
              f"full {rec['T_gp_solvus']:.1f} K")

    solvus = pd.DataFrame(solvus_rows)
    if len(solvus):
        solvus.to_csv(Path(args.out_dir) / "phase_pruning_solvus.csv", index=False)

    summary = {
        "n_alloys": int(len(sample)),
        "temperatures_C": [750.0, 900.0],
        "pdens": args.pdens,
        "excluded_phases": excluded,
        "max_excluded_fraction": max_excluded,
        "excluded_above_1e-4": offending.to_dict(orient="records"),
        "max_six_vs_full_fraction_difference": max_diff,
        "solvus_check": solvus_rows,
        "max_abs_solvus_difference_K": (
            float(solvus["delta_solvus_K"].abs().max()) if len(solvus) else None
        ),
        "max_abs_frac_difference_900C": (
            float(solvus["delta_frac_900C"].abs().max()) if len(solvus) else None
        ),
        "pass": bool(max_excluded <= 1e-4 and (not len(solvus) or
                    float(solvus["delta_solvus_K"].abs().max()) <= 1.0)),
    }
    with open(Path(args.out_dir) / "phase_pruning_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float)[:1200])


if __name__ == "__main__":
    main()
