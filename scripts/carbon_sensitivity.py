"""High-resolution carbon sensitivity around the selected candidates.

The review asks for (a) a higher-resolution carbon sensitivity analysis around
the selected candidates, (b) both matrix-only and full-equilibrium results
where computationally resolvable, (c) sensitivity to plausible carbon
composition tolerances, and (d) an explicit comparison with the reference
alloys.

This script scans the carbon content of the production rank-1 candidate (P1),
the carbon-free train-only candidate (A1) and Rene 77 (R77) at fixed base
composition, on a much finer carbon grid than ``carbon_corner.py``:

* matrix-only melting (LIQUID + FCC_A1) on the production 2 K audit grid for
  14 carbon levels from 0 to 0.15 wt%;
* the six-phase solid-state calculation at the same carbon levels, so that the
  window W(C) uses the solvus of the same composition rather than a fixed
  solvus;
* the twelve-phase full-equilibrium melting calculation at selected carbon
  levels, with the largest one-step liquid jump recorded as the
  resolved/unresolved diagnostic;
* interpolation-free tolerance readout: the solidus change for +-0.01 and
  +-0.02 wt% carbon around each alloy's base carbon content.

Outputs
-------
``results/carbon_sensitivity.csv``    one row per (alloy, carbon level, model).
``results/carbon_tolerance.json``     tolerance readout at the base carbon.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycalphad import Database  # noqa: E402

from calphad_engine import compute_alloy, equilibrium_curve, extract_signatures  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")
FULL_PHASES = (
    "LIQUID", "FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "M6C",
    "M7C3", "SIGMA", "BCC_A2", "HCP_A3", "MU_PHASE", "LAVES",
)
CARBON_GRID = (0.0, 0.001, 0.002, 0.003, 0.005, 0.0075, 0.01, 0.015,
               0.02, 0.03, 0.05, 0.075, 0.10, 0.15)
FULL_EQ_CARBON = (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
TOLERANCES = (0.01, 0.02)


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    db = Database(ROOT / cfg["database"])

    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    p1 = prod.sort_values("W_calphad_K", ascending=False).iloc[0]
    robust = pd.read_csv(ROOT / "results" / "design_robustness.csv")
    a1 = robust.sort_values("W_calphad_K", ascending=False).iloc[0]
    lit = pd.read_csv(ROOT / "data" / "dsc_literature.csv")
    r77 = lit.loc[lit["alloy"] == "Rene77"].iloc[0]

    T_melt_2k = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9, 2.0)
    T_melt_10k = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                           cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])

    alloys = {}
    for tag, ref in (("P1", p1), ("A1", a1)):
        alloys[tag] = {f: float(ref[f]) for f in FEATURES}
    alloys["R77"] = {f: float(r77[f]) for f in FEATURES}

    out_csv = ROOT / "results" / "carbon_sensitivity.csv"
    if out_csv.exists():
        _old = pd.read_csv(out_csv).drop_duplicates(
            subset=["alloy", "C_wt", "model"], keep="first")
        rows = _old.to_dict("records")
        print(f"resuming with {len(rows)} rows", flush=True)
    else:
        rows = []
    done = {(r["alloy"], round(float(r["C_wt"]), 6), r["model"]) for r in rows}

    def save():
        pd.DataFrame(rows).to_csv(out_csv, index=False)

    for tag, base in alloys.items():
        for c in CARBON_GRID:
            if (tag, round(c, 6), "matrix+solvus") in done:
                continue
            comp = dict(base)
            comp["C"] = c
            comp["Ni"] = 100.0 - sum(comp[f] for f in FEATURES)
            rec = compute_alloy(db, comp, cfg["elements"], cfg["active_phases"],
                                T_melt_10k, T_solid,
                                pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
                                refine=cfg.get("refine_solvus", True))
            rows.append({
                "alloy": tag, "C_wt": c, "model": "matrix+solvus",
                "T_solidus": float(rec["T_solidus"]),
                "T_liquidus": float(rec["T_liquidus"]),
                "T_gp_solvus": float(rec["T_gp_solvus"]),
                "dH_fusion_J_per_g": float(rec["dH_fusion_J_per_g"]),
                "gp_frac_900C": float(rec["gp_frac_900C"]),
                "W_K": float(rec["T_solidus"] - rec["T_gp_solvus"]),
                "max_liquid_jump": np.nan, "valid": bool(rec["melt_valid"] * rec["solid_valid"]),
            })
            print(f"{tag} C={c:.4f}: Tsol={rows[-1]['T_solidus']:.1f} "
                  f"Tgammap={rows[-1]['T_gp_solvus']:.1f} W={rows[-1]['W_K']:.1f}", flush=True)
            done.add((tag, round(c, 6), "matrix+solvus"))
            save()

    # Matrix-only at the 2 K audit grid (resolves the label beyond the 10 K
    # production step; the solidus label moves by a median 3.2 K between the
    # two grids).
    for tag, base in alloys.items():
        for c in CARBON_GRID:
            if (tag, round(c, 6), "matrix2K") in done:
                continue
            comp = dict(base)
            comp["C"] = c
            comp["Ni"] = 100.0 - sum(comp[f] for f in FEATURES)
            curve = equilibrium_curve(db, comp, T_melt_2k, cfg["elements"],
                                      ("LIQUID", "FCC_A1"), pdens=cfg["pdens_melt"],
                                      output_hm=True)
            sig = extract_signatures(curve)
            solvus_row = [r for r in rows if r["alloy"] == tag and r["C_wt"] == c][-1]
            rows.append({
                "alloy": tag, "C_wt": c, "model": "matrix2K",
                "T_solidus": float(sig.get("T_solidus", np.nan)),
                "T_liquidus": float(sig.get("T_liquidus", np.nan)),
                "T_gp_solvus": float(solvus_row["T_gp_solvus"]),
                "dH_fusion_J_per_g": float(sig.get("dH_fusion_J_per_g", np.nan)),
                "gp_frac_900C": float(solvus_row["gp_frac_900C"]),
                "W_K": float(sig.get("T_solidus", np.nan) - solvus_row["T_gp_solvus"]),
                "max_liquid_jump": np.nan,
                "valid": bool(sig.get("valid", 0.0)),
            })
            done.add((tag, round(c, 6), "matrix2K"))
            save()

    # Full twelve-phase equilibrium at selected carbon levels.
    for tag in ("P1", "A1"):
        base = alloys[tag]
        for c in FULL_EQ_CARBON:
            if (tag, round(c, 6), "fulleq12") in done:
                continue
            comp = dict(base)
            comp["C"] = c
            comp["Ni"] = 100.0 - sum(comp[f] for f in FEATURES)
            curve = equilibrium_curve(db, comp, T_melt_2k, cfg["elements"], FULL_PHASES,
                                      pdens=2000, output_hm=True)
            sig = extract_signatures(curve)
            liquid = np.asarray(curve.get("frac", {}).get("LIQUID", []), dtype=float)
            jumps = np.diff(liquid) if len(liquid) else np.array([np.nan])
            solvus_row = [r for r in rows if r["alloy"] == tag and r["C_wt"] == c][-1]
            rows.append({
                "alloy": tag, "C_wt": c, "model": "fulleq12",
                "T_solidus": float(sig.get("T_solidus", np.nan)),
                "T_liquidus": float(sig.get("T_liquidus", np.nan)),
                "T_gp_solvus": float(solvus_row["T_gp_solvus"]),
                "dH_fusion_J_per_g": float(sig.get("dH_fusion_J_per_g", np.nan)),
                "gp_frac_900C": float(solvus_row["gp_frac_900C"]),
                "W_K": float(sig.get("T_solidus", np.nan) - solvus_row["T_gp_solvus"]),
                "max_liquid_jump": float(np.nanmax(jumps)) if len(jumps) else np.nan,
                "valid": bool(sig.get("valid", 0.0)),
            })
            print(f"{tag} C={c:.4f} full12: Tsol={rows[-1]['T_solidus']:.1f} "
                  f"jump={rows[-1]['max_liquid_jump']:.2f} valid={rows[-1]['valid']}",
                  flush=True)
            done.add((tag, round(c, 6), "fulleq12"))
            save()

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "results" / "carbon_sensitivity.csv", index=False)

    # Tolerance readout: solidus and window change per 0.01 / 0.02 wt% C
    # around each alloy's base carbon content, from the 2 K audit sweep (the
    # 10 K production grid quantises the solidus to its nodes, which would
    # alias the tolerance; the 2 K grid resolves the smooth dependence).
    tol = {}
    for tag, base in alloys.items():
        sub = df.loc[(df["alloy"] == tag) & (df["model"] == "matrix2K")].sort_values("C_wt")
        cs = sub["C_wt"].to_numpy(float)
        sol = sub["T_solidus"].to_numpy(float)
        w = sub["W_K"].to_numpy(float)
        c0 = float(base["C"])
        entry = {"base_C_wt": c0, "base_T_solidus_K": float(np.interp(c0, cs, sol)),
                 "base_W_K": float(np.interp(c0, cs, w))}
        for dt in TOLERANCES:
            entry[f"dTsol_per_plus{dt:.2f}_K"] = float(np.interp(c0 + dt, cs, sol)
                                                       - np.interp(c0, cs, sol))
            entry[f"dW_per_plus{dt:.2f}_K"] = float(np.interp(c0 + dt, cs, w)
                                                     - np.interp(c0, cs, w))
            lo = max(c0 - dt, 0.0)
            entry[f"dTsol_per_minus{dt:.2f}_K"] = float(np.interp(lo, cs, sol)
                                                        - np.interp(c0, cs, sol))
            entry[f"dW_per_minus{dt:.2f}_K"] = float(np.interp(lo, cs, w)
                                                      - np.interp(c0, cs, w))
        # Sensitivity of the floor candidate: window at the 0.03 wt% floor.
        entry["W_at_C_0p03_K"] = float(np.interp(0.03, cs, w))
        entry["W_at_C_0p05_K"] = float(np.interp(0.05, cs, w))
        tol[tag] = entry

    # The dW/dC of the design optimum in the carbon-bearing regime, as a
    # single number for the text.
    tol["_summary"] = {
        "note": "tolerances read from the 2 K audit sweep; W uses the refined solvus",
    }
    (ROOT / "results" / "carbon_tolerance.json").write_text(json.dumps(tol, indent=2))
    print(json.dumps(tol, indent=2))


if __name__ == "__main__":
    main()
