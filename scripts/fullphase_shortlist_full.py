"""Complete full-equilibrium and phase-assemblage audit of the shortlist.

The review asks for (a) quantitative phase-assemblage comparison for *all*
feasible shortlist candidates, (b) explicit identification of unresolved
melting calculations, (c) resolution sensitivity of the reported rankings and
(d) sensitivity to phase inclusion.

For every production candidate (ranks 1-6) and the carbon-free train-only
candidate A1 this script computes:

* the twelve-phase full-equilibrium melting calculation at the published
  2 K / 2,000-point setting (ranks 1, 3, 6 and A1 already exist in
  ``results/fullphase_design.csv`` and are re-used), with the largest
  one-step liquid-fraction jump as the resolved/unresolved diagnostic;
* the six-phase and twelve-phase assemblages at 750 and 900 degrees C and at
  50 K and 10 K below each candidate's matrix solidus, i.e. the phases that
  a real solution treatment would encounter just below the first liquid.

Outputs
-------
``results/fullphase_design.csv``      extended with the missing candidates.
``results/shortlist_assemblages.csv`` one row per (alloy, phase list, probe).
``results/fullphase_flags.json``      resolved/unresolved flag per candidate.
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

from calphad_engine import equilibrium_curve, extract_signatures  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")
FULL_PHASES = (
    "LIQUID", "FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "M6C",
    "M7C3", "SIGMA", "BCC_A2", "HCP_A3", "MU_PHASE", "LAVES",
)
SIX_PHASES = ("FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "SIGMA", "BCC_A2")


def assemblage(db, cfg, comp, temp: float, phases) -> dict:
    """Phase fractions at one temperature; the grid is two points wide because
    the condition builder derives a step from T[1]-T[0]."""
    curve = equilibrium_curve(db, comp, np.array([temp, temp + 1.0]), cfg["elements"],
                              [p for p in phases if p != "LIQUID"], pdens=2000,
                              output_hm=False)
    fr = curve.get("frac", {})
    return {k: float(v[0]) for k, v in fr.items()}


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    db = Database(ROOT / cfg["database"])

    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    robust = pd.read_csv(ROOT / "results" / "design_robustness.csv")
    a1 = robust.sort_values("W_calphad_K", ascending=False).iloc[0]

    T_2k = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9, 2.0)

    fp_path = ROOT / "results" / "fullphase_design.csv"
    have = pd.read_csv(fp_path) if fp_path.exists() else pd.DataFrame()
    have_alloys = {"rank" + str(x).replace("rank", "") for x in have["rank"]} if len(have) else set()

    # --- 12-phase melting for the missing candidates ------------------------
    new_rows = []
    for _, cand in prod.iterrows():
        tag = f"rank{int(cand['rank'])}"
        if tag in have_alloys:
            continue
        comp = {f: float(cand[f]) for f in FEATURES}
        comp["Ni"] = float(cand["Ni"])
        curve = equilibrium_curve(db, comp, T_2k, cfg["elements"], FULL_PHASES,
                                  pdens=2000, output_hm=True)
        sig = extract_signatures(curve)
        liquid = np.asarray(curve.get("frac", {}).get("LIQUID", []), dtype=float)
        jumps = np.diff(liquid) if len(liquid) else np.array([np.nan])
        w_matrix = float(cand["T_solidus_calphad"] - cand["T_gp_solvus_calphad"])
        t_full = float(sig.get("T_solidus", np.nan))
        new_rows.append({
            "rank": tag,
            "step_K": 2.0, "pdens": 2000, "C_wt": float(cand["C"]),
            "T_solidus_matrix_K": float(cand["T_solidus_calphad"]),
            "T_solidus_full_K": t_full,
            "T_liquidus_full_K": float(sig.get("T_liquidus", np.nan)),
            "T_gp_solvus_K": float(cand["T_gp_solvus_calphad"]),
            "W_matrix_K": w_matrix,
            "W_full_K": t_full - float(cand["T_gp_solvus_calphad"]) if np.isfinite(t_full) else np.nan,
            "dW_full_minus_matrix_K": (t_full - float(cand["T_gp_solvus_calphad"]) - w_matrix
                                       if np.isfinite(t_full) else np.nan),
            "liquid_max": float(np.nanmax(liquid)) if len(liquid) else np.nan,
            "max_liquid_jump_per_step": float(np.nanmax(jumps)) if len(jumps) else np.nan,
            "valid": sig.get("valid", 0.0),
            "error": sig.get("error", ""),
        })
        print(f"{tag} full12: Tsol_full={new_rows[-1]['T_solidus_full_K']:.1f} "
              f"jump={new_rows[-1]['max_liquid_jump_per_step']:.2f} "
              f"valid={new_rows[-1]['valid']} err={new_rows[-1]['error'][:60]}", flush=True)

    if "A1" not in have_alloys:
        a1_comp = {f: float(a1[f]) for f in FEATURES}
        a1_comp["Ni"] = float(a1["Ni"])
        curve = equilibrium_curve(db, a1_comp, T_2k, cfg["elements"], FULL_PHASES,
                                  pdens=2000, output_hm=True)
        sig = extract_signatures(curve)
        liquid = np.asarray(curve.get("frac", {}).get("LIQUID", []), dtype=float)
        jumps = np.diff(liquid) if len(liquid) else np.array([np.nan])
        w_matrix = float(a1["T_solidus_calphad"] - a1["T_gp_solvus_calphad"])
        t_full = float(sig.get("T_solidus", np.nan))
        new_rows.append({
            "rank": "A1",
            "step_K": 2.0, "pdens": 2000, "C_wt": float(a1["C"]),
            "T_solidus_matrix_K": float(a1["T_solidus_calphad"]),
            "T_solidus_full_K": t_full,
            "T_liquidus_full_K": float(sig.get("T_liquidus", np.nan)),
            "T_gp_solvus_K": float(a1["T_gp_solvus_calphad"]),
            "W_matrix_K": w_matrix,
            "W_full_K": t_full - float(a1["T_gp_solvus_calphad"]) if np.isfinite(t_full) else np.nan,
            "dW_full_minus_matrix_K": (t_full - float(a1["T_gp_solvus_calphad"]) - w_matrix
                                       if np.isfinite(t_full) else np.nan),
            "liquid_max": float(np.nanmax(liquid)) if len(liquid) else np.nan,
            "max_liquid_jump_per_step": float(np.nanmax(jumps)) if len(jumps) else np.nan,
            "valid": sig.get("valid", 0.0),
            "error": sig.get("error", ""),
        })

    if new_rows:
        out = pd.concat([have, pd.DataFrame(new_rows)], ignore_index=True)
        out = out.drop_duplicates(subset="rank", keep="last")
        order = [f"rank{int(r)}" for r in prod["rank"]] + ["A1"]
        out["_o"] = out["rank"].astype(str).map({a: i for i, a in enumerate(order)})
        out = out.sort_values("_o").drop(columns="_o").reset_index(drop=True)
        out.to_csv(fp_path, index=False)
        print(f"merged {len(new_rows)} new rows into {fp_path}", flush=True)
        have = out

    # Resolved/unresolved flag: a resolved melting range has a maximum
    # one-step liquid jump well below one; a jump of order one means the
    # calculation melts within a single grid interval.
    flags = {}
    for _, row in have.iterrows():
        jump = row.get("max_liquid_jump_per_step", np.nan)
        flags[str(row["rank"])] = {
            "T_solidus_full_K": float(row["T_solidus_full_K"]) if np.isfinite(row["T_solidus_full_K"]) else None,
            "max_liquid_jump_per_step": float(jump) if np.isfinite(jump) else None,
            "resolved": bool(np.isfinite(jump) and jump < 0.15),
            "valid": bool(row.get("valid", 0.0)),
        }

    # --- phase assemblages ---------------------------------------------------
    as_rows = []
    cands = []
    for _, cand in prod.iterrows():
        comp = {f: float(cand[f]) for f in FEATURES}
        comp["Ni"] = float(cand["Ni"])
        cands.append((f"rank{int(cand['rank'])}", comp, float(cand["T_solidus_calphad"])))
    a1_comp = {f: float(a1[f]) for f in FEATURES}
    a1_comp["Ni"] = float(a1["Ni"])
    cands.append(("A1", a1_comp, float(a1["T_solidus_calphad"])))

    for tag, comp, t_sol in cands:
        probes = {"750C": 1023.15, "900C": 1173.15,
                  "Tsol-50K": t_sol - 50.0, "Tsol-10K": t_sol - 10.0}
        for plist_name, plist in (("six", SIX_PHASES), ("twelve", FULL_PHASES)):
            for label, temp in probes.items():
                fr = assemblage(db, cfg, comp, temp, plist)
                for phase, frac in sorted(fr.items()):
                    if frac > 1e-6:
                        as_rows.append({"alloy": tag, "probe": f"{plist_name}:{label}",
                                        "phase": phase, "fraction": frac})
        print(f"{tag}: assemblages done", flush=True)

    pd.DataFrame(as_rows).to_csv(ROOT / "results" / "shortlist_assemblages.csv", index=False)
    (ROOT / "results" / "fullphase_flags.json").write_text(json.dumps(flags, indent=2))
    print(json.dumps(flags, indent=2))


if __name__ == "__main__":
    main()
