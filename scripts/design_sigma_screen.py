"""Service-temperature phase-stability screen for the designed shortlist.

The design objective is thermal: it does not constrain the phases that form at
750-900 C, where sigma and carbides can embrittle a Ni superalloy.  This script
adds a post-hoc screen: every shortlisted alloy is equilibrated at 750 and
900 C with the production six-phase list, and the sigma, M23C6, MC and BCC
fractions are reported.  Alloys outside the top six are additionally re-run
through the full signature calculation so that the screen is not limited to the
six already validated.

The grid minimiser can return a metastable ordered branch at low temperature
that the ``f_gamma' > 0.85`` repair rule does not catch (e.g. 0.78 gamma' plus
0.19 sigma at a Gibbs energy 345 J/mol above the true state).  The screen
therefore solves each temperature at three point densities and keeps the state
with the lowest Gibbs energy.  ``branch_gap_*`` records how far the production
pdens-400 state lies above that minimum; a positive gap flags a branch artifact
rather than a genuine sigma field.

Output: ``results/design_sigma_screen.csv``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycalphad import Database  # noqa: E402

from calphad_engine import compute_alloy, equilibrium_curve  # noqa: E402

FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    db = Database(ROOT / cfg["database"])
    cand = pd.read_csv(ROOT / "results" / "design_candidates.csv")
    validated = pd.read_csv(ROOT / "results" / "design_validated.csv")
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])

    rows = []
    for _, r in cand.iterrows():
        comp = {f: float(r[f]) for f in FEATURES}
        comp["Ni"] = float(r["Ni"])
        out = {"rank": int(r["rank"]),
               "W_surrogate_K": float(r["W_K"]),
               "knn_dist": float(r["knn_dist"])}
        for T_C in (750.0, 900.0):
            states = []
            for pdens in (int(cfg["pdens_solid"]), 1200, 4000):
                eq = equilibrium_curve(db, comp, [T_C + 273.15, T_C + 274.15],
                                       cfg["elements"], cfg["active_phases"],
                                       pdens=pdens, output_hm=False)
                if not eq.get("valid"):
                    continue
                fr = {k: float(np.asarray(v)[0]) for k, v in eq["frac"].items()}
                g = float(np.asarray(eq["G"], dtype=float)[0])
                if np.isfinite(g):
                    states.append((g, pdens, fr))
            if not states:
                for key in ("sigma", "M23C6", "MC", "BCC", "gp", "branch_gap"):
                    out[f"{key}_{int(T_C)}"] = np.nan
                continue
            g_best, pdens_best, fr = min(states, key=lambda s: s[0])
            g_prod = min((s for s in states if s[1] == int(cfg["pdens_solid"])),
                         key=lambda s: s[0], default=(np.nan, 0, {}))
            out[f"sigma_{int(T_C)}"] = fr.get("SIGMA", np.nan)
            out[f"M23C6_{int(T_C)}"] = fr.get("M23C6", np.nan)
            out[f"MC_{int(T_C)}"] = fr.get("MC", np.nan)
            out[f"BCC_{int(T_C)}"] = fr.get("BCC_A2", np.nan)
            out[f"gp_{int(T_C)}"] = fr.get("GAMMA_PRIME", np.nan)
            out[f"pdens_best_{int(T_C)}"] = int(pdens_best)
            out[f"branch_gap_{int(T_C)}"] = float(g_prod[0] - g_best)
            out[f"sigma_prod_{int(T_C)}"] = g_prod[2].get("SIGMA", np.nan)

        hit = validated.loc[validated["rank"] == r["rank"]]
        if len(hit):
            out["W_calphad_K"] = float(hit.iloc[0]["W_calphad_K"])
        else:
            rec = compute_alloy(db, comp, cfg["elements"], cfg["active_phases"],
                                T_melt, T_solid,
                                pdens_melt=cfg["pdens_melt"],
                                pdens_solid=cfg["pdens_solid"],
                                refine=cfg.get("refine_solvus", True))
            out["W_calphad_K"] = float(rec["T_solidus"] - rec["T_gp_solvus"])
            out["T_solidus_calphad"] = float(rec["T_solidus"])
            out["T_gp_solvus_calphad"] = float(rec["T_gp_solvus"])
        rows.append(out)
        print(f"rank {int(r['rank'])}: W={out['W_calphad_K']:.1f} K, "
              f"sigma750={out['sigma_750']:.3f}")

    df = pd.DataFrame(rows)
    df["sigma_safe"] = (df["sigma_750"].fillna(0.0) <= 0.02) & (df["sigma_900"].fillna(0.0) <= 0.02)
    df.to_csv(ROOT / "results" / "design_sigma_screen.csv", index=False)
    print(df[["rank", "W_calphad_K", "sigma_750", "sigma_900", "M23C6_750",
              "branch_gap_750", "sigma_prod_750", "sigma_safe"]]
          .round(4).to_string(index=False))


if __name__ == "__main__":
    main()
