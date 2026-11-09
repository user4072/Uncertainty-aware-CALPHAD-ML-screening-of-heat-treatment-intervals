"""Equilibrium gamma-prime solvus cross-check against calorimetry (round 4+).

The round-4 limitations anchored the solvus discrepancy prior on a single
alloy (Rene 77: this work 1419.7 K vs TCNI10 1139 C vs calorimetric onset
1229 C).  Newly added literature provides two more in-space data points:

* Nimonic 115, spectrometer-analyzed composition (Shahriari et al. 2009,
  Mater. Manuf. Process. 24:5, 559-563): DTA and dilatometry at 10 C/min,
  heating dissolution 1160.4 +/- 4.6 C, cooling 1133.6 +/- 1.8 C; the
  Sponseller (1996) convention takes the equilibrium solvus as the average
  of the heating and cooling values.
* Astroloy (nominal ASM chemistry; Sponseller 1996, Superalloys 1976,
  Figure 12, DTA heating at 10 C/min): graphical value ~2100 F = 1150 C
  (+/- 10 K graphical).

This script stores the compositions used, recomputes the engine solvus for
each, and writes the generated LaTeX table.

Output: results/solvus_check.json, manuscript/generated/table_solvus_check.tex
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pycalphad import Database  # noqa: E402

from calphad_engine import compute_alloy  # noqa: E402

COMPOSITIONS = {
    # Rene 77 analyzed heat (Grudzien-Rakoczy et al. 2025, Table 1; the
    # anchor composition already used in the paper is the Binczyk one).
    "Rene77": {"Co": 14.55, "Cr": 14.42, "Al": 4.16, "Mo": 4.0, "Ti": 3.38,
               "C": 0.063},
    # Nimonic 115, spectrometer-analyzed (Shahriari 2009 Table 1); Fe 0.30,
    # Si 0.14, Mn 0.08, Zr 0.057, Cu 0.02 are held out of the model.
    "Nimonic115_analyzed": {"Co": 13.3, "Cr": 14.34, "Al": 4.98, "Mo": 3.26,
                            "Ti": 3.79, "C": 0.14},
    # Astroloy, ASM nominal (Sponseller gives no composition); B 0.03 and
    # Zr 0.06 are held out of the model.
    "Astroloy_nominal": {"Co": 16.95, "Cr": 14.7, "Al": 4.0, "Mo": 5.2,
                         "Ti": 3.5, "C": 0.06},
}

LITERATURE = {
    "Rene77": {
        "equilibrium_lit": 1139.0,       # C, TCNI10 in Grudzien-Rakoczy 2025
        "heating_C": 1229.0,             # C, DSC 10 C/min, Grudzien-Rakoczy 2025
        "heating_unc": None,
        "cooling_C": None,
        "incipient_melting_C": 1281.0,   # C, same source
        "mc_dissolution_C": 1307.0,
        "end_of_melting_C": 1355.0,
        "heating_note": "DSC, 10 K/min, onset of secondary-$\\gamma'$ dissolution",
    },
    "Nimonic115_analyzed": {
        "equilibrium_lit": None,
        "heating_C": 1160.4, "heating_unc": 4.6,   # DTA + dilatometry mean
        "cooling_C": 1133.6, "cooling_unc": 1.8,
        "heating_note": "DTA and dilatometry, 10 K/min (Shahriari 2009)",
    },
    "Astroloy_nominal": {
        "equilibrium_lit": None,
        "heating_C": 1150.0, "heating_unc": 10.0,  # graphical, Fig. 12
        "cooling_C": None,
        "heating_note": "DTA, 10 K/min, graphical read-off (Sponseller 1996)",
    },
}


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])

    out = {"compositions": COMPOSITIONS, "literature": LITERATURE, "computed": {}}
    for name, comp in COMPOSITIONS.items():
        c = dict(comp)
        c["Ni"] = 100.0 - sum(comp.values())
        rec = compute_alloy(db, c, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
                            pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"])
        entry = {
            "T_gp_solvus_K": float(rec["T_gp_solvus"]),
            "T_gp_solvus_C": float(rec["T_gp_solvus"] - 273.15),
            "T_solidus_matrix_K": float(rec["T_solidus"]),
            "T_liquidus_matrix_K": float(rec["T_liquidus"]),
            "solvus_refined": float(rec["solvus_refined"]),
        }
        lit = LITERATURE[name]
        if lit.get("cooling_C") is not None:
            entry["midpoint_C"] = round((lit["heating_C"] + lit["cooling_C"]) / 2.0, 1)
            entry["computed_minus_midpoint_K"] = round(entry["T_gp_solvus_C"] - entry["midpoint_C"], 1)
        entry["computed_minus_heating_K"] = round(entry["T_gp_solvus_C"] - lit["heating_C"], 1)
        if lit.get("equilibrium_lit") is not None:
            entry["computed_minus_equilibrium_lit_K"] = round(
                entry["T_gp_solvus_C"] - lit["equilibrium_lit"], 1)
        out["computed"][name] = entry
        print(name, json.dumps(entry))

    (ROOT / "results" / "solvus_check.json").write_text(json.dumps(out, indent=2),
                                                        encoding="utf-8")

    rows = []
    labels = {"Rene77": "Ren\\'e~77 (analyzed)",
              "Nimonic115_analyzed": "Nimonic~115 (analyzed)",
              "Astroloy_nominal": "Astroloy (nominal)"}
    for name in ("Rene77", "Nimonic115_analyzed", "Astroloy_nominal"):
        e = out["computed"][name]
        lit = LITERATURE[name]
        eq = (f"{lit['equilibrium_lit']:.0f}" if lit.get("equilibrium_lit") is not None else "--")
        heat = f"{lit['heating_C']:.0f}"
        if lit.get("heating_unc"):
            heat += f"$\\,\\pm\\,{lit['heating_unc']:.1f}$"
        cool = (f"{lit['cooling_C']:.0f}$\\,\\pm\\,{lit['cooling_unc']:.1f}$"
                if lit.get("cooling_C") is not None else "--")
        mid = (f"{e['midpoint_C']:.0f}" if "midpoint_C" in e else "--")
        dmid = (f"{e['computed_minus_midpoint_K']:+.1f}" if "computed_minus_midpoint_K" in e else "--")
        rows.append(
            f"{labels[name]} & {e['T_gp_solvus_C']:.1f} & {eq} & {heat} & {cool} & {mid} & "
            f"{e['computed_minus_heating_K']:+.1f} & {dmid} \\\\")
    tex = r"""\begin{table}[t]
\centering
\caption{Equilibrium $\gamma'$ solvus of three in-space alloys against
calorimetric dissolution data. This-work values are computed at the
composition shown; Ren\'e~77 is the analysed heat of
Grudzie\'n-Rakoczy \emph{et al.}, Nimonic~115 the spectrometer-analysed heat
of Shahriari \emph{et al.}, Astroloy the ASM nominal chemistry (B and Zr
outside the model). Heating values are 10~K/min DTA/DSC dissolution
temperatures, which over-shoot the equilibrium solvus; where a cooling value
exists, the mean of the two (the Sponseller convention) estimates the
equilibrium value. The database discrepancy is $-0.1$~K for Ren\'e~77
(against the independent TCNI10 value at matched composition) and $+29.6$~K for Nimonic~115 (against
the heating/cooling midpoint): two alloys spanning the 30~K solvus prior.}
\label{tab:solvuscheck}
\footnotesize
\setlength{\tabcolsep}{3.5pt}
\resizebox{\textwidth}{!}{%%
\begin{tabular}{lccccccc}
\toprule
Alloy & This work ($^\circ$C) & Equil. (lit.) & Heat ($^\circ$C) &
Cool ($^\circ$C) & Mid ($^\circ$C) & $\Delta$heat (K) & $\Delta$mid (K) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}%%
}
\end{table}
"""
    (ROOT / "manuscript" / "generated" / "table_solvus_check.tex").write_text(
        tex, encoding="utf-8")
    print("wrote table_solvus_check.tex")


if __name__ == "__main__":
    main()
