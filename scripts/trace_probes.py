"""Trace-solute activation at the two out-of-domain probes.

The production surrogate is seven-solute.  IN713C carries 2.27 wt% Nb and
IN738LC carries 1.7 wt% Ta, 2.6 wt% W and 0.8 wt% Nb.  Ta is not defined in the
database; Nb and W are defined but pinned at a mole fraction of 1e-12 in the
production calculation.  This script repeats the matrix melting calculation for
each probe twice: with the production seven-solute composition, and with the
database-defined omitted solutes activated at their reported mass fractions.
Ta stays folded into Ni in both cases because the database cannot accept it.

The purpose is to separate the part of the solidus offset that is traceable to
a solute the database can represent from the part that is not (Ta plus the
gamma/gamma-prime eutectic, which a liquid+fcc calculation cannot produce).

Run from the repository root:

    py -3.12 scripts/trace_probes.py
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

from calphad_engine import equilibrium_curve, extract_signatures  # noqa: E402
from pycalphad import Database  # noqa: E402

PROBES = {
    "IN713C": {
        "measured": {"T_solidus": 1325.0 + 273.15, "T_liquidus": 1349.0 + 273.15,
                     "dH_fusion_J_per_g": 171.74},
        # Seven-solute projection: Nb dropped, remaining 97.73 wt% renormalised.
        "projection": {"Co": 0.03, "Cr": 13.26, "Al": 5.85, "Mo": 4.1,
                       "Ti": 0.85, "C": 0.1, "Ni": 73.54},
        # True composition: Nb activated; Ni is already the balance including Nb.
        "active": {"Co": 0.03, "Cr": 13.26, "Al": 5.85, "Mo": 4.1,
                   "Ti": 0.85, "C": 0.1, "Nb": 2.27, "Ni": 73.54},
        "elements": ("NB",),
    },
    "IN738LC": {
        "measured": {"T_solidus": 1238.0 + 273.15, "T_liquidus": 1322.0 + 273.15,
                     "dH_fusion_J_per_g": float("nan")},
        # Seven-solute projection: Ta, W and Nb folded into Ni (Ni = 67.0).
        "projection": {"Co": 8.3, "Cr": 15.9, "Al": 3.5, "Mo": 1.7,
                       "Ti": 3.5, "C": 0.1, "Ni": 67.0},
        # W and Nb activated; Ta (1.7 wt%) remains folded into Ni (63.6).
        "active": {"Co": 8.3, "Cr": 15.9, "Al": 3.5, "Mo": 1.7,
                   "Ti": 3.5, "C": 0.1, "Nb": 0.8, "W": 2.6, "Ni": 63.6},
        "elements": ("NB", "W"),
    },
}


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"],
                       cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    base = list(cfg["elements"])

    rows = []
    for alloy, spec in PROBES.items():
        for variant in ("projection", "active"):
            els = base + list(spec["elements"] if variant == "active" else ())
            curve = equilibrium_curve(db, spec[variant], T_melt, els,
                                      ["LIQUID", "FCC_A1"],
                                      pdens=cfg["pdens_melt"])
            sig = extract_signatures(curve)
            if not curve.get("valid"):
                raise RuntimeError(f"{alloy} {variant} returned an invalid curve")
            row = {"alloy": alloy, "variant": variant,
                   "T_solidus": float(sig["T_solidus"]),
                   "T_liquidus": float(sig["T_liquidus"]),
                   "dH_fusion_J_per_g": float(sig["dH_fusion_J_per_g"])}
            for key, val in spec["measured"].items():
                row[f"{key}_measured"] = val
                row[f"{key}_residual"] = row[key] - val
            rows.append(row)
            print(f"{alloy:9s} {variant:10s} "
                  f"Tsol={row['T_solidus']:8.2f} Tliq={row['T_liquidus']:8.2f} "
                  f"dH={row['dH_fusion_J_per_g']:8.2f}", flush=True)

    out = pd.DataFrame(rows)
    res = ROOT / "results"
    out.to_csv(res / "trace_probes.csv", index=False)
    summary = {
        alloy: {
            "projection": out.loc[(out["alloy"] == alloy)
                                  & (out["variant"] == "projection")].iloc[0].to_dict(),
            "active": out.loc[(out["alloy"] == alloy)
                              & (out["variant"] == "active")].iloc[0].to_dict(),
        }
        for alloy in PROBES
    }
    with open(res / "trace_probes.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
