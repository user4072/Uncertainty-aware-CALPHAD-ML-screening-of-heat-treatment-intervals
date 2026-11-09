"""Trace the solvus refinement for dataset id 35 (six-phase list)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import (  # noqa: E402
    equilibrium_curve, refinement_grid, solvus_from_fractions,
)

SIX = ["FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "SIGMA", "BCC_A2"]


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    sig = pd.read_csv(ROOT / "data" / "processed" / "dataset_signatures.csv")
    row = sig.loc[sig["id"] == 35].iloc[0]
    comp = {k: float(row[k]) for k in ("Co", "Cr", "Al", "Mo", "Ti", "C")}
    comp["Ni"] = float(row["Ni"])
    T_solid = np.arange(cfg["temperature"]["solid_min"],
                        cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    solid = equilibrium_curve(db, comp, T_solid, cfg["elements"], SIX,
                              pdens=cfg["pdens_solid"], output_hm=True)
    T = np.asarray(solid["T"], dtype=float)
    gp = np.asarray(solid["frac"]["GAMMA_PRIME"], dtype=float)
    fcc = np.asarray(solid["frac"]["FCC_A1"], dtype=float)
    t_coarse, cens = solvus_from_fractions(T, gp, fcc)
    print(f"coarse solvus {t_coarse:.3f} K censored={cens}")
    m = (T > 1420) & (T < 1500)
    print(pd.DataFrame({"T": T[m], "gp": gp[m], "fcc": fcc[m]}).round(8).to_string(index=False))
    fine_T = refinement_grid(t_coarse, T, half_width=10.0, step=2.0)
    print("fine grid", fine_T)
    fine = equilibrium_curve(db, comp, fine_T, cfg["elements"], SIX,
                             pdens=cfg["pdens_solid"], output_hm=False)
    fT = np.asarray(fine["T"], dtype=float)
    fgp = np.asarray(fine["frac"].get("GAMMA_PRIME", np.zeros(len(fT))), dtype=float)
    ffcc = np.asarray(fine["frac"].get("FCC_A1", np.ones(len(fT))), dtype=float)
    t_fine, cens_fine = solvus_from_fractions(fT, fgp, ffcc)
    print(f"fine solvus {t_fine:.3f} K censored={cens_fine}")
    print(pd.DataFrame({"T": fT, "gp": fgp, "fcc": ffcc}).round(8).to_string(index=False))


if __name__ == "__main__":
    main()
