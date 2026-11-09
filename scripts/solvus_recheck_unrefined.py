"""Re-refine solvi that still fall back to the coarse interpolation.

After the bracket-aware refinement was introduced, 92 of 411 uncensored solvi
still carried ``solvus_refined = 0``.  In these cases the 2 K curve returns no
gamma-prime inside the coarse bracket itself, which indicates that the finer
minimisation dissolves gamma-prime earlier than the 25 K curve does.  This
script widens the refinement window by one coarse interval below the bracket
(three nodes wide) and, if a crossing is found there, replaces the stored
value.  Used first with ``--dry-run`` on a sample to measure the shifts.
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import equilibrium_curve, solvus_from_fractions  # noqa: E402

SIX = ["FCC_A1", "GAMMA_PRIME", "MC", "M23C6", "SIGMA", "BCC_A2"]
FEATURES = ("Co", "Cr", "Al", "Mo", "Ti", "C")
_W = {}


def _init(db_path, elements, phases, T, pdens):
    from pycalphad import Database
    _W.update(db=Database(db_path), elements=elements, phases=phases,
              T=T, pdens=pdens)


def _run(payload):
    pid, comp, t_stored = payload
    db, T, el = _W["db"], _W["T"], _W["elements"]
    solid = equilibrium_curve(db, comp, T, el, _W["phases"],
                              pdens=_W["pdens"], output_hm=False)
    Tc = np.asarray(solid["T"], dtype=float)
    gp = np.asarray(solid["frac"].get("GAMMA_PRIME", np.zeros(len(Tc))), dtype=float)
    fcc = np.asarray(solid["frac"].get("FCC_A1", np.ones(len(Tc))), dtype=float)
    above = (gp > 1e-3) & (fcc >= 0.5)
    idx = np.where(above)[0]
    if not len(idx):
        return dict(id=pid, T_stored=t_stored, T_new=np.nan, gp_lo=np.nan,
                    gp_hi=np.nan, note="no coarse crossing")
    i = int(idx[-1])
    lo = float(Tc[max(0, i - 1)]) - 2.0
    hi = float(Tc[i + 1]) + 2.0 if i + 1 < len(Tc) else float(Tc[-1])
    fine_T = np.arange(max(float(Tc[0]), lo), min(float(Tc[-1]), hi) + 0.5, 2.0)
    fine = equilibrium_curve(db, comp, fine_T, el, _W["phases"],
                             pdens=_W["pdens"], output_hm=False)
    fT = np.asarray(fine["T"], dtype=float)
    fgp = np.asarray(fine["frac"].get("GAMMA_PRIME", np.zeros(len(fT))), dtype=float)
    ffcc = np.asarray(fine["frac"].get("FCC_A1", np.ones(len(fT))), dtype=float)
    tf, cf = solvus_from_fractions(fT, fgp, ffcc)
    return dict(id=pid, T_stored=t_stored, T_new=float(tf), gp_lo=float(gp[i]),
                gp_hi=float(fgp.max()) if len(fgp) else np.nan,
                note="censored fine" if cf else "ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="0 = all flagged")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    sig_path = ROOT / "data" / "processed" / "dataset_signatures.csv"
    sig = pd.read_csv(sig_path)
    flag = sig.loc[(sig["censored_gp_high"] == 0) & (sig["solvus_refined"] == 0)]
    if args.limit:
        flag = flag.head(args.limit)
    print(f"{len(flag)} flagged alloys")

    T = np.arange(cfg["temperature"]["solid_min"],
                  cfg["temperature"]["solid_max"] + 1e-9,
                  cfg["temperature"]["solid_step"])
    payloads = []
    for _, row in flag.iterrows():
        comp = {f: float(row[f]) for f in FEATURES}
        comp["Ni"] = float(row["Ni"])
        payloads.append((int(row["id"]), comp, float(row["T_gp_solvus"])))

    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init,
                             initargs=(str(ROOT / cfg["database"]), cfg["elements"],
                                       SIX, T, int(cfg["pdens_solid"]))) as pool:
        out = list(pool.map(_run, payloads))

    df = pd.DataFrame(out)
    df["shift_K"] = df["T_new"] - df["T_stored"]
    print(df.round(2).to_string(index=False))
    resolved = df["T_new"].notna()
    if resolved.any():
        print(f"resolved {int(resolved.sum())}/{len(df)}; shift median "
              f"{df.loc[resolved,'shift_K'].median():.2f} K, min "
              f"{df.loc[resolved,'shift_K'].min():.2f}, max "
              f"{df.loc[resolved,'shift_K'].max():.2f}")

    if args.write and len(df) == int(((sig["censored_gp_high"] == 0)
                                      & (sig["solvus_refined"] == 0)).sum()):
        for _, r in df.loc[(df["T_new"].notna()) & (df["note"] == "ok")].iterrows():
            j = sig.index[sig["id"] == r["id"]][0]
            sig.loc[j, "T_gp_solvus"] = r["T_new"]
            sig.loc[j, "solvus_refined"] = 1.0
        inconsistent = (sig["T_gp_solvus"] >= sig["T_solidus"] - 5.0) | (
            (sig["gp_frac_900C"] > 0.5) & (sig["T_gp_solvus"] < 1173.15))
        sig.loc[inconsistent, "censored_gp_high"] = 1.0
        sig.to_csv(sig_path, index=False)
        print("updated", sig_path)


if __name__ == "__main__":
    main()
