"""Compare CALPHAD and surrogate predictions with independent melting data.

Six literature alloys are used, in four tiers:

* calibration (DSC, chemically in-domain): Rene 77 and IN100 (Binczyk 2008);
* validation (manufacturer melting range, chemically in-domain): Nimonic 105
  and Nimonic 115 (Special Metals bulletins);
* projection (DSC, 2.3 wt% Nb absent): IN713C (Binczyk 2008);
* stress (DSC, 5.1 wt% Ta+W+Nb absent): IN738LC (Chapman 2004).

Only the calibration and validation tiers enter the error statistics.  V, Nb,
Ta and W are absent from the six-solute surrogate, so every alloy is evaluated
on the same Ni-Co-Cr-Al-Mo-Ti-C basis as the training set.  For IN738LC the
omitted Ta, W and Nb are folded into Ni, which makes that probe an approximate
projection rather than an exact composition; the limitation is reported
explicitly in the results table and quantified by scripts/trace_probes.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import compute_alloy  # noqa: E402
from surrogate import FEATURES, TARGETS  # noqa: E402

DSC_TARGETS = ("T_solidus", "T_liquidus", "dH_fusion_J_per_g")
DSC_COLUMNS = {"T_solidus": "Tsol_C", "T_liquidus": "Tliq_C", "dH_fusion_J_per_g": "dH_J_g"}


def calphad_holdouts(cfg: dict, lit: pd.DataFrame, cache: Path) -> pd.DataFrame:
    if cache.exists():
        return pd.read_csv(cache)

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])
    rows = []
    for _, alloy in lit.iterrows():
        comp = {k: float(alloy[k]) for k in ("Co", "Cr", "Al", "Mo", "Ti", "C", "Ni")}
        rec = compute_alloy(
            db, comp, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
            pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
            refine=cfg.get("refine_solvus", True),
        )
        rows.append({
            "alloy": alloy["alloy"],
            "T_gp_solvus": rec["T_gp_solvus"],
            "T_solidus": rec["T_solidus"],
            "T_liquidus": rec["T_liquidus"],
            "dH_fusion_J_per_g": rec["dH_fusion_J_per_g"],
            "gp_frac_750C": rec["gp_frac_750C"],
            "gp_frac_900C": rec["gp_frac_900C"],
            "melt_valid": rec["melt_valid"],
            "solid_valid": rec["solid_valid"],
        })
    out = pd.DataFrame(rows)
    cache.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(cache, index=False)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    parser.add_argument("--models", default=str(ROOT / "results"))
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    lit = pd.read_csv(ROOT / "data" / "dsc_literature.csv")

    cal = calphad_holdouts(cfg, lit, out_dir / f"dsc_holdout_calphad_pdens{cfg['pdens_solid']}.csv")

    rows = []
    for _, alloy in lit.iterrows():
        row = {"alloy": alloy["alloy"], "source": alloy["source"],
               "role": alloy.get("role", "calibration")}
        for t in DSC_TARGETS:
            dsc = float(alloy[DSC_COLUMNS[t]]) + (273.15 if t != "dH_fusion_J_per_g" else 0.0)
            row[f"{t}_dsc"] = dsc
            row[f"{t}_calphad"] = float(cal.loc[cal["alloy"] == alloy["alloy"], t].iloc[0])

        # Surrogate prediction at the same composition (six solutes only).
        x = np.array([[float(alloy[f]) for f in FEATURES]])
        for t in TARGETS:
            path = Path(args.models) / f"model_gp_{t}.joblib"
            if not path.exists():
                continue
            payload = joblib.load(path)
            mu, sd = payload["model"].predict(x, return_std=True)
            row[f"{t}_gp"] = float(mu[0])
            row[f"{t}_gp_std"] = float(sd[0])
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "dsc_validation.csv", index=False)

    # Leave-one-alloy-out constant-bias calibration on the three Binczyk
    # anchors: correct a prediction by the mean residual of the other two.
    # IN738LC is an out-of-domain stress test (Ta/W omitted and gamma/gamma'
    # eutectic melting not represented), so it is reported but never used to
    # fit the bias.
    summary = {}
    # In-domain anchors: the two DSC measurements (calibration tier) and the
    # two manufacturer melting-range determinations (validation tier).  The
    # chemically off-space alloys are reported below as projections and never
    # enter the error statistics.
    indomain = table["role"].isin(("calibration", "validation")).to_numpy()
    print("\nMelting-temperature validation (temperatures in K, enthalpy in J/g)")
    print(table[[f"{t}_dsc" for t in DSC_TARGETS]].round(1).to_string())
    for t in DSC_TARGETS:
        pred_all = table[f"{t}_calphad"].to_numpy(float)
        obs_all = table[f"{t}_dsc"].to_numpy(float)
        mask = np.isfinite(pred_all) & np.isfinite(obs_all) & indomain
        pred = pred_all[mask]
        obs = obs_all[mask]
        resid = pred - obs
        # With fewer than three anchors the leave-one-out bias correction
        # degenerates (it replaces each residual by the other one), so it is
        # reported as undefined rather than as an improvement.
        if len(pred) >= 3:
            loo_corrected = np.array([
                pred[i] - np.mean(np.delete(resid, i)) for i in range(len(pred))
            ])
            mae_corr = float(np.mean(np.abs(loo_corrected - obs)))
        else:
            mae_corr = float("nan")
        mae_raw = float(np.mean(np.abs(resid)))
        unit = "K" if t != "dH_fusion_J_per_g" else "J/g"
        stress = table.loc[~indomain, ["alloy", "role", f"{t}_dsc", f"{t}_calphad"]].dropna()
        summary[t] = {
            "unit": unit,
            "n_anchors": int(mask.sum()),
            "alloys": table.loc[mask, "alloy"].tolist(),
            "role_tiers": table.loc[mask, "role"].tolist(),
            "mae_raw": mae_raw,
            "mae_loo_bias_corrected": mae_corr,
            "bias": float(np.mean(resid)),
            "residuals": resid.tolist(),
            "observed": obs.tolist(),
            "predicted": pred.tolist(),
            "projections": stress.rename(columns={
                f"{t}_dsc": "observed", f"{t}_calphad": "predicted"
            }).to_dict(orient="records"),
        }
        print(f"  {t:20s} n={int(mask.sum())}  raw MAE={mae_raw:6.1f} {unit}  "
              f"LOO-bias MAE={mae_corr:6.1f} {unit}")
        for _, s in stress.iterrows():
            print(f"    {s['role']:10s} {s['alloy']:9s} obs={s[f'{t}_dsc']:7.1f}  "
                  f"CALPHAD={s[f'{t}_calphad']:7.1f}  {unit}")

        gp_col = f"{t}_gp"
        if gp_col in table:
            gp_all = table[gp_col].to_numpy(float)
            gmask = np.isfinite(gp_all) & np.isfinite(obs_all) & indomain
            if gmask.sum() >= 2:
                gpred, gobs = gp_all[gmask], obs_all[gmask]
                gresid = gpred - gobs
                if len(gpred) >= 3:
                    gloo = np.array([
                        gpred[i] - np.mean(np.delete(gresid, i)) for i in range(len(gpred))
                    ])
                    gp_loo = float(np.mean(np.abs(gloo - gobs)))
                else:
                    gp_loo = float("nan")
                summary[t]["gp"] = {
                    "n_anchors": int(gmask.sum()),
                    "mae_raw": float(np.mean(np.abs(gresid))),
                    "mae_loo_bias_corrected": gp_loo,
                    "bias": float(np.mean(gresid)),
                }
                print(f"    surrogate GP        raw MAE={np.mean(np.abs(gresid)):6.1f} {unit}  "
                      f"LOO-bias MAE={gp_loo:6.1f} {unit}")

    with open(out_dir / "dsc_validation.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)


if __name__ == "__main__":
    main()
