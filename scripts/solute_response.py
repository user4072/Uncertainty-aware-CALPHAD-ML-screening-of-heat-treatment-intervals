"""Per-solute response of the design targets from the deployed surrogates.

The review asks for a mechanistic interpretation of the optimisation: why the
high-Co/low-Al region is favoured, which elements depress the solvus and
which preserve the matrix solidus.  This script probes the deployed GP with
one-dimensional composition sweeps around the rank-1 candidate (all other
solutes fixed) and reports the predicted response range of each target to
each solute, together with the Gaussian-process length scales that quantify
the same sensitivity globally.

The sweeps are surrogate statements about the CALPHAD labels, not new
thermodynamic calculations; they are used only for interpretation.

Outputs
-------
``results/solute_response.csv``
``results/solute_response.json``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import FEATURES, load_dataset  # noqa: E402

RESULTS = ROOT / "results"
TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    space = cfg["composition_space_wt"]

    sel = pd.read_csv(RESULTS / "design_validated.csv")
    rank1 = sel.loc[sel["W_calphad_K"].idxmax()]
    base = np.array([float(rank1[f]) for f in FEATURES])

    ds = load_dataset("dataset", root=str(ROOT))
    med = np.median(ds.comp[ds.split == "train"], axis=0)

    models = {t: joblib.load(RESULTS / f"model_gp_{t}.joblib")["model"] for t in TARGETS}

    n = 41
    rows = []
    for j, f in enumerate(FEATURES):
        lo, hi = space[f]
        grid = np.linspace(lo, hi, n)
        X = np.tile(base, (n, 1))
        X[:, j] = grid
        pred = {t: models[t].predict(X) for t in TARGETS}
        entry = {"solute": f, "range_wt": f"{lo}-{hi}", "base_wt": float(base[j]),
                 "train_median_wt": float(med[j])}
        for t in TARGETS:
            y = pred[t]
            entry[f"{t}_min"] = float(np.min(y))
            entry[f"{t}_max"] = float(np.max(y))
            entry[f"{t}_span"] = float(np.max(y) - np.min(y))
            entry[f"{t}_at_base"] = float(models[t].predict(base.reshape(1, -1))[0])
            # direction: does the target increase with the solute?
            entry[f"{t}_slope_sign"] = "+" if y[-1] > y[0] else "-"
        rows.append(entry)

    df = pd.DataFrame(rows)
    df.to_csv(RESULTS / "solute_response.csv", index=False)

    lens = pd.read_csv(RESULTS / "gp_length_scales.csv")
    summary = {
        "base_composition": {f: float(base[i]) for i, f in enumerate(FEATURES)},
        "solute_response": rows,
        "gp_length_scales": lens.to_dict(orient="records"),
        "note": "Surrogate probes for interpretation only; no new CALPHAD calculations.",
    }
    (RESULTS / "solute_response.json").write_text(json.dumps(summary, indent=2))
    print(df[["solute", "base_wt", "train_median_wt",
              "T_gp_solvus_span", "T_gp_solvus_slope_sign",
              "T_solidus_span", "T_solidus_slope_sign",
              "gp_frac_900C_span", "gp_frac_900C_slope_sign"]].to_string(index=False))


if __name__ == "__main__":
    main()
