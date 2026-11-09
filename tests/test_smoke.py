"""Fast smoke test for the released artefacts.

This does not rerun CALPHAD.  It checks that the dataset, the trained models
and the design outputs load and that the models return physically plausible
values for the Rene 77 projection.  Run with

    py -3.12 tests/test_smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import FEATURES, load_dataset  # noqa: E402

TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")
# Rene 77 projected onto the six-solute space.
RENE77 = np.array([[14.49, 14.28, 4.42, 4.20, 3.38, 0.10]])

EXPECTED = {
    "T_gp_solvus": (1300.0, 1520.0),
    "T_solidus": (1520.0, 1640.0),
    "T_liquidus": (1580.0, 1680.0),
    "dH_fusion_J_per_g": (180.0, 320.0),
    "gp_frac_900C": (0.3, 0.8),
}


def main() -> None:
    ds = load_dataset("dataset", root=str(ROOT))
    # 480 compositions were drawn; four raised a solver exception, leaving 476
    # curves (397 train / 79 test) before the per-target validity masks.
    assert len(ds.comp) == 476, len(ds.comp)
    assert int((ds.split == "train").sum()) == 397
    assert int((ds.split == "test").sum()) == 79
    assert np.isfinite(ds.comp).all()

    for target in TARGETS:
        path = ROOT / "results" / f"model_gp_{target}.joblib"
        assert path.exists(), path
        model = joblib.load(path)["model"]
        mu, sd = model.predict(RENE77, return_std=True)
        mu, sd = float(mu[0]), float(sd[0])
        lo, hi = EXPECTED[target]
        assert np.isfinite(mu) and lo <= mu <= hi, (target, mu)
        assert np.isfinite(sd) and sd > 0.0, (target, sd)
        print(f"  {target:20s} {mu:8.3f} +/- {sd:6.3f}  (allowed {lo}-{hi})")

    design = ROOT / "results" / "design_candidates.csv"
    assert design.exists()
    import pandas as pd
    d = pd.read_csv(design)
    assert len(d) == 12, len(d)
    assert (d["knn_dist"] > 0).all()
    assert (d["W_K"] > 0).all()

    macros = ROOT / "manuscript" / "generated" / "macros.tex"
    assert macros.exists()
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
