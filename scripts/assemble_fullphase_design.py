"""Assemble Table 12 from the refined (2 K, pdens 2000) full-phase calculations.

Ranks 1, 3 and A1 come from ``results/fullphase_convergence.csv``; rank 6 from
``results/fullphase_rank6_2k.json``.  The matrix and solvus values come from the
production CALPHAD re-validation.  A1 is flagged as unresolved because its
liquid fraction still steps from zero to one within 2 K.

Output: ``results/fullphase_design_2k.csv``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    cv = pd.read_csv(ROOT / "results" / "fullphase_convergence.csv")
    cv = cv.loc[cv["step_K"] == 2.0]
    full = {str(r["rank"]): float(r["T_solidus"]) for _, r in cv.iterrows()}
    r6 = json.load(open(ROOT / "results" / "fullphase_rank6_2k.json", encoding="utf-8"))
    full["6"] = float(r6["T_solidus"])

    prod = pd.read_csv(ROOT / "results" / "design_validated.csv")
    rows = []
    for rank in (1, 3, 6):
        p = prod.loc[prod["rank"] == rank].iloc[0]
        tsol_m = float(p["T_solidus_calphad"])
        tgp = float(p["T_gp_solvus_calphad"])
        tsol_f = full[str(rank)]
        rows.append({
            "rank": str(rank), "C_wt": float(p["C"]),
            "T_solidus_matrix_K": tsol_m, "T_gp_solvus_K": tgp,
            "T_solidus_full_K": tsol_f,
            "W_matrix_K": tsol_m - tgp, "W_full_K": tsol_f - tgp,
            "dW_full_minus_matrix_K": tsol_f - tsol_m,
            "resolved": 1,
        })
    a1 = pd.read_csv(ROOT / "results" / "design_robustness.csv").iloc[0]
    tsol_m = float(a1["T_solidus_calphad"])
    tgp = float(a1["T_gp_solvus_calphad"])
    tsol_f = full["A1"]
    rows.append({
        "rank": "A1", "C_wt": float(a1["C"]),
        "T_solidus_matrix_K": tsol_m, "T_gp_solvus_K": tgp,
        "T_solidus_full_K": tsol_f,
        "W_matrix_K": tsol_m - tgp, "W_full_K": tsol_f - tgp,
        "dW_full_minus_matrix_K": tsol_f - tsol_m,
        "resolved": 0,
    })
    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "fullphase_design_2k.csv", index=False)
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
