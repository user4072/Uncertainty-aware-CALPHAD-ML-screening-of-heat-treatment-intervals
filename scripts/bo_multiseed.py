"""Multi-seed replication of the sequential (Bayesian-optimisation) baseline.

A reviewer asked for the sequential search to be repeated over multiple
initializations and reported as a distribution rather than a single
trajectory.  The original run (``bo_baseline.py``) is deterministic: one
train-only GP, one Sobol cloud, greedy LCB picks.  This script repeats it
with stochastic initializations.

Each run spends its *first* evaluation on a random feasible composition
(seeded), then continues greedily: rank the cloud by the lower confidence
bound, compute the CALPHAD label of the best not-yet-evaluated candidate,
add it to the training pool, refit the five scalar GPs.  The budget stays at
six CALPHAD evaluations per run, as in the paper.  The deterministic run of
``bo_baseline.py`` (no random first pick) is kept as a reference row and is
not repeated here.

The train-only initial GP fit is identical across runs, so it is performed
once and deep-copied; only the per-step refits are repeated.

Outputs
-------
``results/bo_multiseed.csv``   one row per evaluation, all seeds.
``results/bo_multiseed.json``  per-seed best/mean, distribution summary.
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import qmc
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from calphad_engine import compute_alloy  # noqa: E402
from surrogate import FEATURES, StandardizedGP, load_dataset  # noqa: E402

TARGETS = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g", "gp_frac_900C")
KNN_K, TRUST_PERCENTILE, MAX_LIQ = 5, 95.0, 1650.0
MIN_C = 0.03
N_STEPS = 6
N_SEEDS = 10
BASE_SEED = 101


def predict(models, X):
    out = {}
    for t, m in models.items():
        mu, sd = m.predict(X, return_std=True)
        out[t] = mu
        out[f"{t}_std"] = sd
    return out


def main() -> None:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml", encoding="utf-8"))
    space = cfg["composition_space_wt"]
    lo = np.array([space[f][0] for f in FEATURES], float)
    hi = np.array([space[f][1] for f in FEATURES], float)
    ds = load_dataset("dataset", root=str(ROOT))
    train = ds.split == "train"

    X = qmc.scale(qmc.Sobol(d=len(FEATURES), scramble=True, seed=7).random(120000), lo, hi)
    al, ti, c = (FEATURES.index(k) for k in ("Al", "Ti", "C"))
    ni = 100.0 - X.sum(1)

    Xtr0 = {t: ds.comp[train & ds.valid[t]].copy() for t in TARGETS}
    ytr0 = {t: ds.y[t][train & ds.valid[t]].copy() for t in TARGETS}

    print("Initial train-only GP fit (shared across seeds) ...", flush=True)
    t_fit = time.time()
    base_models = {}
    for t in TARGETS:
        m = StandardizedGP(n_restarts=3, seed=0).fit(Xtr0[t], ytr0[t])
        m.calibrate(Xtr0[t], ytr0[t], n_splits=5, seed=0)
        base_models[t] = m
        print(f"  {t}: {len(ytr0[t])} labels", flush=True)
    print(f"  initial fit {time.time() - t_fit:.0f} s", flush=True)

    from pycalphad import Database

    db = Database(ROOT / cfg["database"])
    T_melt = np.arange(cfg["temperature"]["melt_min"], cfg["temperature"]["melt_max"] + 1e-9,
                       cfg["temperature"]["melt_step"])
    T_solid = np.arange(cfg["temperature"]["solid_min"], cfg["temperature"]["solid_max"] + 1e-9,
                        cfg["temperature"]["solid_step"])

    rows = []
    seed_summary = {}
    for seed in range(BASE_SEED, BASE_SEED + N_SEEDS):
        rng = np.random.default_rng(seed)
        models = {t: copy.deepcopy(base_models[t]) for t in TARGETS}
        Xtr = {t: Xtr0[t].copy() for t in TARGETS}
        ytr = {t: ytr0[t].copy() for t in TARGETS}
        evaluated = np.zeros(len(X), dtype=bool)
        best = -np.inf
        t0 = time.time()
        for step in range(1, N_STEPS + 1):
            Xall = np.vstack([Xtr[t] for t in TARGETS])
            mu_x, sd_x = Xall.mean(0), Xall.std(0) + 1e-12
            nn = NearestNeighbors(n_neighbors=KNN_K).fit((Xall - mu_x) / sd_x)
            radius = float(np.percentile(nn.kneighbors((Xall - mu_x) / sd_x)[0][:, -1],
                                         TRUST_PERCENTILE))
            inside = nn.kneighbors((X - mu_x) / sd_x)[0][:, -1] <= radius
            pred = predict(models, X)
            W = pred["T_solidus"] - pred["T_gp_solvus"]
            W_std = np.hypot(pred["T_solidus_std"], pred["T_gp_solvus_std"])
            score = W - W_std
            feasible = (
                inside & ~evaluated & (ni >= space["Ni_min"])
                & (X[:, al] + X[:, ti] >= space.get("min_Al_Ti_sum", 0.0))
                & (pred["dH_fusion_J_per_g"] >= 190.0) & (pred["dH_fusion_J_per_g"] <= 290.0)
                & (pred["gp_frac_900C"] >= 0.40) & (pred["T_solidus"] >= 1500.0)
                & (pred["T_liquidus"] <= MAX_LIQ) & (W > 0.0)
                & (X[:, c] <= 0.15) & (X[:, c] >= MIN_C)
            )
            idx = np.where(feasible)[0]
            if len(idx) == 0:
                print(f"seed {seed}: no feasible candidate left", flush=True)
                break
            if step == 1:
                j = int(rng.choice(idx))
                pick = "random-init"
            else:
                j = int(idx[np.argmax(score[idx])])
                pick = "lcb-greedy"

            comp = {f: float(X[j, k]) for k, f in enumerate(FEATURES)}
            comp["Ni"] = float(ni[j])
            tc = time.time()
            rec = compute_alloy(
                db, comp, cfg["elements"], cfg["active_phases"], T_melt, T_solid,
                pdens_melt=cfg["pdens_melt"], pdens_solid=cfg["pdens_solid"],
                refine=cfg.get("refine_solvus", True),
            )
            elapsed = time.time() - tc
            w_cal = float(rec["T_solidus"] - rec["T_gp_solvus"])
            best = max(best, w_cal)
            rows.append({
                "seed": seed, "step": step, "pick": pick,
                **{f: comp[f] for f in FEATURES}, "Ni": comp["Ni"],
                "W_predicted_K": float(W[j]), "W_std_K": float(W_std[j]),
                "W_calphad_K": w_cal,
                "T_solidus_calphad": float(rec["T_solidus"]),
                "T_gp_solvus_calphad": float(rec["T_gp_solvus"]),
                "dH_calphad": float(rec["dH_fusion_J_per_g"]),
                "gp900_calphad": float(rec["gp_frac_900C"]),
                "knn_dist": float(nn.kneighbors(((X[j] - mu_x) / sd_x).reshape(1, -1))[0][0, -1]),
                "best_so_far_K": best,
                "calphad_s": elapsed,
            })
            print(f"seed {seed} step {step} ({pick}): pred {W[j]:.1f}+/-{W_std[j]:.1f} "
                  f"-> CALPHAD {w_cal:.1f} K (best {best:.1f}), {elapsed:.0f}s", flush=True)

            evaluated[j] = True
            Xtr["T_gp_solvus"] = np.vstack([Xtr["T_gp_solvus"], X[j]])
            ytr["T_gp_solvus"] = np.append(ytr["T_gp_solvus"], rec["T_gp_solvus"])
            Xtr["T_solidus"] = np.vstack([Xtr["T_solidus"], X[j]])
            ytr["T_solidus"] = np.append(ytr["T_solidus"], rec["T_solidus"])
            if rec["melt_valid"]:
                for t in ("T_liquidus", "dH_fusion_J_per_g"):
                    Xtr[t] = np.vstack([Xtr[t], X[j]])
                    ytr[t] = np.append(ytr[t], rec[t])
            if rec["solid_valid"]:
                Xtr["gp_frac_900C"] = np.vstack([Xtr["gp_frac_900C"], X[j]])
                ytr["gp_frac_900C"] = np.append(ytr["gp_frac_900C"], rec["gp_frac_900C"])
            for t in TARGETS:
                m = StandardizedGP(n_restarts=1, seed=seed).fit(Xtr[t], ytr[t])
                m.sd_scale = models[t].sd_scale
                models[t] = m

        table = pd.DataFrame([r for r in rows if r["seed"] == seed])
        seed_summary[str(seed)] = {
            "best_calphad_K": float(table["best_so_far_K"].iloc[-1]),
            "mean_calphad_K": float(table["W_calphad_K"].mean()),
            "best_so_far_by_step": [float(v) for v in table["best_so_far_K"]],
            "wall_s": time.time() - t0,
        }
        print(f"seed {seed}: best {seed_summary[str(seed)]['best_calphad_K']:.1f} K "
              f"mean {seed_summary[str(seed)]['mean_calphad_K']:.1f} K "
              f"({seed_summary[str(seed)]['wall_s']:.0f} s)", flush=True)

    out = pd.DataFrame(rows)
    out.to_csv(ROOT / "results" / "bo_multiseed.csv", index=False)

    bests = np.array([v["best_calphad_K"] for v in seed_summary.values()])
    summary = {
        "n_seeds": N_SEEDS,
        "n_steps": N_STEPS,
        "constraint_C_min_wt": MIN_C,
        "base_seed": BASE_SEED,
        "seeds": seed_summary,
        "best_distribution_K": {
            "min": float(bests.min()), "p25": float(np.percentile(bests, 25)),
            "median": float(np.median(bests)), "p75": float(np.percentile(bests, 75)),
            "max": float(bests.max()), "mean": float(bests.mean()),
        },
        "frac_best_ge_240": float((bests >= 240.0).mean()),
        "frac_best_ge_250": float((bests >= 250.0).mean()),
        "deterministic_bo_baseline": json.loads(
            (ROOT / "results" / "bo_baseline.json").read_text(encoding="utf-8")),
    }
    (ROOT / "results" / "bo_multiseed.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["best_distribution_K"], indent=2))


if __name__ == "__main__":
    main()
