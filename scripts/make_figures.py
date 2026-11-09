"""Publication-style figures for the CALPHAD -> surrogate -> DSC pipeline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.ticker import NullLocator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from surrogate import FEATURES, TARGETS, TARGET_LABELS  # noqa: E402

PALETTE = {
    "gp": "#0072B2",
    "pce": "#D55E00",
    "nn": "#009E73",
    "grey": "#666666",
    "gold": "#E69F00",
    "purple": "#CC79A7",
}

plt.rcParams.update({
    "font.size": 8.5,
    "axes.titlesize": 9.5,
    "axes.labelsize": 8.5,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.linewidth": 0.8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "font.family": "DejaVu Sans",
})


def save(fig, out_dir: Path, name: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote {name}.pdf/.png")


# ---------------------------------------------------------------------------
# Dataset overview
# ---------------------------------------------------------------------------

def figure_dataset(out_dir: Path) -> None:
    sig_path = ROOT / "data" / "processed" / "dataset_signatures.csv"
    if not sig_path.exists():
        print("  skip figS1: no dataset")
        return
    sig = pd.read_csv(sig_path)
    mask = (sig["melt_valid"] == 1) & (sig["solid_valid"] == 1)
    for flag in ("censored_gp_high", "censored_solidus_low", "censored_liquidus_high"):
        if flag in sig:
            mask &= sig[flag].fillna(1.0) == 0
    for t in TARGETS:
        mask &= np.isfinite(sig[t])
    d = sig.loc[mask]
    panels = [
        ("Co", "Co (wt%)"), ("Cr", "Cr (wt%)"), ("Al", "Al (wt%)"), ("Mo", "Mo (wt%)"),
        ("Ti", "Ti (wt%)"), ("C", "C (wt%)"), ("Ni", "Ni (wt%)"),
        ("gp_frac_900C", r"$f_{\gamma'}$ (900 C)"),
        ("T_gp_solvus", r"$T_{\gamma'}$ (K)"), ("T_solidus", r"$T_{\rm sol}$ (K)"),
        ("T_liquidus", r"$T_{\rm liq}$ (K)"),
        ("dH_fusion_J_per_g", r"$\Delta H_{\rm fus}$ (J g$^{-1}$)"),
    ]
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.4))
    for ax, (col, label) in zip(axes.ravel(), panels):
        v = d[col].to_numpy(float)
        ax.hist(v, bins=28, color=PALETTE["gp"], alpha=0.75, edgecolor="white", linewidth=0.3)
        ax.set_xlabel(label)
        ax.set_yticks([])
        ax.grid(axis="y", lw=0.2, alpha=0.3)
    fig.suptitle(f"Usable CALPHAD dataset (N = {len(d)})", fontsize=9.5)
    fig.tight_layout()
    save(fig, out_dir, "figS1_dataset")


# ---------------------------------------------------------------------------
# Workflow schematic
# ---------------------------------------------------------------------------

def figure_workflow(out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.0)
    ax.axis("off")

    bw, bh = 2.75, 1.50
    xs = {"l": 0.25, "m": 3.62, "r": 6.99}
    y1, y2 = 2.90, 0.40
    yc1, yc2 = y1 + bh / 2, y2 + bh / 2
    g = 0.06  # standoff so arrowheads never touch the boxes

    boxes = [
        (xs["l"], y1, "#DCE9F5", ["Composition space", "Ni–Co–Cr–Al–Mo–Ti–C", "480-alloy Latin hypercube"]),
        (xs["m"], y1, "#CFE8D8", ["Two linked CALPHAD", "calculations", "matrix melting (liquid + fcc)",
                                  "six-phase solid state"]),
        (xs["r"], y1, "#FBE5C8", ["Five thermal signatures",
                                  r"$T_{\gamma'}$, $T_{\rm sol}^{\rm m}$, $T_{\rm liq}^{\rm m}$,",
                                  r"$\Delta H_{\rm fus}$, $f_{\gamma'}$(900 °C)"]),
        (xs["r"], y2, "#E8DDF1", ["Uncertainty-aware", "surrogates", "GP · PCE · NN",
                                  "PCA-GP for $H(T)$, $C_p(T)$"]),
        (xs["m"], y2, "#F5D5D5", ["Validation against", "literature calorimetry",
                                  "six commercial alloys"]),
        (xs["l"], y2, "#DDE8D5", ["Inverse design", "trust-region screen with",
                                  "calibrated uncertainty"]),
    ]
    for x, y, colour, lines in boxes:
        box = FancyBboxPatch((x, y), bw, bh, boxstyle="round,pad=0.06,rounding_size=0.12",
                             facecolor=colour, edgecolor="#444444", linewidth=0.8)
        ax.add_patch(box)
        n = len(lines)
        sizes = [8.2, 7.4, 6.6, 6.6][:n]
        for i, line in enumerate(lines):
            weight = "bold" if i == 0 else "normal"
            ax.text(x + bw / 2, y + bh * (1 - (i + 0.5) / n), line,
                    ha="center", va="center", fontsize=sizes[i], fontweight=weight)

    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.1, color="#333333",
                                     shrinkA=0, shrinkB=0))

    # horizontal flow, rightward on the top row
    arrow((xs["l"] + bw + g, yc1), (xs["m"] - g, yc1))
    arrow((xs["m"] + bw + g, yc1), (xs["r"] - g, yc1))
    # vertical drop on the right
    arrow((xs["r"] + bw / 2, y1 - g), (xs["r"] + bw / 2, y2 + bh + g))
    # leftward flow on the bottom row
    arrow((xs["r"] - g, yc2), (xs["m"] + bw + g, yc2))
    arrow((xs["m"] - g, yc2), (xs["l"] + bw + g, yc2))

    # re-validation loop as an orthogonal dashed return path routed around the
    # left margin: Inverse design -> left -> up -> across -> down into CALPHAD
    dash = {"linestyle": (0, (5, 3)), "color": "#555555", "lw": 1.1}
    x_margin = 0.10
    x_down = xs["m"] + bw / 2
    y_chan = 4.72
    ax.plot([xs["l"] - g, x_margin], [yc2, yc2], **dash)
    ax.plot([x_margin, x_margin], [yc2, y_chan], **dash)
    ax.plot([x_margin, x_down], [y_chan, y_chan], **dash)
    ax.add_patch(FancyArrowPatch((x_down, y_chan), (x_down, y1 + bh + g),
                                 arrowstyle="-|>", mutation_scale=12,
                                 shrinkA=0, shrinkB=0, color="#555555",
                                 linestyle=(0, (5, 3)), linewidth=1.1))
    ax.text((x_margin + x_down) / 2, y_chan + 0.10, "shortlist re-validated",
            ha="center", va="bottom", fontsize=6.8, color="#555555")

    save(fig, out_dir, "fig1_workflow")


# ---------------------------------------------------------------------------
# Parity and model comparison
# ---------------------------------------------------------------------------

RESOLUTION_NOTES = {
    "T_gp_solvus": "solvus refined on 2 K",
    "T_solidus": "melt grid 10$\\to$2 K: 3.2 K median",
    "T_liquidus": "melt grid 10$\\to$2 K: 5.9 K median",
    "dH_fusion_J_per_g": "melt grid 10$\\to$2 K: 0.03 J g$^{-1}$",
    "gp_frac_900C": "pdens 400 audit",
}


def figure_parity(pred: pd.DataFrame, metrics: pd.DataFrame, out_dir: Path) -> None:
    targets = [t for t in TARGETS + ("gp_frac_900C",) if f"{t}_gp" in pred]
    ncol = 3
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.25 * nrow),
                             layout="constrained")
    axes = np.atleast_1d(axes).ravel()

    for ax, t in zip(axes, targets):
        y = pred[f"{t}_dsc"].to_numpy(float)
        mu = pred[f"{t}_gp"].to_numpy(float)
        sd = pred[f"{t}_gp_std"].to_numpy(float)
        keep = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd)
        y, mu, sd = y[keep], mu[keep], sd[keep]
        row = metrics.loc[metrics["target"] == t]
        r2 = float(row["gp_r2"].iloc[0]) if len(row) else np.nan
        rmse = float(row["gp_rmse"].iloc[0]) if len(row) else np.nan

        lo = min(y.min(), mu.min())
        hi = max(y.max(), mu.max())
        pad = 0.06 * (hi - lo)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color=PALETTE["grey"],
                lw=0.8, ls="--", zorder=1)
        sc = ax.scatter(y, mu, c=sd, cmap="viridis", s=11, edgecolor="none", alpha=0.85, zorder=3)
        # The 95% interval is drawn as a bar so that the uncertainty is legible
        # without relying on colour alone.
        ax.errorbar(y, mu, yerr=1.96 * sd, fmt="none", ecolor=PALETTE["grey"],
                    elinewidth=0.5, capsize=0, alpha=0.45, zorder=2)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("CALPHAD (test)")
        ax.set_ylabel("GP prediction")
        ax.set_title(TARGET_LABELS.get(t, t))
        ax.text(0.04, 0.96, f"$n$ = {len(y)}\n$R^2$ = {r2:.4f}\nRMSE = {rmse:.3g}",
                transform=ax.transAxes, va="top", fontsize=7.2)
        # Residual histogram as an inset: the distribution of the residuals,
        # not only the point cloud, is what the intervals must match.
        ins = ax.inset_axes([0.62, 0.06, 0.34, 0.26])
        res = mu - y
        ins.hist(res, bins=14, color=PALETTE["gp"], alpha=0.75)
        ins.axvline(0.0, color=PALETTE["grey"], lw=0.6, ls="--")
        ins.set_xticks([])
        ins.set_yticks([])
        ins.set_title("residuals", fontsize=5.5, pad=1.5)
        for s in ins.spines.values():
            s.set_linewidth(0.4)
        ax.set_aspect("equal", adjustable="box")

    for ax in axes[len(targets):]:
        ax.axis("off")
    cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=0.6, pad=0.02)
    cbar.set_label("GP predictive std", fontsize=7.5)
    save(fig, out_dir, "fig2_parity")


def figure_model_comparison(metrics: pd.DataFrame, out_dir: Path) -> None:
    targets = list(metrics["target"])
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    x = np.arange(len(targets))
    width = 0.26
    for k, (model, colour, label) in enumerate((
        ("gp", PALETTE["gp"], "Gaussian process"),
        ("pce", PALETTE["pce"], "Polynomial chaos"),
        ("nn", PALETTE["nn"], "Neural network"),
    )):
        vals = metrics[f"{model}_rmse"].to_numpy(float)
        ax.bar(x + (k - 1) * width, vals, width, color=colour, label=label, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([TARGET_LABELS.get(t, t) for t in targets], rotation=12, ha="right")
    ax.set_ylabel("Test RMSE (target units)")
    ax.set_yscale("log")
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.14))
    ax.grid(axis="y", lw=0.4, alpha=0.4)
    save(fig, out_dir, "fig3_model_comparison")


# ---------------------------------------------------------------------------
# Functional H(T) / Cp(T)
# ---------------------------------------------------------------------------

def figure_curves(out_dir: Path) -> None:
    path = out_dir / "curve_test_predictions.npz"
    if not path.exists():
        print("  skip fig4: no curve predictions")
        return
    d = np.load(path)
    T = d["T"]
    Ht, Hp, Ct, Cp = d["H_true"], d["H_pred"], d["Cp_true"], d["Cp_pred"]
    mask = T <= 1450.0
    n = len(Ht)
    picks = np.linspace(0, n - 1, min(3, n)).astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 3.9), sharex="col")
    for col, i in enumerate(picks):
        ax = axes[0, col]
        ax.plot(T[mask], Ht[i, mask], color=PALETTE["grey"], lw=1.1, label="CALPHAD")
        ax.plot(T[mask], Hp[i, mask], color=PALETTE["gp"], lw=1.0, ls="--", label="PCA-GP")
        ax.set_ylabel(r"$H-H_{950}$ (J g$^{-1}$)")
        ax.set_title(f"test alloy {int(d['ids'][i])}")

        ax = axes[1, col]
        ax.plot(T[mask], Ct[i, mask], color=PALETTE["grey"], lw=1.1)
        ax.plot(T[mask], Cp[i, mask], color=PALETTE["gp"], lw=1.0, ls="--")
        ax.set_ylabel(r"$C_p$ (J g$^{-1}$ K$^{-1}$)")
        ax.set_xlabel("Temperature (K)")
        for a in (axes[0, col], axes[1, col]):
            a.grid(lw=0.3, alpha=0.35)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.tight_layout(pad=0.4)
    save(fig, out_dir / "figures", "fig4_enthalpy_cp")


# ---------------------------------------------------------------------------
# DSC validation
# ---------------------------------------------------------------------------

def figure_dsc(out_dir: Path) -> None:
    path = out_dir / "dsc_validation.csv"
    if not path.exists():
        print("  skip fig5: no DSC validation table")
        return
    d = pd.read_csv(path)
    role = d["role"].fillna("calibration").to_numpy()
    ref = np.isin(role, ("calibration", "validation"))
    proj = ~ref
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7))

    ax = axes[0]
    all_t = np.concatenate([
        d[[f"{c}_{s}" for c in ("T_solidus", "T_liquidus") for s in ("dsc", "calphad")]]
        .to_numpy(float).ravel()
    ])
    all_t = all_t[np.isfinite(all_t)]
    lo, hi = all_t.min() - 8.0, all_t.max() + 8.0
    for k, (col, marker, colour, label) in enumerate((
        ("T_solidus", "o", PALETTE["gp"], "Solidus"),
        ("T_liquidus", "s", PALETTE["pce"], "Liquidus"),
    )):
        obs = d[f"{col}_dsc"].to_numpy(float)
        pred = d[f"{col}_calphad"].to_numpy(float)
        dsc_tier = (role == "calibration")
        val_tier = (role == "validation")
        ax.scatter(obs[dsc_tier], pred[dsc_tier], marker=marker, s=26, color=colour,
                   label=f"{label}, DSC", zorder=4, edgecolor="white", linewidth=0.4)
        ax.scatter(obs[val_tier], pred[val_tier], marker=marker, s=26, facecolors="none",
                   edgecolors=colour, linewidth=0.9,
                   label=f"{label}, reported range", zorder=4)
        ax.scatter(obs[proj], pred[proj], marker="D", s=17, facecolors="none",
                   edgecolors=PALETTE["gold"], linewidth=0.8, zorder=3,
                   label="Off-space projection" if col == "T_solidus" else None)
    ax.plot([lo, hi], [lo, hi], color=PALETTE["grey"], ls="--", lw=0.8, zorder=1)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Reported melting temperature (K)")
    ax.set_ylabel("CALPHAD (K)")
    ax.set_title("Melting temperatures")
    ax.set_aspect("equal", adjustable="box")
    h, lab = ax.get_legend_handles_labels()
    order = [0, 3, 1, 4, 2]
    ax.legend([h[i] for i in order], [lab[i] for i in order], frameon=False,
              loc="upper center", bbox_to_anchor=(1.04, -0.24), ncol=3,
              handletextpad=0.35, columnspacing=0.7, fontsize=6.3)

    ax = axes[1]
    order = ["Rene77", "IN100", "Nimonic105", "Nimonic115", "IN713C", "IN738LC"]
    order = [a for a in order if a in set(d["alloy"])]
    d = d.set_index("alloy").loc[order].reset_index()
    x = np.arange(len(d))
    width = 0.38
    res_s = d["T_solidus_calphad"].to_numpy(float) - d["T_solidus_dsc"].to_numpy(float)
    res_l = d["T_liquidus_calphad"].to_numpy(float) - d["T_liquidus_dsc"].to_numpy(float)
    proj_b = ~np.isin(d["role"].fillna("calibration"), ("calibration", "validation"))
    ax.bar(x - width / 2, res_s, width, color=PALETTE["gp"], edgecolor="white",
           linewidth=0.5, label="Solidus")
    ax.bar(x + width / 2, res_l, width, color=PALETTE["pce"], edgecolor="white",
           linewidth=0.5, label="Liquidus")
    for xx, p in zip(x, proj_b):
        if p:
            ax.axvspan(xx - 0.5, xx + 0.5, color=PALETTE["gold"], alpha=0.07, lw=0)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    labels = ["René 77", "IN100", "Nim. 105", "Nim. 115", "IN713C*", "IN738LC*"]
    ax.set_xticklabels(labels[:len(order)], rotation=28, ha="right")
    ax.tick_params(axis="x", labelsize=7.0)
    ax.set_ylabel("CALPHAD $-$ reported (K)")
    ax.set_title("Residuals")
    ax.set_ylim(-45, 100)
    ax.legend(frameon=False, loc="upper left", ncol=2, handletextpad=0.4,
              columnspacing=0.8)
    fig.tight_layout(pad=0.4)
    save(fig, out_dir / "figures", "fig5_dsc_validation")


# ---------------------------------------------------------------------------
# Design map and response surface
# ---------------------------------------------------------------------------

def figure_design(out_dir: Path) -> None:
    cloud_path = out_dir / "design_cloud.csv"
    cand_path = out_dir / "design_candidates.csv"
    if not cloud_path.exists() or not cand_path.exists():
        print("  skip fig6: no design outputs")
        return
    cloud = pd.read_csv(cloud_path)
    cand = pd.read_csv(cand_path)
    valid = pd.read_csv(out_dir / "design_validated.csv")

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.9))

    # (a) Trade-off plane with the Pareto front, the floor-price curve and the
    # CALPHAD-validated anchors at each floor.
    ax = axes[0]
    sc = ax.scatter(cloud["gp_frac_900C"], cloud["W_K"], c=cloud["dH_fusion_J_per_g"],
                    s=3, cmap="magma", alpha=0.35, rasterized=True)
    front_path = out_dir / "pareto_front.csv"
    if front_path.exists():
        front = pd.read_csv(front_path).sort_values("gp_frac_900C")
        ax.plot(front["gp_frac_900C"], front["W_K"], color=PALETTE["gp"], lw=1.2,
                label="Pareto front (predicted)")
    # Floor-price curve: best predicted window at each gamma-prime floor.
    f = cloud["gp_frac_900C"].to_numpy(float)
    W = cloud["W_K"].to_numpy(float)
    floors = np.arange(0.40, 0.61, 0.01)
    price = [np.max(W[f >= fl - 1e-9]) if (f >= fl - 1e-9).any() else np.nan for fl in floors]
    ax.plot(floors, price, color="0.35", lw=0.9, ls=":", label="best $W$ at floor")
    try:
        thr = pd.read_csv(out_dir / "threshold_sensitivity.csv")
        anchors = thr.loc[thr["valid"] > 0].groupby("floor").first().reset_index()
        ax.scatter(anchors["f_gp_pred"], anchors["W_calphad"], marker="o", s=26,
                   facecolor="none", edgecolor=PALETTE["nn"], linewidth=1.0,
                   label="validated best at floor", zorder=5)
    except Exception:  # noqa: BLE001
        pass
    ax.scatter(cand["gp_frac_900C"], cand["W_K"], marker="*", s=70,
               facecolor="none", edgecolor=PALETTE["gp"], linewidth=0.9, label="selected (0.40)")
    try:
        dsc_path = out_dir / "dsc_holdout_calphad_pdens400.csv"
        if not dsc_path.exists():
            dsc_path = out_dir / "dsc_holdout_calphad.csv"
        dsc = pd.read_csv(dsc_path)
        ref = dsc.loc[dsc["alloy"] == "Rene77"].iloc[0]
        ax.scatter([ref["gp_frac_900C"]], [ref["T_solidus"] - ref["T_gp_solvus"]],
                   marker="D", s=30, color=PALETTE["nn"], label="René 77", zorder=4)
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlabel(r"$f_{\gamma'}$ (900$^\circ$C)")
    ax.set_ylabel(r"$W^{\rm m}$ (K)")
    ax.set_title("(a) Trade-off, not single optimum")
    ax.legend(frameon=False, loc="upper left", fontsize=6)
    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
    cbar.set_label(r"$\Delta H_{\rm fus}$ (J g$^{-1}$)", fontsize=7)

    # (b) Mechanism: the window is anti-correlated with the solvus, so the
    # optimisation depresses the solvus while the solidus stays put.
    ax = axes[1]
    ax.scatter(cloud["T_gp_solvus"], cloud["W_K"], s=3, color="0.55", alpha=0.3,
               rasterized=True)
    ax.scatter(valid["T_gp_solvus_calphad"], valid["W_calphad_K"], marker="*",
               s=70, facecolor="none", edgecolor=PALETTE["gp"], linewidth=0.9,
               label="selected (CALPHAD)", zorder=5)
    try:
        dsc = pd.read_csv(out_dir / "dsc_holdout_calphad_pdens400.csv")
        ref = dsc.loc[dsc["alloy"] == "Rene77"].iloc[0]
        ax.scatter([ref["T_gp_solvus"]], [ref["T_solidus"] - ref["T_gp_solvus"]],
                   marker="D", s=30, color=PALETTE["nn"], label="René 77", zorder=4)
    except Exception:  # noqa: BLE001
        pass
    ax.set_xlabel(r"$T_{\gamma'}$ solvus (K)")
    ax.set_ylabel(r"$W^{\rm m}$ (K)")
    ax.set_title(r"(b) Gain is solvus depression ($\rho=-0.91$)")
    ax.legend(frameon=False, loc="upper right", fontsize=6)

    # (c) Partial-dependence surface for Al and Ti at the Rene 77 base.
    ax = axes[2]
    al = np.linspace(3.0, 6.5, 55)
    ti = np.linspace(0.5, 5.5, 55)
    A, T = np.meshgrid(al, ti)
    base = np.array([14.49, 14.28, 4.42, 4.20, 3.38, 0.10])
    X = np.tile(base, (A.size, 1))
    X[:, FEATURES.index("Al")] = A.ravel()
    X[:, FEATURES.index("Ti")] = T.ravel()
    models = {t: joblib.load(out_dir / f"model_gp_{t}.joblib")["model"] for t in
              ("T_gp_solvus", "T_solidus")}
    Wm = (models["T_solidus"].predict(X) - models["T_gp_solvus"].predict(X)).reshape(A.shape)
    cf = ax.contourf(A, T, Wm, levels=18, cmap="viridis")
    ax.contour(A, T, Wm, levels=8, colors="white", linewidths=0.3, alpha=0.6)
    ax.scatter([4.42], [3.38], marker="D", color=PALETTE["nn"], s=24, label="René 77")
    ax.set_xlabel("Al (wt%)")
    ax.set_ylabel("Ti (wt%)")
    ax.set_title("(c) GP window over Al–Ti")
    cb = fig.colorbar(cf, ax=ax, pad=0.02)
    cb.set_label(r"$W^{\rm m}$ (K)", fontsize=7)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout(pad=0.4)
    save(fig, out_dir / "figures", "fig6_design")


def figure_carbon(out_dir: Path) -> None:
    """High-resolution carbon sensitivity around P1, A1 and Rene 77."""
    path = out_dir / "carbon_sensitivity.csv"
    if not path.exists():
        print("  skip figS-carbon: run carbon_sensitivity.py first")
        return
    df = pd.read_csv(path)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    colors = {"P1": PALETTE["gp"], "A1": PALETTE["nn"], "R77": "0.35"}
    # The 2 K audit rows carry the smooth solidus and the refined-solvus
    # window; the 10 K production rows are quantised to the grid nodes.
    model = "matrix2K" if "matrix2K" in set(df["model"]) else "matrix+solvus"
    ax = axes[0]
    for tag, sub in df[df["model"] == model].groupby("alloy"):
        sub = sub.sort_values("C_wt")
        ax.plot(sub["C_wt"], sub["T_solidus"], "o-", ms=2.5, lw=0.9,
                color=colors.get(tag, None), label=tag)
    ax.set_xlabel("C (wt%)")
    ax.set_ylabel(r"matrix solidus (K)")
    ax.set_title("(a) Solidus vs carbon (2 K audit grid)")
    ax.legend(frameon=False, fontsize=6.5)
    ax = axes[1]
    for tag, sub in df[df["model"] == model].groupby("alloy"):
        sub = sub.sort_values("C_wt")
        ax.plot(sub["C_wt"], sub["W_K"], "o-", ms=2.5, lw=0.9,
                color=colors.get(tag, None), label=tag)
    fe = df[(df["model"] == "fulleq12") & (df["max_liquid_jump"] < 0.2)
            & np.isfinite(df["W_K"])]
    for tag, sub in fe.groupby("alloy"):
        sub = sub.sort_values("C_wt")
        ax.plot(sub["C_wt"], sub["W_K"], "^", ms=4, mfc="none",
                color=colors.get(tag, None), label=f"{tag} full eq. (resolved)")
    ax.set_xlabel("C (wt%)")
    ax.set_ylabel(r"$W^{\rm m}$ (K)")
    ax.set_title("(b) Window vs carbon")
    ax.legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4)
    save(fig, out_dir / "figures", "figS2_carbon")


# ---------------------------------------------------------------------------
# Uncertainty calibration
# ---------------------------------------------------------------------------

def figure_uncertainty(pred: pd.DataFrame, out_dir: Path) -> None:
    targets = [t for t in TARGETS if f"{t}_gp" in pred]
    if "gp_frac_900C_gp" in pred:
        targets.append("gp_frac_900C")
    fig, axes = plt.subplots(1, len(targets), figsize=(8.6, 2.0), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, t in zip(axes, targets):
        y = pred[f"{t}_dsc"].to_numpy(float)
        mu = pred[f"{t}_gp"].to_numpy(float)
        sd = np.maximum(pred[f"{t}_gp_std"].to_numpy(float), 1e-12)
        # Per-target labels are stored with NaNs for alloys whose label is
        # censored or invalid; they must be dropped before computing coverage,
        # otherwise a NaN compares False and PICP is biased downwards.
        keep = np.isfinite(y) & np.isfinite(mu) & np.isfinite(sd)
        z = (y[keep] - mu[keep]) / sd[keep]
        ax.hist(z, bins=np.linspace(-4, 4, 25), color=PALETTE["gp"], alpha=0.8,
                edgecolor="white", linewidth=0.4, density=True)
        xs = np.linspace(-4, 4, 200)
        ax.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), color=PALETTE["pce"], lw=0.9)
        picp = float(np.mean(np.abs(z) <= 1.96))
        ax.set_title(f"{TARGET_LABELS.get(t, t)}\nPICP$_{{95}}$={picp:.2f} ($N$={len(z)})",
                     fontsize=7.2)
        ax.set_xlabel("z-score")
        ax.set_yticks([])
    axes[0].set_ylabel("density")
    fig.tight_layout(pad=0.4)
    save(fig, out_dir, "fig7_uncertainty")


# ---------------------------------------------------------------------------
# Learning curves
# ---------------------------------------------------------------------------

def figure_learning_curve(out_dir: Path) -> None:
    path = out_dir / "learning_curve_summary.csv"
    if not path.exists():
        print("  skip fig8: no learning-curve data")
        return
    d = pd.read_csv(path)
    targets = [t for t in TARGETS if t in set(d["target"])]
    ncol = 2
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(7.2, 2.2 * nrow), squeeze=False)
    axes = axes.ravel()
    for ax, t in zip(axes, targets):
        for model, colour, marker, label in (
            ("gp", PALETTE["gp"], "o", "GP"),
            ("pce", PALETTE["pce"], "s", "PCE"),
        ):
            s = d[(d["target"] == t) & (d["model"] == model)].sort_values("n_train")
            if s.empty:
                continue
            ax.errorbar(s["n_train"], s["rmse_mean"], yerr=s["rmse_std"].fillna(0.0),
                        color=colour, marker=marker, ms=3.5, lw=1.0, capsize=2,
                        label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        # The default log minor ticks place 3x10^1 and 6x10^1 on top of each
        # other; label a sparse set of training sizes instead.
        ticks = [25, 50, 100, 200, 350]
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel("Training alloys")
        ax.set_ylabel("Test RMSE")
        ax.set_title(TARGET_LABELS.get(t, t))
        ax.grid(lw=0.3, alpha=0.35)
    for ax in axes[len(targets):]:
        ax.axis("off")
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.4)
    save(fig, out_dir / "figures", "fig8_learning_curves")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default=str(ROOT / "results"))
    args = parser.parse_args()
    out_dir = Path(args.results)
    fig_dir = out_dir / "figures"

    figure_workflow(fig_dir)
    figure_dataset(fig_dir)

    pred_path = out_dir / "surrogate_test_predictions.csv"
    metrics_path = out_dir / "surrogate_metrics.csv"
    if pred_path.exists() and metrics_path.exists():
        pred = pd.read_csv(pred_path)
        metrics = pd.read_csv(metrics_path)
        figure_parity(pred, metrics, fig_dir)
        figure_model_comparison(metrics, fig_dir)
        figure_uncertainty(pred, fig_dir)
    else:
        print("  skip parity/comparison: run train_surrogate.py first")

    figure_curves(out_dir)
    figure_dsc(out_dir)
    figure_design(out_dir)
    figure_carbon(out_dir)
    figure_learning_curve(out_dir)
    print(f"Figures in {fig_dir}")


if __name__ == "__main__":
    main()
