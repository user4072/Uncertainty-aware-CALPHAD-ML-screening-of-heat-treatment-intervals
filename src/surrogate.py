"""Surrogate models for CALPHAD thermal signatures and H(T) curves.

Three model families are provided so that the paper can compare them on the
same split:

* ``GaussianProcessSurrogate``  - ARD Matern-5/2 GP with a white-noise term,
  giving calibrated predictive uncertainty.
* ``PolynomialChaosSurrogate``  - total-degree Legendre polynomial chaos
  expansion fitted by ridge regression (spectral surrogate).
* ``NeuralNetworkSurrogate``    - fully connected MLP.

All models take wt% solute compositions (Co, Cr, Al, Mo, Ti, C) and predict
scalar DSC-relevant signatures.  Functional H(T) curves are handled by a
separate PCA + GP model (``CurveSurrogate``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import RidgeCV
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES: Tuple[str, ...] = ("Co", "Cr", "Al", "Mo", "Ti", "C")
TARGETS: Tuple[str, ...] = ("T_gp_solvus", "T_solidus", "T_liquidus", "dH_fusion_J_per_g")
TARGET_LABELS: Dict[str, str] = {
    "T_gp_solvus": r"$T_{\gamma'}$ (K)",
    "T_solidus": r"$T_{\rm sol}$ (K)",
    "T_liquidus": r"$T_{\rm liq}$ (K)",
    "dH_fusion_J_per_g": r"$\Delta H_{\rm fus}$ (J g$^{-1}$)",
    "gp_frac_900C": r"$f_{\gamma'}$ (900$^\circ$C)",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    comp: np.ndarray            # (n, 6) wt%
    y: Dict[str, np.ndarray]    # target -> (n,)
    split: np.ndarray           # (n,) 'train'/'test'
    ids: np.ndarray
    T_solid: np.ndarray         # (n_T,)
    H_solid: np.ndarray         # (n, n_T) J/mol
    gp_solid: np.ndarray        # (n, n_T)
    T_melt: np.ndarray
    H_melt: np.ndarray
    Cp_melt: np.ndarray
    liquid_melt: np.ndarray
    molar_mass: np.ndarray      # (n,)
    valid: Dict[str, np.ndarray] = field(default_factory=dict)  # target -> (n,) bool


def load_dataset(prefix: str = "dataset", root: str = ".") -> Dataset:
    import os
    sig = pd.read_csv(os.path.join(root, "data", "processed", f"{prefix}_signatures.csv"))
    curves = np.load(os.path.join(root, "data", "processed", f"{prefix}_curves.npz"))

    # Keep every alloy for which both calculations produced a curve.  Validity
    # is handled per target below: a censored solvus does not invalidate the
    # solidus or the melting enthalpy of the same alloy, and discarding the
    # whole row would waste expensive CALPHAD data.
    ids_available = set(int(i) for i in curves["ids"])
    sig = sig.loc[sig["id"].isin(ids_available)].reset_index(drop=True)

    # Align curves with the signature table.
    order = {int(i): k for k, i in enumerate(curves["ids"])}
    idx = np.array([order[int(i)] for i in sig["id"]])

    from calphad_engine import mean_molar_mass  # local import to avoid cycles

    comp = sig[list(FEATURES)].to_numpy(dtype=float)
    m = np.empty(len(comp))
    for i in range(len(comp)):
        d = {f: float(comp[i, j]) for j, f in enumerate(FEATURES)}
        d["Ni"] = float(100.0 - comp[i].sum())
        m[i] = mean_molar_mass(d)
    y = {t: sig[t].to_numpy(dtype=float) for t in TARGETS if t in sig}
    if "gp_frac_900C" in sig:
        y["gp_frac_900C"] = sig["gp_frac_900C"].to_numpy(dtype=float)

    base = ((sig["melt_valid"] == 1) & (sig["solid_valid"] == 1)).to_numpy()
    flag = {}
    for name in ("censored_gp_high", "censored_solidus_low", "censored_liquidus_high"):
        flag[name] = (sig[name].fillna(0.0).to_numpy() == 1.0) if name in sig else np.zeros(len(sig), bool)

    t_sol = sig["T_solidus"].to_numpy(dtype=float)
    valid: Dict[str, np.ndarray] = {
        "T_solidus": base & ~flag["censored_solidus_low"] & np.isfinite(t_sol),
        "T_liquidus": base & ~flag["censored_liquidus_high"] & np.isfinite(y.get("T_liquidus", np.nan * t_sol)),
        "T_gp_solvus": (base & ~flag["censored_gp_high"] & np.isfinite(y.get("T_gp_solvus", np.nan * t_sol))
                        & (y.get("T_gp_solvus", 0.0 * t_sol) < t_sol - 5.0)),
        "dH_fusion_J_per_g": (base & ~flag["censored_solidus_low"] & ~flag["censored_liquidus_high"]
                              & np.isfinite(y.get("dH_fusion_J_per_g", np.nan * t_sol))
                              & (y.get("dH_fusion_J_per_g", 0.0 * t_sol) > 0.0)),
        "gp_frac_900C": base & np.isfinite(y.get("gp_frac_900C", np.nan * t_sol)),
    }

    H_solid = np.asarray(curves["H_solid"], dtype=float)[idx]
    low = np.asarray(curves["T_solid"], dtype=float) <= 1450.0
    valid["curve"] = base & np.isfinite(H_solid[:, low]).all(axis=1)

    return Dataset(
        comp=comp,
        y=y,
        split=sig["split"].to_numpy(),
        ids=sig["id"].to_numpy(),
        T_solid=np.asarray(curves["T_solid"], dtype=float),
        H_solid=H_solid,
        gp_solid=np.asarray(curves["gp_solid"], dtype=float)[idx],
        T_melt=np.asarray(curves["T_melt"], dtype=float),
        H_melt=np.asarray(curves["H_melt"], dtype=float)[idx],
        Cp_melt=np.asarray(curves["Cp_melt"], dtype=float)[idx],
        liquid_melt=np.asarray(curves["liquid_melt"], dtype=float)[idx],
        molar_mass=m,
        valid=valid,
    )


# ---------------------------------------------------------------------------
# Kernel and GP surrogate
# ---------------------------------------------------------------------------

def make_kernel(n_features: int) -> object:
    """ARD Matern-5/2 plus a small white-noise term."""
    return (
        ConstantKernel(1.0, (1e-2, 1e3))
        * Matern(length_scale=np.ones(n_features), length_scale_bounds=(1e-2, 1e2), nu=2.5)
        + WhiteKernel(noise_level=1e-4, noise_level_bounds=(1e-8, 1e-1))
    )


class StandardizedGP:
    """Single-output GP on standardised inputs and targets."""

    def __init__(self, n_restarts: int = 3, seed: int = 0):
        self.n_restarts = n_restarts
        self.seed = seed
        self.x_scaler = StandardScaler()
        self.y_mean = 0.0
        self.y_std = 1.0
        self.sd_scale = 1.0
        self.gp: Optional[GaussianProcessRegressor] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "StandardizedGP":
        Xs = self.x_scaler.fit_transform(X)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) or 1.0
        ys = (y - self.y_mean) / self.y_std
        self.gp = GaussianProcessRegressor(
            kernel=make_kernel(X.shape[1]),
            normalize_y=False,
            n_restarts_optimizer=self.n_restarts,
            random_state=self.seed,
        ).fit(Xs, ys)
        return self

    def calibrate(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5,
                  seed: int = 0) -> "StandardizedGP":
        """Scale the predictive std by a cross-validated z-score factor.

        A white-noise term fitted by marginal likelihood on deterministic
        CALPHAD labels tends to collapse to a very small value, which makes
        the predictive interval over-confident.  The correction is a single
        global factor estimated from out-of-fold residuals; it does not use
        the test split.
        """
        from sklearn.model_selection import KFold

        n_splits = min(n_splits, len(y))
        if n_splits < 2:
            return self
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        z = []
        for tr, te in kf.split(X):
            m = StandardizedGP(n_restarts=1, seed=seed).fit(X[tr], y[tr])
            mu, sd = m.predict(X[te], return_std=True)
            z.append((y[te] - mu) / np.maximum(sd, 1e-12))
        self.sd_scale = float(np.sqrt(np.mean(np.concatenate(z) ** 2)))
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        Xs = self.x_scaler.transform(X)
        mu, sd = self.gp.predict(Xs, return_std=True)
        mu = mu * self.y_std + self.y_mean
        if return_std:
            return mu, sd * self.y_std * self.sd_scale
        return mu


class PolynomialChaos:
    """Total-degree Legendre polynomial chaos via ridge regression.

    Inputs are mapped to [-1, 1] with the training bounds, and the multi-index
    set contains every monomial with total degree <= ``degree``.
    """

    def __init__(self, degree: int = 3, n_folds: int = 5):
        self.degree = degree
        self.n_folds = n_folds
        self.lo: Optional[np.ndarray] = None
        self.hi: Optional[np.ndarray] = None
        self.model: Optional[Pipeline] = None
        self.multi_index: List[Tuple[int, ...]] = []
        self.y_mean = 0.0
        self.y_std = 1.0

    def _design(self, X: np.ndarray) -> np.ndarray:
        U = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        cols = [np.ones(len(U))]
        for idx in self.multi_index:
            if sum(idx) == 0:
                continue
            col = np.ones(len(U))
            for j, power in enumerate(idx):
                if power:
                    # Legendre recurrence via numpy polynomial evaluation.
                    c = np.zeros(power + 1)
                    c[power] = 1.0
                    col *= np.polynomial.legendre.legval(U[:, j], c)
            cols.append(col)
        return np.column_stack(cols)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "PolynomialChaos":
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)
        span = np.where(self.hi > self.lo, self.hi - self.lo, 1.0)
        self.lo = self.lo - 0.02 * span
        self.hi = self.hi + 0.02 * span
        n_features = X.shape[1]
        self.multi_index = []
        for total in range(self.degree + 1):
            for idx in combinations_with_replacement(range(n_features), total):
                counts = np.zeros(n_features, dtype=int)
                for j in idx:
                    counts[j] += 1
                self.multi_index.append(tuple(counts))
        A = self._design(X)
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) or 1.0
        cv = min(self.n_folds, max(2, len(y)))
        self.model = Pipeline([
            ("scale", StandardScaler(with_mean=False)),
            ("ridge", RidgeCV(alphas=np.logspace(-8, 2, 30), cv=cv)),
        ]).fit(A, (y - self.y_mean) / self.y_std)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(self._design(X)) * self.y_std + self.y_mean


class NeuralNetworkSurrogate:
    """Small MLP with early stopping, trained on standardised inputs."""

    def __init__(self, hidden=(64, 64), seed: int = 0):
        self.hidden = hidden
        self.seed = seed
        self.model: Optional[Pipeline] = None
        self.y_mean = 0.0
        self.y_std = 1.0

    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkSurrogate":
        self.y_mean = float(np.mean(y))
        self.y_std = float(np.std(y)) or 1.0
        early = len(y) >= 40
        self.model = Pipeline([
            ("scale", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=self.hidden,
                activation="tanh",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=4000,
                early_stopping=early,
                n_iter_no_change=40,
                validation_fraction=0.15,
                random_state=self.seed,
            )),
        ]).fit(X, (y - self.y_mean) / self.y_std)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X) * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# Functional H(T) surrogate
# ---------------------------------------------------------------------------

@dataclass
class CurveSurrogate:
    """PCA of relative enthalpy curves plus one GP per principal component."""

    n_components: int = 8
    models: List[StandardizedGP] = field(default_factory=list)
    pca: Optional[PCA] = None
    T: Optional[np.ndarray] = None
    mean_curve: Optional[np.ndarray] = None
    scale: float = 1.0

    def fit(self, X: np.ndarray, H: np.ndarray, T: np.ndarray,
            molar_mass: np.ndarray) -> "CurveSurrogate":
        # Relative enthalpy in J/g, anchored at the lowest temperature.
        Hj = (H - H[:, :1]) / molar_mass[:, None]
        self.T = np.asarray(T, dtype=float)
        self.mean_curve = Hj.mean(axis=0)
        self.scale = float(np.std(Hj)) or 1.0
        Z = (Hj - self.mean_curve) / self.scale
        self.pca = PCA(n_components=self.n_components, random_state=0).fit(Z)
        scores = self.pca.transform(Z)
        self.models = [StandardizedGP(n_restarts=2, seed=k).fit(X, scores[:, k])
                       for k in range(scores.shape[1])]
        return self

    def predict(self, X: np.ndarray, return_std: bool = False):
        preds, stds = [], []
        for model in self.models:
            if return_std:
                mu, sd = model.predict(X, return_std=True)
                preds.append(mu)
                stds.append(sd)
            else:
                preds.append(model.predict(X))
        scores = np.column_stack(preds)
        Z = self.pca.inverse_transform(scores)
        Hj = Z * self.scale + self.mean_curve
        if not return_std:
            return Hj
        # Propagate PC score uncertainty through the PCA loadings.
        loadings = self.pca.components_  # (k, n_T)
        var = np.einsum("nk,kt->nt", np.column_stack(stds) ** 2, loadings ** 2) * self.scale ** 2
        return Hj, np.sqrt(var)

    def predict_cp(self, X: np.ndarray) -> np.ndarray:
        Hj = self.predict(X)
        return np.gradient(Hj, self.T, axis=1)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    resid = y_pred - y_true
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {
        "rmse": float(np.sqrt(np.mean(resid ** 2))),
        "mae": float(np.mean(np.abs(resid))),
        "max_abs": float(np.max(np.abs(resid))),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "bias": float(np.mean(resid)),
    }


def uncertainty_metrics(y_true: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> Dict[str, float]:
    sd = np.maximum(np.asarray(sd, dtype=float), 1e-12)
    z = (np.asarray(y_true, dtype=float) - mu) / sd
    return {
        "z_mean": float(np.mean(z)),
        "z_std": float(np.std(z)),
        "picp95": float(np.mean(np.abs(z) <= 1.96)),
        "nlpd": float(np.mean(0.5 * np.log(2 * np.pi * sd ** 2) + 0.5 * z ** 2)),
    }


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------

@dataclass
class SurrogateBundle:
    gp: Dict[str, StandardizedGP]
    pce: Dict[str, PolynomialChaos]
    nn: Dict[str, NeuralNetworkSurrogate]
    curve: Optional[CurveSurrogate] = None

    def predict_gp(self, X: np.ndarray, target: str, return_std: bool = False):
        return self.gp[target].predict(X, return_std=return_std)

    def predict_all(self, X: np.ndarray) -> Dict[str, Dict[str, np.ndarray]]:
        out: Dict[str, Dict[str, np.ndarray]] = {}
        for t in self.gp:
            mu, sd = self.gp[t].predict(X, return_std=True)
            out[t] = {
                "gp": mu, "gp_std": sd,
                "pce": self.pce[t].predict(X),
                "nn": self.nn[t].predict(X),
            }
        return out
