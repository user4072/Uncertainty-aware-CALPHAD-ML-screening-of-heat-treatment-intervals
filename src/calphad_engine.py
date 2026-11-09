"""CALPHAD engine for Ni-based superalloy thermal signatures.

The module wraps pycalphad and the open MatCalc mc_ni database.  It is
deliberately written around one composition at a time so that the expensive
equilibrium calls can be distributed over processes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from pycalphad import Database, equilibrium, variables as v

# IUPAC molar masses (g/mol).
MOLAR_MASS: Dict[str, float] = {
    "Ni": 58.6934,
    "Co": 58.9332,
    "Cr": 51.9961,
    "Al": 26.9815,
    "Mo": 95.95,
    "Ti": 47.867,
    "C": 12.011,
    "Nb": 92.9064,
    "Fe": 55.845,
    "W": 183.84,
    "Hf": 178.49,
    "Ta": 180.948,
    "B": 10.81,
    "V": 50.9415,
}

SOLUTES = ("Co", "Cr", "Al", "Mo", "Ti", "C")


def wt_to_mol(comp_wt: Dict[str, float]) -> Dict[str, float]:
    """Convert a wt% composition dictionary to mole fractions.

    Ni is treated as the balance if it is not supplied explicitly.  The
    returned dictionary always sums to one (within floating-point error).
    """
    comp = {k: float(v) for k, v in comp_wt.items() if float(v) > 0.0}
    if "Ni" not in comp:
        comp["Ni"] = 100.0 - sum(comp.values())
    if comp["Ni"] <= 0.0:
        raise ValueError(f"Non-positive Ni balance for composition {comp_wt}")
    moles = {k: comp[k] / MOLAR_MASS[k] for k in comp}
    total = sum(moles.values())
    return {k: val / total for k, val in moles.items()}


def mol_to_wt(comp_mol: Dict[str, float]) -> Dict[str, float]:
    """Convert mole fractions to wt%."""
    grams = {k: val * MOLAR_MASS[k] for k, val in comp_mol.items()}
    total = sum(grams.values())
    return {k: 100.0 * val / total for k, val in grams.items()}


def mean_molar_mass(comp_wt: Dict[str, float]) -> float:
    x = wt_to_mol(comp_wt)
    return float(sum(x[k] * MOLAR_MASS[k] for k in x))


def make_conditions(comp_wt: Dict[str, float], T_grid: Sequence[float], elements: Sequence[str]):
    """Build pycalphad conditions for one composition.

    Every active component except Ni and vacancies needs an independent
    composition condition; a component left without one makes the system
    over-determined and the solver returns NaN.  Solutes that are absent from
    the alloy are therefore pinned to a numerically negligible mole fraction
    instead of being skipped.
    """
    comps = [e for e in elements]
    x = {k.upper(): float(val) for k, val in wt_to_mol(comp_wt).items()}
    conds = {v.T: (float(T_grid[0]), float(T_grid[-1]), float(T_grid[1] - T_grid[0])), v.P: 101325.0, v.N: 1.0}
    for el in comps:
        key = el.upper()
        if key in ("NI", "VA"):
            continue
        value = x.get(key, 0.0)
        conds[v.X(key)] = float(value) if value > 0.0 else 1.0e-12
    return comps, conds


def _phase_fraction_matrix(eq, n_t: int) -> Tuple[np.ndarray, np.ndarray]:
    """Return (T, phase-fraction matrix) from a pycalphad equilibrium Dataset."""
    # NP has dims (N, P, T, <composition coords>, vertex).  All composition
    # coordinate dimensions are singleton because we calculate one alloy at a
    # time, so a reshape to (n_t, n_vertex) is safe.
    np_arr = np.asarray(eq.NP.values, dtype=float).reshape(n_t, -1)
    ph_arr = np.asarray(eq.Phase.values, dtype=object).reshape(n_t, -1)
    return np_arr, ph_arr


def _parse_equilibrium(eq, output_hm: bool, molar_mass: float) -> Dict[str, object]:
    """Convert one pycalphad Dataset into arrays used by the pipeline."""
    T_out = np.asarray(eq.T.values, dtype=float).ravel()
    n_t = len(T_out)
    if n_t == 0:
        return {"valid": False, "error": "empty equilibrium grid", "T": T_out}

    NP, PH = _phase_fraction_matrix(eq, n_t)
    frac: Dict[str, np.ndarray] = {}
    for i in range(n_t):
        for name, value in zip(PH[i], NP[i]):
            if not name or str(name) in ("nan", "None"):
                continue
            if not np.isfinite(value) or value <= 0.0:
                continue
            frac[str(name)] = frac.get(str(name), np.zeros(n_t))
            frac[str(name)][i] += float(value)

    G = np.asarray(eq.GM.values, dtype=float).ravel()
    H = np.asarray(eq.HM.values, dtype=float).ravel() if output_hm else np.full(n_t, np.nan)
    valid = len(G) == n_t and np.isfinite(G).all()
    if output_hm:
        valid = valid and len(H) == n_t and np.isfinite(H).all()

    # Apparent heat capacity; the latent heat appears as a peak where H(T)
    # changes rapidly across a first-order transition.
    if output_hm and n_t >= 2 and len(H) == n_t and np.isfinite(H).all():
        Cp = np.gradient(H, T_out)
    else:
        Cp = np.full(n_t, np.nan)

    return {
        "valid": bool(valid),
        "T": T_out,
        "G": G,
        "H": H,
        "Cp": Cp,
        "frac": frac,
        "molar_mass": molar_mass,
    }


def equilibrium_curve(
    db: Database,
    comp_wt: Dict[str, float],
    T_grid: Sequence[float],
    elements: Sequence[str],
    phases: Sequence[str],
    pdens: int = 1000,
    output_hm: bool = True,
) -> Dict[str, object]:
    """Calculate equilibrium phase fractions and H(T) for one composition.

    Returns a dictionary with the temperature grid, phase fractions, Gibbs
    energy, enthalpy and apparent heat capacity.  ``valid`` is False when the
    equilibrium solver failed at any temperature.  The grid minimiser
    occasionally misses the stable sigma/bcc solution at low temperature; in
    that case the whole grid is retried with a denser point density, which is
    cheaper than per-temperature recovery and keeps the curve consistent.
    """
    T_grid = np.asarray(T_grid, dtype=float)
    comps, conds = make_conditions(comp_wt, T_grid, elements)
    m = mean_molar_mass(comp_wt)

    last: Dict[str, object] = {"valid": False, "error": "not run", "T": T_grid}
    error = ""

    def finalize(p: Dict[str, object]) -> Dict[str, object]:
        """Repair isolated single-phase points before returning a curve.

        Two independent glitches are handled: a metastable GAMMA_PRIME-only
        branch in solid-state curves, and an isolated single-phase LIQUID
        point in melting curves (a liquid fraction of one at one temperature,
        with solid on both sides, a hundred kelvin below any physical
        solidus).  Both are re-solved at a denser point density and the new
        solution is kept only when it is more stable.
        """
        if not p.get("valid", False):
            return p
        T = np.asarray(p["T"], dtype=float)
        frac = p.get("frac", {})
        H = p.get("H", np.full(len(T), np.nan))
        G = p.get("G")
        repaired = 0
        if "GAMMA_PRIME" in frac:
            try:
                frac, H, G, n = repair_ordered_points(
                    db, comp_wt, elements, phases, T, frac, H, G=G,
                    molar_mass=m, pdens_retry=max(4000, 2 * int(pdens)),
                )
            except Exception:  # noqa: BLE001
                n = 0
            if n:
                p["ordered_branch_repaired"] = int(n)
                repaired += n
        if "LIQUID" in frac:
            try:
                frac, H, G, n = repair_isolated_liquid_points(
                    db, comp_wt, elements, phases, T, frac, H, G=G,
                    molar_mass=m, pdens_retry=max(4000, 2 * int(pdens)),
                )
            except Exception:  # noqa: BLE001
                n = 0
            if n:
                p["liquid_branch_repaired"] = int(n)
                repaired += n
        if repaired:
            p["frac"] = frac
            p["H"] = H
            if G is not None:
                p["G"] = G
            if output_hm and np.isfinite(H).all():
                p["Cp"] = np.gradient(H, T)
        return p

    def run(attempt: int) -> Dict[str, object]:
        eq = equilibrium(
            db,
            comps,
            list(phases),
            conds,
            output="HM" if output_hm else None,
            calc_opts={"pdens": int(attempt)},
        )
        parsed = _parse_equilibrium(eq, output_hm, m)
        parsed["pdens_used"] = int(attempt)
        return parsed

    try:
        parsed = run(pdens)
        if parsed["valid"]:
            return finalize(parsed)
        last = parsed

        n_t = len(parsed["T"])
        bad = int(np.sum(~np.isfinite(np.asarray(parsed["G"], dtype=float))))
        if n_t > 0 and bad / n_t > 0.3:
            # A large fraction failed: the discretisation itself is too
            # coarse, so retry the whole grid before point-wise recovery.
            try:
                parsed = run(max(900, 3 * pdens))
                if parsed["valid"]:
                    return finalize(parsed)
                last = parsed
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"

        if len(last["T"]) > 0:
            parsed = _recover_failed_temperatures(
                db, comp_wt, elements, phases, last, output_hm, m, pdens_retry=2500
            )
            if parsed["valid"]:
                return finalize(parsed)
            last = parsed
    except Exception as exc:  # noqa: BLE001
        error = f"{type(exc).__name__}: {exc}"

    if error:
        last["error"] = error
    return last


def _recover_failed_temperatures(
    db: Database,
    comp_wt: Dict[str, float],
    elements: Sequence[str],
    phases: Sequence[str],
    parsed: Dict[str, object],
    output_hm: bool,
    molar_mass: float,
    pdens_retry: int = 2500,
) -> Dict[str, object]:
    """Re-solve isolated temperatures where the global grid search failed.

    A few compositions in the high-C/high-Mo corner return NaN for the stable
    sigma-involving state at low temperature.  Such points are rare and
    isolated, so each is retried on its own with a much denser grid.
    """
    T = np.asarray(parsed["T"], dtype=float)
    G = np.asarray(parsed["G"], dtype=float).copy()
    H = np.asarray(parsed["H"], dtype=float).copy()
    bad = ~np.isfinite(G)
    if output_hm:
        bad |= ~np.isfinite(H)
    if not bad.any():
        return parsed

    frac = parsed["frac"]
    names = set(frac)
    recovered = 0
    for j in np.where(bad)[0]:
        t = float(T[j])
        comps1, conds1 = make_conditions(comp_wt, [t, t + 1.0], elements)
        try:
            eq1 = equilibrium(
                db, comps1, list(phases), conds1,
                output="HM" if output_hm else None,
                calc_opts={"pdens": int(pdens_retry)},
            )
        except Exception:  # noqa: BLE001
            continue
        p1 = _parse_equilibrium(eq1, output_hm, molar_mass)
        if len(p1["T"]) == 0 or not np.isfinite(np.asarray(p1["G"], dtype=float)[0]):
            continue
        G[j] = float(np.asarray(p1["G"], dtype=float)[0])
        if output_hm:
            H[j] = float(np.asarray(p1["H"], dtype=float)[0])
        p1_names = set(p1["frac"])
        names |= p1_names
        for name in p1_names:
            if name not in frac:
                frac[name] = np.zeros(len(T))
            frac[name][j] = float(np.asarray(p1["frac"][name], dtype=float)[0])
        for name in names - p1_names:
            if name in frac:
                frac[name][j] = 0.0
        recovered += 1

    parsed["G"] = G
    parsed["H"] = H
    if output_hm and np.isfinite(H).all():
        parsed["Cp"] = np.gradient(H, T)
    parsed["valid"] = bool(np.isfinite(G).all() and (not output_hm or np.isfinite(H).all()))
    parsed["recovered_points"] = recovered
    return parsed


def repair_ordered_points(
    db: Database,
    comp_wt: Dict[str, float],
    elements: Sequence[str],
    phases: Sequence[str],
    T: np.ndarray,
    frac: Dict[str, np.ndarray],
    H: np.ndarray,
    G: Optional[np.ndarray] = None,
    molar_mass: Optional[float] = None,
    pdens_retry: int = 4000,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray], int]:
    """Re-solve temperatures where the minimiser returned gamma-prime alone.

    At a small number of compositions---high Co/Cr/Mo with low Al+Ti---the
    grid minimiser returns a metastable single-phase GAMMA_PRIME state
    (``f_gamma' > 0.85`` and ``f_fcc < 0.15``) even at pdens 400, whereas a
    denser grid recovers the stable gamma + gamma' mixture.  Each such
    temperature is re-solved on a denser grid.  The denser solution is kept
    when it is more stable (lower Gibbs energy); when the Gibbs energy is not
    available, it is kept when it restores an FCC matrix fraction.
    """
    T = np.asarray(T, dtype=float)
    n_t = len(T)
    gp = np.asarray(frac.get("GAMMA_PRIME", np.zeros(n_t)), dtype=float)
    fcc = np.asarray(frac.get("FCC_A1", np.zeros(n_t)), dtype=float)
    flagged = (gp > 0.85) & (fcc < 0.15)
    if not flagged.any():
        return frac, np.asarray(H, dtype=float), G, 0

    if molar_mass is None:
        molar_mass = mean_molar_mass(comp_wt)
    H = np.asarray(H, dtype=float).copy()
    G = None if G is None else np.asarray(G, dtype=float).copy()
    names = set(frac)
    repaired = 0

    for j in np.where(flagged)[0]:
        t = float(T[j])
        comps1, conds1 = make_conditions(comp_wt, [t, t + 1.0], elements)
        try:
            eq1 = equilibrium(
                db, comps1, list(phases), conds1,
                output="HM", calc_opts={"pdens": int(pdens_retry)},
            )
        except Exception:  # noqa: BLE001
            continue
        p1 = _parse_equilibrium(eq1, True, molar_mass)
        if len(p1["T"]) == 0:
            continue
        g1 = float(np.asarray(p1["G"], dtype=float)[0])
        if not np.isfinite(g1):
            continue
        f1 = p1["frac"]
        fcc1 = float(np.asarray(f1.get("FCC_A1", np.zeros(1)), dtype=float)[0])
        if G is not None and np.isfinite(G[j]):
            accept = g1 < G[j] - 0.5
        else:
            accept = fcc1 > fcc[j] + 0.02
        if not accept:
            continue
        if G is not None:
            G[j] = g1
        H[j] = float(np.asarray(p1["H"], dtype=float)[0])
        p1_names = set(f1)
        names |= p1_names
        for name in p1_names:
            if name not in frac:
                frac[name] = np.zeros(n_t)
            frac[name][j] = float(np.asarray(f1[name], dtype=float)[0])
        for name in names - p1_names:
            if name in frac:
                frac[name][j] = 0.0
        repaired += 1

    return frac, H, G, repaired


def repair_isolated_liquid_points(
    db: Database,
    comp_wt: Dict[str, float],
    elements: Sequence[str],
    phases: Sequence[str],
    T: np.ndarray,
    frac: Dict[str, np.ndarray],
    H: np.ndarray,
    G: Optional[np.ndarray] = None,
    molar_mass: Optional[float] = None,
    pdens_retry: int = 4000,
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Optional[np.ndarray], int]:
    """Re-solve isolated single-phase LIQUID points in a melting curve.

    The grid minimiser occasionally drops every solid starting point at one
    temperature and returns pure liquid, with solid on both sides and the
    Gibbs energy well above the solid solution (an IN713C matrix at 1478 K
    was returned as liquid with G 1.4 kJ/mol above the FCC_A1 solution).  The
    pattern is unambiguous: a liquid fraction above 0.9 flanked by values
    below 1e-3.  Each such temperature is re-solved at a denser point density
    and the replacement is kept only when it lowers the Gibbs energy by more
    than 0.5 J/mol and no longer returns a single-phase liquid.
    """
    T = np.asarray(T, dtype=float)
    n_t = len(T)
    liq = np.asarray(frac.get("LIQUID", np.zeros(n_t)), dtype=float)
    if n_t < 3:
        return frac, np.asarray(H, dtype=float), G, 0
    flagged = (liq[1:-1] > 0.9) & (liq[:-2] < 1.0e-3) & (liq[2:] < 1.0e-3)
    if not flagged.any():
        return frac, np.asarray(H, dtype=float), G, 0
    flagged = np.concatenate(([False], flagged, [False]))

    if molar_mass is None:
        molar_mass = mean_molar_mass(comp_wt)
    H = np.asarray(H, dtype=float).copy()
    G = None if G is None else np.asarray(G, dtype=float).copy()
    names = set(frac)
    repaired = 0

    for j in np.where(flagged)[0]:
        t = float(T[j])
        comps1, conds1 = make_conditions(comp_wt, [t, t + 1.0], elements)
        try:
            eq1 = equilibrium(
                db, comps1, list(phases), conds1,
                output="HM", calc_opts={"pdens": int(pdens_retry)},
            )
        except Exception:  # noqa: BLE001
            continue
        p1 = _parse_equilibrium(eq1, True, molar_mass)
        if len(p1["T"]) == 0:
            continue
        g1 = float(np.asarray(p1["G"], dtype=float)[0])
        if not np.isfinite(g1):
            continue
        f1 = p1["frac"]
        liq1 = float(np.asarray(f1.get("LIQUID", np.zeros(1)), dtype=float)[0])
        if liq1 > 0.9:
            continue
        if G is not None and np.isfinite(G[j]):
            if g1 >= G[j] - 0.5:
                continue
        elif liq1 >= liq[j]:
            continue
        if G is not None:
            G[j] = g1
        H[j] = float(np.asarray(p1["H"], dtype=float)[0])
        p1_names = set(f1)
        names |= p1_names
        for name in p1_names:
            if name not in frac:
                frac[name] = np.zeros(n_t)
            frac[name][j] = float(np.asarray(f1[name], dtype=float)[0])
        for name in names - p1_names:
            if name in frac:
                frac[name][j] = 0.0
        repaired += 1

    return frac, H, G, repaired


def _last_index_above(arr: np.ndarray, threshold: float) -> Optional[int]:
    idx = np.where(np.asarray(arr) > threshold)[0]
    return int(idx[-1]) if len(idx) else None


def _first_index_above(arr: np.ndarray, threshold: float) -> Optional[int]:
    idx = np.where(np.asarray(arr) > threshold)[0]
    return int(idx[0]) if len(idx) else None


def _crossing_temperature(
    T: np.ndarray,
    values: np.ndarray,
    threshold: float,
    direction: str,
) -> Tuple[float, float]:
    """Linearly interpolated threshold crossing of a phase fraction.

    Returns ``(temperature, censored)``.  ``direction='up'`` finds the first
    upward crossing (e.g. liquid on heating); ``'down'`` finds the last
    downward crossing (e.g. gamma-prime dissolution).
    """
    values = np.asarray(values, dtype=float)
    above = values > threshold
    if direction == "up":
        idx = np.where(above)[0]
        if len(idx) == 0:
            return np.nan, 1.0
        i = int(idx[0])
        if i == 0:
            return float(T[0]), 1.0
        a0, a1 = float(values[i - 1]), float(values[i])
        frac = (threshold - a0) / (a1 - a0) if a1 != a0 else 1.0
        return float(T[i - 1] + frac * (T[i] - T[i - 1])), 0.0

    idx = np.where(above)[0]
    if len(idx) == 0:
        return np.nan, 1.0
    i = int(idx[-1])
    if i == len(T) - 1:
        return float(T[-1]), 1.0
    a0, a1 = float(values[i]), float(values[i + 1])
    frac = (a0 - threshold) / (a0 - a1) if a0 != a1 else 0.0
    return float(T[i] + frac * (T[i + 1] - T[i])), 0.0


def refinement_grid(centre: float, T_bounds: np.ndarray, half_width: float = 8.0,
                    step: float = 2.0) -> np.ndarray:
    """Local temperature grid around a coarse transition temperature."""
    lo = max(float(T_bounds[0]), float(centre) - half_width)
    hi = min(float(T_bounds[-1]), float(centre) + half_width)
    grid = np.arange(lo, hi + 0.5 * step, step)
    if len(grid) < 3:
        grid = np.linspace(lo, hi, 3)
    return grid


def solvus_from_fractions(T, gp, fcc, threshold: float = 1.0e-3):
    """Gamma-prime solvus using FCC_A1 as the matrix indicator.

    The grid minimiser occasionally returns a metastable branch in which the
    ordered phase alone is selected (gamma-prime fraction near one, FCC_A1
    fraction zero) at temperatures where the disordered matrix must be stable.
    A naive last-crossing rule then places the solvus at the highest such
    glitch, which can be hundreds of kelvin too high.  The solvus is therefore
    defined as the last temperature at which gamma-prime is present *and*
    FCC_A1 is the majority phase, i.e. the precipitate is dispersed in the
    matrix.

    Returns ``(temperature, censored)``; ``censored=1`` means the solvus lies
    at or above the top of the calculated window.
    """
    T = np.asarray(T, dtype=float)
    gp = np.asarray(gp, dtype=float)
    fcc = np.asarray(fcc, dtype=float)
    if len(T) == 0:
        return np.nan, 1.0
    matrix = fcc >= 0.5
    above = (gp > threshold) & matrix
    if not above.any():
        return np.nan, 1.0
    i = int(np.where(above)[0][-1])
    step = float(np.median(np.diff(T))) if len(T) > 1 else 0.0
    if i == len(T) - 1:
        return float(T[-1]), 1.0
    if matrix[i + 1]:
        a0, a1 = float(gp[i]), float(gp[i + 1])
        frac = (a0 - threshold) / (a0 - a1) if a0 != a1 else 0.0
        t = float(T[i] + frac * (T[i + 1] - T[i]))
        if t >= T[-1] - 0.5 * step:
            return float(T[-1]), 1.0
        return t, 0.0
    # The matrix disappears between i and i+1: the two-phase region ends in
    # this interval, so the last matrix-bearing point is the best estimate.
    return float(T[i]), 0.0


def extract_signatures(curve: Dict[str, object], threshold: float = 1.0e-3) -> Dict[str, float]:
    """Extract DSC-relevant thermal signatures from an equilibrium curve.

    Temperatures are in kelvin.  Transitions are located by linear
    interpolation of the phase fraction across the detection threshold, which
    removes most of the quantisation imposed by the temperature grid.
    ``censored_*`` flags indicate that a transition lies outside the
    calculated temperature window.
    """
    T = np.asarray(curve.get("T", []), dtype=float)
    if "H" not in curve or len(T) == 0:
        return {"valid": 0.0, "error": curve.get("error", "no curve")}
    frac: Dict[str, np.ndarray] = curve.get("frac", {})  # type: ignore[assignment]
    liquid = frac.get("LIQUID", np.zeros_like(T))
    solid = 1.0 - liquid
    gp = frac.get("GAMMA_PRIME", np.zeros_like(T))
    H = np.asarray(curve["H"], dtype=float)
    Cp = np.asarray(curve["Cp"], dtype=float)
    m = float(curve.get("molar_mass", 1.0))

    out: Dict[str, float] = {}
    out["T_min"] = float(T[0])
    out["T_max"] = float(T[-1])

    # Gamma-prime solvus: last temperature at which gamma-prime is present in
    # an FCC_A1 matrix.  The matrix criterion is essential: the minimiser
    # sometimes selects a metastable ordered single phase at high temperature
    # (see ``solvus_from_fractions``).
    fcc = frac.get("FCC_A1", np.ones_like(T))
    t_gp, cens_gp = solvus_from_fractions(T, gp, fcc, threshold)
    out["T_gp_solvus"] = t_gp
    out["censored_gp_high"] = cens_gp

    # Solidus: first appearance of liquid on heating.
    t_sol, cens_sol = _crossing_temperature(T, liquid, threshold, "up")
    out["T_solidus"] = t_sol
    out["censored_solidus_low"] = cens_sol

    # Liquidus: last temperature with any solid remaining.
    t_liq, cens_liq = _crossing_temperature(T, solid, threshold, "down")
    out["T_liquidus"] = t_liq
    out["censored_liquidus_high"] = cens_liq

    # A transition that lies within half a grid step of the window boundary is
    # effectively censored: linear interpolation can place it just inside the
    # last interval even when the true transition is outside the calculated
    # range.  Such a value is a grid artefact, not a measurement, and for the
    # liquidus it corrupts the fusion-enthalpy baseline.
    if len(T) > 1:
        edge = 0.5 * float(np.median(np.diff(T)))
        if np.isfinite(t_gp) and t_gp >= T[-1] - edge:
            out["censored_gp_high"] = 1.0
        if np.isfinite(t_liq) and t_liq >= T[-1] - edge:
            out["censored_liquidus_high"] = 1.0
        if np.isfinite(t_sol) and t_sol <= T[0] + edge:
            out["censored_solidus_low"] = 1.0
        cens_gp = out["censored_gp_high"]
        cens_sol = out["censored_solidus_low"]
        cens_liq = out["censored_liquidus_high"]

    # Gamma-prime fraction at 750 C and 900 C.
    for label, temp in (("750C", 1023.15), ("900C", 1173.15)):
        out[f"gp_frac_{label}"] = float(np.interp(temp, T, gp)) if T[0] <= temp <= T[-1] else np.nan
        out[f"fcc_frac_{label}"] = float(np.interp(temp, T, frac.get("FCC_A1", np.zeros_like(T)))) if T[0] <= temp <= T[-1] else np.nan

    # Fusion enthalpy: integral of (Cp - linear baseline) across melting.  The
    # baseline slope is taken from the single-phase regions just outside the
    # melting interval, so latent heat does not contaminate it.
    out["dH_fusion_J_per_mol"] = np.nan
    out["dH_fusion_J_per_g"] = np.nan
    if np.isfinite(t_sol) and np.isfinite(t_liq) and cens_sol == 0.0 and cens_liq == 0.0 and np.isfinite(H).all():
        ts = float(t_sol)
        tl = float(max(t_liq, ts + 1.0))
        anchor = 15.0
        lo = max(float(T[0]), ts - anchor)
        hi = min(float(T[-1]), tl + anchor)
        if hi > lo + 2.0:
            cp_lo = float(np.interp(lo, T, Cp))
            cp_hi = float(np.interp(hi, T, Cp))
            slope = (cp_hi - cp_lo) / (hi - lo)
            base_ts = cp_lo + slope * (ts - lo)
            base_tl = cp_lo + slope * (tl - lo)
            baseline = 0.5 * (base_ts + base_tl) * (tl - ts)
            dH = float(np.interp(tl, T, H) - np.interp(ts, T, H)) - baseline
            out["dH_fusion_J_per_mol"] = dH
            out["dH_fusion_J_per_g"] = dH / m

    # A non-positive fusion enthalpy means the melting interval or its
    # baseline was not bracketed by the calculation window; keep it out of
    # the dataset rather than feeding an artefact to the surrogate.
    if not np.isfinite(out["dH_fusion_J_per_g"]) or out["dH_fusion_J_per_g"] <= 0.0:
        out["dH_fusion_J_per_mol"] = np.nan
        out["dH_fusion_J_per_g"] = np.nan

    out["liquid_frac_max"] = float(np.nanmax(liquid))
    out["valid"] = float(bool(curve.get("valid", False)))
    return out


def sample_compositions(
    n: int,
    ranges: Dict[str, Sequence[float]],
    seed: int,
    ni_min_wt: float = 45.0,
    max_attempts: int = 100000,
    min_al_ti_wt: float = 0.0,
) -> pd.DataFrame:
    """Latin-hypercube sample of a wt% composition space.

    ``min_al_ti_wt`` rejects alloys with too little Al + Ti to form a
    measurable gamma-prime fraction, which keeps the sampled design space
    relevant to precipitation-strengthened superalloys.
    """
    from scipy.stats import qmc

    range_keys = [
        k for k, val in ranges.items()
        if isinstance(val, (list, tuple)) and len(val) == 2
    ]
    sampler = qmc.LatinHypercube(d=len(range_keys), seed=seed)
    raw = sampler.random(n=max_attempts)
    lo = np.array([ranges[k][0] for k in range_keys], dtype=float)
    hi = np.array([ranges[k][1] for k in range_keys], dtype=float)
    vals = lo + raw * (hi - lo)
    rows: List[Dict[str, float]] = []
    names = range_keys
    for row in vals:
        comp = {k: float(val) for k, val in zip(names, row)}
        ni = 100.0 - sum(comp.values())
        if ni < ni_min_wt:
            continue
        if min_al_ti_wt > 0.0 and comp.get("Al", 0.0) + comp.get("Ti", 0.0) < min_al_ti_wt:
            continue
        comp["Ni"] = ni
        rows.append(comp)
        if len(rows) >= n:
            break
    if len(rows) < n:
        raise RuntimeError(f"Could only sample {len(rows)} of {n} compositions with Ni >= {ni_min_wt} wt%")
    return pd.DataFrame(rows)


@dataclass
class CurveResult:
    comp_wt: Dict[str, float]
    signatures: Dict[str, float]
    curve: Dict[str, object]


def compute_alloy(
    db: Database,
    comp_wt: Dict[str, float],
    elements: Sequence[str],
    solid_phases: Sequence[str],
    T_melt: Sequence[float],
    T_solid: Sequence[float],
    pdens_melt: int = 400,
    pdens_solid: int = 300,
    refine: bool = True,
) -> Dict[str, object]:
    """Run the two calculations needed for DSC-relevant signatures.

    The matrix melting calculation deliberately uses only LIQUID + FCC_A1.
    Including carbides and gamma-prime in the same global minimisation makes
    the solver oscillate between one- and two-phase solutions in the melting
    interval; the matrix-only calculation is numerically stable and captures
    the solidus/liquidus and latent heat.  The solid-state calculation uses
    the full phase list for the gamma-prime solvus and phase fractions, and is
    refined on a local 2 K grid around the coarse solvus.
    """
    melt = equilibrium_curve(db, comp_wt, T_melt, elements, ["LIQUID", "FCC_A1"], pdens=pdens_melt)
    solid = equilibrium_curve(db, comp_wt, T_solid, elements, solid_phases, pdens=pdens_solid, output_hm=True)
    melt_sig = extract_signatures(melt)
    solid_sig = extract_signatures(solid)

    # Refine the gamma-prime solvus, which is the sharpest DSC feature and is
    # poorly resolved on a 25 K grid.  The fine grid is centred on the coarse
    # matrix-aware solvus, because the fraction can drop from tens of percent
    # to zero within a few kelvin.
    solid_sig["solvus_refined"] = 0.0
    if refine and solid.get("valid", False):
        T_c = np.asarray(solid["T"], dtype=float)
        gp = solid["frac"].get("GAMMA_PRIME")
        fcc = solid["frac"].get("FCC_A1")
        if gp is not None and fcc is not None:
            t_coarse, cens_coarse = solvus_from_fractions(T_c, gp, fcc)
            if cens_coarse == 0.0 and np.isfinite(t_coarse):
                # Refine over the whole coarse bracket, not a window centred
                # on the interpolated value.  When the gamma-prime fraction at
                # the last coarse node above the threshold is small, linear
                # interpolation places the crossing near the *next* node and a
                # +/-10 K window starting there contains no gamma-prime at all,
                # so the refinement silently fails and keeps the biased coarse
                # value.  The true crossing is always inside [T_i, T_i+1].
                above = (np.asarray(gp) > 1.0e-3) & (np.asarray(fcc) >= 0.5)
                idx = np.where(above)[0]
                if len(idx):
                    i_above = int(idx[-1])
                    lo = float(T_c[i_above])
                    hi = float(T_c[i_above + 1]) if i_above + 1 < len(T_c) else float(T_c[-1])
                    fine_T = np.arange(max(float(T_c[0]), lo - 2.0),
                                       min(float(T_c[-1]), hi + 2.0) + 0.5, 2.0)
                else:
                    fine_T = refinement_grid(t_coarse, T_c, half_width=10.0, step=2.0)
                fine = equilibrium_curve(
                    db, comp_wt, fine_T, elements, solid_phases,
                    pdens=pdens_solid, output_hm=False,
                )
                if fine.get("valid", False):
                    t_fine, cens_fine = solvus_from_fractions(
                        np.asarray(fine["T"], dtype=float),
                        fine["frac"].get("GAMMA_PRIME", np.zeros(len(fine["T"]))),
                        fine["frac"].get("FCC_A1", np.ones(len(fine["T"]))),
                    )
                    if cens_fine == 0.0 and np.isfinite(t_fine):
                        solid_sig["T_gp_solvus"] = t_fine
                        solid_sig["censored_gp_high"] = 0.0
                        solid_sig["solvus_refined"] = 1.0

    # Gamma-prime cannot persist above the solidus.  If the solid-state
    # calculation (which is metastable above melting) places the solvus at or
    # above the melting onset, the solvus is not resolvable and is censored.
    t_gp_final = solid_sig.get("T_gp_solvus", np.nan)
    t_sol_melt = melt_sig.get("T_solidus", np.nan)
    if np.isfinite(t_gp_final) and np.isfinite(t_sol_melt) and t_gp_final >= t_sol_melt - 5.0:
        solid_sig["censored_gp_high"] = 1.0

    record: Dict[str, object] = {"comp_wt": dict(comp_wt)}
    record["melt_valid"] = float(melt_sig.get("valid", 0.0))
    record["solid_valid"] = float(solid_sig.get("valid", 0.0))
    record["melt_error"] = melt.get("error", "")
    record["solid_error"] = solid.get("error", "")

    for key in ("T_solidus", "T_liquidus", "dH_fusion_J_per_mol", "dH_fusion_J_per_g",
                "censored_solidus_low", "censored_liquidus_high"):
        record[key] = float(melt_sig.get(key, np.nan))
    for key in ("T_gp_solvus", "censored_gp_high", "solvus_refined", "gp_frac_750C",
                "gp_frac_900C", "fcc_frac_750C", "fcc_frac_900C"):
        record[key] = float(solid_sig.get(key, np.nan))

    record["melt_T"] = np.asarray(melt.get("T", T_melt), dtype=float)
    record["melt_H"] = np.asarray(melt.get("H", np.full(len(T_melt), np.nan)), dtype=float)
    record["melt_Cp"] = np.asarray(melt.get("Cp", np.full(len(T_melt), np.nan)), dtype=float)
    record["melt_liquid"] = np.asarray(melt.get("frac", {}).get("LIQUID", np.zeros(len(T_melt))), dtype=float)
    record["solid_T"] = np.asarray(solid.get("T", T_solid), dtype=float)
    record["solid_H"] = np.asarray(solid.get("H", np.full(len(T_solid), np.nan)), dtype=float)
    record["solid_Cp"] = np.asarray(solid.get("Cp", np.full(len(T_solid), np.nan)), dtype=float)
    record["solid_gp"] = np.asarray(solid.get("frac", {}).get("GAMMA_PRIME", np.zeros(len(T_solid))), dtype=float)
    record["solid_fcc"] = np.asarray(solid.get("frac", {}).get("FCC_A1", np.zeros(len(T_solid))), dtype=float)
    return record
