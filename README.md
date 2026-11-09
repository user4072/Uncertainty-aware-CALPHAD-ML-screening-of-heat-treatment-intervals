# DSC-anchored CALPHAD surrogate pipeline for Ni superalloys

An end-to-end, open-source workflow that turns a CALPHAD database into a
fast, uncertainty-aware surrogate for DSC-relevant thermal signatures of
Ni-based superalloys, validates it against literature DSC, and uses it for
inverse design of the heat-treatment window.

```
Composition / T space
        |
        v
CALPHAD engine (pycalphad + open mc_ni TDB)
  phase fractions vs T, H(T), Cp(T)
        |
        v
Surrogate models (GP / polynomial chaos / neural network; PCA-GP for H(T))
        |
        v
DSC validation (T_solvus, T_solidus, T_liquidus, peak enthalpy dH)
        |
        v
DSC-anchored design of alloys with a wide solidus-solvus window
```

## What is new in this implementation

1. **Two linked CALPHAD calculations per alloy.** A matrix-only melting
   calculation (`LIQUID + FCC_A1`) gives stable solidus, liquidus and fusion
   enthalpy; a separate six-phase solid-state calculation gives the
   gamma-prime solvus and phase fractions. Including all phases in one global
   minimisation makes the solver oscillate near melting and produces
   unphysical liquid fractions.
2. **Numerically robust transitions.** The gamma-prime solvus is refined on a
   2 K local grid, transitions are located by interpolating the phase fraction
   across the detection threshold, and isolated solver failures are recovered
   at a denser grid instead of being silently dropped. The grid minimiser can
   also lock onto a metastable ordered `GAMMA_PRIME` branch (fcc fraction
   collapsing to zero); the production dataset uses `pdens = 400`, a
   matrix-aware solvus (`fcc >= 0.5`), and a repair pass that re-solves the
   flagged temperatures at `pdens = 4000` and keeps the lower-Gibbs-energy
   solution.
3. **Thermodynamically consistent fusion enthalpy.** `dH` is the integral of
   `Cp - Cp_baseline` across melting, with the baseline slope taken from the
   single-phase regions. The naive `H(hi) - H(lo)` difference overestimates
   latent heat.
4. **Three surrogate families on one split.** GP (ARD Matern-5/2 with white
   noise), total-degree Legendre polynomial chaos, and an MLP are compared on
   the same held-out alloys; the GP supplies calibrated uncertainty.
5. **Functional output.** Relative enthalpy `H(T) - H(950 K)` is compressed by
   PCA and each score is regressed with a GP, giving smooth `H(T)` and `Cp(T)`
   predictions rather than only scalar onsets.
6. **DSC discrepancy analysis.** Literature DSC for Rene 77, IN100 and IN713C
   is used to separate surrogate error from database error: the GP adds only
   4.6 K, 3.3 K and 0.6 J/g to CALPHAD, which itself differs from DSC by
   16.6 K, 14.4 K and 50.3 J/g. The mean residual is not distinguishable from
   zero from three anchors, so no global bias correction is applied and the
   leave-one-alloy-out values are reported as a sensitivity check. IN738LC is
   retained as an out-of-domain stress test: 5.1 wt% Ta+W+Nb are absent from
   the open database, and its gamma/gamma-prime eutectic melting is not
   represented by the matrix-only calculation.
7. **Surrogate-based design.** A Sobol design over the composition space is
   ranked by a lower-confidence-bound heat-treatment window
   `W = T_solidus - T_gamma'`, restricted to a kNN trust region of the
   training design and to a physically attainable predicted liquidus, then
   re-checked with CALPHAD. A train-only replication, a carbon-floor screen
   and a 750/900 C sigma screen test the ranking, the composition boundary
   and service-phase stability of the shortlist.
8. **Audits and baselines.** A 20-alloy phase-pruning check, a full 12-phase
   melting comparison for the DSC anchors, a sample-accounting table, and
   linear / quadratic / random-forest baselines test the modelling choices
   instead of assuming them. Bootstrap confidence intervals (4000 resamples),
   seed replicates for GP and MLP, and a literature comparison table bound
   the claims that a single fit cannot support.
9. **Error budget and unanchored quantities.** `dsc_error_budget.py` splits the
   DSC discrepancy into surrogate and CALPHAD contributions and tests whether a
   constant bias is identifiable; `solvus_sensitivity.py` propagates common-mode
   and differential solvus bias into the design gain, with a Monte Carlo
   probability that the gain remains positive; `domain_check.py` separates
   chemical coverage (omitted solute mass) from composition distance (kNN).

## Repository layout

```
configs/base.yaml          composition space, phase list, grids, sampler
databases/                 mc_ni_v2.034.tdb (ODbL, MatCalc)
data/dsc_literature.csv    DSC anchors (Binczyk 2008; Chapman 2004)
src/calphad_engine.py      equilibrium wrapper and signature extraction
src/surrogate.py           GP, PCE, MLP, PCA-GP, metrics
scripts/generate_data.py   parallel CALPHAD dataset generation
scripts/recompute_solid.py re-run the solid-state calculation at a new grid density
scripts/repair_solid.py    repair metastable ordered-branch points and re-censor labels
scripts/train_surrogate.py train/benchmark/save models
scripts/learning_curve.py  data-efficiency study
scripts/metric_uncertainty.py  bootstrap CIs and paired GP-vs-baseline tests
scripts/dsc_holdout.py     literature DSC comparison and bias analysis
scripts/dsc_error_budget.py  surrogate-vs-CALPHAD vs CALPHAD-vs-DSC error budget
scripts/domain_check.py    applicability domain: omitted solutes and kNN distance
scripts/solvus_sensitivity.py  solvus bias sensitivity and Monte Carlo of the gain
scripts/fullphase_melting.py  full 12-phase melting audit for the DSC anchors
scripts/phase_pruning_check.py  six- vs twelve-phase fractions and solvus check
scripts/baseline_models.py linear, quadratic and random-forest baselines
scripts/design_alloys.py   surrogate-driven alloy design + CALPHAD validation
scripts/design_robustness.py  train-only replication of the design screen
scripts/design_sigma_screen.py  sigma/carbide screen at 750 and 900 C
scripts/carbon_corner.py   carbon sensitivity of the best design composition
scripts/make_figures.py    publication figures
results/                   metrics, predictions, models, figures
```

## Environment

Use the Python 3.12 interpreter. The default `python` on this machine is a
3.11 environment whose `PYTHONPATH` points at 3.12 site-packages and cannot
import `symengine`.

```
py -3.12 -m pip install pycalphad==0.11.2 numpy scipy scikit-learn matplotlib pandas pyyaml joblib tqdm
```

The `mc_ni_v2.034` database is distributed under the ODbL. Any modified
database or derived calibration must be shared under the same licence.

## Reproduce

```bash
# 1. CALPHAD dataset (400 train + 80 test alloys)
py -3.12 scripts/generate_data.py --out-prefix dataset

# 2. Repair metastable ordered-branch points and re-censor the labels
py -3.12 scripts/repair_solid.py

# 3. Surrogate training and benchmarks
py -3.12 scripts/train_surrogate.py

# 4. Data-efficiency study
py -3.12 scripts/learning_curve.py

# 5. Literature DSC validation and model audits
py -3.12 scripts/dsc_holdout.py
py -3.12 scripts/dsc_error_budget.py
py -3.12 scripts/domain_check.py
py -3.12 scripts/solvus_sensitivity.py
py -3.12 scripts/fullphase_melting.py
py -3.12 scripts/phase_pruning_check.py
py -3.12 scripts/baseline_models.py
py -3.12 scripts/metric_uncertainty.py

# 6. Surrogate-based design and its audits
py -3.12 scripts/design_alloys.py
py -3.12 scripts/design_alloys.py --min-C 0.03 --out results/design_cfloor
py -3.12 scripts/design_robustness.py
py -3.12 scripts/design_sigma_screen.py
py -3.12 scripts/carbon_corner.py

# 7. Figures
py -3.12 scripts/make_figures.py
```

Data generation is memory-bandwidth-bound: four workers are optimal on a
16-thread desktop; eight workers are slower because the equilibrium solver
saturates memory bandwidth.

## Design decisions and checks

- **Composition space**: Co 0-20, Cr 8-18, Al 3-6.5, Mo 1.5-6.5, Ti 0.5-5.5,
  C 0-0.15 wt%, Ni balance >= 45 wt%, Al+Ti >= 3 wt%. Rene 77, IN100 and
  IN713C all lie inside this space, so the DSC comparison is interpolation,
  not extrapolation.
- **Active phases**: FCC_A1, GAMMA_PRIME, MC, M23C6, SIGMA, BCC_A2. A scan with
  the full 12-phase list found no other stable phase at 750 or 900 C; the
  phase fractions agree to 4e-9 and pruning cuts the solid-state cost by ~4x.
  The gamma-prime solvus is unchanged for four of five check alloys and 18.9 K
  lower for one high-C/high-Mo alloy, so the pruning error is bounded by 19 K
  rather than assumed to be zero (`scripts/phase_pruning_check.py`).
- **`pdens = 400` for the solid-state calculation.** At `pdens = 150` the grid
  minimiser often returns a metastable single-phase gamma-prime state at high
  temperature (161 of 480 alloys in at least one temperature), displacing the
  solvus by up to several hundred kelvin. `pdens = 400` and `pdens = 800`
  agree to numerical precision; the production dataset uses 400. A residual
  55 alloys with isolated ordered-branch points are repaired at
  `pdens = 4000` (`scripts/repair_solid.py`). The melting calculation only
  involves `LIQUID + FCC_A1` and converges at 150.
- **Solvus definition.** The gamma-prime solvus requires the fcc matrix
  fraction to be at least 0.5 before and below 1e-3 after the transition, which
  rejects single-phase gamma-prime branches. A second, subtler ordered branch
  survives this test at 750 C (gamma-prime 0.784 plus sigma 0.19, 345 J/mol
  above the stable gamma + gamma-prime state); the sigma screen therefore
  resolves each service temperature at pdens 400, 1200 and 4000 and keeps the
  lowest-Gibbs state.
- **Grids.** Solid state 950-1550 K (25 K step, 2 K solvus refinement);
  melting 1450-1690 K (10 K step). pycalphad excludes the upper bound of a
  condition, so the config maxima are 1575 K and 1700 K; the extra interval is
  the censoring margin.
- **Per-target labels.** Censoring is applied per target, so each surrogate is
  trained on 301-354 of the 357 usable alloys rather than on a complete-case
  subset.
- The melting calculation omits carbides and eutectic phases. This is
  deliberate and is the main source of discrepancy for Nb-bearing IN713C.

## Results

Results are generated into `results/`:

- `surrogate_metrics.csv` - test R2/RMSE/MAE for GP, PCE and NN per target.
- `surrogate_test_predictions.csv` - held-out predictions with GP std.
- `metric_uncertainty.csv` / `.json` - bootstrap 95% CIs and paired GP tests.
- `seed_audit/` - GP and MLP replicates with different random seeds.
- `curve_metrics.json` - PCA-GP reconstruction error for H(T) and Cp(T).
- `learning_curve.csv` / `learning_curve_summary.csv` - data-efficiency study.
- `dsc_validation.csv` - DSC vs CALPHAD vs surrogate for the four alloys.
- `dsc_error_budget.csv` / `.json` - surrogate-vs-CALPHAD and CALPHAD-vs-DSC
  errors, mean residual with a 95% t-interval, and the LOO sensitivity check.
- `dsc_domain.csv` / `.json` - omitted solute mass, kNN distance and percentile
  for every DSC alloy.
- `solvus_sensitivity_common.csv`, `solvus_sensitivity_differential.csv`,
  `solvus_sensitivity_monte_carlo.csv` / `.json` - common-mode and differential
  solvus bias, and P(gain > 0) under solvus uncertainty.
- `fullphase_melting.csv` - 12-phase melting audit for the DSC anchors.
- `phase_pruning_check.csv` / `phase_pruning_summary.json` - six- vs
  twelve-phase fractions and solvus offsets.
- `baseline_metrics.csv` - linear, quadratic and random-forest baselines on the
  same splits and per-target masks.
- `design_candidates.csv` / `design_validated.csv` - designed alloys and their
  CALPHAD re-validation.
- `design_robustness.csv` - train-only replication of the design screen.
- `design_cfloor/` - design screen repeated with C >= 0.03 wt%.
- `carbon_corner.csv` - solidus of the best composition versus carbon content.
- `design_sigma_screen.csv` - sigma/carbide fractions of the shortlist at
  750 and 900 C, resolved at three grid densities.
- `figures/` - publication figures.
- `RESULTS.md` - one-page summary regenerated by `scripts/make_tables.py`.
- `../manuscript/main.pdf` - compiled manuscript draft.

### Headline numbers

- 480 Latin-hypercube alloys, 357 usable (297 train / 60 test).
- Held-out GP error: solvus 8.67 K (R2 0.982), solidus 4.04 K (0.967),
  liquidus 3.53 K (0.974; PCE 3.43 K), fusion enthalpy 0.164 J/g (0.99993),
  gamma-prime fraction at 900 C 0.0096 (0.9965). The GP is best or tied on
  4 of 5 targets. Paired bootstrap: P(GP better) is 0.98-1.00 on the targets
  where it wins, 0.50 on the solvus tie and 0.12 against the quadratic ridge
  on the liquidus.
- Functional PCA-GP: H(T) RMSE 5.4 J/g (R2 0.9987, shape 0.9992), Cp RMSE
  0.10 J/g/K (shape R2 0.72).
- Three DSC anchors (Rene 77, IN100, IN713C): solidus MAE 16.6 K
  (leave-one-alloy-out bias corrected 25.6 K), liquidus 14.4 K (9.7 K), fusion
  enthalpy 50.3 J/g (77.2 J/g). IN738LC, held out as a stress test, is
  under-predicted in solidus by 88.9 K because 5.1 wt% Ta+W+Nb are absent and
  gamma/gamma-prime eutectic melting is not represented.
- Error budget at the anchors: the surrogate differs from CALPHAD by 4.6 K
  (solidus), 3.3 K (liquidus) and 0.6 J/g (enthalpy), against CALPHAD-vs-DSC
  differences of 16.6 K, 14.4 K and 50.3 J/g. The residual is a database and
  melting-model error. All three 95% t-intervals for the mean residual include
  zero, so no constant bias correction is applied.
- The gamma-prime solvus supplies 91% of the production gain and has no direct
  DSC anchor. A common-mode solvus error cancels from the gain; a differential
  +/-25 K changes it from 79.9 to 129.9 K; Monte Carlo gives P(gain > 0) =
  98.4% at 30 K solvus standard deviation (95.6% at 40 K).
- Full 12-phase melting is not more accurate than the matrix calculation:
  solidus/liquidus MAE 35.1/35.7 K versus 16.6/14.4 K, and the full liquid
  fraction jumps by 0.8-1.0 within one 5 K step.
- Baselines: the GP beats the best of linear/quadratic/forest on four of five
  targets (solvus 8.67 vs 9.55 K, solidus 4.04 vs 5.77 K, dH 0.164 vs
  0.576 J/g, f_gamma' 0.0096 vs 0.0200); the quadratic ridge is slightly
  better on the liquidus (3.26 vs 3.53 K).
- Design: 120 000 Sobol candidates, 56 509 feasible inside the constraints and
  the kNN trust region (radius 1.70 in standardised composition space).
  The best CALPHAD-validated window is 274.5 K against 150.1 K for Rene 77
  (+124.5 K). It comes from a train-only replication and lies 0.006 outside
  the production trust radius; the production shortlist reaches 254.9 K
  (GP 244.9 +/- 9.5 K, +104.9 K). Window decomposition: +29.9 K from a higher
  solidus and -94.6 K from a lower solvus, so 76% of the gain is solvus
  depression; for the production alloy 91% is solvus depression. f_gamma'
  falls from 0.529 to 0.421 at 900 C (A1) or 0.409 (production).
- The 274.5 K alloy is carbon-free and sits on the boundary of the design box.
  At its base composition the matrix solidus is 1602.3 K at C=0 and 1572.1 K
  at 0.02 wt% C. A screen with C >= 0.03 wt% removes it, returns four of the
  same six production compositions, and leaves the best window at 254.9 K.
  The carbon-bearing value is the realistic design result.
- Surrogate screening cost 12 s versus about 10.4 days of direct CALPHAD on
  four workers. The GP under-predicts large windows by 7 K on average and
  17 K at worst (regression to the mean), so CALPHAD values are quoted as the
  result and surrogate values as screening estimates. Ten of the twelve
  shortlisted alloys are sigma-free at 750 C; ranks 7 and 12 form 6.7% and
  4.7% sigma, and all are sigma-free at 900 C.

## Limitations

- Only three literature DSC alloys are available in the supported chemistry
  (Ni-Co-Cr-Al-Mo-Ti-C), so a composition-dependent discrepancy model is not
  identifiable. The mean residual is not distinguishable from zero for any
  target, so no bias correction is applied and the LOO values are a
  sensitivity check. The fourth anchor (IN738LC) is an out-of-domain stress
  test, not a calibration point.
- V and Nb are omitted from the surrogate. IN100 contains 0.82 wt% V and
  IN713C contains 2.27 wt% Nb; their DSC residuals are therefore partly due to
  chemistry outside the model.
- The open database has no Ta, Re, W or Hf, so CMSX-4, MAR247, IN738LC and
  other refractory-rich alloys are out of scope for quantitative predictions.
  Omitted mass, not kNN distance, is the correct domain test: IN738LC sits
  inside the 99th-percentile composition radius.
- Solid-state calculations are equilibrium calculations. Real DSC heating at
  10 K/min is not fully equilibrated, so kinetic effects (gamma-prime
  dissolution, carbide reactions) are not captured.
- The melting calculation treats the matrix only; carbide/eutectic melting in
  Nb- or C-rich alloys is not represented. The full 12-phase audit shows this
  approximation is better conditioned, not that it is uniformly more accurate.
- No designed alloy has been cast or measured, and no γ′ solvus of a designed
  composition has been experimentally determined. The reported window is a
  calculated thermodynamic interval.

## Round-4 revision scripts

- `scripts/bo_multiseed.py` - sequential search over ten random initializations.
- `scripts/carbon_sensitivity.py` - high-resolution carbon sweep (matrix, solvus, full equilibrium).
- `scripts/fullphase_shortlist_full.py` - twelve-phase melting + phase assemblages for every shortlisted alloy.
- `scripts/uncertainty_grid.py` - P(gain>0) over priors, correlations and discrepancy families.
- `scripts/pareto_analysis.py` - Pareto front, floor-price curve, solidus-preservation counts.
- `scripts/censoring_diagnostics.py` - local reliability of the screen under composition-dependent censoring.
- `scripts/resolution_vs_error.py` - numerical resolution vs regression error, side by side.
- `scripts/solute_response.py` - one-solute-at-a-time mechanism probes.
- `scripts/workflow_budget.py` - end-to-end cost accounting and break-even.

## References

- F. Binczyk et al., ATD and DSC analyses of nickel superalloys, Archives of
  Foundry Engineering.
- L. Chapman et al., Application of high temperature DSC technique to nickel
  based superalloys, J. Mater. Sci. 39 (2004) 7229-7236.
- P. N. Quested et al., Measurement and estimation of thermophysical properties
  of nickel based superalloys (2009).
- Z. J. Miao et al., Microstructure and solidification behaviour of phosphorus
  and boron doped IN718, Mater. Sci. Technol. (2012).
- A. K. Rai et al., Thermophysical properties of Ni based super alloy 617
  (2017).
- Y. Tian et al., Determining multi-component phase diagrams with desired
  characteristics using active learning, Adv. Sci. (2020).
- MatCalc `mc_ni_v2.034` thermodynamic database, ODbL.
