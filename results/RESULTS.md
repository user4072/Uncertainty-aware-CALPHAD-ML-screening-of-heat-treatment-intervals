# Pipeline results

- Alloys generated: 480; usable: 357 (74.4%).
- Train/test (usable): 297/60.
- GP test R2: worst 0.9657, mean 0.9880 (per-target training set 301-354 alloys).
- GP test RMSE: solvus 2.29 K, solidus 4.10 K, liquidus 3.31 K, dH 0.165 J/g, gamma' fraction 0.0070.
- PCA-GP H(T): RMSE 5.42 J/g, R2 0.99870, shape R2 0.9992; Cp RMSE 0.103 J/g/K.
- Reference-anchor MAE: solidus 20.3 K (LOO 27.1), liquidus 14.7 K (LOO 8.6), dH 11.7 J/g (LOO undefined (n=2)).
- Error budget: GP-vs-CALPHAD 3.1/2.7/0.5 (solidus/liquidus/dH) against CALPHAD-vs-reported 20.3/14.7/11.7; the GP-CALPHAD discrepancy is 15%, 19% and 4% of the CALPHAD-reported discrepancy in magnitude (a ratio of pairwise errors, not an additive share).
- Bias identifiability: the liquidus 95% interval [+1.9, +27.5] K excludes zero, so the restricted-phase model melts systematically high; the solidus [-32.5, +45.7] K and enthalpy [-12.0, +35.5] J/g intervals include zero. No correction is applied because the liquidus residual still ranges over 4.6-21.7 K between anchors.
- Applicability domain: omitted solute mass is 0.82 wt% (IN100), 2.3 wt% (IN713C) and 5.1 wt% (IN738LC); the two Nimonic anchors carry 0.06 wt% of trace B and Zr. IN738LC lies at the 36th percentile of the six-solute distance, so a distance-only check would miss the failure.
- Gamma-prime solvus: this work 1146 C, TCNI10 equilibrium 1139 C (gap 8 K), DSC secondary solvus 1229 C (gap 82 K). A common 80 K solvus bias leaves the gain at 114.3 K (absolute window 184.8 K); a +/-25 K differential bias gives 89.3-139.3 K.
- Solvus Monte Carlo: with a 30 K independent solvus sigma, P(gain > 0) = 98.1% and the 5th percentile is 23.9 K; at 40 K the values are 95.7% and 4.5 K.
- Best train-only CALPHAD window: 275.0 K (GP 268.6 +/- 4.5 K; Rene 77 baseline 150.5 K, gain +124.4 K); carbon-free boundary alloy.
- Production (carbon-bearing) CALPHAD window: 264.8 K (Rene 77 baseline 150.5 K, gain +114.3 K; 100% from solvus depression).
- Carbon floor C >= 0.03 wt%: best window 255.8 K; 2 of 6 candidates match the production shortlist.
- Full 12-phase melting: solidus MAE 20.0 K, liquidus 14.0 K; matrix calculation 20.3/14.7 K.
- Best non-GP baseline RMSE: solvus 3.43 (cubic); liquidus 3.3 (cubic); solidus 5.65 (cubic); dH 0.337 (cubic); f_gamma' 0.0137 (svr). The GP is lower on every target; the liquidus is a 0.01 K tie with the cubic polynomial. See Table S8.
- Design screen: 120000 Sobol candidates, 56471 feasible inside the constraints and the kNN trust region; surrogate screen 7.7 s versus 4.8 days of direct CALPHAD on four workers.
- Labels are masked per target (censoring is not propagated across targets), so each surrogate is trained on 301-354 of the 357 usable alloys; the learning curves use a common complete-case subset.
