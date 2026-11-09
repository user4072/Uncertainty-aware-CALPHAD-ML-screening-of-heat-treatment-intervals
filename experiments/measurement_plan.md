# Measurement decision and protocol: one DSC campaign

Decision taken: measure the rank-1 designed alloy of Table 6 (the 264.8 K
candidate), with Ren\'e 77 as a same-session control specimen. This is one
campaign, two specimens, one instrument, one operator. If only one specimen can
be run, run rank 1; if two can be run, the control is worth more than a second
heating rate because the paper's design claim is comparative.

## 1. Why this alloy

Nominal composition (wt%): Co 19.61, Cr 14.67, Al 3.04, Mo 6.08, Ti 2.22,
C 0.022, Ni balance (54.36). Charge for 30 g in Section 4.

1. It is the headline result. The paper claims a 264.8 K interval, 114.3 K
   wider than Ren\'e 77; measuring anything else leaves the headline
   unvalidated.
2. It lies exactly in the six-solute space. No V, B, Zr, Nb, Ta or W, so the
   measurement tests the model rather than a projection.
3. Carbon is 0.022 wt%. Carbides are few, so the matrix-only calculation is a
   fair approximation, yet the alloy stays carbon-bearing and avoids the
   carbon-free artefact of A1. If the melt lands at 0.03--0.05 wt% C, accept it
   and recompute the model at the analysed carbon; do not reject the sample.
4. The predicted window is the widest of the shortlist. The solvus and solidus
   are separated by about 265 K, so the $\gamma'$ dissolution endotherm near
   1030 C and the melting endotherm near 1297 C are resolved in a single trace
   without peak overlap. This is the practical reason to prefer it over ranks
   2--6, whose windows are 10 K narrower.
5. It is $\sigma$-free at both 750 and 900 C in the converged phase check, so
   an optional 900 C ageing for the $\gamma'$ fraction is not confounded by a
   TCP phase.
6. The predicted solidus equals that of Ren\'e 77 in the same calculation
   (1297.1 C for both). The design claim is therefore a *null* on the solidus
   and a 114.3 K shift on the solvus, which is a sharp, falsifiable pairing.

## 2. Predicted values to compare against

Matrix model, production CALPHAD settings (the values the paper reports):

| quantity | matrix CALPHAD | GP | full 12-phase |
|---|---|---|---|
| $\gamma'$ solvus | 1032.3 C (1305.4 K) | 1030.9 C | 1032.3 C |
| solidus | 1297.1 C (1570.2 K) | 1290.6 C | 1308.9 C (1582.0 K) |
| liquidus | 1376.7 C (1649.9 K) | 1373.2 C | 1370.5 C (1643.6 K) |
| $\Delta H_{\rm fus}$ | 242.7 J/g | 242.6 J/g | -- |
| $f_{\gamma'}$(900 C) | 0.413 | 0.41 | -- |
| $W^{\rm m}$ | 264.8 K | 259.7 K | 276.6 K |

Ren\'e 77 in the same calculation: solvus 1146.5 C, solidus 1297.1 C,
$\Delta H$ 247.3 J/g, $f_{\gamma'}$ 0.529, $W^{\rm m}$ 150.5 K. Literature
values at 10 K/min: solidus 1302 C, liquidus 1352 C, $\Delta H$ 237.4 J/g,
secondary-$\gamma'$ onset 1229 C, apparent window 73 K.

The paired prediction is the important one: the rank-1 minus Ren\'e 77
difference in solidus is 0 K, and in computed window 114.3 K. A paired
measurement cancels calibration and laboratory bias, which an absolute
comparison cannot.

## 3. What is measured

One DSC trace at 10 K/min yields four endpoints:

1. Solidus: extrapolated onset of the first endotherm (peak A), tangent
   convention.
2. Liquidus: extrapolated end of the last endotherm (offset convention).
3. $\Delta H_{\rm fus}$: integrated peak-A area above a linear baseline with
   the same 15 K anchors used in the calculation.
4. Apparent $\gamma'$ dissolution onset: the broad endotherm near 1030 C.
   Report it as an apparent, rate-dependent value, not as the equilibrium
   solvus. For Ren\'e 77 the same quantity sits 82.5 K above the calculated
   solvus, so the interpretation is known.

Optional, only if the instrument time exists: repeat at 5 and 20 K/min and
extrapolate the solidus and liquidus linearly to zero heating rate, following
the Chapman round-robin recommendation. This makes the absolute comparison to
equilibrium values defensible. If only one rate is possible, 10 K/min keeps
comparability with the existing two anchors and the Ren\'e 77 literature value.

## 4. Specimen fabrication and chemistry

1. Vacuum arc-melt (or vacuum induction-melt) a 30 g button under Ti-gettered
   Ar; remelt and flip at least five times. Do not use a graphite crucible or
   graphite mould.
2. Charge for 30 g, nominal: Co 5.882 g, Cr 4.400 g, Al 0.912 g, Mo 1.825 g,
   Ti 0.665 g, Ni 16.309 g, C added as Ni--4.5 wt% C master alloy (0.148 g,
   replacing 0.142 g of Ni). Add 2--3% excess Al and Ti for burn-off; the
   analysed composition, not the charge, defines the model evaluation.
3. Analyse by ICP-OES for Co, Cr, Al, Mo, Ti, Ni (three spots) and by LECO for
   C, O, N. Report the mean and the spot-to-spot range. Target Co <= 19.9 so
   the alloy stays inside the 20 wt% bound; if it exceeds it, the test becomes
   a projection and must be reported as such.
4. Homogenise at 1200 C for 24 h in vacuum or flowing Ar (below the predicted
   solidus, above the solvus), then furnace-cool. This removes the
   segregation that would otherwise produce an incipient-melting first liquid
   below the equilibrium solidus. Cut two specimens, top and bottom of the
   button, and run both.
5. DSC specimen mass 150--200 mg, alumina crucible with lid, flowing Ar
   50 mL/min, heating to 1450 C. Calibrate temperature with pure Ni (1455 C),
   Au (1064 C) and Ag (962 C); calibrate enthalpy with sapphire. Subtract an
   empty-crucible baseline at the same rate. Two heating cycles per specimen,
   cooling at 10 K/min between them.
6. Inspect the crucible and specimen after the run for reaction or mass loss;
   report both.

## 5. Pre-registered interpretation

Set these before the run so the result cannot be rationalised afterwards.

1. Solidus within 20 K of 1290--1297 C: consistent with the matrix model and
   with the current four-anchor MAE. A residual larger than 40 K is a failure
   of the design prediction for this composition and is reported as such.
2. Liquidus within 15 K of 1373--1377 C.
3. $\Delta H$ within 12 J/g of 243 J/g.
4. Paired test: measured solidus difference (rank 1 minus Ren\'e 77) within
   15 K of zero, and measured apparent window of rank 1 larger than that of
   Ren\'e 77 by at least 30 K. The absolute measured window will be smaller
   than 264.8 K, as it is for Ren\'e 77 (73 K measured against 150.5 K
   calculated); the defensible claim is the sign and approximate size of the
   difference, not the absolute number.
5. Do not use the 12 K matrix--full solidus difference as a pass/fail test.
   It is at the edge of what one campaign can resolve; report which prediction
   is closer, without claiming discrimination.

## 6. How it enters the manuscript

Add one row to Table 2 with the role "prospective design test" and keep it out
of the four-anchor statistics, exactly as IN713C and IN738LC are kept out.
Report the signed residuals and the paired difference in Section 3.5, and one
sentence in the Conclusions. If the measurement arrives after submission, the
same table row and paragraph become the centrepiece of the response to
reviewers; no other change to the paper is needed.

Expected outcomes and what they mean:

- Solidus within 10 K and paired difference near zero: the central design
  claim is confirmed; the paper gains an experimental anchor on a designed
  alloy, which is its main missing element relative to Pei et al.
- Solidus within 20 K but apparent window much narrower than computed:
  consistent with the kinetic compression seen for Ren\'e 77; the claim becomes
  a ranking claim, which the paper already states.
- Solidus 30--50 K low: inspect for residual segregation, carbide eutectic or
  oxygen; if the microstructure is clean, the database bias for this corner is
  larger than the reference-set average and must be reported.

## 7. Sequencing and lead time

Week 0: approve the composition, send the melt and chemistry request, order a
Ren\'e 77 bar. Week 1--3: melting and chemistry (one week in-house, two to
three weeks outsourced). Week 3--4: homogenisation and DSC. Week 4--5:
analysis, recompute the surrogate and CALPHAD at the analysed composition,
integrate.

Do not hold the submission for this measurement. Submit when the manuscript is
ready and keep the campaign for the revision; a reviewer asking for exactly
this experiment is the most likely outcome, and the campaign answers it with
new data rather than argument.

## 8. Request text for the laboratory

> We require one vacuum-melted button, 30 g, of nominal composition
> Ni--19.6Co--14.7Cr--3.0Al--6.1Mo--2.2Ti--0.022C (wt%), with analysed Co, Cr,
> Al, Mo, Ti and Ni by ICP-OES and C, O, N by LECO. The button should be
> homogenised at 1200 C for 24 h in vacuum or flowing Ar and furnace-cooled.
> We then need DSC on two specimens (150--200 mg, alumina crucible, flowing Ar,
> 10 K/min to 1450 C, two heating cycles, empty-crucible baseline, temperature
> calibration with Ni/Au/Ag and enthalpy calibration with sapphire), with
> extrapolated-onset solidus, offset liquidus and integrated fusion enthalpy.
> Please also run a commercial Ren\'e 77 specimen under the identical protocol
> as a control. Report raw curves as well as the analysed endpoints.
