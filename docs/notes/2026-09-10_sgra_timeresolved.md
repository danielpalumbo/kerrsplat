# Sgr A* through the time-resolved likelihood: first real runs (2026-09-10)

The first real time-resolved fits: Sgr A*, EHT 2017 April 11, low band, the HOPS file with the
ALMA feed rotation applied and the leakage calibrated
(`hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits`; 21 scans over 4.9 hours, which is
865 GM/c³ at 4.15 × 10⁶ M⊙, `scan_times`), through `chi2_timeresolved` with one slow-light frame
per scan: the closure phases and log closure amplitudes of every scan (`closure_scans`, 97
quantities) for the gain-free stage, the four correlation products through per-scan R and L
gains (`observed_scans`, 1616 values, 394 gains, ALMA the phase reference) for the
self-calibrated stage, a 2.4 ± 0.05 Jy prior on every frame's flux, a 2% noise floor, baselines
below 0.1 Gλ dropped. The sky: eight thermal parcels on a 5 M ring (the 52 μas ring is 10 M
across at 8.15 kpc), spin 0.94, inclination 150°, densities pre-scaled so that the first frame
carries the prior's flux, velocities, fields and pattern rates free. Script
`validation/sgra_fit/sgra_jones.jl`; one joint iteration over 21 frames costs 4.3 s on the
2080 SUPER (600 iterations, 43 minutes).

## What happened

| protocol | stage | χ²/N start → end | sky's closure χ²/N (128 over the frames) | gains |
|---|---|---|---|---|
| self-calibration straight from the ring, 600 iterations | products | 1636 → 222 | 107 | 0.36–1.06 with scatter up to 1.6; L/R ratios 0.7–1.6 |
| closures from the ring, 600 iterations, everything free | closures | 47 → 15.4 | 13.1 | |
| self-calibration from that sky, 600 iterations | products | 8127 → 246 | 107 | 0.47–0.88, ALMA 0.56 |

Per scan the closure χ²/N of the sky runs from 2 to 36 across the track and the products'
from 70 to 440. The fitted pattern rates sit at 0.03–0.07 rad/M in either sign (Keplerian at
5 M is 0.08): the optimizer reaches for variability but the sky it has is wrong at every epoch,
and the self-calibration then spends the gains on the amplitude mismatch, as the M87 runs
warned it would from a poor sky. Neither the pipeline nor the data are at fault in an
identifiable way: `closure_scans` reproduces the whole-observation closure χ² to 1e-10 (gate),
the M87 path with the same machinery reached the noise floor, and Sgr A* on April 11 is a
variable, low-inclination source that static geometric models fit at χ²/N of a few; the
failure is the optimization from a ring with nineteen free rows per parcel and motion, in the
600 iterations a 45-minute budget allows.

## The staged protocol, and what it showed

The closure stage rerun as the M87 fits were staged (`--staged`, 300 iterations each): the
geometry and densities first with the pattern rates at zero, then the plasma rows, then
everything. The trace by stage:

| stage | free rows | closure χ²/N (with the flux prior) at iterations 100, 200, 300 |
|---|---|---|
| 1, static geometry | centres, sizes, orientations, densities | 8.7, 5.6, 5.1 |
| 2, plasma | densities, temperatures, fields, field angles, velocities | 5.14, 5.13, 5.12 |
| 3, everything | all, pattern rates free | 77, 40, 32 |

A static eight-parcel sky fits the closures of the whole track at χ²/N 5.1 (from 47), the
plasma rows add nothing the closures see, and freeing the pattern rates at the common Adam
step wrecks the sky within a hundred iterations: 0.005 rad/M per iteration is a Keplerian
rate after twenty, and the rates ran to ±0.06–0.09 rad/M. That is the failure of the runs
above too, whose rates started free. The self-calibration from the staged sky (products
χ²/N 9179 → 229, gains 0.50–0.90 with ALMA and the SPT at 0.56) repeats the earlier one's
amplitude mismatch: with the model's flux held at 2.4 Jy by the prior, the gains absorb the
difference between a compact ring's amplitudes and the data's, so the day's compact flux on
the baselines kept (> 0.1 Gλ) is the next thing to settle.

## Next

A static closure fit alone (`--static`, no pattern rates) for the whole budget, then
self-calibration from it with the pattern rates still frozen and a weaker flux prior; motion
afterwards with its own, much smaller step (`--eta-omega`, a per-row multiplier of the Adam
step: Adam's step is scale-free, so a slow row needs a smaller step, not a smaller gradient);
a snapshot baseline to separate the source's variability from the model's inability.
STATIC_RESULT
