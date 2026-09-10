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

## Next

A staged closure protocol as the M87 fits used (geometry and densities with a static sky, then
the plasma rows, then motion; `--staged`), longer schedules, a snapshot-model baseline (the
same parcels fitted scan by scan without motion, to separate the source's variability from the
model's inability), and the flux prior checked against the day's compact flux. STAGED_RESULT
