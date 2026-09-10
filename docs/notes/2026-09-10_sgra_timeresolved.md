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

## The static sky

The static closure fit (`--static`, 900 iterations, 65 minutes on the GPU): closure χ²/N with
the flux prior 47 → 4.83 (the sky's closure χ²/N over the 128 closures of its own frames 4.59),
still falling by 0.05 per 25 iterations at the end. The trace: 17.4 after 25 iterations, 10.3
after 75, 6.0 after 275, 5.04 after 400, a spike to 7.9 at 425 that took 250 iterations to
undo, then 5.3 → 4.83 over the last 400. Per scan the static sky fits most of the track at
χ²/N 0.4–4.7 and misses two scans, the 12th and 13th (UT 11.45 and 11.65 h, t ≈ 430–470 M),
at 12.7 and 13.2; they are also the scans the moving fits missed worst, so that hour holds
either the source's own change or a systematic, and a moving sky started from this one has to
show the difference. The model's flux sits on the prior at 2.401 Jy in every frame (a static
sky has one flux).

Self-calibration from the static sky with the pattern rates still frozen and the flux prior
weakened to 2.4 ± 0.5 Jy (600 iterations): products χ²/N 4971 → 81.5, better than the 229 of
the staged sky but the same failure. The gains absorb what the joint loop cannot fit (|g_R|
0.16 ± 0.06 at the SMA, 0.35 ± 0.09 at the SPT, 0.72–0.85 elsewhere, gain phases at the SPT
random), the model's flux drifts to 2.05 Jy, and the sky's own closure χ²/N goes from 4.6 to
50: the products at unit gains start at χ²/N 4971 although the closures fit, so the amplitude
scale of the static sky is far from the data's on the SMA and SPT baselines, and the joint
Adam loop moves the sky to meet the gains instead of the gains to meet the sky. A gain-only
solve with the sky held (Levenberg–Marquardt over the instrument alone, the protocol every
imaging pipeline uses between sky updates) is the next tool: it tells whether the static sky's
amplitudes can be calibrated at all with sane gains, and it starts the joint fit from
calibrated gains rather than from unit ones.

## The instrument solve, and what the amplitudes said

The instrument-only solve (`calibrate!`: Levenberg–Marquardt over the gains with the static
sky's model visibilities held, the phases started from the reference baselines and solved
first) on the static sky: products χ²/N 4971 → 211 in 8 seconds, with the gains again
collapsing (ALMA 0.46, the SMA 0.25) and the parallel hands worse than the cross-hands (RR
202, LL 257, RL 119, LR 170). The per-baseline table of the sky's Stokes I amplitude against
the data's, without gains, explained it: the model is too bright on every baseline, by 1.2 on
the shortest (AZ–LM, 1.1 Gλ) and by 2.6–5 beyond 3 Gλ (AZ–SM 2.6, LM–SP 2.9, AZ–SP 5.0 at
8.5 Gλ), the ratio growing with baseline length. That is the signature of Sgr A*'s
interstellar scattering, which the model lacked: the diffractive kernel (Johnson et al. 2018;
23 × 12 μas FWHM at 230 GHz, position angle 82°) multiplies the visibilities by 0.6–0.9 at
3 Gλ and by 0.04–0.4 at 8.5 Gλ depending on the orientation. The closure stage did not see
it because its SNR cut keeps the bright, short baselines, where the kernel is near one, and a
static ring fits those at χ²/N 5 with or without it. The model visibilities of every scan now
go through `ScatteringKernel` (`taper`, gated against ehtim's `sgra_kernel_uv`), on by default
in the script (`--no-scatter` to drop it).

With the kernel on the same static sky (fitted without it), the amplitude ratios fall to
1.02–1.19 on the ALMA/APEX–SMT, ALMA/APEX–LMT and LMT–SMA baselines and to 0.4–0.6 on the
ALMA/APEX–SPT and ALMA/APEX–SMA ones, with 1.2–2 left on the LMT/SMA/SMT–SPT baselines: the
scale is right and the residual pattern is the sky's, fitted without the kernel's anisotropy
(the sky's own closure χ²/N with the kernel is 10.6, from 4.6 without). The instrument solve
on it still collapses (χ²/N 205). The static closure fit is being repeated with the kernel,
followed by the instrument solve and a self-calibration that starts from the solved gains.

Motion from the static sky (before the kernel) with the pattern rates moving at a twentieth
of the common step (`--eta-omega 0.05`, 600 iterations): closure χ²/N 4.83 → 4.45 by
iteration 300, then a spike to 21 at iteration 400 that the rest of the schedule brought back
only to 11.2; the rates stayed within ±0.003 rad/M, a thirtieth of Keplerian at 5 M, so the
small step made the motion harmless and also inert. The spike is not the rates': the static
run had the same one at iteration 425 (5.0 → 7.9). A parcel row that Adam kicks past a
threshold (a size or a density) is the likely cause and is worth a per-iteration trace of the
parameters when the kernel fits are in.

## With the kernel

The static closure fit with the kernel (900 iterations): closure χ²/N with the flux prior
40.7 → 4.66, the sky's own closures 4.82, and this time a clean trace (8.6 after 100
iterations, 5.2 after 300, 4.75 after 500, 4.66 at the end) with no spike. Per scan the fit is
0.2–3.4 everywhere except the same two scans as before, the 12th and 13th (UT 11.45 and
11.65 h) at 21.6 and 35.2, which now hold almost the whole residual: 11 closures each, so
about 620 of the 617 closure χ² of the sky sit in that hour and the other nineteen scans fit
a static ring at the noise. Whatever happens at UT 11.5 h on April 11 (a change of the source,
or of a station) is where the time-resolved model has to earn its keep.

The instrument solve on this sky still collapsed (χ²/N 219, ALMA at 0.19), and the
per-baseline table now showed the ALMA–SPT and APEX–SPT amplitudes seven times too faint
(model 0.017 Jy against 0.178) with the SMA baselines 2–4 times too bright, on a sky whose
closures fit at 4.8. That is impossible if the closure amplitudes hold those baselines, and
they did not: `scan_quadrangles` built one quadrangle per four-station set, (ij, kl, ik, jl)
with the stations sorted, which never contains the il and jk baselines, so the baseline
between a scan's first and last station in the station order, ALMA–SPT here, entered no log
closure amplitude of any scan, and APEX–SPT entered none either whenever ALMA was absent from
the quadruple. The builder now gives two of the three quadrangles per quadruple, (ij, kl, ik,
jl) and (ik, jl, il, jk), which are independent and between them use all six baselines
(`test_uvfits` checks both). The M87 closure fits of 2026-09-04 ran with the incomplete set;
their conclusions rested on the self-calibrated products, not on the closures alone, but their
closure χ²/N will read differently with the full set. The static fit, the instrument solve
and the self-calibration are being repeated with it.

## With every baseline in the closures

The static closure fit with the kernel and the full closure set (143 closures instead of
128, 900 iterations): χ²/N 37.3 → 4.59, the sky's own closures 4.59, converged (5.15 after
300 iterations, 4.67 after 500, 4.59 from 700 on). The two scans at UT 11.5 h again hold the
residual (15.6 and 40.8); the other nineteen sit at 0.03–4.0.

The dual-feed instrument solve on it still ends at χ²/N 181 with the SMA gain at 0.25, and
the per-baseline table still has the SPT baselines at 0.3–0.4 of the data and the SMA ones at
2–4 times it. Two things are now clear from the split by product and by station. First, this
amplitude pattern is station-like (every SPT baseline low by about the same factor, every SMA
baseline high), and closure amplitudes are blind to exactly that: with the full closure set
the sky can still trade a station-like amplitude pattern against the true one, and it has.
Second, the model's L/R ratios come out at 0.29 (SPT) to 0.71 (ALMA) where the prior allows
±10%, and LL fits worse than RR by a factor 1.8: the closure stage constrains Stokes I alone,
so the sky's Q, U and V are whatever the initial parcels had, and the dual-feed products then
misfit the parallel hands through V and the cross-hands through Q and U with nothing the
gains can do about it. The next step is the standard one: solve the instrument on Stokes I
alone (`--stokes-i`, one complex gain per station on the parallel-hand average, the
polarization left out), then self-calibrate Stokes I, and only then let the polarization rows
meet the products.

The Stokes I instrument solve on the same sky: χ²/N 1268 → 14.1 (303 after the phase start,
the phase-only solve stalling at 259, the full solve reaching 14 in six iterations), and per
scan 0.02–2.3 for seventeen of the twenty-one scans, with two scans (the 3rd and 10th) at 13
and 14 and the last at 4.2. So the sky fits the Stokes I products with gains almost at its
closure level, and the polarization was what the dual-feed products could not swallow. The
gains say what the closures could not: the SPT's amplitude gain is 2.2 on average with a
scatter of 3 and the SMA's 0.39, the absorption of a station-like amplitude error of the sky
(its SPT baselines faint, its SMA baselines bright) that closure amplitudes cannot see and
only the gains' priors can push back on. That is the job of the joint Stokes I
self-calibration from these gains.

The joint Stokes I self-calibration (static sky, the gains started from the solve, flux prior
2.4 ± 0.5 Jy, 600 iterations): products χ²/N 14.1 → 6.72, still falling by 0.1 per 50
iterations at the end (11.9 after 50, 8.0 after 300, 6.8 after 550). Per scan the products fit
at 0.01–3.5 except the 12th (UT 11.45 h) at 5.5. The gains moved toward their priors as the
sky took over the station-like pattern: ALMA 1.01 ± 0.40, APEX 0.94, SMT 0.81, LMT 1.06, SPT
1.42 ± 0.68 (from 2.2), SMA 0.49 ± 0.25 (from 0.39). The sky's own closure χ²/N rose from 4.6
to 9.7, almost all of it in the 12th and 14th scans (45 and 17) again, as the products, not the
closures, are what the joint fit minimizes.

## Where this stands

The chain that works for Sgr A* on this day is the one every imaging pipeline uses, and every
piece of it had to be put in today: the scattering kernel on the model visibilities, closure
amplitudes over every baseline, a Stokes I closure fit, an instrument solve on Stokes I with
the phases started from the reference baselines, and a joint Stokes I self-calibration from
the solved gains. From a ring of eight parcels it reaches Stokes I products at χ²/N 6.7 with
gains near their priors, where the morning's runs sat at 230 and 80 with the gains collapsed.

Open, in the order to take them:

1. The SMA's amplitude gain of 0.49 is not physical (its a priori calibration in 2017 was
   good to 10–20%); the model is still twice too bright on the SMA baselines. Whether this is
   the sky (an east–west structure the parcels cannot make) or the data (the SMA's a priori
   amplitudes on this day) is checkable against the published gain solutions.
2. The hour around UT 11.5 h holds the closure residual in every fit; a moving sky (the pattern
   rates at their own step) from the Stokes I self-calibrated sky, and a look at the
   per-station data of those scans.
3. The polarization: with the Stokes I sky and gains held, the parcels' field rows against the
   dual-feed products (the L/R ratios and d-terms of the instrument model), then everything
   jointly. The closure stage constrains Stokes I alone; the dual-feed products from a
   closure-only sky are unusable, as the morning showed.
4. A longer Stokes I self-calibration (the trace has not flattened) and the joint LM polish
   for the Laplace errors.
