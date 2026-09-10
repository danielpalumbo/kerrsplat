# Self-fits to synthetic VLBI data of the slow-light movie (2026-09-09)

The plan's gate 7 in the data domain, and the head-to-head with PI-DEF's reconstructions from
simulated EHT data: the four-parcel half-orbit truth (`docs/notes/2026-09-08_half_orbit_selffit.md`,
n ≤ 1, M87's mass and distance, 230 GHz) observed with the EHT 2017 array and fitted from its
own synthetic closure quantities or visibilities through the time-resolved likelihood
(`Fit.chi2_timeresolved`, `timeresolved_gradient!`; script `validation/winding/winding_vlbi.jl`).

## Setup

- **Coverage.** The scans of the public M87 2017 day-101 low-band uvfits file, scan-averaged
  (22 scans, 216 baselines among the seven stations), with their frame times spread over the
  first 25 M of the movie in 5 groups (or one time per scan, 22 frames): a real array's (u, v)
  sampling lent to a synthetic source whose time axis is chosen freely (`Fit.coverage`).
- **Data.** The frame at each scan's time rendered on the uniform 48² screen over 16 M
  (0.33 M pixels), its model visibilities on the scan's baselines with complex Gaussian noise
  of 1% of the frame's total flux density per real and imaginary part (`Fit.synthetic_scans`),
  and either the closure phases of all triangles and the log closure amplitudes of all
  quadrangles of the scan (uncertainties propagated as the real-data path does, legs below 3σ
  dropped; 445 quantities in all) or the calibrated Stokes visibilities themselves (1728 real
  values).
- **Fit.** All 76 parcel parameters (the temporal envelope frozen) from the same perturbed
  start as the movie fits (positions off by 0.4–0.75 M, densities, temperatures and fields by
  10–20%), 1000 Adam iterations on the GPU (one dual sweep per frame per iteration; 18 minutes),
  then six Levenberg–Marquardt iterations on the time-resolved residuals
  (`timeresolved_residuals`, `polish!`; five minutes) with the Laplace covariance at the end.

## Results

| data | χ²/N truth | Adam → polished χ²/N | position errors (M) | nₑ, Θe, B errors | Laplace σ(x), σ(y) (M) | Laplace σ(ln nₑ), σ(ln Θe), σ(ln B) |
|---|---|---|---|---|---|---|
| closures, 5 frames (445) | 1.035 | 1.017 → 0.971 | 0.12–0.39 | 11–25%, 5–25%, 4–27% | 1.7–2.8, 1.1–2.3 | 3.7–9.4, 2.1–3.6, 4.0–10 |
| visibilities, 5 frames (1728) | 1.039 | 1.072 → 1.020 | 0.013–0.19 | 7–18%, 7–14%, 5–19% | 0.036–0.072, 0.023–0.091 | 2.6–3.8, 0.48–0.79, 0.71–1.2 |
| closures, 22 frames (445) | 0.957 | 0.880 → 0.842 | 0.07–0.45 | 10–22%, 0.8–22%, 4–31% | 0.50–1.5, 0.45–0.97 | 6.3–15, 16–22, 8.0–19 |

(The pattern rates come back to 0.7–1.5% from closures and 0.2–0.6% from visibilities in every
case: the motion is what the time resolution measures.)

## Reading

- **Closure quantities alone, at the 2017 coverage, do not constrain this model.** The polished
  fit sits below the noise floor with the parameters 0.1–0.4 M and 10–25% off, and the Laplace
  errors say why: 1–3 M in position, factors of e² to e¹⁰ in density, temperature and field.
  The likelihood is flat over ranges wider than the parcels' separations; the fit's modest
  errors are inherited from the start. Four Gaussian parcels with 76 parameters against 445
  closure quantities from seven stations is an under-determined problem, before any question
  of degeneracy between the parcels' emission properties.
- **Calibrated visibilities recover the geometry and the motion, not the plasma.** Absolute
  phases and amplitudes pin the positions to a few hundredths of a gravitational radius
  (fit errors within one to three Laplace σ) and the pattern rates to a few tenths of a per
  cent, but nₑ, Θe and B stay unconstrained by factors of two to forty: one band and 216
  baselines cannot separate the emissivity's three ingredients, the same degeneracy the
  single-band movie fits show, now with far fewer data.
- **Time resolution helps the geometry, not the plasma.** With every scan at its own frame
  time (22 frames, the same 445 closure quantities) the Laplace errors in position halve
  (0.5–1.5 M) and the pattern rates come back to 0.1–0.3% for three parcels, but the emission
  rows become, if anything, flatter (σ(ln) of 6 to 22: directions the closures do not see at
  all). The fit costs 78 minutes against 18 for five frames (one dual sweep per frame per
  iteration).
- Against the image-domain movie fits of the same truth (99,072 pixel values at one band,
  χ²/N 1.010 and every parcel to 0.002–0.02 M), the data-domain fit at the 2017 coverage is
  a different regime: the array, not the model or the optimizer, sets the limit. What the
  method needs for real data follows directly: the multiband bracketing of the turnover
  (which pinned Θe and B to 0.3% in the image domain), gains solved jointly rather than
  calibrated away, and a denser array (ngEHT-class coverage) for the positions from closures.
  All three are within the existing machinery: `synthetic_scans` takes any coverage, the
  self-calibration path has per-scan gains, and the likelihood sums over bands.

## Self-calibration through the instrument model (2026-09-10)

The same truth on the same coverage, but the data are the correlation products RR, LL, RL, LR
corrupted by a seeded instrument with Comrade's structure (`docs/STATUS.md`, `Fit.InstrumentModel`):
per-scan complex R and L gains with 0.15 rms in log-amplitude and phase (372 free entries over
22 scans with a reference station, ALMA), d-terms of 0.05 rms per part (32), and the real feed
rotation of the EHT array from the antenna table (angles from −3.1 to 3.9 rad); thermal noise
1% of the flux (1728 real values). Sky and instrument are fitted jointly from the same
perturbed sky and a unit instrument (`Fit.selfcal!`: 500 Adam iterations, the sky's gradient
by the dual sweep on the GPU and the instrument's by ForwardDiff on the host, 9 minutes), then
six joint Levenberg–Marquardt iterations over the 480 parameters (6 minutes) with the Laplace
covariance (`validation/winding/winding_vlbi.jl --mode selfcal`).

| | χ²/N (truth 1.34) | positions (M) | Laplace σ(x), σ(y) (M) | gains, rms error (worst) | d-terms, rms error (worst) |
|---|---|---|---|---|---|
| unit instrument, start | 295 | 0.41–0.75 | | | |
| after Adam | 1.33 | | | 0.045 | 0.021 |
| after the polish | 1.20 | 0.09–0.26 | 0.03–0.15 | 0.033 (3.9σ) | 0.020 (2.3σ) |

Against the calibrated-visibility fit above (positions to 0.013–0.19 M, σ 0.02–0.09 M), the
instrument's freedom costs a factor of two to three in the positions and nothing in the
pattern rates (0.05–0.75%); the plasma rows stay unconstrained (σ(ln Θe) 0.5–1.2, σ(ln B)
0.9–1.7, σ(ln nₑ) 1–3.9), as with calibrated data. The gains come back to 0.03 in
log-amplitude and phase, within their Laplace errors (median 0.018), the d-terms to 0.02
(median σ 0.003, the worst 2.3σ); with 400 instrument parameters against 1728 values the fit
sits below the truth's χ², absorbing noise, which is the price Comrade's priors are there to
limit and which the closure-only and calibrated variants bracket from either side.

## What the machinery does

`ScanData(time, data)` carries a scan's frame time and its `VisibilityData` or `ClosureData`;
`TimeResolved` is a vector of scans, and `frame_times` its distinct times. `chi2_timeresolved`
renders each distinct time once (slow light; pixel integration through `binning`) and sums the
static χ² of the scans at that time; `timeresolved_gradient!` is one `image_loss_gradient!`
(dual sweep) per frame, with priors on the host; `timeresolved_residuals` gives the scaled
residual vector for `polish!` and the Laplace covariance. `coverage(obs, times)` takes a real
observation's scans with chosen frame times; `synthetic_scans` renders the frames and adds the
noise. Gate `test_timeresolved`: zero-noise synthetic visibilities equal the frames' DFT, the
per-scan χ² by hand, the device gradient against host Enzyme (closures 1e-14 CPU / 1.6e-13
CUDA, visibilities with a prior 4e-15 / 1.3e-14), the residual norms against the χ², and the
polish on the residuals. `ObservedScan` carries the rows of an observation compared through the
instrument (`instrument = (model, gains, dterms)` in the χ², the gradient, which then also
returns the instrument's gradient through `dinstrument`, and the residuals); `selfcal!` is the
joint Adam loop, `levenberg_marquardt!` with `pack`/`unpack` the joint polish. Gate
`test_selfcal`: the sky, gain and d-term gradients against host Enzyme and ForwardDiff to
1e-16, the instrument recovered from unit gains with the sky fixed within 2.8σ, a joint polish
and a joint Adam loop lowering χ². Real observations still need the UT-to-M mapping of scan
times.
