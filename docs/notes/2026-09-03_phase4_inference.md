# Phase 4/5: inference pieces — status (2026-09-03)

`KerrSplat.Fit` (PRs #16, #17) holds the inference layer built on the polarized splats:

- `StokesMovie`, `chi2` over minibatches of frames and frequencies, `freeze` masks and `fit!`
  (Adam on Enzyme reverse gradients with staged unfreezing). Gate: two polarized splats fitted to
  a noisy three-frame movie reach the noise floor (χ² = 1202 for 1200 data points), positions to a
  fraction of a pixel, pattern rates to 2–3%.
- Partition hygiene: `prune`, `merge` (exact for co-located identical parcels because the transfer
  coefficients add), `densify` (mass-conserving split along the largest axis).
- Spacetime derivatives: ForwardDiff duals for (a, θo) run through the whole pipeline (dual-safe
  K₀, K₁ wrappers were the only missing piece), matching finite differences to 3e-6: the
  mixed-mode strategy (forward for the few spacetime parameters, reverse for the many splat
  parameters) is available end to end on the CPU backend.
- Fisher audit: `fisher` (Jacobian of every masked data value by duals, F = Jᵀ Σ⁻¹ J) and `audit`
  (eigen-decomposition, weakest combinations named). Finding on a single splat with 2/1/1/0.5%
  noise in I/Q/U/V at 8² pixels: at 230 GHz alone the weakest direction is 0.70 ln nₑ − 0.65 ln B
  + 0.28 ln Θe with σ = 0.58 (the nₑ–B–Θe degeneracy of plan §7.7); adding 345 GHz raises its
  eigenvalue only 1.6× (less than the doubling of the data) but lifts the temperature-dominated
  second direction 7.4× (σ 0.25 → 0.09). Multi-frequency data constrain Θe far better than the
  nₑ–B trade-off, which needs the polarized and optically thick information (or priors) to close.

Later the same day (PRs #25–#28): `fit!(params, movie, cache, L, stages)` with `Stage` (rows to
free, iterations, cosine-annealed learning rate, minibatch, frequency subset) and `Hygiene`
(prune, merge, densify every n iterations; a one-splat start densifies into the two splats of the
target movie); `recovery_metrics` on a voxel grid (density PSNR, weighted Θe and B errors; the
noise-floor fit gains 3.7 dB in density and recovers B to 1.5%); FITS movies in the ehtim layout
(`read_stokes_movie`, `write_stokes_fits`); and `fit_spacetime`, Levenberg–Marquardt for (a, θo)
on the dual Jacobian (inclination to 0.07°, spin to 0.04 on a coarse image, χ² at the noise floor).

Open: GPU-side reverse gradients (Enzyme 0.13.199 is the newest release; the device-side failure
of docs/notes/2026-09-04_phase1_thin_splats.md stands), motion mode C, power-law rotativities,
the visibility-domain likelihood (Comrade), fits to ipole-rendered GRMHD movies (gate 7 ii).
