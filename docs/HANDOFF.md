# Handoff: picking KerrSplat up on another machine

Written 2026-09-04 for whoever (or whichever Claude session) continues this work elsewhere. The
repository is the only durable record: session memories of the Claude instance that wrote most of
the code live outside the repository and do not travel.

## Where things are

- `CLAUDE.md`: working conventions (branches, PRs, tests before PRs, Claude may merge) and the
  numerical facts that were expensive to learn. The environment section describes *this*
  workstation (RTX 2080 SUPER, driver 570); see below for what changes elsewhere.
- `docs/STATUS.md`: one page mapping every layer to its validation gate, the conventions
  (units, Stokes basis, Fourier sign), timings, open items and the PR map.
- `docs/notes/`: findings by phase (geodesic conditioning, transfer design, slow light,
  inference, cross-model fits, real-data path, GRMHD fits, half-orbit decomposition) and
  `upstream_issues.md` (bugs found in Krang, ipole, symphony).
- `docs/plans/`: the original plan and its addendum; the GPU geodesic plan.
- `validation/`: reference tables (symphony, ipole RIAF), the cross-model, GRMHD, M87 and
  half-orbit experiments; each directory's README or script header says how it was run.
- `test/`: `runtests.jl` is the full suite (≈ 1 h, run through a pipe), `ci.jl` the CPU subset
  GitHub Actions runs on every push.

## Setting up

1. Julia 1.10 (juliaup `lts`). `Pkg.instantiate()` in the project; the Manifest pins Krang.jl
   to the git commit `f36f43a` (never the registered release) and Enzyme to 0.13.199.
2. CUDA: pick the runtime that the driver supports, e.g. `CUDA.set_runtime_version!(v"12.8")`
   for driver 570 (`nvidia-smi` shows the driver's CUDA version). Without a pin CUDA.jl may
   select a toolchain the card cannot load. The kernels are Float64 (Float32 is unstable in
   Krang's geodesics): a consumer card runs them at 1/32–1/64 rate, a datacenter card is many
   times faster. Every CUDA session needs `CUDA.limit!(CUDA.LIMIT_STACK_SIZE, 4096)`; never
   `julia -g2`.
3. Run `julia -t 8 --project=. test/ci.jl` first (CPU only, ≈ 15 min including compiles), then
   the CUDA gates through the full suite. Enzyme gates take ≈ 7 min of compile each.
4. Optional external tools, used only for validation and fixtures (the fixtures and reference
   tables are committed): ipole (built with conda HDF5 and GSL, recipe in
   `validation/ipole_riaf/README.md` and `docs/notes/2026-09-03_phase2_transfer_design.md`),
   symphony (`validation/symphony/Makefile`), eht-imaging 1.2.10 in a conda environment
   (`pip install ehtim`; `validation/uvfits`).

## Data that is not in the repository

Scripts take these by path or flag; they were on the original workstation only.

- EHT 2017 M87 Stokes-I uvfits (public SR1 release, day 101 band low,
  `SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits`): the closure fits, `validation/m87_fit`.
- A full-polarization HOPS file of the same night (`hops_3601_M87+netcal.uvfits`, four
  correlation products) and a Sgr A* full-polarization file: the self-calibrated fits.
- ipole images of KHARMA snapshots (Sgr A*, `KHARMA_M_Rh10_a*_i*_Balign1.h5`) and the
  time-averaged M87 GRMHD library (`tavgs_fits_480/*.fits`): `validation/grmhd_fit`.
- No GRMHD fluid dumps: movie fits against GRMHD remain open for that reason.

The half-orbit self-fit experiment (`validation/winding/winding_selffit.jl --backend cuda`,
`docs/notes/2026-09-08_half_orbit_selffit.md`) ran to completion on 2026-09-08 with the dual
sweep: χ²/N 1.155, 1.055, 1.019 for n = 0, n ≤ 1, n ≤ 2 in 45 minutes for all three cases (the
CPU path, tiled host Enzyme, reproduces case n = 0 to twelve digits). Pixel integration
(`Geodesics.Binning`: sub-sampled grids and polar annuli, `binning` keyword of `chi2`,
`chi2_gradient!`, `image_loss_gradient!`, `fit!`, `fisher`, `spacetime_residuals`; the script's
`--subsamples K`) integrates the pixels over their points; the note records the sub-sampled
rerun.

## State on 2026-09-05

Merged and validated: analytic geodesics on CPU and CUDA; full polarized transfer (thermal,
power-law, κ) against symphony and ipole; Gaussian plasma splats with slow light, pattern
rotation and advection; the inference layer (movie χ², staged fits with hygiene, priors,
Fisher audit, spin and inclination by duals); visibilities, closures, per-scan gains, a
uvfits reader and scan averaging validated against ehtim on real data (EHT Fourier sign
convention); reflection parity for observers below the equator and negative spins; the
half-orbit (sub-image) decomposition validated against Krang; Enzyme reverse mode inside the
CUDA kernel over stored samples: thin rays exact to 6e-13, and full polarized rays through the
chunked reverse sweep (`Splats.polarized_gradient!`, `Fit.chi2_gradient!`,
`Fit.image_loss_gradient!`; exact to 5e-13 on CUDA, but slower than the 8-thread CPU on this
card, see `docs/notes/2026-09-04_phase1_thin_splats.md`). Full suite: 4,882,390 checks,
81 minutes, 2026-09-05 (before the chunked sweep, whose gates ran standalone on CPU and CUDA and
in the CPU subset). Real-data results: M87 closure
fit χ²/N 1.5; self-calibrated polarimetric fit χ²/N 7 with six parcels; GRMHD snapshot fit to
5% of the Stokes I norm.

Daniel's request of 2026-09-08, done 2026-09-09 (`docs/notes/2026-09-08_half_orbit_selffit.md`,
section "Multifrequency"): the n ≤ 1 half-orbit self-fit at 86 + 345 GHz, bracketing the
parcels' turnover (178 GHz), against 230 GHz alone: Θe and B come back to ≲ 1–2% for three of
the four parcels (single band: 5–18%, 2.5–7%), Fisher σ(ln Θe) 13–20× and σ(ln B) 8–12×
tighter, densities 2–3× better; a third band adds little. The Levenberg–Marquardt polish
(`Fit.polish!`, 2026-09-09; the script's `--polish N`, `--init file`) finishes the descent the
Adam fits leave (Δχ² ≈ 10³ above the truth along the nₑ–B valley of an edge parcel) and gives
the Laplace errors; the note records the polished recoveries. Still open: a field of view that
holds every parcel's lensed images.

The time-resolved visibility likelihood (one slow-light frame per scan, needed for Sgr A*) exists
since 2026-09-09 (`Fit.chi2_timeresolved`, `timeresolved_gradient!`, `synthetic_scans`,
`coverage`; `validation/winding/winding_vlbi.jl` fits the half-orbit truth to synthetic closure
data on the EHT 2017 coverage, the plan's gate 7 in the data domain; its note records the
result). Real observations still need a UT-to-M time mapping and per-scan gains in the
time-resolved loss (the static self-calibration path has them).

Open, roughly in order of value: more parcels in the joint self-calibration update with
densification; R/L gains and leakage; a smooth background component for extended flux and a
noise floor; spin and inclination fits on real data; per-ray interval lists for hundreds of
splats; the dual sweep (`Splats.polarized_gradient!`, the default of `Fit.chi2_gradient!` and
`Fit.image_loss_gradient!` since 2026-09-06: the adjoint over the compositing from 4-vectors,
per-sample derivatives by forward-mode duals; 2.8 s per 128² × 300 gradient of six parcels on the 2080 SUPER, 34× the Enzyme sweep,
`docs/notes/2026-09-06_dual_sweep.md`; half-orbit truncation included since 2026-09-08) for
`KnotSplats` and the power-law and κ populations (each needs an `element_adjoint!`; the
Enzyme sweep, `method = :enzyme`, remains the slow reference for any model; movie fits take the
dual sweep with `fit!(...; gradient = :dual)` on a stored-sample cache since 2026-09-08); the
Comrade.jl route; GRMHD movie fits
(need dumps); the upstream reports in `docs/notes/upstream_issues.md` (Daniel's call).

## Conventions that bit us

- Krang returns NaN at a = 0 (use 1e-3). Screen basis north = +β, east = −α; V = Σ I e^{+2πi(ul+vm)}.
- Velocity and field components of a splat follow the axes (r̂, φ̂, −θ̂): the third is vertical.
- A closure named like `f3` makes `8f3(x)` a Float32 literal times x; name closures without
  trailing digits. A name assigned both inside a closure Enzyme differentiates and in the
  enclosing function is boxed, and Enzyme silently returns a zero gradient for it.
- Enzyme reverse passes over the CPU kernel tape the whole screen (≈ 1 GB per 8e4
  pixel-samples): tile large screens.
- Code that Enzyme must differentiate inside a CUDA kernel: no `sincos` (use `sincos_pair`),
  no polynomial tables of nine or more coefficients through `evalpoly`/`@horner` (use
  `@muladd_chain`), no mutually recursive helpers, and keep the geodesic march outside the
  differentiated region (stored samples). Details in `docs/notes/2026-09-04_phase1_thin_splats.md`.
