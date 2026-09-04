# kerrsplat

Differentiable Gaussian splatting of time-dependent plasma into the Kerr spacetime, fitted to
spectro-polarimetric (Stokes I, Q, U, V; multi-frequency) movies of the near-horizon region.
Julia, built on [Krang.jl](https://github.com/dchang10/Krang.jl) analytic geodesics, with the
hot path on NVIDIA GPUs via CUDA.jl + KernelAbstractions and gradients from Enzyme (reverse mode,
splat parameters) and ForwardDiff duals (forward mode, spacetime parameters).

Status (2026-09-03): the GPU geodesic plan through step 6, Phase 1 (optically-thin splats),
Phase 2 (polarized radiative transfer) and the first Phase 3 items. The package builds Krang's
per-pixel constants on the GPU (or the CPU backend of KernelAbstractions), marches the rays
either with Krang's closed forms at every sample (the reference, 128 ns per sample on the
RTX 2080 SUPER) or with the addition-theorem recurrence for r(τ), θ(τ) plus anchored quadrature
of the Mino-time rates for t̃(τ), φ(τ) (10 ns per sample, with ForwardDiff duals for the
spacetime parameters), validates both against Krang on the CPU and against BigFloat references,
and integrates the full Stokes transfer equation along the rays with synchrotron coefficients
validated against ipole's symphony fits, an exact constant-coefficient step validated against
BigFloat, Walker–Penrose frame angles validated against Krang, Stokes I fluxes validated on the
Gold et al. (2020) test problems and full Stokes images validated against ipole's RIAF model
(0.2–2% pixel norms). Gaussian splats carry either a thin emissivity or a thermal plasma
(density, temperature, field, velocity) and render, with slow light and a pattern rotation, as
Stokes movie cubes differentiable with Enzyme (fits of positions, plasma, field, velocity and
pattern rate from synthetic movies are part of the tests).

## Package layout

- `src/KerrSplat.jl` — package root.
- `src/Geodesics/` — `KerrSplat.Geodesics`: `Camera` (Bardeen screen coordinates),
  `PixelConstants` (Krang's per-pixel constants as a structure of arrays, sorted by radial root
  case), `GeodesicSamples` (stored per-ray samples t̃, r, θ, φ, ν_r, ν_θ), `GeodesicCache` and
  `regenerate!(cache, a, θo[, camera]; marcher = Direct() | Recurrence(64))`. `Direct` is
  Krang's closed-form evaluation at every sample; `Recurrence` advances r and θ by the Jacobi
  addition theorems, integrates the Mino-time rates for t̃ and φ with the singular parts
  removed in closed form, and re-anchors everything to Krang every 64 samples, keeping each
  ray's largest anchor residual as an error estimate. `Fused(64)` with `fused_march!(f, out,
  cache)` runs the same marcher inside a consumer `f(acc, j, k, sample, Δτ, pix)` without
  storing samples; `tiles(camera, n)` splits large screens.
- `src/Transfer/` — `KerrSplat.Transfer`: synchrotron coefficients (thermal Dexter/Pandya
  fits, Kirchhoff absorptivities, Dexter and Shcherbakov rotativities, Pandya power law;
  `coefficients.jl`), the exact constant-coefficient 4×4 transfer step (`step.jl`), the
  per-sample frames (redshift, pitch angle, Walker–Penrose screen angle; `frames.jl`) and the
  fused-march consumers `UnpolarizedTransport` and `RadiativeTransport` that compose the steps
  front to back for any model of overlapping fluid elements (`transport.jl`). Conventions: cgs,
  ipole's field-aligned Stokes basis, screen basis north = +β, east = −α.
- `src/Splats/` — `KerrSplat.Splats`: Gaussian plasma splats with a temporal envelope and a
  pattern rotation, either with a thin emissivity (14 parameters, `thin_image`) or as one-zone
  thermal plasma parcels (21 parameters: density, temperature, field strength and direction,
  ZAMO velocity; `polarized_image`, `polarized_cube`), rendered through the fused marcher with
  slow light and differentiable with Enzyme.
- `test/` — gates 1–5 of the GPU plan (every stored quantity against Krang on the CPU, the
  recurrence against a BigFloat evaluation of the closed forms, the quadrature against a
  BigFloat integration of the rates), the Phase 2 gates (symphony tables, BigFloat transfer
  steps, Krang's polarization transport, Gold et al. 2020 fluxes, ipole's RIAF images in
  `validation/`), the slow-light flare echo, the motion gates and the Enzyme gradient and fit
  tests, on both the CPU and the CUDA backend. The suite writes its progress to stderr; run it
  through a pipe (`… 2>&1 | tee log`) to see it, since output to a file is buffered until exit.
- `validation/` — the symphony table generator (built against a local ipole checkout) and the
  ipole RIAF reference images with the recipe that produced them.
- `docs/notes/` — findings that matter for later gates (conditioning of Krang's closed forms
  near the polar axis and the critical curve) and candidate upstream issues.
- `bench/geodesics_bench.jl` — timings of the stages (compare plan §3).

## Environment

Julia 1.10. `Project.toml` and `Manifest.toml` pin Krang.jl to git commit `f36f43a` (see
`CLAUDE.md` for why the registered release is not usable). On this workstation CUDA.jl must
use the CUDA 12.8 runtime; `LocalPreferences.toml` carries that pin. To set up:

    JULIA_PKG_USE_CLI_GIT=true julia --project=. -e 'import Pkg; Pkg.instantiate()'

Run the tests with `julia -t 8 --project=. test/runtests.jl 2>&1 | tee log` (or `Pkg.test()`);
GPU tests run when `CUDA.functional()`. The full suite takes over two hours (BigFloat references
and Enzyme compilations); `test/ci.jl` is the CPU-only subset that GitHub Actions runs on every
push and pull request. Kernels that contain Krang code need a per-thread stack larger than
CUDA's default; `regenerate!` raises it through `Geodesics.prepare_backend!`.

- `docs/plans/` — the project plan (2026-07-09), the one-zone-splat addendum (2026-07-10), and
  the GPU geodesic plan (2026-09-02). Read them in that order.
- `smoketests/` — Enzyme reverse-mode gradients through Krang geodesics and polarization
  (the feasibility evidence for the plan).
- `gpu_probes/` — CUDA measurements behind the GPU geodesic plan, plus the environment fixes
  needed to run Krang inside CUDA kernels (see its README).

See `CLAUDE.md` for the conventions that apply to all work in this repository.

## License

KerrSplat is released under the Apache License, Version 2.0 (see `LICENSE`; attribution
notices in `NOTICE`). Dependencies are installed by the package manager under their own
licenses. The reference tables and images under `validation/` were produced with the
GPL-licensed codes ipole and symphony, which are not included: the small driver programs
there are built against local checkouts of those codes, and the committed files are their
numerical output.
