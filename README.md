# kerrsplat

Differentiable Gaussian splatting of time-dependent plasma into the Kerr spacetime, fitted to
spectro-polarimetric (Stokes I, Q, U, V; multi-frequency) movies of the near-horizon region.
Julia, built on [Krang.jl](https://github.com/dchang10/Krang.jl) analytic geodesics, with the
hot path on NVIDIA GPUs via CUDA.jl + KernelAbstractions and gradients from Enzyme (reverse mode,
splat parameters) and ForwardDiff duals (forward mode, spacetime parameters).

Status (2026-09-03): geodesic module skeleton (plan §9 step 2). The package builds Krang's
per-pixel constants and the per-sample geodesic coordinates on the GPU (or the CPU backend of
KernelAbstractions) and validates them against Krang's own CPU evaluation.

## Package layout

- `src/KerrSplat.jl` — package root.
- `src/Geodesics/` — `KerrSplat.Geodesics`: `Camera` (Bardeen screen coordinates),
  `PixelConstants` (Krang's per-pixel constants as a structure of arrays, sorted by radial root
  case), `GeodesicSamples` (stored per-ray samples t̃, r, θ, φ, ν_r, ν_θ), `GeodesicCache` and
  `regenerate!(cache, a, θo[, camera])`. The current marcher is Krang's direct closed-form
  evaluation at every sample; it is the reference against which the recurrence marcher of the
  GPU plan (§4) will be validated.
- `test/` — gate 1 of the GPU plan: every stored quantity against Krang on the CPU, on both
  the CPU and the CUDA backend.
- `bench/geodesics_bench.jl` — timings of the stages (compare plan §3).

## Environment

Julia 1.10. `Project.toml` and `Manifest.toml` pin Krang.jl to git commit `f36f43a` (see
`CLAUDE.md` for why the registered release is not usable). On this workstation CUDA.jl must
use the CUDA 12.8 runtime; `LocalPreferences.toml` carries that pin. To set up:

    JULIA_PKG_USE_CLI_GIT=true julia --project=. -e 'import Pkg; Pkg.instantiate()'

Run the tests with `julia -t 8 --project=. test/runtests.jl` (or `Pkg.test()`); GPU tests run
when `CUDA.functional()`. Kernels that contain Krang code need a per-thread stack larger than
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
