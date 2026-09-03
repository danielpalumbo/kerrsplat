# kerrsplat

Differentiable Gaussian splatting of time-dependent plasma into the Kerr spacetime, fitted to
spectro-polarimetric (Stokes I, Q, U, V; multi-frequency) movies of the near-horizon region.
Julia, built on [Krang.jl](https://github.com/dchang10/Krang.jl) analytic geodesics, with the
hot path on NVIDIA GPUs via CUDA.jl + KernelAbstractions and gradients from Enzyme (reverse mode,
splat parameters) and ForwardDiff duals (forward mode, spacetime parameters).

Status (2026-09-03): geodesic module through plan §9 step 4. The package builds Krang's
per-pixel constants on the GPU (or the CPU backend of KernelAbstractions), marches the rays
either with Krang's closed forms at every sample (the reference, 128 ns per sample on the
RTX 2080 SUPER) or with the addition-theorem recurrence for r(τ), θ(τ) plus anchored quadrature
of the Mino-time rates for t̃(τ), φ(τ) (10 ns per sample, with ForwardDiff duals for the
spacetime parameters), and validates both against Krang on
the CPU and against BigFloat references.

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
- `src/Splats/` — `KerrSplat.Splats`: Gaussian plasma splats (position, shape, temporal
  envelope, amplitude as 13 unconstrained parameters per splat) and `thin_image`, their
  optically-thin image through the fused marcher (Phase 1 of the roadmap: unpolarized,
  frequency-independent, ZAMO-frame emitters, slow light), differentiable with Enzyme.
- `test/` — gates 1–5 of the GPU plan: every stored quantity against Krang on the CPU, the
  recurrence against a BigFloat evaluation of the closed forms and the quadrature against a
  BigFloat integration of the rates (`test/highprec_reference.jl`), on both the CPU and the
  CUDA backend.
- `docs/notes/` — findings that matter for later gates (conditioning of Krang's closed forms
  near the polar axis and the critical curve) and candidate upstream issues.
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
