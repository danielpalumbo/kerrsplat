# KerrSplat status (2026-09-03)

What exists, how it was validated, and what is open. Plans: `docs/plans/` (read the evaluation
plan, the addendum, then the GPU geodesic plan). Detailed findings: `docs/notes/`.

## Layers and their gates

| Layer | Module | Gate (independent reference) | Result |
|---|---|---|---|
| Per-pixel constants, direct samples | `Geodesics` | Krang on the CPU, CPU and CUDA | 1e-12 (gate 1) |
| Recurrence marcher r(τ), θ(τ) | `Geodesics` | BigFloat evaluation of the closed forms | root-limited, adaptive anchoring near the critical curve (gate 2) |
| Anchored quadrature t̃(τ), φ(τ) | `Geodesics` | BigFloat integration of the rates | 2e-9 / 2e-8 (gate 3) |
| Spacetime duals (a, θo) | `Geodesics` | finite differences | 1e-5 (gate 4) |
| Fused mode, tiles | `Geodesics` | stored samples | exact (gate 5) |
| Thin splats, Enzyme gradients, fit | `Splats` | host loop on Krang's samples; central differences | 1e-9; 2e-7 |
| Synchrotron coefficients | `Transfer` | symphony tables from ipole's sources (`validation/symphony`) | 6e-14 |
| Exact 4×4 transfer step | `Transfer` | BigFloat matrix exponentials and integrals | eps per radian of rotation |
| Frames and Walker–Penrose angle | `Transfer` | Krang's `synchrotronPolarization` | 1e-15 |
| Stokes I transport, units | `Transfer` | Gold et al. (2020) published fluxes; ipole built here | 0.6–0.9% (ipole itself 0.7–1.0%) |
| Full Stokes transport | `Transfer` | ipole's RIAF images at 230 GHz and 2 THz (`validation/ipole_riaf`) | 0.2–2% pixel norms, 0.0005 rad EVPA |
| Polarized one-zone splats | `Splats` | host loop; CUDA vs CPU; Enzyme vs 4th-order stencil; fit | 4e-18; 7e-14; 2e-7; loss ÷140 |
| Slow light | `Splats` | flare echo arrival times vs the lookback of the crossing rays | ±0.3 M |
| Motion modes B and C | `Splats` | rotated copy; Keplerian orbit; mode C vs mode B | 1e-16; 1e-12; 0.14% |
| χ², staged minibatched Adam, hygiene | `Fit` | noisy movie reaches the noise floor; merge exact | χ²/N = 1.0; 1e-12 |
| Spacetime derivatives of polarized images | `Transfer`/`Splats` | finite differences | 3e-6 |
| Fisher audit | `Fit` | (finding) nₑ–B–Θe degeneracy at one frequency | σ 0.58 → 0.45 with a second frequency |
| Power-law rotativities (Jones & O'Dell) | `Transfer` | symphony's numerical susceptibility integration | signs everywhere, ρ_V 3%, ρ_Q 30% in the validity window |
| Fit schedules with hygiene | `Fit` | one splat densifies into two on a two-splat movie | χ² ÷ 6 in 100 iterations, centres within a pixel |
| Field-level recovery on a voxel grid | `Splats` | truth of the noise-floor fit | density PSNR +3.7 dB, Θe error 0.22 → 0.17, B to 1.5% |
| FITS movies (ehtim layout) | `Fit` | write/read round trip | 1e-9 |
| Spin and inclination fit (Levenberg–Marquardt on duals) | `Fit` | noisy image of two splats | θo to 0.07°, a to 0.04 at 2.25 M pixels, χ² at the noise floor |
| Visibility-domain χ² (direct transform) | `Fit` | explicit transform from sky coordinates; Enzyme vs stencil | 1e-10; 1e-5 |
| Closure phases and log closure amplitudes | `Fit` | invariance under station gains; Enzyme vs stencil | 1e-12; 1e-5 |
| κ-distribution coefficients | `Transfer` | upstream symphony's fits (2160 rows); symphony numerics for j_V | 1e-15 (ρ_Q 2e-13); j_V fit 29% |

Timings on the RTX 2080 SUPER (256² × 1000 samples): direct evaluation 128 ns per sample,
r/θ recurrence 1.9 ns, full quadrature 10 ns (0.67 s per regeneration), fused thin splats 15 ns
per sample-splat. Polarized transport (`bench/polarized_bench.jl`, 128² × 1000 samples, one
frequency): 97 ns per sample with one splat, 170 ns with four, 366 ns with sixteen (about 18 ns
per additional splat: the Gaussian weight decides whether the frames, coefficients and exact step
run), i.e. 6 s for a 128² Stokes image of sixteen splats and about a quarter of an hour for a
cube of twenty frames and eight frequencies. Interval culling would remove most of the
per-splat cost for many splats.

## Conventions that were settled by validation

- Krang's t̃ grows inward along a ray (elapsed time to the observer minus r_obs + 2 ln r_obs);
  every consumer evaluates the plasma at t_em = t_obs − t̃.
- Stokes basis: ipole's plasma tetrad (Q axis ⟂ projected field); screen basis north = +β,
  east = −α (Krang's `evpa`), right-handed with the propagation direction. The opposite choice
  reverses the Faraday sense and fails against ipole at the 60% level.
- Krang's analytic geodesics return NaN at a = 0; Schwarzschild models use a = 1e-3.
- The full test suite prints progress to stderr and takes over two hours (several Enzyme
  compilations of seven minutes each); run it through a pipe, `julia -t 8 --project=. test/runtests.jl 2>&1 | tee log`.
- Enzyme compilation inside a KernelAbstractions CPU kernel deadlocks when several worker tasks
  reach the first call together (`-t 8`); the in-kernel gradient test compiles on one work item
  first (this is what stalled two suite runs for hours).

## Open items

- GPU-side reverse-mode gradients: Enzyme 0.13.199 (the newest release) fails device-side inside
  kernels; the host path on the CPU backend is what the fits use. Forward-mode duals work on both.
- κ populations inside the splats (the coefficients exist; the hypergeometric factor is a host
  quantity per population).
- Interval culling of splats along rays (performance; accuracy first).
- Fits to ipole-rendered GRMHD movies (gate 7 ii; a fit to ipole's RIAF image is in
  `validation/riaf_fit/`). The visibility and closure χ² exist on the direct transform; station-gain
  fitting and the Comrade.jl route do not.
- Upstream issues for Krang and JacobiElliptic (`docs/notes/upstream_issues.md`, eleven items,
  Daniel's call).

## Pull requests (2026-09-03)

#7 integrates the earlier stack into `main`; #8–#12 are Phase 2 (coefficients, step, frames,
Gold fluxes, polarized transport); #13 polarized splats; #14 slow light; #15 pattern rotation
and cubes; #16 Fit; #17 spacetime duals and Fisher; #18 advection; #19 this page; #20
power-law rotativities; all merged into `main` on 2026-09-03 together with #21 (Apache 2.0),
#24 (in-kernel gradient warm-up), #25 (fit schedules), #26 (field recovery), #27 (FITS) and
#28 (spacetime fit).
