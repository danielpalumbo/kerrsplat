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
| Mode C as a soft constraint (`PatternPrior`) | `Fit` | Keplerian rate through the ZAMO tetrad; Enzyme vs stencil | 1e-12; 1e-6 |
| Staged fit for an arbitrary loss (`fit!(params, loss, stages)`) | `Fit` | the movie form's history and parameters, step for step | 1e-12 |
| χ², staged minibatched Adam, hygiene | `Fit` | noisy movie reaches the noise floor; merge exact | χ²/N = 1.0; 1e-12 |
| Spacetime derivatives of polarized images | `Transfer`/`Splats` | finite differences | 3e-6 |
| Fisher audit | `Fit` | (finding) nₑ–B–Θe degeneracy at one frequency | σ 0.58 → 0.45 with a second frequency |
| Power-law rotativities (Jones & O'Dell) | `Transfer` | symphony's numerical susceptibility integration | signs everywhere, ρ_V 3%, ρ_Q 30% in the validity window |
| Fit schedules with hygiene | `Fit` | one splat densifies into two on a two-splat movie | χ² ÷ 6 in 100 iterations, centres within a pixel |
| Field-level recovery on a voxel grid | `Splats` | truth of the noise-floor fit | density PSNR +3.7 dB, Θe error 0.22 → 0.17, B to 1.5% |
| FITS movies (ehtim layout) | `Fit` | write/read round trip | 1e-9 |
| Spin and inclination fit (Levenberg–Marquardt on duals) | `Fit` | noisy image of two splats | θo to 0.07°, a to 0.04 at 2.25 M pixels, χ² at the noise floor |
| Visibility-domain χ² (direct transform, EHT sign convention) | `Fit` | explicit transform from sky coordinates; Enzyme vs stencil | 1e-10; 1e-5 |
| Scan averaging (`average_scans`, ehtim's rules) | `Fit` | ehtim's `avg_coherent(scan_avg)` on real M87 data; hand-built doubled rows | 5e-9 Jy; exact |
| uvfits reader (`read_uvfits`, scan closures) | `Fit` | ehtim's parse of an ehtim-written observation and of real M87 data; transform of the source image vs ehtim's noiseless visibilities | 1e-7 Jy, 2e-7 in u, v; 6e-8 Jy |
| Closure phases and log closure amplitudes | `Fit` | invariance under station gains; Enzyme vs stencil | 1e-12; 1e-5 |
| κ-distribution coefficients | `Transfer` | upstream symphony's fits (2160 rows); symphony numerics for j_V | 1e-15 (ρ_Q 2e-13); j_V fit 29% |
| Composite models (background + splats) | `Transfer` | empty model, element order, RIAF + splat | rounding |
| Cross-model fit to ipole's RIAF (1 and 2 frequencies, with and without shrinkage) | `validation/riaf_fit` | ipole images; analytic RIAF fields | images to 0.6–3.6%; fields only up to the parcel degeneracy |
| Fit to a GRMHD snapshot (KHARMA Sgr A*, ipole image, 130°) | `validation/grmhd_fit` | ipole full-Stokes image | 5.3/2.1/2.4/0.5% of the I norm; flux 9% low |
| Real EHT data: M87 2017 (closure fit; self-calibrated polarimetric fit) | `validation/m87_fit` | public uvfits files; ehtim-validated reader | closure χ²/N 1.5; visibility χ²/N 7 with six parcels (open) |
| Priors and hierarchical shrinkage | `Fit` | explicit sums; Enzyme vs analytic gradient | 1e-10 |
| Station gains (self-calibration χ², per station or per station and scan) | `Fit` | gained data at the true gains; Enzyme vs stencil | prior only; 1e-5 |
| Power-law and κ splat sets | `Splats` | thin-limit additivity of the three populations; CUDA vs CPU | 2e-6; 1e-11 |
| Reflection parity (observers beyond 90°, negative spin, U and V handedness) | `Splats` | equatorial and azimuthal mirror images of the mirrored source (axial field) | 1e-11 |
| GPU gradients: Enzyme inside the CUDA kernel over stored samples (`thin_gradient!`; the polarized consumer over ≤ 8 samples) | `Splats` | host Enzyme gradient | 6e-13 (thin, 64² × 300 in 1.2 s); 2e-15 (polarized chunk) |
| Full polarized rays on the GPU: the chunked reverse sweep (`polarized_gradient!(...; method = :enzyme)`), the movie χ² gradient (`Fit.chi2_gradient!`) and any image loss through a host seed (`Fit.image_loss_gradient!`) | `Splats`, `Fit` | host Enzyme gradient of the same loss; Enzyme's CPU gradient of `chi2` | 5e-13 (32² × 300, CUDA); 6e-13 (χ² gradient, CUDA); exact but 4× slower than the 8-thread CPU at 32² |
| Movie fits on the GPU: `fit!(params, movie, cache, L, stages; gradient = :dual)` takes χ² and gradient from `chi2_gradient!` on the cache's backend (priors on the host); the fit loop evaluates loss and gradient in one call on both paths (`history` is the loss at the start of each iteration) | `Fit` | the Enzyme-gradient fit on the same schedule and prior | trajectories agree to 1e-9 over 8 iterations (`test_fit_schedule`) |
| Pixel integration: `Geodesics.Binning` (points grouped into pixels; `binned_grid`, `binned_polar`, `concatenate`, `bin`) through `chi2`, `chi2_gradient!`, `image_loss_gradient!`, `fit!`, `fisher`, `spacetime_residuals` (`binning` keyword) | `Geodesics`, `Fit` | hand means; host Enzyme through `chi2` with the same binning; `spacetime_residuals` reproduces χ² | `test_binning` (CPU and CUDA) |
| Levenberg–Marquardt polish (`Fit.polish!`): damped Gauss–Newton on the free splat parameters from a fitted point, Jacobian by ForwardDiff duals through the transfer at fixed geodesics (`model_values`, `data_values`, `prior_residuals`), Laplace covariance at the end | `Fit` | the truth's χ² and the Laplace errors; `fisher` (same Jacobian); `penalty` (prior residuals) | `test_polish`: four iterations take a start 5% off the truth from χ² 614 to 540 (truth 548), within 1.2 Laplace σ; covariance = inverse Fisher to 1e-8 |
| Multifrequency n ≤ 1 self-fit (`--frequencies 86,345`, per-band noise; note section "Multifrequency"): bands bracketing the parcels' turnover (178 GHz) against 230 GHz alone | `Fit` | the single-band fit; marginal Fisher errors from the finite-difference audit | after 1000 Adam iterations and five LM iterations: all four parcels to Θe ≤ 0.35% and B ≤ 0.31% at 86 + 345 GHz (230 GHz alone: 0.3–1.1% and 0.1–5.7%), nₑ 0.1–7% (3–15%); Fisher σ(ln Θe) 13–20× and σ(ln B) 8–12× tighter than one band, σ(ln nₑ) 2×; the Laplace errors at the polished points agree with the Fisher at the truth; a third band adds little |
| Half-orbit self-fits (`validation/winding/winding_selffit.jl --backend cuda`, `docs/notes/2026-09-08_half_orbit_selffit.md`): the four-parcel truth fitted to movies truncated at n = 0, n ≤ 1, n ≤ 2 on screens growing to 20,224 pixels, joint Fisher of spin and inclination | `Fit`, `Splats` | the CPU run of case n = 0 (twelve digits); χ²/N against the noise floor | χ²/N 1.155, 1.055, 1.019 (1.134, 1.059, 1.043 with 2 × 2 points per pixel); positions pinned to 0.003–0.07 M by the lensed passages; the Fisher σ(a), σ(θo) at the truth are not converged in screen sampling (n = 0: 5.8e-4 to 3.1e-5 for 1–4 points per pixel side) and are reported as such; 45 minutes for the three point-sampled cases |
| The dual sweep (`polarized_gradient!`, the default; `polarized_tails!` + `polarized_dual_sweep!`): the adjoint over the compositing from tails and adjoint 4-vectors, the per-sample derivatives by forward-mode duals through `step_operator` and the splat coefficients; half-orbit truncation through per-ray cutoffs (`winding_cutoff!`, `nmax`/`slab`) (`docs/notes/2026-09-06_dual_sweep.md`) | `Transfer`, `Splats`, `Fit` | `test_step_adjoint` (finite differences over the 413 step cases); host Enzyme gradient of the same loss, truncated and not (`test_polarized_gradient_winding`: cutoffs vs the host counter, gradients 3e-15–2e-14); the Enzyme sweep | 2e-15 (CPU, 8² × 40); 1.4e-14 vs the Enzyme sweep (CUDA, 32² × 300); 2.8 s per 128² × 300 gradient of six parcels on the 2080 SUPER (Enzyme sweep: 95 s); 0.83 s at 32² against 2.27 s for the CPU fits' host Enzyme gradient |
| Half-orbit decomposition (`WindingState`, rays truncated after the n-th midplane passage) | `Transfer`, `Splats`, `Fit` | Krang's `Gθ` crossing times and `emission_radius` sub-image geometry; identity without truncation; Enzyme vs stencil; CUDA | 2e-7 in Mino time; 1e-13; 1e-5 |

Timings on the RTX 2080 SUPER (256² × 1000 samples): direct evaluation 128 ns per sample,
r/θ recurrence 1.9 ns, full quadrature 10 ns (0.67 s per regeneration), fused thin splats 15 ns
per sample-splat. Polarized transport (`bench/polarized_bench.jl`, 128² × 1000 samples, one
frequency): 97 ns per sample with one splat, 170 ns with four, 366 ns with sixteen, 520 ns with
sixty-four (the bounding-sphere early-out of `outside_support` rejects far splats for a `sincos`
and a few multiplications, 20% faster than the full weight at sixty-four splats), i.e. 6 s for a
128² Stokes image of sixteen splats and about a quarter of an hour for a cube of twenty frames
and eight frequencies. Per-ray interval lists would remove the remaining per-splat cost.

## Conventions that were settled by validation

- Krang's t̃ grows inward along a ray (elapsed time to the observer minus r_obs + 2 ln r_obs);
  every consumer evaluates the plasma at t_em = t_obs − t̃.
- Stokes basis: ipole's plasma tetrad (Q axis ⟂ projected field); screen basis north = +β,
  east = −α (Krang's `evpa`), right-handed with the propagation direction. The opposite choice
  reverses the Faraday sense and fails against ipole at the 60% level.
- Krang's analytic geodesics return NaN at a = 0; Schwarzschild models use a = 1e-3.
- The full test suite prints progress to stderr and takes about 80 minutes (4,882,390 checks
  in 81 minutes on 2026-09-05, with the GPU in-kernel gradient gates; several Enzyme
  compilations of seven minutes each); run it through a pipe,
  `julia -t 8 --project=. test/runtests.jl 2>&1 | tee log`.
- Enzyme compilation inside a KernelAbstractions CPU kernel deadlocks when several worker tasks
  reach the first call together (`-t 8`); the in-kernel gradient test compiles on one work item
  first (this is what stalled two suite runs for hours).

## Open items

- GPU-side gradients: the dual sweep (2026-09-06, half-orbit truncation 2026-09-08) gives the
  polarized gradient of thermal parcels on CUDA and the CPU backend without a tape;
  `KnotSplats` and the power-law and κ populations still need their `element_adjoint!` (the
  Enzyme sweep, `method = :enzyme`, covers any model at fifty times the forward cost per sample).
- The κ splats hold their hypergeometric factors fixed during a fit (κ and w are not fitted).
- Per-ray interval lists for many splats (the per-sample bounding-sphere early-out exists).
- Fits to ipole-rendered GRMHD movies (gate 7 ii): a KHARMA snapshot image is fitted to a few
  per cent (`validation/grmhd_fit`, note `docs/notes/2026-09-04_grmhd_snapshot_fit.md`); movies
  need fluid dumps or an ipole movie, which this machine does not have.
- Upstream issues for Krang and JacobiElliptic (`docs/notes/upstream_issues.md`, eleven items,
  Daniel's call).

## Pull requests (2026-09-03)

#7 integrates the earlier stack into `main`; #8–#12 are Phase 2 (coefficients, step, frames,
Gold fluxes, polarized transport); #13 polarized splats; #14 slow light; #15 pattern rotation
and cubes; #16 Fit; #17 spacetime duals and Fisher; #18 advection; #19 this page; #20
power-law rotativities; all merged into `main` on 2026-09-03 together with #21 (Apache 2.0),
#24 (in-kernel gradient warm-up), #25 (fit schedules), #26 (field recovery), #27 (FITS) and
#28 (spacetime fit).
