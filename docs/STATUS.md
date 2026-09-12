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
| Per-row step multipliers (`Stage(steps = (; omega = 0.05))`, `selfcal!(...; steps)`, `step_scale`) | `Fit` | first Adam update of the scaled row equals the factor times the plain one, other rows untouched | 1e-12 |
| Instrument-only solve with the sky held (`scan_models`, `calibrate!`) | `Fit` | the joint LM polish with the sky frozen (`timeresolved_residuals` through `pack`/`unpack`), history, solution and covariance | 1e-8 |
| Reference-baseline phase start (`reference_phases!`), phases-first solve, χ² by product (`chi2_products`) | `Fit` | chained phases vs the truth from scrambled phases; the default solve from scrambled and from zero phases agree; the split sums to `chi2_instrument` | 0.3 rad; 1e-6; exact |
| Sgr A* diffractive scattering kernel (`ScatteringKernel`, `taper`; the `kernel` of every `ScanData`) | `Fit` | ehtim's `sgra_kernel_uv` for two parameter sets (`test/data/sgra_kernel_ehtim.csv`); synthetic scans, χ², device gradient and residuals through it | 1e-12; 1e-9 |
| Sgr A* 2017 April 11 (low band) static Stokes I self-calibration (`sgra_jones.jl --stokes-i`: closures → instrument solve → joint self-cal) | `Fit` | (experiment) products χ²/N 47 → 3.5 over 21 scans with the gains within 12% of one on average; the residual in the hour at UT 11.5 h | `docs/notes/2026-09-10_sgra_timeresolved.md` |
| Joint spacetime-and-splat fit (`fit_joint!`: Adam on the splats from the dual sweep, Levenberg–Marquardt on (a, θo[, ln L]) from `spacetime_jacobian`, forward duals through a fused march; `spacetime_valgrad`, `spacetime_chi2`, `spacetime_movie_residuals`) | `Fit` | the spacetime gradient vs central finite differences of the stored-sample χ²; the dual residuals square to it and 2Jᵀr equals the gradient; recovery from a wrong spacetime with perturbed splats | 1e-9; 1e-8; a 0.92, θo 60.4° from 0.8, 52° (truth 0.9, 60°) at 8² × 40 |
| Per-ray parcel lists for the large-N dual sweep (`ray_lists`, `RayLists`, `RaySubset`, `Transfer.ray_model`; `cull`) | `Splats` | the dense sums: tails image and gradient with and without the half-orbit truncation; lists sorted, unique, culling 80% of 48 parcels | image identical, gradient 2e-16 |
| Spin profile of the joint fit (`joint_selffit.jl --fix-spin`: the spin held, everything else fitted jointly) on the inner-parcel n ≤ 2 movie | `Fit` | (experiment) χ²/N 1.415, 1.235, 1.169, 1.030, 1.086 at spins 0.3, 0.5, 0.7, 0.9 (truth), 0.98; the inclination to a degree | the minimum at the truth, 8,600 χ² for 0.08 of spin |
| Uniqueness table, second row: the spin profile with the n ≤ 1 images only (`--nmax 1`) | `Fit` | (experiment) χ²/N 1.130, 1.025, 1.060 at spins 0.7, 0.9, 0.98: the n = 2 ring carries a third of the spin's discrimination | Δχ² 5,400 for +0.08 (8,600 with n ≤ 2) |
| Uniqueness table, Stokes rows (`--stokes IQU`, `--stokes I`) | `Fit` | (experiment) Δχ² for +0.08 of spin: 8,600 (IQUV), 5,100 (IQU), 2,400 (I); polarization carries two thirds of the spin's discrimination, Stokes I alone still resolves it | see the note |
| Uniqueness table, frames and frequency rows (preliminary, fits unconverged at 80 iterations) | `Fit` | (experiment) Δχ² for +0.08 of spin: 1,600 with two frames instead of six, 9,300 with a second frequency (8,600 baseline) | rerunning longer |
| Large-N starting point (`shell_parcels`: n small parcels filling a shell, Keplerian rates, random fields) and the hygiene at that size | `Splats`/`Fit` | shell geometry, unit quaternions, rates; a three-parcel truth fitted from the shell with prune and merge lowers χ² and ends with fewer parcels | `test_large_n_init` |
| Weight cutoff 1e-6 (was 1e-12; `WEIGHT_CUTOFF`, the 5.3σ support sphere) | `Splats` | (performance) the large-N gradient on the GPU, 512–8192 parcels; the boundary discontinuity of the total is the cutoff × one pair's share, ~1e-11 | 1.7× faster; FD gates unchanged at 1e-5 |
| Closure amplitudes over every baseline (`scan_quadrangles`: two quadrangles per quadruple) | `Fit` | (finding) one quadrangle per quadruple left the first–last station baseline of every scan out of all log closure amplitudes (ALMA–SPT on Sgr A*, 7× off while the closures fit) | `test_uvfits`: the pair covers six legs and shares two |
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
| Instrument model with Comrade's structure (`Fit.InstrumentModel`, `src/Fit/instrument.jl`): coherency basis [RR RL; LR LL], RIME V = J₁ C J₂†, J = R† G D R for feed-rotation-corrected data (G D R raw), gains exp(lg + i gp) per feed with the L feed as a ratio to R, per-scan segmentation, reference-station phase gauge, d-terms per track, Comrade's priors (N(0, 0.2) log amplitudes, 1.0 for the LMT, 0.1 ratios, 0.2 d-term parts); the reader keeps the correlation products (`Observation.coh`, `σ_coh`) | `Fit` | ehtim's Jones corruption of the synthetic EHT 2017 observation (seeded gains, R/L ratios, phases, d-terms; corrected and raw feed rotation) | `test_instrument`: corrupted products to 1e-11, the assembled Jones matrices to 3e-16 in both forms; the reader's products vs ehtim's circular parse to 2e-7 (float32); unit instrument = Stokes χ²; gauge and priors by hand |
| Feed rotation angles (`Fit.feed_angles`, `station_angles`, `gmst`, `Mount`, `EHT_MOUNTS`; `antenna_positions` from the AN table): ehtim's formulas for the elevation and parallactic angle with the IAU 1982 sidereal time | `Fit` | ehtim's angles for the fixture observation | `test_feed_rotation`: elevation to 3e-5 rad, parallactic and feed angles to 2.4e-4 rad (the UT1−UTC offset astropy applies, amplified near transit) over 264 station-times |
| M87 2017 full-polarization data through the instrument model (`validation/m87_fit/m87_jones.jl`; note `2026-09-04_phase5_real_data.md`, last section): correlation products of the HOPS file, per-scan R/L gains, d-terms, the array's feed rotation, Comrade's priors, 2% noise floor, joint Adam plus LM polish | `Fit` | the no-leakage, tight-prior and fixed-sky variants; the sky's closure χ² | products χ²/N 275 → 1.33 with leakage (4.1 without, 2.8 with the sky fixed); the d-terms are Paper VII's rotated by ∓90°, the global RL phase of ALMA's 45° feed offset the HOPS netcal file lacks: with `rotate_crosshands(obs, π/2)` APEX's D_R agrees to 0.1%, ALMA lies along −i as the paper requires, SMT/PV/LMT within 1–3% (the methods' spread); gates `test_crosshand_rotation` (the χ² gauge), `test_dterm_recovery` (ehtim's seeded d-terms to 3e-12); `average_scans` averages the products on their own; with a 0.6 Jy flux prior (`image_prior` of the time-resolved likelihood, `total_flux`) and 3000 joint iterations the products reach χ²/N 0.98 at 0.57 Jy, the sky's closure χ²/N 3.86 → 1.67, PV and the SMT d-terms within 1–2% of every published method |
| Real time-resolved data: `scan_times` (scan mean UT hours → GM/c³), `closure_scans` (real closure data scan by scan, legs below 3σ dropped), the Sgr A* script `validation/sgra_fit/sgra_jones.jl` (April 11 low band: closures, then self-calibration; `docs/notes/2026-09-10_sgra_timeresolved.md`) | `Fit` | `test_timeresolved`: scan times against the formula, per-scan closure χ² sums to the whole-observation χ² (1e-10) | first Sgr A* runs end to end (4.3 s per joint iteration over 21 frames) but far from the data: closures χ²/N 15, products 246 from a ring start; a staged protocol is next |
| Joint self-calibration in the time-resolved likelihood (`ObservedScan`, `instrument` in `chi2_timeresolved`/`timeresolved_gradient!`/`timeresolved_residuals`, `instrument_gradient`, `selfcal!`, `levenberg_marquardt!` with `pack`/`unpack`) and the synthetic self-calibration of the half-orbit truth on the EHT 2017 coverage (`winding_vlbi.jl --mode selfcal`, note section "Self-calibration") | `Fit` | host Enzyme and ForwardDiff gradients; the seeded instrument; the calibrated-visibility fit | `test_selfcal`: gradients to 1e-16, instrument recovered within 2.8σ; experiment: χ²/N 295 → 1.20 (truth 1.34), positions 0.09–0.26 M (σ 0.03–0.15), gains to 0.033 rms (worst 3.9σ), d-terms to 0.020 (worst 2.3σ), 15 min on the GPU |
| Synthetic-VLBI self-fit of the half-orbit truth on the EHT 2017 coverage (`validation/winding/winding_vlbi.jl`, `docs/notes/2026-09-09_synthetic_vlbi.md`): closures or calibrated visibilities, 1000 Adam iterations on the GPU plus the LM polish with Laplace errors | `Fit` | the truth's χ²; the image-domain fits of the same truth | closures alone (445 quantities) leave the likelihood flat (Laplace σ 1–3 M, e²–e¹⁰ in nₑ, Θe, B) at χ²/N 0.97; calibrated visibilities pin positions to 0.02–0.09 M and pattern rates to 0.2–0.6% but not the plasma (σ(ln Θe) 0.5–0.8, σ(ln B) 0.7–1.2): the array sets the limit |
| Time-resolved visibility likelihood (`Fit.chi2_timeresolved`, `timeresolved_gradient!`): scans carry their frame time and their own closure or visibility data, one slow-light frame per distinct time, one dual sweep per frame on the device; `coverage` lends a real array's scans to a synthetic movie and `synthetic_scans` makes closure or visibility data with thermal noise | `Fit` | `synthetic_scans` at zero noise = the frames' visibilities; the per-scan χ² by hand; host Enzyme gradient of `chi2_timeresolved` | `test_timeresolved`: closure χ² gradient 1e-14 (CPU), 1.6e-13 (CUDA); visibilities with a prior 4e-15, 1.3e-14 |
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
