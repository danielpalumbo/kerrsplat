# Half-orbit self-fits on the GPU (2026-09-08)

Daniel asked (2026-09-05) how self-fits of the splat model behave when the movie contains only
the direct image (n = 0), the direct image and the first lensed passage (n ≤ 1), or up to the
second (n ≤ 2), with the screen resolution growing with the order, after the decomposition of
Johnson et al. (2020) by the half-orbit counter (`Transfer.WindingState`, `nmax`/`slab`,
validated against Krang's crossing times and emission radii, `test_winding`). The experiment
(`validation/winding/winding_selffit.jl`) was halted after the first case on 2026-09-05 because
the tiled host Enzyme gradients would have needed about twelve hours for the three cases. With
the dual sweep and its half-orbit truncation (`docs/notes/2026-09-06_dual_sweep.md`) the χ² and
gradient of a truncated movie come from one stored-sample cache on the GPU, and the whole
experiment runs in under an hour, most of it the CPU Fisher audit.

## Setup (unchanged from 2026-09-05)

Four thermal parcels (σ = 0.7 M in the plane, 0.15 M vertically) on Keplerian orbits at
r = 5–7 M with rigid pattern rotation, a = 0.94, θo = 17°, 230 GHz, M87's mass; two frames 25 M
apart; noise 1% of the peak in I and 0.5% / 0.5% / 0.2% in Q, U, V. The screen is a uniform 48²
grid over 16 M plus, for n ≥ 1, an annulus of fine pixels where Krang's n-th emission radius
falls in the source region (Δρ = 0.03 M for n = 1, 0.006 M for n = 2). The truth is rendered
with rays truncated after the n-th passage through the slab |z| < 0.6 M, and the model fitted
with the same truncation from a perturbed start (positions by 0.2–0.6 M, log densities,
temperatures and fields by 0.1–0.2, pattern rates by 2%), 300 Adam iterations at η = 0.005 with
a cosine schedule; t₀ and ln w frozen. The Fisher information of spin and inclination at the
truth is joint with the free parcel parameters (ForwardDiff duals through the whole pipeline,
tile by tile on the CPU).

The GPU path (`--backend cuda`) reproduces the CPU run of case n = 0 to twelve digits
(χ² 21289.2555940 on both, identical recoveries and Fisher errors), in 1.0 minute of fitting
instead of hours.

## Results (point-sampled screens, one point per pixel)

| case | pixels | samples | fit time | χ²/N start → end | position errors (M) | nₑ, Θe, B errors | pattern rate | σ(a) joint / alone | σ(θo) joint / alone |
|---|---|---|---|---|---|---|---|---|---|
| n = 0 | 2304 | 80 | 1.0 min | 15.4 → 1.155 | 0.01, 0.14, 0.34, 0.16 | 10–21%, 6–14%, 4–20% | 0.04–0.7% | 5.8e-4 / 1.9e-4 | 0.027° / 0.008° |
| n ≤ 1 | 12384 | 160 | 5.1 min | 12.7 → 1.055 | 0.014, 0.064, 0.032, 0.042 | 2–23%, 1–12%, 0.2–12% | 0.05–0.27% | 7.4e-5 / 3.0e-5 | 0.0021° / 0.0008° |
| n ≤ 2 | 20224 | 240 | 15.6 min | 7.2 → 1.019 | 0.073, 0.010, 0.020, 0.003 | 3–30%, 2–13%, 0.6–13% | 0.04–0.28% | 1.9e-5 / 1.0e-5 | 0.00077° / 0.00042° |

(Per-parcel numbers in `validation/winding/output/case_n/summary.txt`, untracked; the
cumulative sub-image fluxes on the uniform grid are 0.0514, 0.0534 and 0.0537 for n = 0, ≤ 1
and ≤ 2, so the first lensed passage carries 4% of the direct flux there and the second 0.5%.)

## Reading

- The direct image alone leaves the fit degenerate at the noise level: χ²/N reaches 1.155 with
  two parcels still 0.16 and 0.34 M from their true positions and densities off by 10–20%,
  because the data prefer a slightly different arrangement of the same total emission. Adding
  the first lensed passage pins the positions to a few hundredths of M for three of the four
  parcels and brings χ²/N to 1.055: the lensed image sees each parcel from a second direction.
- The Fisher errors of spin and inclination at the truth tighten by 8× and 13× from n = 0 to
  n ≤ 1, joint with the parcel parameters; the ratio joint/alone (≈ 3 in both cases) says how
  much the parcels' freedom costs the spacetime measurement.
- The second lensed passage adds another factor of four in the Fisher errors (σ(a) 1.9e-5,
  σ(θo) 0.0008° at this noise level, 30× and 35× tighter than the direct image alone) and pins
  three of the four positions to 0.003–0.02 M, although it carries only 0.5% of the direct flux:
  the fine annulus (Δρ = 0.011 M, 17,920 pixels) samples the n = 2 ring at its own width. The
  fit reaches χ²/N 1.019 in 300 iterations, closer to the noise floor than the two lower orders.
- A caveat on n ≤ 2: the fine annulus is point-sampled at Δρ = 0.011 M (its ring count is capped
  at 140), wider than the n = 2 ring itself (about 0.002 M), so the data and the model share a
  sampling that no instrument has. The self-fit is consistent on that screen; the pixel-integrated
  rerun below addresses this.
- The thermodynamic errors (nₑ, Θe, B at the 3–30% level) do not improve systematically with the
  order: they are the one-zone synchrotron degeneracy of a single frequency (the emissivity fixes
  a combination of density, temperature and field), which more lensing orders do not break and
  a second frequency would. Per-parcel numbers are not comparable across cases, because each
  case draws its own perturbed start. Daniel asked (2026-09-08) for the follow-up: a
  multifrequency fit of the n ≤ 1 case with frequencies above and below the parcels' turnover
  frequency, to see whether that pins down the emission properties (HANDOFF, open items).

## Pixel integration (2026-09-08, `Geodesics.Binning`, `--subsamples 2`)

Every uniform pixel integrated over 2 × 2 points and every annulus cell over 4 × 2 points in
(ρ, ψ), so that the n = 2 annulus samples its ring at 0.003 M; data and model share the
integration.

| case | pixels / points | fit time | χ²/N end | position errors (M) | Fisher σ(a) joint, duals | σ(θo) joint, duals |
|---|---|---|---|---|---|---|
| n = 0 | 2304 / 9216 | 0.9 min | 1.134 | 0.013, 0.15, 0.45, 0.17 | 3.1e-5 | 0.0066° |
| n ≤ 1 | 12384 / 89856 | 10.3 min | 1.059 | 0.001, 0.070, 0.035, 0.049 | 2.6e-5 | 0.0042° |
| n ≤ 2 | 20224 / 152576 | 32.2 min | 1.043 | 0.108, 0.007, 0.003, 0.010 | 1.2e-5 | 0.00058° |

The fits behave as on the point-sampled screens: the same χ²/N floors (the data are the binned
truth plus the same relative noise), the same parcels recovered to the same accuracy. The
Fisher numbers do not: for n = 0 the joint σ(a) at the truth is 5.8e-4, 3.1e-5, 2.2e-4 and 6.0e-4
for 1, 2, 3 and 4 points per pixel side (duals through the whole pipeline), and a
finite-difference audit of the spin and inclination columns (`--fisher fd`, step 1e-4, parcel
columns by duals through the transfer at fixed geodesics) reproduces the dual numbers at one
point per pixel (5.80e-4 vs 5.78e-4) but gives 5.7e-5 against 3.1e-5 at two. So the derivatives
are not glitching; the truncated images carry features sharper than any of these screens (the
cut edges where a ray's passage count changes, and the demagnified lensed structure), and the
information a screen sees in them depends on where its points fall. **The Fisher errors of
this experiment are therefore not converged in screen sampling, at any order, and the
tightening with n in the table above is an upper bound on what the sampling delivers, not a
measurement**; the recoveries of the fits are the robust result. Converging the spacetime Fisher
needs the instrument's own smoothing (a beam convolution of the binned image before the
residuals, or a soft slab edge), which is the natural next step for this experiment.

## Cost

Per iteration on the 2080 SUPER: 0.2 s for n = 0 (2304 rays × 80 samples × 2 frames), 1.0 s for
n ≤ 1 (12384 × 160 × 2) and 3.1 s for n ≤ 2 (20224 × 240 × 2): 0.6–0.7 ns per ray-sample-frame
of full polarized gradient. The Fisher audit (CPU, ForwardDiff duals, 12-wide chunks, tiled)
took 7.4, 6.3 and 8.0 minutes. The whole experiment, 45 minutes; the same three cases by tiled
host Enzyme were estimated at twelve hours on 2026-09-05. With 2 × 2 points per pixel (7–8×
the rays) the fits took 0.9, 10.3 and 32.2 minutes: the card is under-occupied by the
point-sampled screens, so pixel integration is nearly free at n = 0 and costs 2–2.5× at the
higher orders.
