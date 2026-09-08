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

## Results

| case | pixels | samples | fit time | χ²/N start → end | position errors (M) | nₑ, Θe, B errors | pattern rate | σ(a) joint / alone | σ(θo) joint / alone |
|---|---|---|---|---|---|---|---|---|---|
| n = 0 | 2304 | 80 | 1.0 min | 15.4 → 1.155 | 0.01, 0.14, 0.34, 0.16 | 10–21%, 6–14%, 4–20% | 0.04–0.7% | 5.8e-4 / 1.9e-4 | 0.027° / 0.008° |
| n ≤ 1 | 12384 | 160 | 5.1 min | 12.7 → 1.055 | 0.014, 0.064, 0.032, 0.042 | 2–23%, 1–12%, 0.2–12% | 0.05–0.27% | 7.4e-5 / 3.0e-5 | 0.0021° / 0.0008° |
| n ≤ 2 | CASE2 |

(Per-parcel numbers in `validation/winding/output/case_n/summary.txt`, untracked; the
sub-image fluxes on the uniform grid are 0.0514 for n = 0 and 0.0534 cumulative for n ≤ 1, so
the first lensed passage carries 4% of the direct flux there.)

## Reading

- The direct image alone leaves the fit degenerate at the noise level: χ²/N reaches 1.155 with
  two parcels still 0.16 and 0.34 M from their true positions and densities off by 10–20%,
  because the data prefer a slightly different arrangement of the same total emission. Adding
  the first lensed passage pins the positions to a few hundredths of M for three of the four
  parcels and brings χ²/N to 1.055: the lensed image sees each parcel from a second direction.
- The Fisher errors of spin and inclination at the truth tighten by 8× and 13× from n = 0 to
  n ≤ 1, joint with the parcel parameters; the ratio joint/alone (≈ 3 in both cases) says how
  much the parcels' freedom costs the spacetime measurement.
- CASE2_READING

## Cost

Per iteration on the 2080 SUPER: 0.2 s for n = 0 (2304 rays × 80 samples × 2 frames), 1.0 s for
n ≤ 1 (12384 × 160 × 2), CASE2_COST. The Fisher audit (CPU, ForwardDiff duals, 12-wide chunks,
tiled) took 7.4, 6.3 and CASE2_FISHER minutes.
