# Fitting the spacetime in parallel with the splats (2026-09-11)

Daniel, 2026-09-11: fitting spacetime parameters in parallel with the splats is crucial. What
existed was `fit_spacetime`, Levenberg–Marquardt on spin and inclination at fixed splats on
the CPU, with the Jacobian by forward-mode duals through the geodesic cache and the transport
(gate 4, 1e-5 against finite differences). This note adds the joint fit.

## Mixed mode, block-wise, one loop

`Fit.fit_joint!(params, x, movie, cache, camera; L, ...)` fits the splat matrix and the
spacetime block `x = [a, θo]` (or `[a, θo, ln L]`) to a Stokes movie. Every iteration:

1. the Float64 stored samples of `cache` are regenerated at the current spin and inclination
   (one recurrence march on the fit's backend);
2. the splats take one Adam step from the dual sweep over those samples (`chi2_gradient!`,
   with the per-row step multipliers and the priors of the movie fits);
3. after `warmup` iterations, the spacetime takes one Levenberg–Marquardt step with the
   splats held: `spacetime_jacobian` renders the residuals on a cache whose scalars are
   two- or three-partial duals, regenerated with the fused marcher at the dual spin and
   inclination, so the Jacobian with respect to `x` comes out of one pass costing a few
   marches' worth of work independent of the pixel count; the damped Gauss–Newton step is
   clipped to the bounds and accepted when one Float64 fused pass at the trial spacetime
   lowers the χ².

Gauss–Newton is the right optimizer for a block of two or three parameters whose curvature
the duals give for free. The first version of the loop took an Adam step on the spacetime
from the dual gradient (`spacetime_valgrad`, exact to 1e-9 against finite differences) and
did not work: the splats adapt to the wrong spacetime within a few iterations, and the
spacetime gradient then flips sign from one iteration to the next as the sky moves under it,
so a fixed-size normalized step wandered (θo moved 0.8° of the 8° it was off in 60
iterations while χ² fell fifty-fold through the splats). The curvature-scaled step does not
have that problem: it moves the spacetime to the best fit of the current residuals in one
step, and the accept test keeps it honest when the sky is still bad.

## Mass

Mass enters data in M units only through the length unit L = GM/c², which scales the
emissivity along the path exactly as the densities do: where the emission is optically thin,
ln L is degenerate with ln nₑ. It is identifiable through absorption and Faraday depth, and,
for data in physical units, through the angular scale (the pixel size Δα L/D) and the frame
times, which is where the mass constraint of a real observation lives. The M-unit movie form
therefore fits `[a, θo]` with L given; the three-parameter block exists for the physical-unit
path and its derivatives are gated with the others.

## The dual march on CUDA

The dual-typed geodesic cache had only ever run on the CPU backend. On CUDA it runs as is:
at 32² pixels and 60 samples the two- and three-partial Jacobians of the four Stokes totals
of a two-parcel image with respect to (a, θo[, ln L]) agree with the CPU's to every printed
digit, and cost 0.17 and 0.21 s per evaluation on the 2080 SUPER after compilation (the
Float64 pass 0.07 s). The per-thread stack sizing already scaled with the dual width.

The dual cache can also store its samples: the recurrence marcher runs once on dual scalars
and every frame then renders through the tails kernel (`render_frame!`), so the spacetime
block costs one dual march per visit instead of one per frame. Same derivatives to the digit
(the gates below), and the CPU joint test runs 2.3 times faster with two frames.

## The schedule decides

The CPU test case (two parcels, 8² pixels at 2.25 M, 40 samples, two frames at 230 GHz, the
truth at a = 0.9 and θo = 60°, the fit started at 0.8 and 52° with the parcels perturbed by
0.05 in every row, 60 iterations at η = 0.03):

| schedule | χ² at the end | a | θo | trace of θo every 10 iterations |
|---|---|---|---|---|
| no warmup, 1 LM step per iteration | 585 | 0.917 | 57.5° | 52.0, 57.3, 57.4, 57.9, 57.7, 57.5 |
| no warmup, 3 LM steps per iteration | 635 | 0.922 | 60.4° | 52.0, 59.8, 60.3, 60.5, 60.5, 60.4 |
| warmup 5, 3 LM steps | 620 | 0.801 | 53.8° | 52.0, 53.8, 55.0, 53.8, 53.8, 53.8 |
| no warmup, 3 LM steps, splats held at the truth | 548 | 0.901 | 59.9° | 52.0, 59.9, 59.9, 59.9, 59.9, 59.9 |

The three χ² values of the free-sky rows are the same fit to the noise (512 values); what
differs is where the spacetime went. With five iterations of warmup the splats absorb the
inclination error and the spacetime never recovers it, a local minimum that the joint problem
has and the fixed-sky problem does not. With the spacetime moving from the first iteration
and three inner steps it lands at 0.4° and 0.02 in spin, as good as the fit with the splats
held at the truth. The defaults of `fit_joint!` are therefore no warmup and three inner steps,
and the rule they express is general: when two blocks are fitted jointly and one can mimic
the other, the block that cannot be mimicked has to lead.

`test_joint_fit` pins the machinery and the recovery: the dual gradient of the stored-sample
χ² against central finite differences (1e-9), the dual residuals squaring to the same χ² with
2Jᵀr equal to the gradient, and the joint fit at 8² × 40 recovering a = 0.922 and θo = 60.4°
from 0.8 and 52° in 60 iterations (168 spacetime steps accepted); at the CI size (6² × 16,
30 iterations) 58.0° and 0.78, within the looser bounds the CI asks for.

## Results at scale

The GPU self-fit (`validation/joint/joint_selffit.jl`): six parcels on orbits at 4–7 M with
Keplerian pattern rates, at a = 0.9 and θo = 60°, rendered with the n ≤ 2 sub-images
(40² pixels over 20 M, 160 samples, four frames over 60 M at 230 GHz, 25600 values, noise at
2% of the peak in I), fitted from a = 0.5 and θo = 45° with the truth perturbed by 0.1 in
every row. The first run (150 iterations, η = 0.02, 35 minutes) went to θo = 54.7° by
iteration 20 and then froze at 53.0° and a = 0.49 while χ²/N crept from 2.4 to 1.92 through
the splats (1.01 at the truth). Two things were wrong. The Levenberg–Marquardt damping was
carried from one visit of the spacetime block to the next, so once a few rejections had
grown it, every later step was accepted and microscopic: it now restarts at every visit. And
a sky perturbed by 0.1 in every row (positions, log sizes, orientations, log densities,
angles) is far from the truth, far enough that the χ² at the true spacetime with that sky
(164k) is above the χ² at the wrong one (129k), so nothing pulls the spacetime home until
the sky has improved, by which time it has adapted. The joint problem from a rough sky is
the fresh-start problem, and it has two honest answers: the spacetime leading with a slow
sky (a small η early), and several spacetime starts compared by their final χ², which here
are well separated (1.9 against 1.0).

The second run (the damping reset, the truth perturbed by 0.05, η = 0.01, 120 iterations, 12
minutes with the stored dual cache) settled by iteration 30 at a = 0.45 and θo = 56° with
χ²/N 1.56: a joint local minimum, the sky fitting the wrong spacetime three times worse than
the truth fits. From iteration 72 on every spacetime Jacobian came back non-finite (the guard
skipped the steps and the splats went on): a state-dependent singularity that the probe at
fixed skies never reproduced on either backend. The one boundary-singular function on the
dual path was the pitch angle, `acos(clamp(cos θB, −1, 1))`, whose partials are NaN whenever
the clamp engages; it is now `safe_acos`, and the guard writes the state to a file when it
fires so that the next occurrence can be replayed. The basin question is the 3 × 3 grid of
spacetime starts, a ∈ {0.3, 0.6, 0.9} × θo ∈ {45°, 60°, 75°}, each fitted for 60 iterations
from the same perturbed sky and compared by its final χ². GRID_RESULT

## Where the spin lives

The first three grid runs, all from a = 0.3, ended at a = 0.19, 0.30 and 0.38 with θo at
56–58° and χ²/N 1.6–1.7: the inclination is found from any start, the spin barely moves from
where it began. In these fits every row of every parcel is free, including the pattern rates,
and the truth's rates are the Keplerian rates of its orbits at a = 0.9. Free rates absorb the
spin's effect on the motion, and what remains for the spin is the lensing, which four frames
of a 40² screen constrain weakly. The dynamics are the other handle: the pattern prior
(`PatternPrior`, mode C as a soft constraint) ties each parcel's rate to the Keplerian rate
of its fluid at the current spin, and `fit_joint!(...; pattern = σ)` now rebuilds it at every
iteration, feeds it to the splat gradient through the priors and to the spacetime block as
residuals whose partials with respect to the spin come through the metric (gated against
finite differences in `test_joint_fit`). With it the spin is constrained by the motion as the
inclination is by the shape. PATTERN_RESULT
