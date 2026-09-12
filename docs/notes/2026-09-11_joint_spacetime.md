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
from the same perturbed sky and compared by its final χ².

| start a, θo | end a | end θo | χ²/N after 60 iterations |
|---|---|---|---|
| 0.3, 45° | 0.19 | 56.2° | 1.64 |
| 0.3, 60° | 0.30 | 57.5° | 1.61 |
| 0.3, 75° | 0.38 | 57.7° | 1.74 |
| 0.6, 45° | 0.55 | 57.0° | 1.38 |
| 0.6, 60° | 0.62 | 56.9° | 1.68 |
| 0.6, 75° | 0.66 | 58.0° | 1.67 |
| 0.9, 45° | 0.80 | 59.8° | 1.28 |
| 0.9, 60° | 0.89 | 60.5° | 1.42 |
| 0.9, 75° | 0.94 | 59.4° | 1.62 |

The inclination is found from every start (56–60° for 60°, nearer the truth when the spin
is). The spin ends within 0.1 of where it started, whatever the inclination, and the final
χ² orders the spins only faintly (1.3–1.6 at a ≈ 0.9 against 1.6–1.7 at a ≈ 0.3, with the
fits not converged at 60 iterations). So on this movie the joint fit with free pattern rates
is nearly blind to the spin, and the multi-start does not rescue it because the basins are
not separated in χ²: the sky mimics the spin.

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
inclination is by the shape, in principle.

In practice, on this movie, it does nothing: with the prior at σ = 0.02 rad/M the fits from
(0.5, 45°) and (0.6, 60°) end at a = 0.45 and 0.62, θo = 56° and 57°, χ²/N 1.62 and 1.63,
the same as without it. Two reasons, both physical. The prior ties each rate to the
azimuthal coordinate velocity of the parcel's own fluid, whose ZAMO-frame velocity rows are
free, so the spin enters only through frame dragging, and at these radii that is small:
between a = 0.45 and a = 0.9 the Keplerian rate at 5 M differs by 0.003 rad/M, six times less
than the σ used, and the free velocities cover the rest. And the lensing signature of the
spin lives in the n ≥ 1 rings, which at 0.5 M per pixel this screen does not resolve. The
inclination survives both because it changes the whole image's shape. What would constrain
the spin is what constrains it in nature: emission near the ISCO (2–3 M, where the rates
depend on the spin strongly), pixels of a fifth of a gravitational radius to resolve the
rings, more frames of the fast inner motion, and, for the dynamics, a prior on the fluid
velocity itself rather than on the rate alone. That is the next self-fit, and it is the
first row of the uniqueness table the large-N program is after: which data constrain which
parameter.

The first of those runs (parcels at 2.5–5 M, 80² pixels over 16 M, 200 samples, six frames,
the pattern prior at σ = 0.005, from (0.5, 45°), 100 iterations in 27 minutes): the
inclination comes back to 58.6° (from 45°, for 60°), the spin goes from 0.5 to 0.43, and
χ²/N settles at 1.51 against 1.00 at the truth (153,600 values). Resolving the rings and
putting the emission at 2.5–5 M fixed the inclination to a degree and did nothing for the
spin, whose orbital signature the free velocities still absorb.

The run with Keplerian truth velocities and the fluid prior (σ_u = 0.05 with the pattern
prior at 0.005, the same screen and start, 19 minutes): χ²/N 4.08 → 1.24 with the
inclination at 58.0° and the spin at 0.438, unmoved since iteration 30, when the spacetime
Jacobian began coming back non-finite at every other visit (69 skipped steps, the state
written to a file each time). The prior did make the fit tighter (1.24 against 1.51), but
the spin question is not answered by this run, because the spacetime block stopped moving
at the moment it would have had to.

The replay of the dumped state found the singularity: one pixel of the 6400, one sample at
r = 2.20, a radial turning point of that ray, where Krang's momentum takes `√max(0, R)` and
the clamp's constant zero gives the square-root rule NaN partials on a dual; the frame's
redshift, pitch angle and polarization angle inherit them, and through them the whole
Jacobian. The frame now takes its momentum from `Transfer.momentum_bl_d` with the safe
square root (`test_turning_point_frame`). The rerun with the fix has no non-finite visits at
all, 75 accepted spacetime steps, and the same answer: χ²/N 1.244, θo = 57.9°, a = 0.438.
So the spin's stagnation was never the NaN. It is a local minimum of the joint problem: with
the velocities free and drawn to the Keplerian values of whatever spin the fit holds, the
prior is self-consistent at any spin, and from 0.5 the Levenberg–Marquardt block sees no
descent toward 0.9, although the truth fits 37,000 units of χ² better. The local optimizer
cannot cross that; a profile over the spin can, and its χ² separation is decisive. The spin
profile (`--fix-spin`: the spin held at each of 0.3, 0.5, 0.7, 0.9 and 0.98 with everything
else, the inclination included, fitted jointly) is the next run, and the shape of χ² along
it is the first entry of the uniqueness table: what these data say about the spin.

## The spin profile

The same movie (parcels at 2.5–5 M with Keplerian velocities and rates, 80² pixels of 0.2 M,
200 samples, n ≤ 2, six frames, 153,600 values at 2% noise in I), the spin held at each of
five values and everything else fitted jointly from the perturbed sky, the inclination
included (80 iterations each, about 25 minutes):

| spin held at | χ²/N at the end | θo | Δχ² from the minimum |
|---|---|---|---|
| 0.3 | 1.415 | 50.6° | 59,000 |
| 0.5 | 1.235 | 58.4° | 31,500 |
| 0.7 | 1.169 | 59.5° | 21,400 |
| 0.9 (the truth) | 1.030 | 59.6° | 0 |
| 0.98 | 1.086 | 58.3° | 8,600 |

The profile has its minimum at the truth, at the noise floor, and rises by 8,600 units of χ²
for 0.08 of spin on the high side and 21,000 for 0.2 on the low side: on this movie the spin
is determined to a few hundredths, and the inclination to a degree, from a start 0.4 off in
spin and 15° off in inclination. What the joint fit could not do from a single start, the
profile does, because the local optimizer's failure was a basin, not a degeneracy: the χ²
landscape along the spin is well separated once the sky is refitted at each value. This is
the first row of the uniqueness table, and the shape of the argument for the rest of it:
hold the parameter, refit everything else, read the curvature. (The point at a = 0.3 skipped
49 of its spacetime visits on a non-finite Jacobian of another kind, at a geometry far from
the truth; it is being replayed. Its inclination step was thereby hampered, which can only
have made 0.3 look worse than it is, and the points at 0.5 and 0.7, with no skipped visits,
carry the low side of the profile on their own.)

The replay of the a = 0.3 state found the other kind: the stored samples themselves. Two
rays of the 6400, a pair sharing η = 12.35 and λ = 3.21 (the pixels (α, β) and (α, −β) share
their conserved quantities and their radial motion), have NaN spin and inclination partials
in every radial sample from the first, with the polar samples finite: the ray sits on the
boundary between radial root cases at that spin, where the roots' dependence on the spin is
singular and a clamp in the root finding turns it into NaN on a dual. That is a genuine
singularity of the map from the spacetime to the ray, of measure zero, and the treatment is
the driver's: `spacetime_jacobian` now zeroes the rows whose values are finite and whose
partials are not, one pixel in thousands, and warns if they exceed one per thousand. The
per-cause guards (`safe_acos`, `momentum_bl_d`) remain for the cases that are removable.

## The uniqueness table, first rows

The same profile with the data degraded one axis at a time, the three points that carry the
spin's discrimination (the spin held at 0.7, 0.9 and 0.98; the inclination and the splats
fitted jointly; 153,600 values in every row so the χ² are comparable):

| data | χ²/N at 0.7 | at 0.9 (truth) | at 0.98 | Δχ² for +0.08 | Δχ² for −0.2 |
|---|---|---|---|---|---|
| I, Q, U, V with the n ≤ 2 images | 1.169 | 1.030 | 1.086 | 8,600 | 21,400 |
| I, Q, U, V with the n ≤ 1 images | 1.130 | 1.025 | 1.060 | 5,400 | 16,200 |

Removing the n = 2 ring takes away a third of the spin's discrimination on the high side and
a quarter on the low side; the spin is still determined to a few hundredths from the direct
and the n = 1 images alone at this resolution and with the dynamics. The next rows, the
Stokes parameters removed in turn (`--stokes IQU`, `--stokes I`), are running. UNIQUENESS
