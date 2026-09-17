# The triband ngEHT self-fit (2026-09-14)

Daniel's request (2026-09-13): an example self-fit to synthetic ngEHT data at 86, 230 and 345 GHz
with the most optimistic array of the reference-array paper (Doeleman et al. 2023, Galaxies 11,
107), an M87* campaign, each station observing the bands the paper assigns it, with animations.
The large-N program's data-domain instance: an over-complete shell of parcels fitted to the
campaign's visibilities at the three bands through the time-resolved likelihood.

## The campaign

`validation/ngeht/make_campaign.py` generates it with `ngehtsim` (Pesce et al.), whose `ngEHT`
preset is the paper's Phase-2 reference array: 21 stations (ALMA, APEX, BAJA, CNI, GAM, GLT,
HAY, IRAM, JCMT, JELM, KP, KVNYS, KVNPC, LAS, LLA, LMT, OVRO, NOEMA, SMA, SMT, SPT; BAJA, CNI,
JELM and LAS as 9 m dishes) with the receivers the paper gives them: ALMA, APEX and SMA at
230 GHz alone, HAY, IRAM, KP, OVRO and NOEMA at 86 and 230, the rest at all three bands. M87 in
April 2026 with 'good' weather from the package's site tables, 600 s scans every 1800 s over
24 h, 8 GHz of bandwidth, thermal noise only (no gains, no leakage: the self-calibration path
exists and is a later row). One uvfits per band and day; the source in the files is a
placeholder Gaussian whose visibilities the fit never sees, since `Fit.coverage` lends the
(u, v) sampling and the thermal noise of every scan to the synthetic movie.

The paper's 345 GHz detections rest on frequency phase transfer from the lower bands.
`ngehtsim`'s own multi-frequency FPT path breaks when the stations differ per band, so the
generator's `--fpt 600` lets the fringe finder integrate coherently over the scan at the bands
above 300 GHz instead (SNR 5 over 600 s rather than over 10 s), the proxy for it. A day then
gives:

| band | stations | visibilities | median σ | longest baseline |
|---|---|---|---|---|
| 86 GHz | 17 | 2,480 | 0.7 mJy | 3.4 Gλ |
| 230 GHz | 20 | 3,060 | 1.0 mJy | 8.4 Gλ |
| 345 GHz | 10–11 | 470–510 | 3.8 mJy | 11.7 Gλ |

(SPT never sees M87; CNI and GAM drop out at 345 GHz even in good weather; without the FPT
proxy the 345 GHz array is 9 stations and 250 visibilities reaching 5.3 Gλ.)

## The truth, the frames, the fit

The six-parcel truth of the large-N self-fits (`docs/notes/2026-09-10_large_n.md`: parcels at
2.5–5 M with Keplerian rates, random temperatures and fields around Θe = 30 and B = 20 G) at
M87's mass and distance (6.5e9 M☉, 16.8 Mpc; GM/c³ = 8.9 h), its densities scaled so that the
230 GHz flux density is 0.6 Jy, so the campaign's thermal noise applies as it is. Five days of
campaign are 13.5 M, a quarter of a Keplerian orbit at 4 M. The scans of every four hours share
a frame (0.45 M of motion), thirty frames per band over five days; the frame at each scan's time
is rendered on the 64² screen of 0.25 M pixels with 160 samples and the n ≤ 2 truncation, and
its visibilities on the scan's baselines get the campaign's noise (`Fit.synthetic_scans` with
`noise = nothing`).

The fit (`validation/ngeht/triband_selffit.jl`) starts from three hundred shell parcels of
0.25 M at 2.2–5.5 M with random fields, the densities scaled to the truth's flux, and runs the
hygiene schedule of the shell fits at the true spacetime; the three bands' time-resolved χ²
are summed, their gradients from the dual sweep with the frames batched eight to a launch
(`timeresolved_gradient!(...; batch_frames)`, new for this: a campaign of hundreds of scans
would otherwise launch one underfilled kernel per frame), in Float32 on the 2080 SUPER
(`Geodesics.precision` for the time-resolved data, new as well). The recovered fields are
compared with the truth on a voxel grid (`recovery_metrics`).

## Results

**The Float64 pilot** (48² pixels of 0.33 M, 120 samples, 200 shell parcels, 150 iterations,
frames per six hours, the five days; 80 minutes on the 2080 SUPER in Float64): χ²/N per band
from 57,600 / 15,200 / 1,200 at the start (230 / 86 / 345 GHz) to 232 / 89 / 16 at the end
against 1.00 / 1.01 / 0.97 at the truth, 200 → 72 parcels through five hygiene events, and on the
voxel grid the density's PSNR 19.7 → 21.2 dB, its relative error 0.104 → 0.087, the temperature
error 0.136 → 0.117 and the field error 0.161 → 0.184. Two things to read off. The recovered
fields are about what the image-domain shell fits reach at the noise floor (22.4 dB, 0.20 and
0.14 in `docs/notes/2026-09-10_large_n.md`), so the campaign's data constrain the plasma as
well as a 2%-noise movie does. But the χ² is 150 times the floor, and the last fifty iterations
moved it by one percent: at the campaign's sensitivity (a median σ of 1 mJy on a 0.6 Jy source)
the visibilities demand the image to a few tenths of a percent, where the movie fits needed two
percent, and Adam on the over-complete basis plateaus far above it. The distance to the floor is
the optimizer's, not the data's: the same truth lies in the basis, and the image-domain fits reach
it.

**The full run** (64² pixels of 0.25 M, 160 samples, 300 shell parcels, 300 iterations, frames per
four hours, the five days, 241,264 values; Float32 on the 2080 SUPER with the analytic seed, 5.0
hours; a first attempt with the Enzyme host seed was stopped after 21 hours without finishing):

| | 230 GHz | 86 GHz | 345 GHz | all |
|---|---|---|---|---|
| values | 122,368 | 99,440 | 19,456 | 241,264 |
| χ²/N at the truth | 0.996 | 1.005 | 0.974 | 0.998 |
| χ²/N at the start | 51,600 | 18,000 | 1,060 | 33,700 |
| χ²/N at the end | 44.2 | 21.5 | 4.5 | 31.6 |

300 → 74 parcels through nine hygiene events. This run is not converged, and its numbers are an
interim record, not a result: the χ² history (`triband_history.csv`) reads 33,700 → 1,455 at
iteration 50, 2,400 when the second stage frees the fields at 100, 83 at 200, 42 at 250 and
31.5 at the end, still falling by a third per fifty iterations when the cosine schedule closes.
The truth scores 1.0 on the same data, so the floor is reachable; a converged fit of the
over-complete basis is expected to reach it with a plasma distribution that need not be the
truth's (Daniel, 2026-09-15), which is why recovered-field metrics are not quoted for this run.
It was resumed from its end state (`--resume`, one stage with everything free, 600 iterations
at η 0.01 → 1e-4, 9.8 hours): χ²/N 31.6 → 7.44 (9.4 / 5.9 / 2.7 at 230 / 86 / 345 GHz), 74 → 70
parcels, and not converged either: the last hundred iterations moved it by 3% as the step
closed, and the two hygiene events of the run (iterations 100 and 300) each threw the χ² up by
an order of magnitude (776 and 63) with a hundred iterations spent recovering. So Adam with a
decaying step on the over-complete basis stalls an order of magnitude above the floor, and the
hygiene schedule, useful early, is harmful late. Two follow-ups: a continuation with hygiene
off at a small constant-ish step, to separate the schedule from the curvature; and the
principled tool for the last decade, a Gauss–Newton polish on the time-resolved residuals on
the card, matrix-free (J·v by a directional pass, Jᵀr by the adjoint sweep in hand, conjugate
gradients between them). The recovered-field numbers are withheld until a fit reaches the
floor. The movie of the resumed run is `viz/output/triband_resumed_movie.mp4`.

The continuation with hygiene off (600 more iterations at η 0.003 → 3e-4, 9.5 hours): χ²/N
7.44 → 3.09 (3.83 / 2.37 / 2.05 per band), monotone, no spikes, still falling by 3% per hundred
iterations at the end. So the hygiene schedule was most of the stall, and Adam alone keeps
descending but slowly: three times the floor after 1,500 iterations in all.

**The Gauss–Newton polish on that state** (ten Levenberg–Marquardt steps of twenty
unpreconditioned conjugate-gradient iterations, Float32, 63 minutes): χ²/N 3.09 → 2.93, every
step accepted and the damping divided by three each time down to 2e-7, the χ² moving by a
third of a percent per step. The damping was never the limit; the linear solve was: twenty
iterations of plain conjugate gradients on rows that span decades of scale (positions in M,
logarithms, angles, rates in rad/M) resolve only the stiffest directions, and the step is a
fraction of the Gauss–Newton step. Per wall-clock hour this matched Adam's late descent and
did no better. The polish now preconditions the iterations with a Hutchinson estimate of
the diagonal of JᵀJ (sixteen probes, each a dual tails pass and an adjoint sweep), which also
serves as the Marquardt diagonal, and moves the damping by the gain ratio of the actual to the
model's decrease.

**The preconditioned polish** (four steps of sixty iterations, sixteen probes, 93 minutes, from
the plain polish's state): two steps rejected at a damping of 1e-3 and 1e-2 (the first with the
conjugate-gradient residual five times its start, a runaway step along a null direction of the
over-complete basis in single precision, the quadratic model predicting a quarter of the χ²
and the χ² rising instead), then two accepted at 0.1 and 0.03 with gain ratios 0.96 and 0.86:
χ²/N 2.93 → 2.81 → 2.61, four and seven percent per step against a third of a percent for the
plain iterations, twenty minutes per step. The iterations now stop on a non-positive curvature
estimate or a stalled residual, their recurrences run in Float64 around the Float32 products,
and the diagonal is refreshed every fourth step.

**Twelve guarded steps from that state** (104 minutes, nine per step): every step accepted with
gain ratios 0.15–0.90, χ²/N 2.61 → 2.05 (2.45 / 1.63 / 1.68 per band), two to three percent
per step, no faster per hour than before, and the linear solves poor (the residual at the stall
guard 0.3–1.2 of its start). The conjugate gradients on the matrix-free normal equations
resolve a few dozen directions of 1,470 per step. The problem is small enough for the
explicit Jacobian: `jacobian!` forms it by tails passes on duals with eight partials (eight
columns per pass, the parcel lists once per frame chunk), a few gigabytes of host memory for
the campaign's residuals, and `polish_dense!` takes the exact damped step from JᵀJ and Jᵀr
accumulated in Float64, trying several dampings on the same Jacobian; the last normal matrix's
inverse is the Laplace covariance.

**The first dense run** (six iterations, 53 minutes per Jacobian of 184 passes, 5.3 hours in all)
made no step: every damping from 1e-2 to 1e4 was rejected with the trial χ² near 1e10 even where
the model predicted a decrease of a few hundred, one step at a damping of 3e3 gained 600, and
the damping then ran to 1e27 over four iterations of nothing. The signature of flat columns: a
parameter the residuals barely see (a parcel of no flux, the rate of a parcel at rest) has a
Jacobian column of single-precision noise, a diagonal entry of 1e-14 of the largest, and the
damped solve sends it anywhere the model considers free, into a state the transport cannot
render. The Marquardt diagonal is now floored at 1e-6 of its largest entry (the Hutchinson
version had this floor from the start) and the iterations end once the damping passes 1e6.

**The dense run with the floor** (three iterations, 2.6 hours): every step accepted at a
damping of 0.1 and 0.03 with gain ratios 0.73, 0.77 and 0.58, χ²/N 2.05 → 1.95 → 1.87 → 1.77
(2.08 / 1.45 / 1.54 per band), the largest step 0.08–0.14 in the parameters' units, the
diagonal of JᵀJ spanning forty decades. So the exact solve gains five percent per iteration
where the preconditioned partial solves gained two to three, at six times the cost: at this
damping the step is set by the model's nonlinearity, not by the solver, and the cheaper solver
wins per hour.

**Forty preconditioned steps from the 1.77 state** (forty iterations each, the stall guard at
twenty, 7.9 hours): every step but one accepted, χ²/N 1.77 → 1.25 (1.31 / 1.17 / 1.28 per band
against the truth's 1.00 / 1.01 / 0.97), the descent steady at one to two percent per step
with no sign of a floor (1.62 at step 4, 1.46 at 12, 1.38 at 20, 1.32 at 28, 1.25 at 40), the
damping wandering between 1e-4 and 7e-2 with the gain ratio.

**Sixty more** (12.7 hours): χ²/N 1.25 → 1.11 (1.14 / 1.07 / 1.14 per band), every step
accepted with gain ratios of 0.8–0.95, but the quadratic model predicting decreases of a few
hundred on a χ² of 268,000 and the last six steps moving it by 0.4% in all, the damping at its
floor of 1e-8 and the solves' residuals ending at 0.2–5 of their start. A gain ratio near one
with a tiny predicted decrease says the step is the solver's, not the model's: conjugate
gradients on the normal equations of the over-complete basis (exact null directions, Float32
products) diverge, and the step returned is whatever the last iterate held. The solve is now
LSQR on the Jacobian scaled by D^{-1/2} (`lsqr_step!`: the same J·v and Jᵀu per iteration,
the bidiagonalization in Float64, a monotone residual, gated against the dense damped solve on
the explicit Jacobian to 1e-11); its run from the 1.11 state follows.

## The spacetime free in the data domain

`fit_joint!` now takes, in place of a movie, a vector of `BandScans` (a band's frequency, its
time-resolved scans, the pixel size and the distance): the splats' gradient is the batched
time-resolved sweep summed over the bands, and the spacetime block's Levenberg–Marquardt step
takes its residuals from the scans of every `lm_every`-th frame rendered on the dual cache
(`spacetime_scan_residuals`), since a dual render of every frame of a campaign would cost as
much as the sweep itself. The gate `test_joint_scans` checks the Jacobian of the scan residuals
against finite differences of the summed χ² (2e-5) and runs 24 joint iterations on two bands of
three scans (χ² down seventeenfold, the spin and inclination near the truth). The triband driver
runs it with `--free-spacetime 1` (Float64, the geodesics being regenerated at every iteration;
the pattern and Keplerian priors tied to the current spin as in the image-domain joint fits). At
this card's speed the full-size joint run is a ten-hour job; it is the first job for a cluster
node. For the visibility scans the host seed of each frame's sweep is the adjoint transform of
the weighted residuals (`visibility_seed!`, threaded over baselines) rather than an Enzyme pass,
which was the fit's bottleneck: the card sat idle between launches.

## The likelihood on the card and the Gauss–Newton polish

With the analytic seed the host's share of a time-resolved iteration was still the largest: ninety
frames' direct transforms and adjoints at a second each against twenty seconds of sweeps, the
card idle between launches. `src/Fit/device_visibilities.jl` moves the visibility likelihood onto
the backend: the scans of a frame concatenated as `FrameScans` (baselines, observed Stokes
visibilities, their noise, the scattering taper, in the scalar type of the fit), `vis_kernel!`
(the direct transform of the frame's image onto the baselines, one thread per baseline over the
sorted pixel order the tails kernel leaves), `seed_kernel!` (the adjoint transform of the
weighted residuals into the sweep's seed) and `frame_chi2_seed!` (both, and the χ²). The batched
sweep takes the forward images from `sweep_passes` on the device and never copies them; the
gate `test_timeresolved` holds the device path to the host path at 1e-9, and the Float32 χ² is
compared through stored samples only (a fused Float32 march recomputes the geodesics in single
precision, 9% off at 12² × 40 on the card).

The polish (`src/Fit/gauss_newton.jl`) is Levenberg–Marquardt on the same residual vector, the
Jacobian never formed. J·v comes from one tails pass with the parameters as one-partial duals
seeded along v (the tails kernel's accumulator is typed by the tails array, so a dual parameter
matrix gives the image's directional derivative; the parcel lists come from the values) followed
by the frame's device transform of the dual image; Jᵀw comes from the adjoint sweep seeded by the
adjoint transform of w (`seed_kernel!` with weights w/σ); conjugate gradients on
(JᵀJ + λ diag) p = −Jᵀr give the step, the diagonal being |Jᵀr| per row as the Marquardt scale,
and λ moves with the outcome (÷3 on a decrease, ×10 on a rejection). The gate `test_gauss_newton`
checks J·v against finite differences (2e-9), the adjoint identity ⟨Jv, w⟩ = ⟨v, Jᵀw⟩ (3e-14) and
that two steps lower a stalled χ² (10,885 → 2,946). Each conjugate-gradient iteration costs a
tails pass on duals and a full gradient sweep, so a step of twenty iterations costs about forty
Adam iterations; the driver's `--polish N --cg K --lambda λ` runs it on a `--resume`'d state at
the held spacetime.

## The animations

`viz/triband_movie.jl` renders the campaign frame by frame: the truth and the fit at the three
bands with their polarization ticks, the (u, v) coverage filling in scan by scan, the midplane
density and temperature of the truth and the fit, and the χ² trace as a still; MP4 and six
stills under `viz/output/` (not tracked).
