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
descending but slowly: three times the floor after 1,500 iterations in all. The Gauss–Newton
polish (`polish_timeresolved!`) runs on this state next.

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

## The animations

`viz/triband_movie.jl` renders the campaign frame by frame: the truth and the fit at the three
bands with their polarization ticks, the (u, v) coverage filling in scan by scan, the midplane
density and temperature of the truth and the fit, and the χ² trace as a still; MP4 and six
stills under `viz/output/` (not tracked).
