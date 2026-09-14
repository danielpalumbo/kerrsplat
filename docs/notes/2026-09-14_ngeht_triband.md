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

RESULTS_PENDING

## The animations

`viz/triband_movie.jl` renders the campaign frame by frame: the truth and the fit at the three
bands with their polarization ticks, the (u, v) coverage filling in scan by scan, the midplane
density and temperature of the truth and the fit, and the χ² trace as a still; MP4 and six
stills under `viz/output/` (not tracked).
