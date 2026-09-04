# Phase 5: the real-data path — status (2026-09-04)

What now connects the pipeline to EHT data (PRs #40, #43, #44), and what the first run on M87
says.

## Reading uvfits natively

`Fit.read_uvfits` reads the AIPS random-groups files the EHT distributes through CFITSIO's
group routines (no Python, no Comrade): UU, VV scaled to wavelengths by the antenna table's
reference frequency, baselines decoded as 256 a₁ + a₂, both DATE parameters summed, the
correlation products averaged over channels and IFs without weights (ehtim's rule) and
converted to Stokes as I = (RR + LL)/2, Q = (RL + LR)/2, U = (RL − LR)/(2i), V = (RR − LL)/2
with the noise in quadrature. `Observation` keeps times, stations and integration times;
`VisibilityData(obs)` feeds the visibility χ², and `scan_triangles`/`scan_quadrangles` build
closure index sets from the simultaneous baselines of every time stamp (a negative index means
the conjugate row, which `closure_phases` and `log_closure_amplitudes` honour).

Gates: ehtim 1.2.10's parse of an ehtim-written polarized observation (the 2017 stations from
the real M87 antenna table, two Gaussians with Q, U, V, realistic SEFDs; the fixtures are in
`test/data`) is reproduced row by row (u, v to 2e-7, the float32 product ehtim forms; Stokes
visibilities to 1e-7 Jy; noise to 1e-9 Jy), and the real public M87 2017 Stokes-I file
(`SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits`) on all 7447 rows (`validation/uvfits`).

## The Fourier sign convention

ehtim's transform is V(u, v) = Σ I exp(+2πi(u l + v m)) with l the sky offset toward the east
(`ftmatrix`, "changed to agree with BU data (Jan 2017)": (u, v) of the baseline 1–2 is
(x₁ − x₂)/λ). EHT uvfits data are read by ehtim without a flip, so this is the convention in
which the data come; `visibilities` used the opposite sign until #43 and now follows it. The
gate: the direct transform of the source image (read with `read_stokes_fits`) reproduces
ehtim's noiseless visibilities to 6e-8 Jy, a mirrored sky fails by 1e-2, and the noisy
observation gives a reduced χ² of 0.98. `test_visibilities`' explicit reference transform
follows the same sign.

## Observers below the equator and negative spins (reflection parity)

M87's jet points toward us and the ring's southern side is brighter, which needs the observer
at 163° from the spin axis (or, equivalently, at 17° with the spin reversed). Neither regime
had been exercised in the polarized path. `test_reflection` (#44) pins both by symmetry: the
Kerr spacetime is invariant under z → −z and, with the spin reversed, under y → −y, so the
image from 163° must be the β-mirror of the image of the reflected source from 17°, and the
image at spin −a the α-mirror of the y-reflected source at +a, with the field mirrored as an
axial vector (B → −B on top of the geometric mirror), the mirrored velocity component reversed
and U, V changing sign. All four Stokes parameters agree to 1e-11 in both routes, for
optically thick and moving sources. Mirroring the field as a polar vector instead changes V
by more than 10% and Q, U by up to 40% in the thick case (Faraday rotation reverses with the
field), which is what a first attempt at this check found: the velocity components follow the
field's axes (r̂, φ̂, −θ̂; `u3` is the vertical one), now stated in the docstring.

## First fit to M87 (2017, day 101, band low)

`validation/m87_fit/m87_fit.jl`: six thermal splats started on a ring of radius 4.5 M in the
equatorial plane with a toroidal field and a toroidal ZAMO velocity of 0.35, fitted to the
closure phases (6251 triangles) and log closure amplitudes (3662 quadrangles) of the ten-second
data at a = 0.94, θo = 163°, M = 6.5e9 M⊙, D = 16.8 Mpc, on a 32² screen of 20 M (2.4 μas
pixels), with a Gaussian prior on the total flux (0.6 ± 0.06 Jy). Adam on Enzyme gradients,
300 iterations, on the CPU backend with six threads.

Run 1 (2026-09-04 00:31; the field and velocity started *vertical*, because the ordering of the
velocity components was only established by the reflection gate afterwards, and the flux prior
had σ = 0.06 Jy): 17 minutes for 300 iterations; the closure χ² per closure quantity fell from
29 to 1.37 (phases alone 1.76), the total flux drifted to 1.03 Jy (closures do not constrain it
and the prior was too weak against ~10⁴ closure quantities). The splats settled on a ring of
radius 4.1–5.4 M with nₑ ≈ 2–3e5 cm⁻³, Θe ≈ 16–25, B ≈ 7–10 G; two of them acquired
relativistic velocities (γβ ≈ 1.7) and show up as beamed streaks in the image, the rest form a
ring of ≈ 40 μas diameter.

The same image tested the Fourier sign convention end to end on real data: rotating the model by
180° (equivalent to the opposite sign, i.e. conjugate visibilities) raises the closure χ²/N from
1.37 to 21 and the closure-phase χ²/N from 1.76 to 33; the α- and β-mirrors give 8 and 12. The
convention adopted in #43 is the one in which EHT closure phases are reproduced.

Run 2 (toroidal field and toroidal velocity of 0.35 at the start, flux prior 0.6 ± 0.01 Jy, 400
iterations, 19 minutes): closure χ²/N = 1.54 at a total flux of 0.61 Jy (the tighter prior costs
0.17 in χ²/N against run 1's free flux). The six splats now form a coherent ring at r = 3.7–4.9 M
within ±1.4 M of the equatorial plane, all rotating in the same sense with a toroidal ZAMO
velocity γβ_φ ≈ 0.65–0.80 (0.55–0.6 c; the Keplerian value at 4.3 M is ≈ 0.5 c), an inflow
component of 0.1–0.35 and a vertical one of 0.15–0.35, and almost uniform plasma: nₑ = 2.1–2.2e5
cm⁻³, Θe = 20.6–21.7, B = 6.7–7.3 G, the canonical one-zone numbers for M87's ring. The image
(`figures/m87_2017_run2.png`) is a ≈ 40 μas ring brightest in the south-west. Only Stokes I
closures were fitted, so its polarization is a prediction of a thermal ring with a toroidal
field at this inclination: a net linear fraction of 17% and a net circular fraction of −6%,
both far above what the EHT's polarimetric papers report for M87, which says the field and
Faraday structure of this six-splat model are not those of M87; a run that fits Q, U (and V)
is the natural next step.

What this run does not do yet: vary the spin and inclination (both are available through
`fit_spacetime`), or model the extended emission.

## Self-calibrated polarimetric fits (`--mode selfcal`, PR #50)

The full-polarization file of the same night (`hops_3601_M87+netcal.uvfits`, all four
correlation products, 216 scan-averaged rows over 22 scans, 170 with Q, U, V) fitted on the
complex visibilities of all four Stokes parameters with one complex gain per station and scan
(the same gain for every Stokes parameter; no R/L gain ratio, no leakage), Gaussian priors on
the log-amplitudes, free phases, Adam on the splat parameters and the gains together:

| run | start | data | iterations | χ²/N (visibilities) | closure χ²/N | flux (Jy) | gain log-amplitude rms |
|---|---|---|---|---|---|---|---|
| 1 | toroidal ring, σ_gain 0.1 | all baselines | 400 | 330 | 82 | 2.97 | 0.72 |
| 2 | run 2 of the closure fit, σ_gain 0.05 | all baselines | 600 | 95 | 29 | 0.94 | 0.32 |
| 3 | run 2 of the closure fit, σ_gain 0.05 | \|uv\| ≥ 0.1 Gλ | 600 | 15 | 38 | 0.34 | 0.11 |
| 4 | run 3 continued (gains restarted), η_gain 0.03 | \|uv\| ≥ 0.1 Gλ | 2000 | 7.8 | 6.0 | 0.38 | 0.094 |
| 5 | run 3 continued, `--densify 16` (alternating epochs, gains restarted) | \|uv\| ≥ 0.1 Gλ | 600 | 30 | 66 | 0.14 | 0.33 |
| 6 | run 4 continued with its gains (`--init-gains`), η 0.01, η_gain 0.005 | \|uv\| ≥ 0.1 Gλ | 1000 | 7.0 | 4.6 | 0.38 | 0.096 |

Two lessons. The intra-site baselines (JC–SM at 0.1 Mλ, AA–AP at 1.6 Mλ) measure 1.2 Jy while
the baselines at 0.5–2 Gλ see 0.33 Jy: the jet contributes 0.6 Jy that a compact ring cannot
produce, and with those points in the fit the gains absorb a factor 1.4 per station (run 2).
Dropping them (run 3, the EHT practice of excluding intra-site data or adding a large-scale
component) brings the gains to an 11% spread but the fit is far from converged after 600
iterations (18 minutes): the visibility χ²/N is 15 and the closure χ²/N 38, worse than the
closure-only fit, because the amplitudes and the polarized visibilities with their small
errors now dominate and six thermal parcels with one field direction each cannot follow the
data's polarization structure. Run 4 (2000 more iterations from run 3's parcels, 33 minutes)
halves the visibility χ²/N to 7.8 and brings the closure χ²/N to 6.0 with the gains at a 9%
spread; the loss still oscillates by ±10% between iterations at that gain step, and the image
is a thin 40 μas ring with a few bright pixels, the sharpest structure six parcels can make.
Run 5 tried densification
through the generic `fit!` (#53) in alternating epochs of twenty splat iterations and twenty gain
steps, with the gains restarted from zero: the model flux collapsed to 0.14 Jy with the gains
compensating (a factor 1.4 per station) while the parcels split to 21, and the χ² stayed near 30,
a failure of the alternating scheme rather than of the densification (the gains must be carried
along, `--init-gains`, and stepped together with the parcels). Run 6 continues run 4 with its gains
carried along and smaller steps: χ²/N 7.0, closure χ²/N 4.6, the gains at a 10% spread, the
image (`figures/m87_2017_selfcal_run6.png`) a thin ring of 40 μas brightest in the south-east
with a net linear polarization of 4% and −1.6% circular; the descent is slow (0.8 in χ²/N per
thousand iterations), so six parcels are the limit of this run rather than the optimizer. The
structural next steps are a larger parcel set with densification in the joint update, R/L gains
and leakage terms, a smooth background component for the extended flux, and a noise model with
a systematic floor, i.e. the ingredients of the EHT polarimetric analyses.
