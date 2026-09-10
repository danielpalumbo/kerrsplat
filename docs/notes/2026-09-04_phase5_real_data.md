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

## The full-polarization data through the instrument model (2026-09-10, `m87_jones.jl`)

The same `hops_3601_M87+netcal.uvfits` file refitted with the instrument model that follows
Comrade.jl's structure (`docs/STATUS.md`, `Fit.InstrumentModel`; Daniel's request of
2026-09-09): the data are the four correlation products themselves (RR, LL, RL, LR, kept by
the reader and averaged on their own over scans, so that the JCMT's single hand enters), the
model V = J₁ C J₂† with J = R† G D R (per-scan complex R and L gains with ALMA as the phase
reference, d-terms per station over the track, the feed rotation of the EHT array from the
antenna table), Comrade's priors (log amplitudes N(0, 0.2), the LMT 1.0; the L/R ratios
N(0, 0.1); d-term parts N(0, 0.2)) and a 2% fractional noise floor on the products, as Comrade's
tutorials add. Baselines below 0.1 Gλ are dropped as in the earlier runs (the intra-site
baselines see the jet's extended flux), leaving 189 rows, 22 scans, 1330 product values,
372 free gain entries and 28 d-term parts. The sky is the six-parcel model of self-calibration
run 6 above, its eighteen non-temporal rows free, fitted jointly with the instrument
(`Fit.selfcal!`, 300 Adam iterations from a unit instrument, one dual sweep per iteration on
the GPU) and polished by six joint Levenberg–Marquardt iterations with the Laplace covariance.

| variant | χ²/N of the products, start → end | closure χ²/N of the sky alone, start → end (240 quantities) | flux (Jy) |
|---|---|---|---|
| leakage, corrected feeds, σ_lg 0.2 (baseline) | 275 → 1.33 | 3.86 → 3.79 | 0.373 |
| no leakage | 275 → 4.08 | 3.86 → 4.14 | 0.369 |
| tight amplitude priors, σ_lg 0.05 | 275 → 1.48 | 3.86 → 3.68 | 0.402 |
| instrument alone, sky fixed | 275 → 2.78 | 3.86 | 0.382 |

The instrument model with leakage is what the data need: without d-terms the products stay at
χ²/N 4.1, and the fixed-sky variant shows that most of the descent is the instrument's (2.8 with
the starting sky). The sky's own closure χ² barely moves (3.86 → 3.79; the closure-only fit of
2026-09-04 on the Stokes-I file, with 4,096 closures rather than 240 scan-averaged ones,
reached 1.5), so the products are fitted through the instrument rather than by moving the
parcels, which is the behaviour Comrade's priors are designed to give. The gain amplitudes stay
near unity (0.89–1.13 per station; the LMT's scatter 0.19 under its wide prior) and the flux is
the starting sky's; the phases are free (rms 0.4–1.6 rad per station).

The d-terms per station after the polish, with their Laplace errors (baseline):

| station | D_R | D_L |
|---|---|---|
| AA | −0.029 − 0.018i (±0.007) | +0.047 − 0.003i (±0.006) |
| AP | +0.015 + 0.061i (±0.013) | −0.033 + 0.036i (±0.013) |
| AZ | +0.086 − 0.046i (±0.005) | −0.105 − 0.043i (±0.006) |
| JC | (no R feed) | −0.139 + 0.037i (±0.022) |
| LM | +0.028 − 0.022i (±0.005) | −0.001 − 0.008i (±0.007) |
| PV | +0.006 + 0.112i (±0.012) | +0.028 + 0.123i (±0.016) |
| SM | +0.092 + 0.066i (±0.06) | −0.173 + 0.014i (±0.022) |

Magnitudes of a few per cent at ALMA and the LMT and of ten per cent at the SMT, Pico Veleta
and the SMA, with formal errors of 0.5–1.5% (the SMA's R feed 6%), the range of the 2017
D-terms of EHT Collaboration (2021, Paper VII). What this run does not have: an amplitude
reference (the flux is fixed only by the sky model), and the priors' widths tuned to this data
set rather than Comrade's defaults.

### Against the published D-terms: the missing ALMA feed phase

Station by station, the d-terms above are the published ones (Paper VII, Tables 3–5) rotated by
−90° for R and +90° for L at every station: D_R,ours = −i D_R,pub, D_L,ours = +i D_L,pub. This
is what a global R−L phase of 90° in the data does (RL → −i RL with the sky's EVPA rotated by
−45°; the algebra is gated in `test_crosshand_rotation`), and Paper VII's Appendix D names the
phase: ALMA's Band 6 feeds are rotated by 45° with respect to their projection on the focal
plane, which leaves a phase offset between the post-converted RCP and LCP signals, applied by
the collaboration as a global phase to the RL (LR*) products "before performing the analysis".
The HOPS netcal file lacks it; `Fit.rotate_crosshands(obs, π/2)` applies it (the script's
`--crosshand-phase 90`), and the refit then gives, against the April 11 low-band values of the
five imaging and posterior-exploration methods of Table 5 (LMT, SMT, PV) and the campaign
intra-site values of Tables 3–4 (ALMA, APEX):

| station | ours, D_R (%) | published D_R | ours, D_L | published D_L |
|---|---|---|---|---|
| AA | +0.5 − 5.8i (±0.8) | along −i, amplitudes 2.6–7.1 over the campaign (Table 3) | −0.5 − 5.1i (±0.7) | along −i, 2.8–6.1 |
| AP | −8.7 + 3.1i (±1.3) | −8.67 + 2.96i (±0.70) | +4.0 + 3.1i (±1.3) | 4.66 + 4.58i (±1.20) |
| AZ | +3.0 + 9.4i (±0.6) | 2.9–4.1 + 6.9–8.8i (five methods) | −2.6 + 8.1i (±0.6) | −3.9 to −5.9 + 9.3–11.0i |
| LM | +0.6 + 3.9i (±0.5) | 0.7–2.8 + 0.5–4.4i | −0.2 + 1.0i (±0.7) | −0.4 to −1.4 + −0.5–0.9i |
| PV | −12.2 − 2.3i (±1.1) | −11.3 to −14.2 + −1.2–3.6i | +15.9 − 1.5i (±1.5) | 12.9–16.2 + −1.6–1.6i |

ALMA's d-terms come out along the negative imaginary axis with similar amplitudes, as the
paper says they must (the X–Y phase offset of the linear feeds); APEX's D_R agrees to 0.1%;
the SMT, PV and LMT agree within 1–3%, the spread between the published methods. The SMA and
the JCMT are not comparable in this fit (seven and nine scans, the JCMT's single hand, and the
SMA's extra R–L phase rotation for its own feed offset that the paper applies separately).
The unrotated fit had the sky's EVPA off by 45°: any polarimetric result from the 2017 HOPS
netcal products needs this rotation. With it the products' χ²/N is 1.42 (the sky of run 6 was
fitted in the unrotated frame and has not fully re-rotated in 300 iterations; the closure χ²
3.90). Gate `test_dterm_recovery`: ehtim's own seeded d-terms are recovered from its corrupted
products through this fit to 3e-12, which pins the convention on the simulator's side.

### With an amplitude anchor (`--flux 0.6 --sigma-flux 0.01`, 2026-09-10)

The instrument fit's amplitude scale is held only by the gain priors around unity; a Gaussian
prior on the model's total flux density (an `image_prior` of the time-resolved likelihood,
`total_flux`) anchors it. With the rotation and a 0.6 ± 0.01 Jy prior, 300 iterations give
χ²/N 1.59 at 0.46 Jy (the sky of run 6 still re-rotating into the absolute frame) and 1000
iterations 1.14 at 0.56 Jy, with the sky's own closure χ²/N down from 3.86 to 2.50: the sky
now carries the amplitude and the absolute EVPA. The gain amplitudes settle at 0.74–0.94 (the
netcal amplitude scale of this file sits some 20% below the prior's flux, within the a-priori
calibration uncertainties the gain priors allow) and the d-terms move by up to 2%
(ALMA −0.1 − 3.6i, APEX −6.5 + 3.3i, SMT +2.8 + 9.5i, LMT +1.7 + 4.9i, PV −11.8 + 0.2i for R),
the SMT and PV within 1% of the published values and the others within 2–3%.
