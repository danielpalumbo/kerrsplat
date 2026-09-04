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

Results: pending (the run is in progress; this section is updated when it finishes).
