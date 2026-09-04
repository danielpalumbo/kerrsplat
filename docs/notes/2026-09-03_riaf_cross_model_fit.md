# Cross-model recovery: splats fitted to ipole's RIAF image (2026-09-03)

Plan §7.5 item 7 (ii) asks for a fit of the splat representation to a movie rendered by an
independent code from an independent plasma model. Without a GRMHD dump on this machine, the
stand-in is ipole's RIAF (`model/riaf`, a = 0.9375, θ = 85°, 230 GHz): the reference image of
`validation/ipole_riaf/riaf_fine_*.csv` block-averaged to 16² pixels (2.44 M), with the noise
of the fits' gates (2/1/1/0.5% of the peak in I/Q/U/V), fitted by `validation/riaf_fit/riaf_fit.jl`
on the CPU (200 samples per ray, 300 iterations, 12 minutes with 4 threads).

## Setup

Six polarized splats on a ring at r = 4 M in the midplane with Keplerian ZAMO velocities and a
toroidal field, densities 3e6 cm⁻³, Θe = 30, B = 30 G. Stage 1 (60 iterations, plasma rows only)
with hygiene every 40 iterations allowing densification to 12 splats; stage 2 (240 iterations,
everything free, learning rate annealed 0.03 → 0.003).

## Result

| Quantity | Value |
|---|---|
| χ² start → end (1024 data points) | 9147 → 18.3 |
| Splats | 6 → 12 at the first hygiene pass |
| Image residual norms relative to ‖I‖: I, Q, U, V | 3.6%, 1.3%, 1.1%, 0.6% |
| Recovered centres | r = 2.7–5.0 M, |z| ≤ 0.5 M (the emitting torus) |
| Recovered Θe (RIAF at r = 3–6 M: 20–11) | 9–28 |
| Recovered B (RIAF at r = 3–6 M: 43–21 G) | 16–44 G |
| Fields on a voxel grid (RIAF-density-weighted where both have density): density ratio, Θe error, B error | 0.68, 36%, 63% |
| Fraction of the RIAF's density (r < 10 M, |z| < 2 M) inside the splats' 1% contours | 38% |

The image is reproduced in all four Stokes parameters (figure in `validation/riaf_fit/output/`,
untracked). The plasma fields are recovered only up to the nₑ–B–Θe degeneracy of single-frequency
data that the Fisher audit identified (`docs/notes/2026-09-03_phase4_inference.md`): the splats
place the right temperatures and fields in the emitting region but trade density against field
strength, and they do not represent the RIAF's extended low-density tails, which do not emit at
230 GHz. A multi-frequency version of this fit, and the GRMHD movie once a dump is available,
are the next steps of gate 7.

## Two frequencies (230 and 345 GHz)

The same fit with ipole's 345 GHz image added (`--freqs 2`; `validation/ipole_riaf/riaf_345_*.csv`,
2048 data points, 17 minutes):

| Quantity | one frequency | two frequencies |
|---|---|---|
| χ²/N at the end | 0.018 | 0.077 |
| Splats | 12 | 12 |
| Density ratio fitted/true (weighted) | 0.68 | 0.61 |
| Θe relative error | 36% | 42% |
| B relative error | 63% | 111% |
| RIAF density inside the splats' 1% contours | 38% | 33% |

Both images are reproduced (χ² per point well below one at the assumed noise), but the second
frequency does not improve the recovered fields; it makes them worse. The Fisher audit's
prediction (a second frequency lifts the temperature direction) holds for a single parcel near
the truth; a twelve-parcel fit started from a ring has the partition non-uniqueness of addendum
§5.2 on top: the smooth power-law RIAF can be represented by parcels that trade density against
field strength and temperature between neighbours in many ways, and the spectral information
constrains the sum of their contributions, not the split. This is the case for the priors,
hierarchical shrinkage and field-level regularization of the addendum, and for merge hygiene
between overlapping parcels, before field-level conclusions are drawn from such fits.
