# The FP32 transport question (2026-09-12)

Consumer and workstation GPUs run FP64 at 1/32 to 1/64 of their FP32 rate, so a workstation
purchase turns on whether the transport and the dual sweep can run in Float32 over the
Float64 geodesics (the march itself cannot: Krang's semi-analytic path is unstable in
Float32). `Geodesics.precision(cache, Float32)` copies a regenerated cache with every scalar
converted, no march, so the stored-sample kernels can be run at another precision.

## What a first probe showed

On the CPU, 24² pixels, 60 samples, eight parcels with Keplerian rates, with and without the
half-orbit truncation:

- The tails kernel runs with Float32 parcels, frequency, length unit, samples and pixel
  constants, and its image is finite and agrees with the Float64 one to 1.1e-4 of the peak
  over the four Stokes parameters and to 8e-7 in Stokes I.
- That agreement overstates the case, because the coefficient chain is not Float32-clean:
  `thermal_synchrotron` returns Float64 coefficients for Float32 arguments (Float64 literals
  in the fitting functions promote), so the kernel computed its coefficients in Float64 and
  only composited and stored in Float32. The dual sweep, which passes the coefficients to
  `sample_adjoint` alongside Float32 tails, fails on the type mismatch outright.
- The invariant Stokes vectors I/ν³ reach 1.4e-45, the smallest Float32 subnormal: at
  230 GHz the invariants of cgs intensities sit near 1e-37, below Float32's normal range
  (1.2e-38), so the compositing in Float32 needs a change of units (ν in units of 1e11 Hz,
  or intensities in Jy per steradian) before its accuracy can even be measured.

## What it means for the purchase

FP32 is not a drop-in. Making the transport genuinely Float32 is a bounded job: type-generic
literals through `thermal_synchrotron`, `invariants` and the frames, a unit rescaling of the
invariants, and then the accuracy test proper (the image to 1e-4 is encouraging for the
compositing, but the Faraday-thick and n = 2 cases are the ones to check). A day or two,
after which the RTX PRO 6000 route can be decided on measurement. Until then the FP64
numbers stand: the H200 route needs no code change.

## Making the transport Float32-clean

Seven changes (the last one twice), each found by running the Float32 tails image and dual
sweep against Float64 and walking the first divergent ray sample by sample (`test_precision`; the probe scripts are
not kept). Float64 results are unchanged: bit for bit in the step operator, at the rounding
level elsewhere.

1. **The coefficient chain is typed by its arguments.** `@muladd_chain` converts every series
   coefficient with `oftype(x, c)`, the Dexter fits take `x::T`, and the literals and physical
   constants of `thermal_synchrotron`, its Pandya variant and `planck` are wrapped. In Float64
   the conversions are the identity.
2. **The invariants are formed with ν̂ = ν/NU0, NU0 = 1e11 Hz, instead of ν**: j/ν̂², ν̂α, ν̂ρ,
   the affine path length L/ν̂ Σ Δτ, the observer's I = ν̂³ I_inv, and the gradient seeds
   ν̂³ ∂l/∂I. Every observable is unchanged (the optical depth ν̂α · L/ν̂ and the emitted
   intensity j/ν̂² · L/ν̂ · ν̂³ are ν̂-free), and the invariant Stokes vectors move from 1e-37 to
   the order of the physical intensities. `Transfer.νhat` is the one place the scaling lives;
   any new conversion between a physical and an invariant quantity goes through it (the
   unpolarized step was the one site missed at first: the Gold fluxes came out 1e-33 too small).
3. **The Planck function and the rotativities are formed from in-range ratios.** 2h/c² is
   1.5e-47, below Float32's range, so `planck` is `PLANCK0` ν̂³/(eˣ − 1) with PLANCK0 = 2h NU0³/c²;
   the Faraday chain had (2πν)⁴ = 4e48 in a denominator, so it uses ω₀/ω and ω_p²/ω². Before
   this the Float32 absorptivities were zero (Kirchhoff's law divided by an underflowed
   Planck function) and ρ_Q was zero: the image was 2% off.
4. **Constants are in the scalar type behind a dual** (`_scalar(T)`, `S(...)` in the chains).
   `T(ME)` with `T` a dual type is a dual constant; dividing a dual by it goes through the
   square of its value in the derivative rule, and mₑ² = 8e-55 underflows Float32, so every
   partial of ρ_Q and ρ_V was NaN while their values were right.
5. **The step operator works in the products K′Δ** (`step_operator` rescales by the power of two
   just below Δ, which rounds nothing, and hands the old algorithm a length in [1, 2)). With
   K′ and Δ apart, the partial of Δ³M₄K′³ passes through 1e45 and that of 1/Θ through Θ² ~ 1e-62
   at the tails of a density (coefficients 1e-20, Δ ~ 1e11): NaN in the adjoint of α_I. Scaling
   by Δ itself instead of a power of two costs a factor four in Float64 (independent roundings
   of each αΔ before squaring; the BigFloat gate at 2e-15 per radian catches it).
6. **The branch thresholds of the step are in the precision of the type**: the linear branch
   below (|K′|Δ)² = 1e-200 in Float64 (which is zero in Float32) and 1e-25 in Float32, the
   cubic branch below ΘΔ²(1 + |K′Δ|²) = eps/20 (1e-17 in Float64, 6e-9 in Float32).
7. **The polarization cap is computed on coefficients scaled by an exact power of two** (the
   one below the largest of |I|, |Q|, |V|, a plain number): jQ² underflows at the tails
   (jQ ~ 1e-24), the ratios Q/I, V/I overflow where the field lies within a fraction of a
   degree of the ray (jI vanishes there while jV, with its cot θ, does not; found at fit size,
   83 samples in 1e8, after the small tests had passed with the ratio form), and a dual divisor
   of 1e-37 squares to an underflow in the derivative rule. Any of the three gives NaN partials.

Result (`test_precision`, 6² rays × 16 samples, two parcels, CPU): the Float32 tails image
agrees with Float64 to 3e-7 of the peak (1e-4 in the first probe, where the chain still
promoted), and the Float32 dual-sweep gradient to 1.5e-6 of its largest entry (gates 1e-4 and 1e-3). The κ and
power-law chains are typed the same way but not probed in Float32 (`Bessels.gamma` promotes).

A whole fit can run in Float32: `Geodesics.precision` converts a `StokesMovie` as it converts a
cache, and `shell_selffit.jl --precision Float32` fits Float32 copies of the movie, the
parameters and the stored samples (the geodesics stay Float64, the reported χ² is Float64). 

## The timing

`bench_precision.jl` on the RTX 2080 SUPER (FP64 at 1/32 of FP32), the tails and the dual sweep
of four batched frames over per-ray lists, Float64 geodesics in both cases:

| screen × samples × parcels | Float64 | Float32 | ratio |
|---|---|---|---|
| 64² × 160 × 300 | 6.37 s | 0.71 s | 9.0 |
| 128² × 160 × 600 | 24.2 s | 4.0 s | 6.0 |

The Float32 image agrees with Float64 to 1.3e-6 of the peak at both sizes, and after the cap's
rescaling the Float32 gradient at the first size is finite and agrees to 7e-6 (8.8× on the rerun).
A 300-parcel shell fit of 300 iterations in Float64 takes 1901 s, 6.3 s per iteration: the sweep
is the whole cost, so a Float32 fit stands to gain most of the sweep's factor.

Two things are open (2026-09-13, 05:30). At 128² × 160 × 600 one set of gradient entries is still
non-finite after the cap fix: a rarer sample of the same kind as the others, being walked on the
CPU (the probe at that size). And the Float32 shell fit itself (`--precision Float32`) died after
203 s, about a hundred iterations, with a DomainError thrown inside a kernel, which the card
reports without a stack trace; the same fit at reduced size on the CPU backend is running to get
one. Until both are closed the Float32 path is validated for the sweep at fit size, not for a
fit.
