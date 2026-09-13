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
