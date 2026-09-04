# Phase 2: polarized radiative transfer — design and status (2026-09-03)

Phase 2 of `docs/plans/kerrsplat_evaluation_and_plan.md` §7.6 (with the per-splat additive
formulation of the addendum §2). This note fixes the conventions and the order of work so that
each PR can be validated on its own.

## Reference implementations available on this machine

- **ipole** (`~/local_scripts/ipole`, C, GPL): the polarized validation reference of the plan.
  Its symphony fitting formulae build standalone against GSL (`validation/symphony/`), which is
  how the coefficient tables were generated. The full code needs HDF5, which is not installed
  here (no `libhdf5`, no `h5cc`), so images from ipole itself need either a conda HDF5 or the
  Jipole port; deferred to the polarized-image gate.
- **Gold et al. (2020) GRRT test suite**: ipole's `model/analytic/model.c` and
  `tests/analytic/analytic_test.py` define five unpolarized analytic models with published total
  fluxes (1.6465, 1.4360, 0.4418, 0.2710, 0.0255 Jy at 230 GHz, 2% tolerance). They need no
  reference images, so they are the image-level gate for the unpolarized transfer and the unit
  scaling.
- **Krang** `synchrotronPolarization` (Walker–Penrose transport of the emitted polarization to
  the screen) is the reference for the frame-rotation angle.

## Conventions (ipole's)

- cgs units; Θe = kT/(mₑc²); the fluid-frame frequency ν = ν_obs/g with g = 1/(−p·u), E = 1.
- Stokes basis aligned with the magnetic field projected on the plane of the sky: Q along the
  projected field so that synchrotron emission has j_Q > 0 and j_U = 0; V in the IEEE/IAU sense
  (ipole flips the sign of symphony's Q, U and corrects the sign of the Pandya V fits).
- Thermal coefficients: Dexter (2016) j_{I,Q,V}, Kirchhoff absorptivities, Dexter ρ_Q,
  Shcherbakov (2008) ρ_V (ipole's `E_DEXTER_THERMAL` with `dexter_fit = 0` for ρ_V). Power law:
  Pandya+ (2016) j and α; symphony has no power-law rotativities (ipole exits), and the
  Marszewski+ (2021) fits are not transcribed here yet.
- Invariants: j/ν², ν α, ν ρ. With Krang's E = 1 normalization and the Mino-time parameterization
  (dλ = Σ dτ, λ in units of M), the invariant transfer equation reads
  d(I/ν³)/dλ = (GM/c²)/ν_obs · (j/ν² − ν α · I/ν³), which reduces to dI/ds = j − α I at infinity.

## Order of work

1. **Coefficients** (this PR, `transfer-coefficients`): `KerrSplat.Transfer` with the fits above,
   Bessels.jl for K₀, K₁, K₂ and Γ (verified on CUDA and with Enzyme), and the symphony tables as
   the gate (2016 thermal + 864 power-law rows; agreement 6e-14 and 7e-16 on CPU and CUDA).
2. **Analytic step** (`transfer-step`): `transfer_step(j, α, ρ, Δ)` returns the exact operator
   exp(−KΔ) and the emission integral ∫₀^Δ exp(−Ku) du j for constant coefficients, in Landi
   Degl'Innocenti's bounded-matrix form (even powers of K′ plus the two matrices M₂, M₃ that carry
   the odd powers), with the scalar functions evaluated without cancellation in every regime
   (series for small arguments, split exponentials at large optical depth, the nilpotent case
   α⃗² = ρ⃗², α⃗ ⟂ ρ⃗ handled exactly). Gate: BigFloat matrix exponentials and integrals over 411
   cases spanning optical depths 1e-9…2600 and rotation angles up to 3.5e5 rad; the attainable
   accuracy is eps per radian of rotation (the angle itself is only known to that), measured
   ≤ 1e-15 per radian, plus Kirchhoff equilibrium, rotation about ρ⃗, the semigroup property,
   the unpolarized limit, ForwardDiff, and the kernel on CPU and CUDA. The observer-first sample
   order of the fused marcher composes the step operators front to back: `RadiativeState` keeps
   P = O₁…Oᵢ₋₁ and accumulates S += P Eᵢ, the polarized form of front-to-back compositing.
3. **Frames**: photon momentum, fluid frame from a ZAMO velocity, redshift, pitch angle, and the
   Walker–Penrose rotation angle χ per splat; tests against Krang's `synchrotronPolarization`.
4. **Gold et al. (2020) suite**: unpolarized analytic models through the fused marcher with the
   unit scaling (M, D, Jy); 2% on the published fluxes.
5. **Polarized splats and images**: one-zone splats (n_e, Θe, B, direction, ZAMO velocity),
   per-splat Mueller assembly M = Σ R(χ_k) M_k R(χ_k)ᵀ, the two-splat overlap and Faraday-screen
   tests of the addendum, polarized images against ipole (HDF5 or Jipole) and Enzyme gradients.
