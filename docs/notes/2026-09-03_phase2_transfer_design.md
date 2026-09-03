# Phase 2: polarized radiative transfer — design and status (2026-09-03)

Phase 2 of `docs/plans/kerrsplat_evaluation_and_plan.md` §7.6 (with the per-splat additive
formulation of the addendum §2). This note fixes the conventions and the order of work so that
each PR can be validated on its own.

## Reference implementations available on this machine

- **ipole** (`~/local_scripts/ipole`, C, GPL): the polarized validation reference of the plan.
  Its symphony fitting formulae build standalone against GSL (`validation/symphony/`), which is
  how the coefficient tables were generated. The full code needs HDF5, absent from the system;
  a conda environment provides it: `mamba create -n ipole-build -c conda-forge hdf5 gsl`, then
  in a copy of the checkout `make MODEL=<model> CC=gcc HDF5_DIR=$HOME/miniforge-pypy3/envs/ipole-build GSL_DIR=/usr`
  and run with `LD_LIBRARY_PATH=$HOME/miniforge-pypy3/envs/ipole-build/lib`.
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
3. **Frames** (`transfer-frames`): `local_frame(met, r, θ, η, λ, νr, νθ, α, β, θo, ũ, B)` returns
   the redshift g = 1/(−p·u), the pitch-angle cosine and the position angle χ of the local Q axis
   on the screen, for a fluid element of ZAMO 3-velocity ũ = γβ⃗ (Krang's ZAMO axes r̂, φ̂, −θ̂)
   and fluid-frame field B. Local basis: ipole's plasma tetrad, e₁ = k̂ × B̂ (Q axis, ⟂ projected
   field), e₂ = k̂ × e₁, right-handed with the propagation direction. Screen basis: north = +β,
   east = +α with the direction toward the observer completing a right-handed triad, so
   χ = atan(e_α, e_β); whether +α is east on Krang's screen (Krang's `evpa` uses −e_α) is a global
   sign of U to settle against ipole images in step 5. Gate: the boost against Krang's
   `jac_fluid_u_zamo_d`, g and χ against Krang's `synchrotronPolarization` (1e-15 on 400 random
   rays, samples, velocities and fields), the pitch angle against an explicit construction, and
   the kernel on CPU and CUDA.
4. **Gold et al. (2020) suite** (`transfer-gold2020`): `UnpolarizedTransport` (a fused-march
   consumer with the front-to-back accumulator `UnpolarizedState`, the unit scaling of the
   `transport.jl` header: Δ = (L/ν_obs) Σ Δτ over the invariants j/ν², να) renders ipole's
   analytic models 1–5 (test/test_gold2020.jl). Total fluxes at 128², 2000 samples per ray:
   1.6604, 1.4488, 0.4453, 0.2726, 0.0257 Jy against the published 1.6465, 1.4360, 0.4418,
   0.2710, 0.0255 Jy, i.e. 0.6–0.9% high for every model, unchanged at 256² × 4000 and at
   48² × 500. ipole itself (built against a conda HDF5 in the session scratch directory; see
   below) gives 1.6576, 1.4501, 0.4488, 0.2741, 0.02591 Jy, also 0.7–1.0% above the table, and
   within −0.8% … +0.2% of us; its model-3 flux moves from 0.4495 toward our 0.4453 when its
   camera is moved from 1000 M to 10000 M (0.4475) or its geodesic steps are refined tenfold
   (0.4484), so the residual is ipole's discretization. ipole's own test accepts 2%, which is
   the gate here. Model 2 (Schwarzschild) runs at a = 1e-3 because
   Krang's analytic solution returns NaN at a = 0 (upstream_issues.md item 11).
5. **Polarized transport** (`transfer-polarized`): `RadiativeTransport(model, ν_obs, L)` sums
   the screen-basis invariants of overlapping fluid elements (the addendum's additivity) and
   takes one exact step per sample. Gates: the Gold model through the polarized path equals the
   unpolarized path; a cold Faraday screen rotates the EVPA of an emitter behind it by ½∫ρ_V ds
   and two co-located screens add; superposed emitters add and their order is irrelevant; and
   ipole's RIAF model (`test/riaf_model.jl`, `validation/ipole_riaf/`) at 230 GHz (τ_F up to
   9 rad, τ up to 15) and 2 THz: pixel-norm residuals of I, Q, U, V ≤ 0.6% and 2%, totals to
   0.1%, EVPA to 0.0005 rad. The comparison fixed the screen handedness: north = +β, east = −α
   (Krang's `evpa` convention); with east = +α, U flips at every frequency and the Faraday-thick
   230 GHz images disagree at the 60% level, because the rotation sense is then reversed relative
   to the geometry. Also found and guarded: rotativities of 1e-170 from density tails underflowed
   the squared angles of the emission integral.
6. **Polarized splats** (`polarized-splats`): `Splats.PolarizedSplats` (20 parameters per splat:
   the thin splat's geometry and envelope, ln nₑ, ln Θe, ln B, the field direction in the fluid
   frame, the ZAMO 3-velocity) as elements of `RadiativeTransport`; `polarized_image`. Gates: the
   fused march against a host loop over stored samples (4e-18), CUDA against the CPU backend
   (7e-14), Enzyme reverse gradients of an image loss for all 40 parameters of two overlapping
   splats against a fourth-order stencil (1.6e-7; 0.19 s per gradient at 10² × 100 after a
   7-minute first compile), and an Adam fit recovering a perturbed splat (loss down 140×).

Phase 2 is complete with this step. Open: power-law rotativities (Marszewski+ 2021), κ
distributions, GPU-side Enzyme gradients, interval culling for many splats.
