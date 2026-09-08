# The dual sweep: fast polarized gradients on the GPU (2026-09-06)

The chunked Enzyme reverse sweep of 2026-09-05 (`docs/notes/2026-09-04_phase1_thin_splats.md`)
gave exact polarized gradients inside the CUDA kernel but was slower than the CPU: Enzyme's
device reverse pass of one transfer step costs about fifty times its forward pass. This note
records its replacement, `Splats.polarized_gradient!(...; method = :dual)`, which needs no
reverse-mode tape at all and is now the default of `Fit.chi2_gradient!` and
`Fit.image_loss_gradient!`.

## The adjoint of a ray from four-vectors alone

Along one ray with samples k = 1…N (front to back), each sample has a step operator O_k and
emission E_k (identity and zero for a skipped sample), and the observed invariant Stokes vector
is T₁ with the tail recurrence

    T_{N+1} = 0,   T_k = E_k + O_k T_{k+1}

(T_k is what arrives at the observer side of sample k from everything behind it). For a loss
l = w₁ · T₁ the adjoints of the step of sample k are

    ∂l/∂O_k = w_k T_{k+1}ᵀ,   ∂l/∂E_k = w_k,   w_{k+1} = O_kᵀ w_k,

with w_k the adjoint of the Stokes vector in front of sample k, propagated inward from the
observer by the transposed operators. The reverse over the compositing therefore never needs
the 4×4 products P_k of the front-to-back form: a backward pass over the stored samples keeps
the tails T_k (one 4-vector per sample and ray, `polarized_tails!`; `tails[:, 1]` is the image),
and a forward pass carries w_k and forms each sample's adjoint on the spot
(`polarized_dual_sweep!`). Two passes, each the cost of a forward transport plus the local
derivatives below.

## The local derivatives by forward-mode duals

Reverse mode is only needed through the compositing, which is now explicit. What remains per
sample is the Jacobian of one step with respect to its coefficients and of the coefficients
with respect to the splat parameters, both small enough for forward-mode duals (ForwardDiff,
which runs inside CUDA kernels without any of Enzyme's device constraints):

- **The step.** `transfer_step` was split: `step_operator(α, ρ, Δ) -> (O, Ej)` holds the whole
  dependence on the coefficients and `E = Ej j` is linear in the emissivities. So j̄ = Ejᵀ w
  exactly, and ᾱ, ρ̄ are the seven partials of the scalar wᵀ O T + wᵀ Ej j evaluated with
  `α`, `ρ` seeded as duals (`Transfer.sample_adjoint`): the contraction with the adjoint of the
  matrices is free, since a scalar's partials are exactly that contraction.
- **The splats.** Every coefficient of a thermal splat is proportional to its electron density,
  which is the only place the Gaussian weight and ln nₑ enter. So for the thirteen geometric
  rows (centre, scales, quaternion, t₀, ln w, pattern rate) and ln nₑ the parameter gradient
  is the scalar c̄ · c times ∂ln G/∂θ, where ∂ln G comes from thirteen cheap partials through
  `splat_weight` alone (run unchanged on a `ParamColumn` of duals). The seven fluid rows
  (Θe, B, the field angles, the ZAMO velocity) go through the local frame and the synchrotron
  fits with seven partials (`splat_coefficients` and `screen_coefficients`, shared with the
  primal element so that the derivative is of exactly the function the image uses).

Per-ray partial sums go to a `npix × 21 × nsplat` scratch array and are reduced at the end, so
the result does not depend on thread order (no atomics).

## Validation

- `test_step_adjoint`: `sample_adjoint` against `transfer_step` (agreement to the step's own
  accuracy, ~1e-14 per radian of rotation) and against sixth-order central differences of
  wᵀ(O T + E) in α⃗, ρ⃗ over the 413 step cases (worst 1.5e-7, at the nearly nilpotent case,
  see below; the rest at the finite-difference noise of ~1e-8).
- `test_polarized_gradient` (both methods against the host Enzyme gradient of the same loss
  through the fused march): 2e-15 on the CPU backend at 8² × 40; CUDA at 32² × 300 in the
  suite. Dual against Enzyme sweep on CUDA at 32² × 300: 1.4e-14.
- `test_chi2_gradient`, `test_image_loss_gradient`: both methods against Enzyme's CPU gradient
  of `chi2` and of the closure χ² of a frame.
- The refactored step (`E = Ej j`) against the BigFloat references: O 9e-14, E 6e-14
  angle-scaled, as before.

## What it costs (RTX 2080 SUPER, Float64, six parcels, N = 300 stored samples)

| screen | forward image (march + transport) | tails | dual sweep | gradient (tails + sweep) | Enzyme chunked sweep |
|---|---|---|---|---|---|
| CUDA 32² | 0.52 s | 0.16 s | 0.67 s | 0.83 s | 8.2 s |
| CUDA 64² | 0.77 s | 0.25 s | 1.10 s | 1.35 s | |
| CUDA 128² | 0.89 s | 0.62 s | 2.15 s | 2.77 s | 95 s |
| CPU backend, 8 threads, 32² | 0.25 s | 0.15 s | 0.56 s | 0.71 s | 1.28 s |
| CPU backend, 8 threads, 64² | 0.45 s | 0.61 s | 2.25 s | 2.86 s | |

Enzyme's host gradient through the fused march (the path the CPU movie fits take) costs 2.27 s
per gradient at 32² with eight threads on the same parcels. So the dual sweep costs about four
times the forward transport over stored samples on either backend (the seven-partial duals are
the price) and is 2.7× faster than the CPU fits' gradient already at 32² on this card, ten
times faster than the Enzyme device sweep there and 34× at 128² (2.8 s against 95 s for the
full polarized gradient of 16384 rays). At 32² the card is under-occupied (1024 threads) and
the CPU backend's own dual sweep is as fast; from 64² up the GPU pulls ahead. With two parcels
the figures are 0.37 s (32²), 0.54 s (64²) and 1.22 s (128²) on CUDA. Agreement between the
dual and the Enzyme sweeps in these runs: 4e-15 to 3e-14; dual against the host gradient
1.2e-14.

## Traps found

- ForwardDiff ≥ 1 defines `x == 0` for a dual as "value zero *and* partials zero". The step's
  removable-singularity guards (`a == 0 ? one(a) : -expm1(-a)/a`, `sin(b)/b`) therefore took
  the 0/0 branch for duals with a zero value and gave NaN values, not just NaN partials. The
  guards now test the value (`Transfer.vanishes`), and `phi` returns its first-order form at
  zero so that its derivative there is −1/2.
- `sqrt` of a dual whose value is zero has NaN partials, which then poisoned comparisons.
  Λ₁, Λ₂ come from `safe_sqrt` (zero, with zero partials, at a zero value): the step depends on
  them only through even functions, so the derivative is right.
- The step's unpolarized branch (|K′|Δ < 1e-100) returned the operator without any K′ term;
  its derivative with respect to the polarized coefficients was zero there. It now keeps the
  first-order term (invisible in the value, correct in the derivative).
- At an exactly nilpotent K′ (|α⃗| = |ρ⃗| and α⃗ ⟂ ρ⃗ to 1e-17) the cubic branch is exact in value
  but its derivative misses the quartic terms (3% at |K′|Δ ≈ 1 in the test case); near the
  degeneracy the closed forms' derivatives lose accuracy as eps/(ΘΔ²) while their values do not
  (1.5e-7 at ΘΔ² ≈ 1e-9). A codimension-two set that real samples do not hit; any
  differentiation of these formulae shares it, Enzyme's included. A series form of the
  coefficient functions in (Λ₁²Δ², Λ₂²Δ²) would remove it if it ever matters.
- `LocalFrame(g, cosθB, zero(T))` with `T` the metric's Float64 and `g` a dual is a
  MethodError; the degenerate branches of `local_frame` are now typed by their arguments.

## Half-orbit truncation (2026-09-08)

The truncated transport (`polarized_image!(...; nmax, slab)`, the `WindingState` consumer)
skips a sample once the ray's passage count exceeds `nmax`, and the count never decreases, so
the truncation of a ray is one index: the first sample beyond its `nmax`-th passage.
`winding_cutoff!` runs the counter over the stored samples (exactly the consumer's `wind`, on
the same validity test) and both passes of the dual sweep stop there (`polarized_tails!`,
`polarized_dual_sweep!`, `polarized_gradient!`, `Fit.chi2_gradient!`, `Fit.image_loss_gradient!`
take `nmax`, `slab`; the Enzyme sweep does not). Gate `test_polarized_gradient_winding`: the
cutoffs equal the counter run on the host for every ray, and the gradients of the truncated
loss agree with host Enzyme through the winding consumer to 6e-15 and 3e-15 (CPU, n ≤ 0 and
n ≤ 1, 6² × 40) and 1.6e-14 and 1.4e-14 (CUDA, 24² × 300). The half-orbit self-fit experiment
(`validation/winding/winding_selffit.jl --backend cuda`) takes its χ² and gradient from this
path over one stored-sample cache instead of tiled host Enzyme.

## What is still open

The dual sweep covers `PolarizedSplats` (thermal parcels); `KnotSplats` and the power-law and κ
populations need their own `element_adjoint!` (the same recipe: density scaling for the
geometry, duals for the population rows). The Enzyme sweep (`method = :enzyme`) stays as the
reference for any model.
