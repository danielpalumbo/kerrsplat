# Phase 1: optically-thin splats through the fused marcher

**Date:** 2026-09-04 (early). Main plan §7.2, §7.6 Phase 1; addendum §1. Branch `splats-thin`,
stacked on the geodesic PRs.

## What is in it

`KerrSplat.Splats`: one splat is a column of 13 unconstrained numbers,
(μ ∈ ℝ³ in quasi-Cartesian Kerr–Schild coordinates, s ∈ ℝ³ log-scales, q ∈ ℝ⁴ a quaternion
normalized on use, t₀, ln w, ln A), with emissivity

    j(t, x) = A exp(−½ (x − μ)ᵀ Σ⁻¹ (x − μ)) exp(−½ ((t − t₀)/w)²),   Σ = R(q) diag(e^{2s}) R(q)ᵀ.

The renderer is a fused consumer (`ThinRenderer`) of the geodesic marcher: at every sample it
evaluates the emission event (t_obs − t̃, r, θ, φ) → (t, x) and accumulates

    I = ∫ g² j(t, x) Σ_BL dτ,

with Σ_BL = r² + a² cos²θ converting Mino time to the affine parameter (dλ = Σ dτ, from
dr/dλ = ±√R/Σ) and g = 1/(−p_μ u^μ) the redshift factor for a ZAMO emitter, from the analytic
momentum p_μ(η, λ, r, θ, ν_r, ν_θ). g² is the optically-thin, frequency-integrated-at-fixed-
observed-frequency factor for a frequency-independent emissivity (I_ν = ∫ g² j_ν' dλ). The
probes' smoke test used g³ without Σ; that was a toy.

Deliberately not yet in Phase 1 (addendum §1): the per-splat fluid velocity (all emitters are
ZAMOs here; the addendum's ũ_k ∈ ℝ³ in the ZAMO frame comes with the polarized transfer,
where the boost also sets the pitch angle), per-splat B, electron distributions, frequency
dependence, absorption. Units: geometric (GM/c², GM/c³), addendum §4.2.

## Why these choices

* Slow light comes for free: t̃ is Krang's regularized time, increasing along the ray, so the
  emission time is t_obs − t̃ up to the one global constant that fixes the zero point.
* The consumer runs inside the marcher (`fused_march!`), so no samples are stored and the
  per-sample cost is the marcher's 10 ns plus the splat sum; with a few hundred splats the
  splat sum will dominate and the interval-culling of addendum §6.3 becomes the next
  performance item, not the geodesics.
* The parameter matrix is the differentiated object: `ThinRenderer` closes over it, and a
  host renderer built on `march_ray` (a plain Julia loop) is what Enzyme reverse-mode
  differentiates in the tests (gate 6 of §7.5); the GPU path is the same consumer in a kernel.

## Gradients on the GPU: status (rewritten 2026-09-05)

Enzyme reverse mode runs inside the CUDA kernel, one ray per thread, and matches the host
gradient (`Splats.thin_gradient!`, gate `test_stored_gradient`: 1e-12 at 12² × 120 and
6e-13 at 64² × 300, the latter in 1.2 s; the polarized consumer over eight samples:
1.6e-15, gate `test_polarized_kernel_gradient`). Four things stood in the way, none of them the
version of Enzyme (0.13.200 with CUDA.jl 6.3.1 is strictly worse: hundreds of unsupported
GC-frame and safepoint calls on top of the same failures):

1. `sincos` lowers to libdevice's `__nv_sincos`, for which Enzyme has no reverse rule; the
   kernel throws "No augmented forward pass found for __nv_sincos" at run time. The hot path
   uses `Geodesics.sincos_pair` (sine and cosine separately) and `jacobi_state` takes the
   amplitude's sine and cosine itself rather than through JacobiElliptic's `ellipj`.
2. Inside the differentiated region Enzyme compiles the geodesic march's special functions
   (Jacobi amplitude, `atanh`, `asin`, `^`, …) through their checked host implementations,
   which reference `DomainError` and cannot run on the device. The march is constant for
   splat gradients, so the kernel differentiates the consumer over *stored* samples
   (`store_samples = true`, `Direct` or `Recurrence` marcher); the march itself is never
   differentiated.
3. Polynomial tables: `evalpoly` (also behind `Base.Math.@horner`) is outlined into a call
   taking the coefficient tuple by reference once the tuple has nine or more entries, and
   Enzyme cannot cache that call on the device ("caching call: julia_evalpoly"; ptxas then
   reports an unresolved `jl_nothing`). Bessels.jl's K₀, K₁ are vendored into
   `Transfer/bessel.jl` (bit-identical, MIT) with an explicit `@muladd_chain`.
4. Mutually recursive helpers (the transfer step's trigonometric integrals called each other
   for their opposite regimes) cannot be taped on the device; each regime is now spelled out.

Limits: the per-thread stack cannot exceed 64 KB on this card (the driver allocates it for
every resident thread), which holds the tape of a thin ray of 300 samples or of eight
polarized samples. A full polarized ray therefore needs a chunked reverse sweep (states saved
at chunk boundaries in the forward pass, chunks differentiated from the last to the first with
the adjoint of the incoming state carried along), which is the next step; `Bessels.gamma` on
the power-law and κ paths and `thermal_synchrotron_pandya` (an Enzyme assertion) are untested.
Probes for the bisection live in the scratch directory of the 2026-09-05 session; the rules
are in `CLAUDE.md`.
