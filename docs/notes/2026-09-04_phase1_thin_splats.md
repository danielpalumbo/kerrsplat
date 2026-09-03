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
