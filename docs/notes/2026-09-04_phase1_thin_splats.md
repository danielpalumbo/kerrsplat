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

## Gradients on the GPU: status

* Host path (`thin_image_march`, a plain loop over `march_ray`): Enzyme reverse mode works and
  matches central finite differences to 1.9e-7 (the FD floor) at 3.6× the forward cost. Two
  idioms are needed and are recorded in the tests: the loss is passed as `Const` with
  `set_runtime_activity` (JacobiElliptic's amplitude routine and a captured constant target
  array trip Enzyme's static analysis), and loops over root cases must be type-stable (a loop
  over a heterogeneous tuple dispatches dynamically, which Enzyme rejects).
* KernelAbstractions CPU backend: both routes work and agree with the host gradient to 3e-15:
  (a) Enzyme through the kernel launch via KernelAbstractions' Enzyme rules (`autodiff` of a
  wrapper that calls `kernel(args...; ndrange)`), 160 s of compilation; (b) `autodiff_deferred`
  inside the kernel, one ray per thread, the ray writing its value into a `Duplicated` output
  whose shadow carries the seed and all consumer arrays `Duplicated`, 16 s of compilation.
  Route (b) is what `test_kernel_gradient` keeps.
* CUDA (Enzyme 0.13.199): route (a) is refused by the extension ("Active kernel arguments not
  supported on GPU"; it classifies a scalar kernel argument as active — removing the scalar
  accumulator argument from the fused kernel was not enough). Route (b) compiles once no value
  is actively returned and no constant pointer is stored into the active consumer (otherwise
  Enzyme's runtime-activity error paths pull string formatting into the kernel), but the
  kernel then throws a device-side exception whose type cannot be printed, also for a consumer
  over *stored* samples (so the marcher's tape is not the cause; the consumer's momentum
  Jacobians and exponential are enough to trigger it), with a 64 KB stack and a 1 GB malloc
  heap. To be revisited with a newer Enzyme (the one that the environment could not download
  today) and, if it persists, with Enzyme's own CUDA.jl examples as a bisection baseline.
  Until then, gradients at scale run through the KernelAbstractions CPU backend (threaded) or
  the host loop.
