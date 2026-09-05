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
polarized samples. A full polarized ray therefore runs as a chunked reverse sweep
(`Splats.polarized_gradient!`, 2026-09-05): a forward kernel folds every ray over all samples
and keeps the radiative state at each chunk boundary (`chunk_size(N)` samples per chunk, the
largest divisor of N not above eight), then the chunks are differentiated from the last to the
first, each launch seeded with the adjoint of its outgoing state and returning the adjoint of
its incoming one (`polarized_reverse_sweep!`). One trap: the adjoint buffers must start from a
true zero, not `zero(RadiativeState)`, which is the compositing identity P = 1 (a spurious
identity seed made the gradient wrong by factors of thousands). `Fit.chi2_gradient!` builds the
movie χ² and its gradient on the backend from this, and `Fit.image_loss_gradient!` seeds the
sweep with a host Enzyme gradient of any function of the image (closure χ², self-calibration)
(gates `test_polarized_gradient`, `test_chi2_gradient`, `test_image_loss_gradient`). Making the sweep fast turned
up more device rules: the series of the moment integrals in the transfer step is now a set of
Horner chains rather than a loop, the adjoint kernels wrap the splat set in
`Transfer.StaticCount` so that the sum over splats is unrolled with `ntuple` (one compile per
splat count on the GPU), and the chunk fold is one straight-line method per chunk size; loops
inside the differentiated device function keep Enzyme's per-iteration cache in device malloc
(a two-sample loop cost 30× a one-sample chunk), a recursion on the index is not inferable, a
closure capturing a `Type` dispatches dynamically, and ptxas runs out of memory on the unrolled
code beyond two samples per chunk. Cost on the 2080 SUPER (6 parcels, 32² × 300): the forward
pass 0.13 s, the reverse sweep 26 ms per launch of 1024 rays with one sample per chunk (80 ms per
launch of 4096 rays at 64², 283 ms per launch of 16384 rays at 128²), 7.8 s per frame at 32²,
24 s at 64² and 85 s at 128², against 2.1 s per frame at 32² for Enzyme on the CPU backend
with eight threads (the throughput improves with the ray count but never overtakes the CPU). The device
reverse pass of one polarized sample is about fifty times its forward cost (three to five on
the CPU): Enzyme's device code generation spills the transfer step's tape to local memory. So
the GPU sweep is exact and bounded in memory (no whole-screen tape) but not faster than the CPU
on this card; the way to a fast GPU gradient is a hand-written adjoint of the transfer step
(analytic derivatives of the 4×4 exponential) or forward-mode duals for the per-sample
coefficient Jacobians with a hand-written reverse over the compositing. `Bessels.gamma` on the power-law and κ paths and
`thermal_synchrotron_pandya` (an Enzyme assertion) remain untested on the device.
Probes for the bisection live in the scratch directory of the 2026-09-05 session; the rules
are in `CLAUDE.md`.
