# t̃ and φ by anchored quadrature: design and findings

**Date:** 2026-09-03 (late). Plan step 4 / gate 3. Companion to
`2026-09-03_direct_azimuth_conditioning.md`.

## Design (as implemented in `src/Geodesics/quadrature.jl`)

The Mino-time rates dt/dτ = (r²+a²)(r²+a²−aλ)/Δ + a(λ − a sin²θ) and
dφ/dτ = a(2r − aλ)/Δ + λ/sin²θ are integrated by one Simpson panel per sample interval (the
recurrence marches at half the sample spacing) between anchors where Krang's closed forms reset
the sums. Three singular pieces are removed before Simpson sees the integrand, each with an
elementary antiderivative:

1. **Large r** (both ends of a scattering ray): |dr/dτ|(1 + 2/r), integral |Δ(r + 2 ln r)| per
   monotone leg; the interval containing the radial turning point is split there (dr/dτ = 0,
   r = r₄ known), with three direct Jacobi evaluations for that one interval.
2. **Horizon** (plunging rays): both poles of 1/Δ = 1/((r − r₊)(r − r₋)) by partial fractions,
   Σ± Q± (dr/dτ)/s± · r±/(r(r − r±)) with Q± the residues and s± = dr/dτ continued to r±
   (√R(r±) = |r±² + a² − aλ| because Δ(r±) = 0), so that each subtracted term carries exactly
   the pole it removes; integral Σ± (Q±/s±) Δ ln((r − r±)/r). The inner pole is never reached,
   but for near-extremal spin it sits only 2√(1 − a²) inside the horizon (0.09 M at a = 0.999)
   and its tail is not smooth on the sample spacing of the final plunge: subtracting only r₊
   left 5e-6 in t̃ at the last sample of an a = 0.999 ray; both poles normalized with s₊ gave
   2e-7 there; with s₋ for the r₋ term the quadrature matches Krang's own error to two digits
   all the way in (2.2e-8 at r − r₊ = 0.01 M, 1e-8 at 0.02, 3e-9 at 0.05, where the
   BigFloat reference itself is limited by the pole a sample away). The gate-3 comparison with
   the BigFloat reference still stops at r₊(1 + 0.15), where every consumer has long discarded
   the samples (plan §10).
3. **Polar axis**: with φ_am = am(X) and 1/sin²θ = C/(1 − n sn²X), the integral
   ∫ dX/(1 − n sn²X) = ∫ g(φ) dφ/(1 − n sin²φ) with g = 1/dn = g_c (1 + m' cos²φ)^(−1/2),
   g_c = 1/√(1 − μ), m' = μ/(1 − μ) ∈ (−1, 0). The first two terms of g in cos²φ integrate
   elementarily, A(φ) = arctan(√(1−n) tan φ)/√(1−n) and φ/n − (1−n)/n A(φ) (both unfolded by
   the winding number of X), leaving a remainder λ C [1 − g_c dn (1 − ½ m' cn²)]/(1 − n sn²)
   that is bounded and vanishes at the spike like cn⁴. Without any of this, rays passing within
   a few degrees of the axis needed 30–1000 substeps; with the first term alone, one panel
   reached 1e-11 at 1000 samples on a ray grazing the axis at 14 mrad but still left 6e-8 per
   polar crossing at 7 mrad (a = 0.999, τ_total = 2.2); with both terms that ray is at 6e-11,
   Krang's own level there.

Convergence (CPU prototype, N samples, one panel each, errors against Krang, absolute):

| ray | N = 200 φ | N = 200 t̃/|t̃| | N = 1000 φ | N = 1000 t̃/|t̃| |
|---|---|---|---|---|
| plunging (case 1) | 2e-9 | 1e-10 | 4e-12 | 5e-12 |
| scattering | 3e-11 | 4e-10 | 1e-13 | 5e-12 |
| axis-grazing, θ_min = 0.014 | 1.5e-8 | 1e-9 | 1.3e-10 | 1.5e-12 |
| vortical | 2e-11 | 5e-12 | 1e-11 | 2e-11 |

Cost on the RTX 2080 SUPER: 10.8 ns/sample (256² × 1000 samples in 0.71 s) against 128 ns for
Krang's direct evaluation and 1.9 ns for r, θ alone; the anchors (one closed-form evaluation
every 64 samples) are 2 ns of that. Not yet tuned (three logarithms and an arctangent per
sample can be reduced).

## Findings

1. **Krang's closed forms carry isolated glitches of ~1e-7 in t̃.** At a = 0.7, θo = 120°,
   α = 1.064, β = −4.468, sample 274 of 1000, Krang's t̃ deviates from the cubic through its
   neighbours by 6.7e-8 while the neighbours deviate by ≤ 3.5e-9; the quadrature (which
   carries its sum through that sample) shows nothing. The gate-3 tolerance against Krang for
   t̃ is therefore 3e-7 absolute; the sharp check is the BigFloat rate reference.
2. **Krang's t̃ and φ are wrong after the polar turning point of vortical rays with β < 0**, by
   1e-2 at the end of the ray (a = 0.94, θo = 60°, α = β = −0.213), the same bookkeeping that
   breaks its ν_θ there: finite differences of Krang's own φ(τ) disagree with the rate
   λ/sin²θ + a(2r − aλ)/Δ by 1.5 % on 63 samples. The quadrature *without* re-anchoring agrees
   with the BigFloat rate reference to 4e-10 on that ray; anchored to Krang every 64 samples
   it inherits Krang's error (its anchor residual there, 6.7e-3, is the largest on the
   screen — the residual arrays do their job). `docs/notes/upstream_issues.md`, Krang item 5.
3. **Krang's I0_inf is accurate to ~1e-12**, which shifts the whole Float64 trajectory in τ.
   Both Float64 paths (direct and quadrature) share the shift; against the BigFloat reference
   it shows up as δ · dt/dτ, i.e. 4e-8 in t̃ at the first samples (r ≈ 260) and nothing inside
   50 M. Comparisons with the truth are therefore made on samples inside 50 M as increments
   from the first such sample.
4. **Near-critical rays remain root-limited** (7e-9 in φ at 1 − k ≈ 1e-5), identically for both
   Float64 paths, as for r.
5. **A BigFloat rate reference is cheap enough for tests**: 16-point Gauss–Legendre per sample
   interval on the BigFloat closed-form coordinates (Newton-polished roots) converges
   geometrically for samples inside 50 M on rays with θ_min > 0.02; ~4 s per 1000-sample ray.

## End-to-end image (gate 5)

The feasibility probes' test splat (Gaussian emissivity × g³ from the analytic momentum),
integrated along the stored samples of each marcher on the GPU and compared with a host loop
over Krang's direct evaluation (128² × 1000 samples): the direct marcher agrees to 5e-11 of the
peak, the recurrence marcher to 1e-10 except on the central vortical pixels, where it differs
by 7e-6 — and there it is the *host* that is wrong (finding 2 above; Krang's φ, t̃ and ν_θ
after the polar turning point). The cache's anchor residuals flag exactly those rays (0.1–0.2
against a median of 1e-10), which is what the test uses. Krang's
`boyer_lindquist_to_quasi_cartesian_kerr_schild*` cannot compile in CUDA kernels (a `@warn`),
so `Geodesics.quasi_cartesian_kerr_schild` provides the same map (upstream issue 10).

## Fused mode (plan step 6)

`march_ray(f, acc, …)` folds a consumer over a ray's samples inside the kernel; the stored
mode is the consumer that writes `GeodesicSamples`, and `fused_march!(f, out, cache)` runs any
isbits consumer `f(acc, j, k, sample, Δτ, pix)` with no per-sample storage (a `Fused(M)` cache
allocates none: 256² × 1000 samples need 2.2 GB stored, nothing fused). The fused splat image
equals the stored-sample one bitwise on the CPU and to 2e-13 on CUDA (the two kernels are
compiled separately and contract floating-point operations differently). With the test splat
as consumer the fused kernel costs 14.9 ns per sample (8.5 for the march, ~6 for the
consumer's momentum Jacobians and exponential), 256² × 1000 in 0.97 s. `tiles(camera, n)`
splits a screen into consecutive pixel blocks for stored-mode runs that do not fit.

## Duals (plan step 5, gate 4)

ForwardDiff `Dual{2}` for (a, θo) propagate through K1, the direct marcher and the
recurrence + quadrature marcher without changes to the kernels, on the CPU backend and on
CUDA. The one trap was the radial turning point of scattering rays, where the split evaluates
dr/dτ = ±√R at R = 0: the value is fine (0) but ∂√R/∂a is infinite there, and the partials of
t̃ and φ became NaN on every scattering ray. dr/dτ is identically zero at the turning point
for every spin, so the kernel uses an exact zero (`ray_point_turning`).

Agreement of the recurrence duals with the direct path's duals on well-conditioned rays:
r 1e-12, θ 1e-14, t̃ 3e-10, φ 1.4e-9 (scaled by max(|∂/∂a|, |∂/∂θo|, 1)); against central finite
differences of the direct path (h = 1e-5) the direct duals agree to 1e-8 … 1e-7, the FD noise
floor (plan §3). On axis-grazing and near-critical rays the comparisons degrade as the values do.

On CUDA the dual kernels need a 16 KB per-thread stack (12 KB fails with an illegal memory
access; Float64 needs 4 KB); `regenerate!` sets it from the element type.
