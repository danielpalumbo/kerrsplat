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
2. **Horizon** (plunging rays): Q (dr/dτ)(r₊/(r(r − r₊)))/s₊ with Q the coefficient of
   1/(r − r₊) at r₊ and s₊ = dr/dτ there; integral (Q/s₊) Δ ln((r − r₊)/r).
3. **Polar axis**: with φ_am = am(X) and 1/sin²θ = C/(1 − n sn²X), the elementary part
   λ C f g_c A(φ_am), A(φ) = arctan(√(1−n) tan φ)/√(1−n) unfolded by the winding number of
   X, g_c = 1/√(1 − μ); the remainder λ C (1 − g_c dn X)/(1 − n sn²X) is bounded and O(μ).
   Without this, rays passing within a few degrees of the axis needed 30–1000 substeps; with
   it, one panel reaches 1e-11 at 1000 samples on a ray grazing the axis at 14 mrad.

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
