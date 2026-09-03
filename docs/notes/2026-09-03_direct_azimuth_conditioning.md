# Conditioning of Krang's closed-form azimuth on polar-grazing rays

**Date:** 2026-09-03. Found while validating the GPU geodesic skeleton (gate 1 of
`../plans/kerrsplat_gpu_geodesics_plan.md`).

## Observation

Running Krang's `emission_coordinates` inside CUDA kernels reproduces the CPU result to
about 1e-11 (t̃, scaled by |t̃|), 1e-13 (r) and 1e-12 (θ) on a 64² screen of ±10 M at seven
(a, θo) combinations. The unwrapped azimuth φ, however, differs by up to 2.9e-7 (absolute) on
the pixels of the α ≈ 0 screen columns, and the resulting spherical position by 1.45e-7 relative
(a = 0.94, θo = 1°, α = 0.159, β = 10). Smaller versions of the same effect: 5.4e-10 at
(a = 0.5, θo = 30°, α = 0.159, β = 8.7) and 4.6e-10 at (a = 0.2, θo = 75°).

## Cause

Those rays have λ = −α sin θo ≈ 0 and pass within a few milliradians of the polar axis. In
Krang's polar integrals the elliptic parameter u₊ → 1, so the third-kind integral Π(n = u₊, …)
that gives G_φ diverges (G_φ̂ = 1134 for the worst pixel) and is extremely sensitive to
rounding. Perturbing any single input by one ulp on the CPU (`scratchpad/diag_cond.jl` in the
session; reproduced by `test/reference.jl`'s `perturbed_pixels`) moves φ by exactly the
GPU–CPU difference:

| pixel | input +1 ulp | max Δφ | max Δposition / r |
|---|---|---|---|
| a = 0.94, θo = 1°, α = 0.159, β = 10 | β or a | 2.9e-7 | 1.45e-7 |
| a = 0.5, θo = 30°, α = 0.159, β = 8.7 | a | 1.1e-9 | 5.4e-10 |
| a = 0.2, θo = 75°, α = 1.43, β = 8.4 | β or a | 6.8e-12 | 6.8e-12 |
| a = 0.94, θo = 60°, α = 5, β = 3 (ordinary) | any | 3e-14 | 3e-14 |

So the GPU is not at fault: Krang's direct φ has a condition number of order 1e9 on such rays
in double precision, on any hardware. The error persists along the rest of the ray after the
polar crossing (the azimuth jump of ≈ π near the axis is what is ill-conditioned), so it is a
position error of order 1e-7 M at radii of a few M, not just an error near the axis.

A second, smaller effect on the same rays is not captured by input perturbations: Krang forms
u₊ = Δθ + √(Δθ² + η/a²) with Δθ ≈ −(η + λ²)/(2a²) large and negative, a cancellation whose
rounding (which differs between GPU and CPU through FMA contraction) is amplified by the
complete Π(n = u₊ → 1) in G_φ̂: residual GPU–CPU differences of 1e-10 to 1e-9 at pixels whose
one-ulp input sensitivity is 1e-12. The stable form u₊ = (η/a²)/(√(Δθ² + η/a²) − Δθ) removes
the cancellation; it should be used when the recurrence constants are built in K1 (plan §4),
and is worth an upstream patch.

Two smaller, unrelated effects seen at the same time:

* The complex-conjugate root pairs of case-3/4 pixels come out in either order depending on
  rounding (GPU and CPU differ). Every Krang formula uses the pair symmetrically, and all
  derived constants agree to 1e-15, so the tests compare roots as a set.
* At small spin (a = 0.2) Krang's polar parameterization loses digits through the
  (η + λ²)/a² cancellation; the sensitivity is ~1e-11 and grows toward a → 0 (a = 0 is
  unsupported upstream).

## Consequences for the plan

1. **Test criterion (implemented).** GPU results are required to match the CPU to within
   `tol + 8 × sens`, where `sens` is the one-ulp input sensitivity of the CPU evaluation at
   that pixel and `tol = 2e-9` covers the u₊ cancellation above. On well-conditioned pixels
   the measured agreement is 1e-12 (r, θ) and 1e-10 (t̃, constants); on ill-conditioned ones
   the criterion demands only that the GPU is no worse than the formula's own rounding.
2. **Gate 2/3 need an independent reference.** The recurrence and the quadrature of dφ/dτ
   (plan §4) are well conditioned near the axis (dφ/dτ = a(2r − aλ)/Δ + λ/sin²θ is a rational
   function integrated in Mino time), so validating them against the *direct* φ at the plan's
   1e-10 criterion is impossible on these rays: the reference itself is only good to 1e-7.
   The independent reference should be the geodesic ODE in Mino time integrated in extended
   precision (d²r/dτ² = R′(r)/2, d²θ/dτ² = Θ′(θ)/2 are smooth through turning points), started
   from Krang's first sample; that checks r, θ and the *differences* Δφ, Δt̃ along each ray,
   which are convention-free.
3. **BigFloat through Krang is not a shortcut.** Several constants in Krang are formed in
   Float64 before conversion (`T(√3/2)`, `T(1/3)`, `T(π/2)`), so extended precision does not
   improve its results (it even misclassifies root cases because `_isreal2` uses `eps(T)`).
   Worth an upstream patch, but the ODE reference is the right tool regardless.
4. **The α ≈ 0 columns matter physically** only through the emission at those positions;
   an error of 1e-7 M is far below the splat scales of interest, but it is an accuracy floor
   of the direct formulation that the recurrence/quadrature path should beat, not inherit.
