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

---

# Addendum (same day): radius near the critical curve, and the Jacobi amplitude

Found while validating the recurrence marcher (plan step 3, gate 2) against a BigFloat
re-evaluation of the closed forms (`test/highprec_reference.jl`: Krang's Float64 roots
Newton-polished in 256-bit arithmetic, then the same Gralla–Lupsasca expressions in BigFloat).

## Krang's direct r(τ) loses accuracy from 1 − k ≈ 1e-2 inward

Errors of the direct evaluation against the BigFloat reference (a = 0.94, θo = 60°, pixels at
screen radius ρ_c(1 ± δ) around the critical curve; "r < 50" restricts to samples inside 50 M):

| side | δ | 1 − k | max rel. error in r | r < 50 |
|---|---|---|---|---|
| out (4 real roots) | 1e-3 | 4.4e-2 | 7e-12 | 1e-12 |
| out | 1e-4 | 1.4e-2 | 1.0e-10 | 2e-11 |
| out | 1e-5 | 4.5e-3 | 5.8e-10 | 1.4e-10 |
| out | 1e-6 | 1.4e-3 | 2.3e-9 | 7.7e-10 |
| out | 1e-8 | 1.4e-4 | 7.4e-8 | 2.5e-8 |
| in (2 real roots) | 1e-5 | 1.3e-6 | 2.6e-10 | 2.6e-10 |
| in | 1e-6 | 1.3e-7 | 2.2e-9 | 2.2e-9 |
| in | 1e-7 | 1.3e-8 | **1.3** | 0.2 |

θ stays at 1e-13 to 1e-15 throughout. Two mechanisms:

1. **Root conditioning (dominant).** Near the critical curve r₃ ≈ r₄ is a near-double root of
   the radial quartic. With coefficients rounded at ε ≈ 1e-16 the roots are determined only to
   ε/r₄₃, so r₄₃ itself (and 1 − k, which sets the number of half-orbits before escape) carries
   a relative error ε/r₄₃² — 1e-8 at 1 − k ≈ 1e-4. Every downstream quantity inherits it. Both
   the direct and the recurrence path use the same roots and show the same error. The fix, if
   ever needed, is to solve the quartic in extended precision in K1 (cheap: K1 is 18 ms for
   65,536 pixels); on a 256² screen of ±10 M only 16 pixels have 1 − k < 1e-4 and 1648 have
   1 − k < 1e-2, so the practical impact is a ring of pixels with r accurate to 1e-10…1e-8.
2. **`JacobiElliptic._am` asymptotic branch.** For 1 − m < √eps ≈ 1.5e-8 the amplitude switches
   to A&S 16.15.4, a first-order expansion whose error term m₁ cosh(u) grows without bound: at
   u ≈ 2K ≈ 21 the returned sn, cn are wrong by 0.4. Krang's direct evaluation returns nonsense
   there (negative radii). Up to 1 − m = 1e-7 the AGM branch is accurate to 4e-15. Worth an
   upstream issue; `NEAR_CRITICAL_ONE_MINUS_K = 1e-7` flags such pixels in `PixelConstants`
   (none occur on ordinary grids).

A third, smaller effect is specific to Krang's form of the four-real-root radius,
r = (r₃₁ r₄ − r₃ r₄₁ sn²)/(r₃₁ − r₄₁ sn²): the denominator cancels near the observer end for
near-critical rays, so the absolute rounding of sn² becomes a relative error ~ r/r₄₃ in r. The
recurrence uses the equivalent r = (r₃ r₄₁ cn² − r₁ r₄₃)/(r₄₁ cn² − r₃₄), which cancels between
quantities carried with relative accuracy. With Float64 roots the improvement is invisible
(mechanism 1 dominates), but it costs nothing.

## Per-ray constants must be the ones Krang used

Krang stores I0_inf per pixel and evaluates the Jacobi functions with a parameter k rebuilt at
every sample from the roots. Any marcher that rebuilds k differently — even by one ulp, e.g.
`abs(z)` (hypot) instead of Krang's `√abs2(z)` for |r₃ − r₂| — pays for it on near-critical
case-3 rays: at the first samples the argument is X ≈ 1.9 K ≈ 16 and ∂cn/∂k ≈ e^X/8 ≈ 1e6, so
the inconsistency between the k inside I0_inf and the k inside cn moved r by 5e-9 while the
self-consistent pairs (Krang's, and the BigFloat reference's) agree to 4e-14. `radial_marcher`
therefore copies Krang's expressions token for token. The anchored quadrature of t̃ and φ (plan
step 4) must respect the same rule for every constant it shares with Krang's anchors.

## Recurrence accuracy

With re-anchoring every 64 samples, the recurrence for r and θ is indistinguishable from the
direct evaluation against the BigFloat reference at every tested pixel (θ to 1e-13; r to 1e-13
away from the critical curve and to the root-limited values above near it), the momentum signs
ν_r, ν_θ and the validity flags agree with Krang's at every sample, and the per-sample cost is
a few nanoseconds instead of the ~12 elliptic evaluations of the direct path.
