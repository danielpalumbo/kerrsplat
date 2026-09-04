# Candidate upstream issues (Krang.jl, JacobiElliptic.jl)

Collected while building the GPU geodesic layer (2026-09-03). Each item is reproduced in
this repository's tests or session diagnostics; none has been filed upstream yet.

## Krang.jl (main at f36f43a)

1. **Concrete pixel type for device arrays.** `SlowLightIntensityPixel{T}` leaves 16 of 17
   type parameters abstract, so the KernelAbstractions extension allocates a non-isbits array
   that CUDA rejects. `KerrSplat.Geodesics.ConcretePixel{T}` is the fully concrete alias.
   (GPU plan §2 item 2.)
2. **`generate_rays(pixels, N; A=CuArray)` calls `first(pixels)` on the device array** (scalar
   indexing). (GPU plan §2 item 4.)
3. **`unsafe_trunc(Int, τ/τ̂)` in `_θs` has no `ForwardDiff.Dual` method**, which is a dynamic
   dispatch inside GPU kernels with duals. `KerrSplat.Geodesics` defines the one-line method.
   (GPU plan §2 item 5.)
4. **Per-thread stack.** Krang's Float64 radial integrals need more than CUDA's default 1 KB
   stack (`CUDA.limit!(CUDA.LIMIT_STACK_SIZE, 4096)` suffices for pixel construction and for
   `emission_coordinates` in a loop of 200). A note in the docs would save users an illegal
   memory access.
5. **ν_θ on vortical geodesics after a polar turning point.** For η < 0 pixels,
   `emission_coordinates(pix, τ)` returns `νθ = false` for the samples after the ray turns
   in θ while its own θ(τ) is decreasing there (finite differences of θ(τ) give
   dθ/dτ < 0, which on every other geodesic corresponds to `νθ = true`). Example: a = 0.94,
   θo = 60°, α = ±0.2128, β = −0.2128 (β < 0 side), the last dozen of 200 uniform Mino-time
   samples; the β > 0 mirror pixels are fine. The
   sign feeds `p_bl_d`, so p_θ has the wrong sign on those samples. The cause is the
   `isindir` reconstruction in `_θs` (`τ1 ≈ τ` test) for the vortical branch. The same
   bookkeeping makes `emission_coordinates`' t̃ and φ wrong on those samples (by 1e-2 at
   the end of the ray): finite differences of its φ(τ) disagree with the Mino-time rate by
   1.5 %, while a quadrature of the rate agrees with a BigFloat reference to 4e-10
   (docs/notes/2026-09-03_quadrature_findings.md).
9. **Isolated ~1e-7 glitches in t̃.** `emission_coordinates` at a = 0.7, θo = 120°,
   α = 1.064, β = −4.468, τ = 0.20976 returns t̃ off the smooth curve through its neighbours
   by 6.7e-8 (neighbours ≤ 3.5e-9). Probably the `arg == k → arg += eps` guards around
   `Pi` or a Carlson iteration hitting its cap.
6. **Constants formed in Float64.** `T(√3/2)`, `T(1/3)`, `T(π/2)`, `T(2π)` in
   `get_radial_roots`, `_θs`, `Gθ`, … are exact only for T = Float64, so evaluating Krang in
   `BigFloat`/`Double64` does not gain precision (and `_isreal2`'s `eps(T)` tolerance then
   misclassifies root cases). Writing them as `sqrt(T(3))/2`, `T(1)/3`, `T(π)/2` would make the
   package usable as its own extended-precision reference.
7. **Polar constant u₊ by cancellation.** `_θs`/`_absGθo_Gθhat` form u₊ = Δθ + √(Δθ² + η/a²) with
   Δθ ≈ −(η + λ²)/(2a²) large and negative for rays grazing the polar axis; the stable form
   u₊ = (η/a²)/(√(Δθ² + η/a²) − Δθ) removes a rounding amplification that Π(n = u₊ → 1) in
   G_φ then magnifies (docs/notes/2026-09-03_direct_azimuth_conditioning.md).
8. **Near-critical radii are root-limited.** Near the critical curve r₃ ≈ r₄ and Float64
   roots carry relative errors ε/r₄₃² in r₄₃; the four-real-root radius formula additionally
   amplifies the absolute rounding of sn² by r/r₄₃ (an equivalent cn² form avoids that). See the
   same note's addendum. Solving the quartic in extended precision would remove the dominant
   error for a negligible cost.

10. **`ϕ_kerr_schild` / `ϕ_BL` cannot compile in CUDA kernels** because of the `@warn` on
    the horizon (`has_metal()` only gates Metal): `boyer_lindquist_to_quasi_cartesian_kerr_schild*`
    fail with `InvalidIRError` on the GPU. `KerrSplat.Geodesics.quasi_cartesian_kerr_schild`
    is the same map without the warning.

## JacobiElliptic.jl (0.3.10)

1. **`_am` asymptotic branch for 1 − m < √eps.** A&S 16.15.4 is used for m₁ < √eps ≈ 1.5e-8,
   but its error term grows like m₁ cosh(u); at u ≈ 2K ≈ 21 (m = 1 − 1e-8) sn and cn are
   wrong by 0.4, whereas the AGM branch is accurate to 4e-15 at m = 1 − 1e-7. Either extend the
   AGM branch (it converges fine there) or use the `tanh` limit with an error bound.
2. **`_sqrt` is defined only for Float32/Float64/duals**, which blocks `BigFloat` evaluation
   (the rest of the Carlson code is generic). `_sqrt(x) = sqrt(x)` as a fallback would do.
3. **`ellipj` is not reachable at top level** although listed in the exports (defined in
   `CarlsonAlg` only); `JacobiElliptic.CarlsonAlg.ellipj` works.
