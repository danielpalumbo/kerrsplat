# K2 in recurrence mode: r(τ) and θ(τ) by the Jacobi addition theorems (plan §4, step 3).
#
# Krang's closed forms (misc.jl `_rs_case*`, emission_coordinates.jl `_θs`) all have the shape
# "Jacobi function of an argument linear in Mino time", so consecutive samples are related by
# the addition theorem with a per-ray step constant. Every sample then costs a rational update;
# the elliptic functions are evaluated only at the first sample, for the step constants, and at
# the re-anchoring samples (every M-th), which reset the accumulated rounding error.
#
# Conventions verified against Krang at every sample (session diagnostics, 2026-09-03):
# * ν_r = (τ < I0_inf): true before the radial turning point, i.e. the photon (which travels
#   toward decreasing τ) moves outward;
# * ν_θ = (dθ/dτ < 0): the sign of p^θ along the photon's direction;
# * ordinary polar motion: cos θ = sign(β) √u₊ sn((τ + sign(β) G_θ^o) / f | m), m = u₊/u₋ < 0,
#   f = 1/√(−u₋ a²); vortical (η < 0): cos²θ = u₋ + (u₊ − u₋) sn²((τ + σ sign(β) G_θ^o)/f | 1 − m)
#   with σ = sign(cos θo), cos θ keeping the sign σ, m = u₊/u₋ > 1.

"""
    Case2, Case3, Case4

Krang's radial root cases: four real roots (cases 1 and 2, scattering or turning inside the
horizon), two real roots (case 3), no real roots (case 4). Singleton dispatch tags for the
per-case kernels.
"""
struct Case2 end
struct Case3 end
struct Case4 end

# ---- radial marchers ---------------------------------------------------------------------

"""
    RadialCase2{T}, RadialCase3{T}, RadialCase4{T}

Per-ray constants of Krang's radial closed forms, r = f(Jacobi function of X(τ)), with
X(τ) = c (I0_inf − τ) and parameter k ∈ (0, 1). Built by `radial_marcher(case, pix)`.
"""
struct RadialCase2{T}
    k::T
    c::T
    fo::T
    r1::T
    r3::T
    r41::T
    r43::T
    τ_horizon::T   # last valid τ when the turning point r4 lies inside the horizon (case 1); Inf otherwise
end
struct RadialCase3{T}
    k::T
    c::T
    fo::T
    n0::T          # r = (n0 + n1 cn X) / (d0 + d1 cn X)
    n1::T
    d0::T
    d1::T
end
struct RadialCase4{T}
    k::T
    c::T
    fo::T
    a2::T          # r = −(a2 (g0 − sc X)/(1 + g0 sc X) + b1)
    b1::T
    g0::T
end

@inline function radial_marcher(::Case2, pix::Krang.SlowLightIntensityPixel)
    # constants exactly as in Krang's `_rs_case1_and_2` (see the case-3 note below)
    met = Krang.metric(pix)
    T = typeof(met.spin)
    radial_roots = real.(Krang.roots(pix))
    r1, r2, r3, r4 = radial_roots
    _, r31, r32, r41, r42, _ = Krang._get_root_diffs(radial_roots...)
    k = r32 * r41 / (r31 * r42)
    c = √(r31 * r42) / 2
    fo = Krang.I0_inf(pix)
    rh = Krang.horizon(met)
    τ_horizon = r4 < rh ? fo - Krang.Ir_s_case1_and_2(met, rh, (r1, r2, r3, r4), true) : T(Inf)
    return RadialCase2(k, c, fo, r1, r3, r41, r4 - r3, τ_horizon)
end

@inline function radial_marcher(::Case3, pix::Krang.SlowLightIntensityPixel)
    # Bit-for-bit Krang's `_rs_case3` constants: the stored I0_inf was computed with this k,
    # and cn(X | k) at the large arguments of the first samples of near-critical rays has
    # ∂cn/∂k ~ e^X/8 ~ 1e6, so a one-ulp difference in k (e.g. from `abs` instead of `√abs2`)
    # costs 1e-9 in r.
    roots = Krang.roots(pix)
    r1, r2, _, _ = roots
    r21, r31, r32, _, _, _ = Krang._get_root_diffs(roots...)
    r1, r2, r21 = real.((r1, r2, r21))
    A = √abs2(r32)
    B = √abs2(r31)
    k = ((A + B)^2 - r21^2) / (4 * A * B)
    c = √(A * B)
    return RadialCase3(k, c, Krang.I0_inf(pix), -A * r1 + B * r2, A * r1 + B * r2, B - A, A + B)
end

@inline function radial_marcher(::Case4, pix::Krang.SlowLightIntensityPixel)
    T = typeof(Krang.metric(pix).spin)
    roots = Krang.roots(pix)
    r2, r4 = roots[2], roots[4]
    a1 = abs(imag(r4))
    a2 = abs(imag(r2))
    b1 = real(r4)
    b2 = real(r2)
    C = sqrt((a1 - a2)^2 + (b1 - b2)^2)
    D = sqrt((a1 + a2)^2 + (b1 - b2)^2)
    k = 4 * C * D / (C + D)^2
    c = (C + D) / 2
    g0 = sqrt(max((4 * a2^2 - (C - D)^2) / ((C + D)^2 - 4 * a2^2), zero(T)))
    return RadialCase4(k, c, Krang.I0_inf(pix), a2, b1, g0)
end

const RadialMarcher{T} = Union{RadialCase2{T},RadialCase3{T},RadialCase4{T}}

"""
    radial_parameter(pix) -> k

The elliptic parameter k ∈ (0, 1) of Krang's radial closed form for this pixel (any root
case). 1 − k → 0 at the critical curve, where the closed forms (and their Jacobi functions)
lose accuracy; see [`NEAR_CRITICAL_ONE_MINUS_K`](@ref).
"""
@inline function radial_parameter(pix::Krang.SlowLightIntensityPixel)
    n = num_real_roots(Krang.roots(pix))
    return n == 4 ? radial_marcher(Case2(), pix).k :
           n == 2 ? radial_marcher(Case3(), pix).k : radial_marcher(Case4(), pix).k
end

"""
    NEAR_CRITICAL_ONE_MINUS_K

Pixels with 1 − k below this are flagged near-critical (`near_critical(pc, j)`): below
1.5e-8 JacobiElliptic's amplitude switches to a first-order asymptotic formula that is wrong
for large arguments (Krang's direct evaluation returns nonsense there as well), and between
that and ~1e-7 the closed forms are conditioned worse than 1e-10. Consumers should treat such
rays' samples with suspicion; they are a vanishing fraction of any screen.
"""
const NEAR_CRITICAL_ONE_MINUS_K = 1e-7

"Whether sorted slot `j` is a near-critical pixel (see [`NEAR_CRITICAL_ONE_MINUS_K`](@ref))."
@inline near_critical(pc::PixelConstants, j::Integer) = @inbounds (1 - pc.k_r[j]) < NEAR_CRITICAL_ONE_MINUS_K

"""
    anchor_interval(M, one_minus_k)

Re-anchoring interval for a ray with radial parameter k: `M` for 1 − k ≥ 2e-5, shrinking like
√(1 − k) toward the critical curve (down to 4). The recurrence's per-step rounding error grows
like (1 − k)^(−1/2) (2e-15 at 1 − k = 0.1, 3e-12 at 1e-7, gate-2 measurements), and near the
critical curve the plunging samples close to the horizon amplify it; the shorter interval keeps
the accumulated error below 1e-11 at negligible cost (near-critical rays are a small fraction
of any screen).
"""
@inline function anchor_interval(M::Integer, one_minus_k::T) where {T}
    scaled = M * sqrt(max(one_minus_k, zero(T)) / T(2e-5))
    return max(4, min(M, unsafe_trunc(Int, min(scaled, T(M)))))
end

"Argument of the radial Jacobi function at Mino time τ."
@inline radial_argument(rm::RadialMarcher, τ) = rm.c * (rm.fo - τ)

"""
    radius(rm, x)

Radius from the radial Jacobi state. For four real roots Krang writes
r = (r₃₁ r₄ − r₃ r₄₁ sn²X)/(r₃₁ − r₄₁ sn²X); the denominator vanishes at the observer end
(sn²X → r₃₁/r₄₁) and, for near-critical rays (r₄₃ → 0), is a cancellation of two O(1) terms
that turns the absolute rounding of sn² into a relative error ~ r/r₄₃ in r. The equivalent
form used here, r = (r₃ r₄₁ cn²X − r₁ r₄₃)/(r₄₁ cn²X − r₄₃), cancels between quantities the
recurrence carries with *relative* accuracy (cn ~ sech X stays small but well determined), so
the error scales as r/r₃₁ instead.
"""
@inline function radius(rm::RadialCase2, x::JacobiState)
    cn2 = rm.r41 * x.cn * x.cn
    return (rm.r3 * cn2 - rm.r1 * rm.r43) / (cn2 - rm.r43)
end
@inline radius(rm::RadialCase3, x::JacobiState) = (rm.n0 + rm.n1 * x.cn) / (rm.d0 + rm.d1 * x.cn)
@inline function radius(rm::RadialCase4, x::JacobiState)
    sc = x.sn / x.cn
    return -(rm.a2 * (rm.g0 - sc) / (1 + rm.g0 * sc) + rm.b1)
end

"Radial momentum sign (photon direction): before the turning point the photon moves outward."
@inline radial_sign(rm::RadialMarcher, τ) = τ < rm.fo

"Last Mino time at which Krang's radial evaluation is valid."
@inline radial_valid_until(rm::RadialCase2) = rm.τ_horizon
@inline radial_valid_until(rm::RadialMarcher{T}) where {T} = T(Inf)

# ---- polar marcher -----------------------------------------------------------------------

"""
    PolarMarcher{T}

Per-ray constants of Krang's polar closed form (see the header of this file). The Jacobi
parameter of the closed form is negative; the recurrence runs at the transformed parameter
`μ ∈ (0, 1)` with arguments scaled by `scale` (see [`negative_parameter_transform`](@ref)).
"""
struct PolarMarcher{T}
    μ::T
    scale::T
    tempfac::T
    offset::T      # sign(β) G_θ^o (ordinary) or σ sign(β) G_θ^o (vortical)
    sqrt_up::T
    um::T
    up_m_um::T
    sign::T        # sign(β) (ordinary) or σ = sign(cos θo) (vortical)
    vortical::Bool
end

@inline function PolarMarcher(pix::Krang.SlowLightIntensityPixel)
    met = Krang.metric(pix)
    a = met.spin
    T = typeof(a)
    θo = Krang.inclination(pix)
    η = Krang.η(pix)
    λ = Krang.λ(pix)
    _, β = Krang.screen_coordinate(pix)
    signβ = sign(β)
    Gθo, _ = Krang.absGθo_Gθhat(pix)
    # exactly Krang's `_θs` constants (same rounding)
    Δθ = (1 - (η + λ^2) / a^2) / 2
    dsc = sqrt(Δθ^2 + η / a^2)
    up = Δθ + dsc
    um = Δθ - dsc
    m = up / um
    tempfac = inv(sqrt(abs(um * a^2)))
    vortical = η < zero(T)
    if vortical
        k = one(T) - m
        σ = θo > T(π / 2) ? -one(T) : one(T)
        offset = σ * signβ * Gθo
        s = σ
    else
        k = m
        offset = signβ * Gθo
        s = signβ
    end
    μ, scale = negative_parameter_transform(k)
    return PolarMarcher(μ, scale, tempfac, offset, sqrt(up), um, up - um, s, vortical)
end

"Transformed argument of the polar Jacobi state at Mino time τ."
@inline polar_argument(pm::PolarMarcher, τ) = pm.scale * (τ + pm.offset) / pm.tempfac

"""
    polar_angle(pm, x) -> (θ, νθ)
    polar_angle_cos(pm, x) -> (θ, νθ, cos θ)

Polar angle and polar momentum sign (photon direction) from the polar Jacobi state.
"""
@inline function polar_angle_cos(pm::PolarMarcher{T}, x::JacobiState) where {T}
    snX = x.sn / (pm.scale * x.dn)                 # sn of the original (negative) parameter
    if pm.vortical
        cosθ = pm.sign * sqrt(max(pm.um + pm.up_m_um * snX * snX, zero(T)))
        νθ = pm.sign * x.sn * x.cn > zero(T)
    else
        cosθ = pm.sign * pm.sqrt_up * snX
        νθ = pm.sign * x.cn > zero(T)
    end
    cosθ = clamp(cosθ, -one(T), one(T))
    return acos(cosθ), νθ, cosθ
end
@inline polar_angle(pm::PolarMarcher, x::JacobiState) = polar_angle_cos(pm, x)[1:2]

# ---- validity ----------------------------------------------------------------------------

"""
    krang_visible(pix, θs) -> Bool
    krang_visible(vis::VisibilityConstants, cos θs) -> Bool

Krang's screen-boundary consistency check in `emission_coordinates(pix, τ)`: reproduced so
that the recurrence marcher flags exactly the samples Krang flags. The second form uses the
per-ray constants and cos θs only (no trigonometric call), with Krang's own expressions
`αboundary = a sin θs` and `βboundary² = (cos²θo − cos²θs)(α² − a² sin²θs)/(cos²θs − 1)`.
"""
@inline function krang_visible(pix::Krang.SlowLightIntensityPixel, θs::T) where {T}
    α, β = Krang.screen_coordinate(pix)
    θo = Krang.inclination(pix)
    met = Krang.metric(pix)
    if cos(θs) > abs(cos(θo))
        αmin = Krang.αboundary(met, θs)
        βbound = abs(α) >= (αmin + eps(T)) ? Krang.βboundary(met, α, θo, θs) : zero(T)
        (abs(β) + eps(T)) < βbound && return false
    end
    return true
end

struct VisibilityConstants{T}
    a::T
    absα::T
    α2::T
    absβ_eps::T
    abscosθo::T
    cos2θo::T
end
@inline function VisibilityConstants(pix::Krang.SlowLightIntensityPixel)
    α, β = Krang.screen_coordinate(pix)
    T = typeof(α)
    cθo = cos(Krang.inclination(pix))
    return VisibilityConstants(Krang.metric(pix).spin, abs(α), α * α, abs(β) + eps(T), abs(cθo), cθo * cθo)
end
@inline function krang_visible(v::VisibilityConstants{T}, cosθs) where {T}
    cosθs > v.abscosθo || return true
    c2 = cosθs * cosθs
    s2 = one(T) - c2
    αmin = v.a * sqrt(s2)
    v.absα >= αmin + eps(T) || return true              # βboundary = 0
    temp = (v.cos2θo - c2) * (v.α2 - v.a * v.a * s2) / (c2 - one(T))
    βbound = sqrt(max(temp, zero(temp)))
    return !(v.absβ_eps < βbound)
end

# ---- the kernel ---------------------------------------------------------------------------

# One thread per ray of one root case (pixels are case-sorted, so `offset` is the first slot of
# the case minus one). N samples, re-anchored every M samples (fewer near the critical curve,
# see `anchor_interval`). Only r, θ and the momentum signs are produced here; t̃ and φ are
# written as NaN until the quadrature of plan §4 step 4 lands.
@kernel function recurrence_march_kernel!(S, pc, met::Krang.Kerr, θo, case, ::Val{N}, ::Val{M}, offset) where {N,M}
    j0 = @index(Global, Linear)
    j = j0 + offset
    pix = build_pixel(pc, j, met, θo)
    T = typeof(Krang.total_mino_time(pix))
    rm = radial_marcher(case, pix)
    pm = PolarMarcher(pix)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    τ_valid = radial_valid_until(rm)
    Manchor = anchor_interval(M, one(T) - rm.k)
    Δr = jacobi_step_constants(-rm.c * Δτ, rm.k)
    Δθ = jacobi_step_constants(pm.scale * Δτ / pm.tempfac, pm.μ)
    xr = jacobi_state(radial_argument(rm, Δτ), rm.k)
    xθ = jacobi_state(polar_argument(pm, Δτ), pm.μ)
    for k in 1:N
        τ = k * Δτ
        if k > 1
            if (k - 1) % Manchor == 0
                xr = jacobi_state(radial_argument(rm, τ), rm.k)
                xθ = jacobi_state(polar_argument(pm, τ), pm.μ)
            else
                xr = jacobi_step(xr, Δr, rm.k)
                xθ = jacobi_step(xθ, Δθ, pm.μ)
            end
        end
        r = radius(rm, xr)
        θ, νθ = polar_angle(pm, xθ)
        νr = radial_sign(rm, τ)
        ok = (τ <= τ_valid) & krang_visible(pix, θ)
        store_sample!(S, j, k, GeodesicSample(T(NaN), r, θ, T(NaN), νr, νθ, ok))
    end
end

"""
    recurrence_march!(S, pc, ranges, met, θo, Val(N), Val(M); workgroup = 128)

Fill `S` with r, θ, ν_r, ν_θ from the addition-theorem recurrence, one kernel launch per root
case (`ranges` from [`case_permutation`](@ref)). t̃ and φ are set to NaN (plan §4 step 4).
"""
function recurrence_march!(S::GeodesicSamples, pc::PixelConstants, ranges, met::Krang.Kerr, θo,
                           ::Val{N}, ::Val{M}; workgroup::Integer = 128) where {N,M}
    npixels(S) == npixels(pc) || throw(DimensionMismatch("sample and pixel counts differ"))
    nsamples(S) == N || throw(DimensionMismatch("sample storage has $(nsamples(S)) columns, kernel is compiled for $N"))
    M >= 1 || throw(ArgumentError("re-anchoring interval must be positive"))
    backend = KA.get_backend(S.t)
    for (case, rng) in ((Case2(), ranges.case2), (Case3(), ranges.case3), (Case4(), ranges.case4))
        isempty(rng) && continue
        recurrence_march_kernel!(backend, workgroup)(S, pc, met, θo, case, Val(N), Val(M), first(rng) - 1;
                                                     ndrange = length(rng))
    end
    return S
end
