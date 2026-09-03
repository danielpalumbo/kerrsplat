# K2 in full recurrence mode: t̃ and φ by anchored quadrature of the Mino-time rates
# (plan §4, step 4), on top of the r, θ recurrence of recurrence.jl.
#
# Along a ray the regularized time and the azimuth advance by
#     dt/dτ = (r²+a²)(r²+a²−aλ)/Δ + a(λ − a sin²θ),   dφ/dτ = a(2r − aλ)/Δ + λ/sin²θ,
# rational in r and cos²θ (both known at every substep from the recurrence) and regular at the
# radial and polar turning points. Composite Simpson over each sample interval (the recurrence
# marches at half the sample spacing) integrates them; three singular pieces are removed first
# and integrated in closed form so that Simpson only ever sees smooth, bounded integrands:
#
#  * large r (the ends of the ray, r ~ 1/τ): dt/dτ ≈ r² + 2r + …, subtract |dr/dτ| (1 + 2/r),
#    whose integral is |Δ(r + 2 ln r)| on each monotone leg (the interval containing the
#    radial turning point of a scattering ray is split there);
#  * the horizon (plunging rays, 1/Δ = 1/((r − r₊)(r − r₋))): both poles by partial fractions,
#    subtracting (dr/dτ)/s₊ · Σ± Q± r±/(r(r − r±)) with Q± the residues at r± and s₊ = dr/dτ at
#    r₊; the integral is (1/s₊) Σ± Q± Δ ln((r − r±)/r). The r₋ pole is never reached but for
#    near-extremal spin it sits only r₊ − r₋ ≈ 2√(1−a²) inside the horizon and its tail is not
#    smooth on the sample spacing of the final plunge (a = 0.999: 5e-6 in t̃ without it);
#  * the polar axis (λ/sin²θ, a spike of width ~θ_min at every polar turning point): with
#    φ_am = am(X) and 1/sin²θ = C/(1 − n sn²X), ∫ λ dτ/sin²θ = λ C f [ g_c A(φ_am) +
#    (1/f)∫(1 − g_c dn X)/(1 − n sn²X) dτ ], where A(φ) = ∫ dφ/(1 − n sin²φ) =
#    arctan(√(1−n) tan φ)/√(1−n) (unfolded by the winding number of X), g_c = 1/√(1 − μ),
#    and the remaining integrand is bounded and O(μ) (μ is the polar parameter, small for
#    axis-grazing rays).
#
# Every M samples (fewer near the critical curve) the Jacobi states and (t̃, φ) are reset to
# Krang's direct evaluation; the largest |quadrature − direct| over a ray's anchors is stored
# per pixel as its error estimate. Validated against Krang at every sample (gate 3).

"""
    QuadratureConstants{T}

Per-ray constants of the subtractions described in the header of quadrature.jl.
"""
struct QuadratureConstants{T}
    a::T
    λ::T
    η::T
    rp::T          # outer horizon r₊
    rmn::T         # inner horizon r₋
    sp::T          # dr/dτ at the horizon on the inbound leg
    Qt::T          # residue of dt/dτ at r₊ (partial fractions of 1/Δ)
    Qtm::T         # residue at r₋
    Qϕ::T          # same for dφ/dτ
    Qϕm::T
    fo::T          # Mino time of the radial turning point
    r4::T          # its radius (four-real-root rays; unused otherwise)
    scattering::Bool
    # polar split: 1/sin²θ = C/(1 − n sn²(X|μ)), X = (τ + offset)/f
    C::T
    n::T
    sqrt_eps::T    # √(1 − n)
    gc::T          # 1/√(1 − μ)
    f::T
    offset::T
    K2::T          # 2K(μ): am(X + 2K) = am(X) + π
end

@inline function QuadratureConstants(pix::Krang.SlowLightIntensityPixel, rm::RadialMarcher, pm::PolarMarcher{T}, case) where {T}
    met = Krang.metric(pix)
    a = met.spin
    λ = Krang.λ(pix)
    η = Krang.η(pix)
    rp = one(T) + sqrt(one(T) - a^2)
    rmn = one(T) - sqrt(one(T) - a^2)
    E = rp^2 + a^2 - a * λ
    sp = -abs(E)
    Qt = (rp^2 + a^2) * E / (rp - rmn)
    Qtm = -(rmn^2 + a^2) * (rmn^2 + a^2 - a * λ) / (rp - rmn)
    Qϕ = a * (2 * rp - a * λ) / (rp - rmn)
    Qϕm = -a * (2 * rmn - a * λ) / (rp - rmn)
    scattering = case isa Case2 && !isfinite(radial_valid_until(rm))   # turning point outside the horizon
    r4 = case isa Case2 ? real(Krang.roots(pix)[4]) : T(NaN)
    μ = -pm.μ / (one(T) - pm.μ)                    # the (negative) parameter of the closed form
    up = pm.sqrt_up^2
    if pm.vortical
        C = inv(one(T) - pm.um)
        n = (up - pm.um) / (one(T) - pm.um)
    else
        C = one(T)
        n = up
    end
    K2 = 2 * JacobiElliptic.K(pm.μ) / pm.scale
    return QuadratureConstants(a, λ, η, rp, rmn, sp, Qt, Qtm, Qϕ, Qϕm, rm.fo, r4, scattering, C, n,
                               sqrt(max(one(T) - n, zero(T))), inv(sqrt(one(T) - μ)), pm.tempfac, pm.offset, K2)
end

"Antiderivative of the large-r subtraction |dr/dτ|(1 + 2/r) along a monotone leg."
@inline F_large_r(r) = r + 2 * log(r)


"""
    polar_A(qc, τ, sn, cn)

The elementary polar antiderivative A(am X) = ∫ dφ/(1 − n sin²φ), unfolded by the winding
number of X = (τ + offset)/f (am(X + 2K) = am(X) + π; tan(am X) = sn X / cn X).
"""
@inline function polar_A(qc::QuadratureConstants{T}, τ, sn, cn) where {T}
    X = (τ + qc.offset) / qc.f
    nw = floor((X + qc.K2 / 2) / qc.K2)
    return (atan(qc.sqrt_eps * sn / cn) + nw * T(π)) / qc.sqrt_eps
end

"""
    smooth_integrands(qc, r, cosθ, s, sn, cn, dn) -> (g_t, g_φ)

The Simpson integrands: the Mino-time rates minus the three singular pieces. `s = dr/dτ`
(signed); `sn, cn, dn` are the polar Jacobi functions at the closed form's own parameter.
"""
@inline function smooth_integrands(qc::QuadratureConstants{T}, r, cosθ, s, sn, cn, dn) where {T}
    a = qc.a
    λ = qc.λ
    Δ = r^2 - 2 * r + a^2
    s2 = 1 - cosθ * cosθ
    ft = (r^2 + a^2) * (r^2 + a^2 - a * λ) / Δ + a * (λ - a * s2)
    fϕ = a * (2 * r - a * λ) / Δ
    wp = s * qc.rp / (r * (r - qc.rp)) / qc.sp
    wm = s * qc.rmn / (r * (r - qc.rmn)) / qc.sp
    gt = ft - abs(s) * (1 + 2 / r) - qc.Qt * wp - qc.Qtm * wm
    gpol = λ * qc.C * (1 - qc.gc * dn) / (1 - qc.n * sn * sn)
    gϕ = fϕ - qc.Qϕ * wp - qc.Qϕm * wm + gpol
    return gt, gϕ
end

"""
    RayPoint

Everything the quadrature needs at one Mino time: r, θ, ν_r, ν_θ, dr/dτ, the polar Jacobi
functions at the closed form's parameter, and the two smooth integrands.
"""
struct RayPoint{T}
    r::T
    θ::T
    cosθ::T
    νr::Bool
    νθ::Bool
    s::T
    sn::T
    cn::T
    gt::T
    gϕ::T
end

@inline function ray_point(qc::QuadratureConstants{T}, rm::RadialMarcher, pm::PolarMarcher, xr::JacobiState, xθ::JacobiState, τ) where {T}
    r = radius(rm, xr)
    θ, νθ, cosθ = polar_angle_cos(pm, xθ)
    νr = radial_sign(rm, τ)
    s = (νr ? -one(T) : one(T)) * sqrt(max(Krang.r_potential(Krang.Kerr(qc.a), qc.η, qc.λ, r), zero(T)))
    sn = xθ.sn / (pm.scale * xθ.dn)
    cn = xθ.cn / xθ.dn
    dn = inv(xθ.dn)
    gt, gϕ = smooth_integrands(qc, r, cosθ, s, sn, cn, dn)
    return RayPoint(r, θ, cosθ, νr, νθ, s, sn, cn, gt, gϕ)
end

# direct (non-recurrence) evaluation of a ray point, for the turning-point split
@inline function ray_point_direct(qc::QuadratureConstants, rm::RadialMarcher, pm::PolarMarcher, τ)
    xr = jacobi_state(radial_argument(rm, τ), rm.k)
    xθ = jacobi_state(polar_argument(pm, τ), pm.μ)
    return ray_point(qc, rm, pm, xr, xθ, τ)
end

# the ray point at the radial turning point itself: r = r₄ and dr/dτ ≡ 0 for every spacetime,
# so s is an exact zero (√R has an infinite derivative at R = 0, which would poison dual numbers)
@inline function ray_point_turning(qc::QuadratureConstants{T}, rm::RadialMarcher, pm::PolarMarcher) where {T}
    τ = qc.fo
    xθ = jacobi_state(polar_argument(pm, τ), pm.μ)
    r = qc.r4
    θ, νθ, cosθ = polar_angle_cos(pm, xθ)
    s = zero(T)
    sn = xθ.sn / (pm.scale * xθ.dn)
    cn = xθ.cn / xθ.dn
    dn = inv(xθ.dn)
    gt, gϕ = smooth_integrands(qc, r, cosθ, s, sn, cn, dn)
    return RayPoint(r, θ, cosθ, true, νθ, s, sn, cn, gt, gϕ)
end

@inline simpson3(fa, fm, fb, h) = h / 6 * (fa + 4 * fm + fb)

"""
    interval_increment(qc, pa, pm_, pb, τa, τb) -> (Δt̃, Δφ)

Increment over one monotone (in r) interval from the points at its ends and midpoint: the
Simpson panel of the smooth integrands plus the closed-form pieces.
"""
@inline function interval_increment(qc::QuadratureConstants, pa::RayPoint, pm_::RayPoint, pb::RayPoint, τa, τb)
    h = τb - τa
    ra, rb = pa.r, pb.r
    ΔFr = (rb - ra) + 2 * log(rb / ra)                                   # Δ(r + 2 ln r)
    ΔFp = log((rb - qc.rp) * ra / ((ra - qc.rp) * rb)) / qc.sp            # Δ ln((r − r₊)/r) / s₊
    ΔFm = log((rb - qc.rmn) * ra / ((ra - qc.rmn) * rb)) / qc.sp          # Δ ln((r − r₋)/r) / s₊
    It = simpson3(pa.gt, pm_.gt, pb.gt, h) + abs(ΔFr) + qc.Qt * ΔFp + qc.Qtm * ΔFm
    Iϕ = simpson3(pa.gϕ, pm_.gϕ, pb.gϕ, h) + qc.Qϕ * ΔFp + qc.Qϕm * ΔFm +
         qc.λ * qc.C * qc.f * qc.gc * (polar_A(qc, τb, pb.sn, pb.cn) - polar_A(qc, τa, pa.sn, pa.cn))
    return It, Iϕ
end

"""
    march_ray(f, acc, pc, j, met, θo, case, Val(N), Val(M)) -> (acc, resid_t, resid_ϕ)

March the ray in sorted slot `j` with the recurrence plus anchored quadrature and fold the
consumer `f` over its samples:

    acc = f(acc, j, k, sample::GeodesicSample, Δτ, pix)      for k = 1 … N

where `pix` is Krang's pixel (η, λ, metric, screen coordinate, …) and `Δτ` the sample
spacing. `f` must be a bits type (a closure over device arrays or an isbits functor) because
this runs inside kernels. Returns the final accumulator and the ray's largest anchor residuals
in t̃ and φ. Both K2 front-ends use it: `quadrature_march!` folds a sample store, `fused_march!`
folds the caller's consumer without storing anything (plan §4 "fused mode", §7).
"""
@inline function march_ray(f::F, acc, pc, j, met::Krang.Kerr, θo, case, ::Val{N}, ::Val{M}) where {F,N,M}
    pix = build_pixel(pc, j, met, θo)
    T = typeof(Krang.total_mino_time(pix))
    rm = radial_marcher(case, pix)
    pm = PolarMarcher(pix)
    qc = QuadratureConstants(pix, rm, pm, case)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    h = Δτ / 2
    τ_valid = radial_valid_until(rm)
    Manchor = anchor_interval(M, one(T) - rm.k)
    Δr = jacobi_step_constants(-rm.c * h, rm.k)
    Δθ = jacobi_step_constants(pm.scale * h / pm.tempfac, pm.μ)
    vis = VisibilityConstants(pix)
    # first sample: states and Krang's t̃, φ (the first anchor)
    xr = jacobi_state(radial_argument(rm, Δτ), rm.k)
    xθ = jacobi_state(polar_argument(pm, Δτ), pm.μ)
    pa = ray_point(qc, rm, pm, xr, xθ, Δτ)
    d1 = direct_sample(pix, Δτ)
    t = d1.t
    ϕ = d1.ϕ
    rt = zero(T)
    rϕ = zero(T)
    acc = f(acc, j, 1, GeodesicSample(t, pa.r, pa.θ, ϕ, pa.νr, pa.νθ, (Δτ <= τ_valid) & krang_visible(vis, pa.cosθ)), Δτ, pix)
    for k in 2:N
        τa = (k - 1) * Δτ
        τb = k * Δτ
        # midpoint and end point of the interval
        xr = jacobi_step(xr, Δr, rm.k)
        xθ = jacobi_step(xθ, Δθ, pm.μ)
        pm_ = ray_point(qc, rm, pm, xr, xθ, τa + h)
        if (k - 1) % Manchor == 0
            xr = jacobi_state(radial_argument(rm, τb), rm.k)
            xθ = jacobi_state(polar_argument(pm, τb), pm.μ)
        else
            xr = jacobi_step(xr, Δr, rm.k)
            xθ = jacobi_step(xθ, Δθ, pm.μ)
        end
        pb = ray_point(qc, rm, pm, xr, xθ, τb)
        if qc.scattering && (τa < qc.fo < τb)
            # split at the radial turning point (dr/dτ = 0 there, r = r4)
            pt = ray_point_turning(qc, rm, pm)
            p1 = ray_point_direct(qc, rm, pm, (τa + qc.fo) / 2)
            p2 = ray_point_direct(qc, rm, pm, (qc.fo + τb) / 2)
            It1, Iϕ1 = interval_increment(qc, pa, p1, pt, τa, qc.fo)
            It2, Iϕ2 = interval_increment(qc, pt, p2, pb, qc.fo, τb)
            t += It1 + It2
            ϕ += Iϕ1 + Iϕ2
        else
            It, Iϕ = interval_increment(qc, pa, pm_, pb, τa, τb)
            t += It
            ϕ += Iϕ
        end
        if (k - 1) % Manchor == 0
            d = direct_sample(pix, τb)
            if d.ok
                rt = max(rt, abs(t - d.t))
                rϕ = max(rϕ, abs(ϕ - d.ϕ))
                t = d.t
                ϕ = d.ϕ
            end
        end
        acc = f(acc, j, k, GeodesicSample(t, pb.r, pb.θ, ϕ, pb.νr, pb.νθ, (τb <= τ_valid) & krang_visible(vis, pb.cosθ)), Δτ, pix)
        pa = pb
    end
    return acc, rt, rϕ
end

# consumer that stores every sample (K2 stored mode)
struct StoreSamples{S}
    S::S
end
@inline function (c::StoreSamples)(acc, j, k, s::GeodesicSample, Δτ, pix)
    store_sample!(c.S, j, k, s)
    return acc
end

# K2 stored mode: one thread per ray of one root case (`offset` = first slot of the case − 1)
@kernel function quadrature_march_kernel!(S, pc, resid_t, resid_ϕ, met::Krang.Kerr, θo, case, ::Val{N}, ::Val{M}, offset) where {N,M}
    j0 = @index(Global, Linear)
    j = j0 + offset
    _, rt, rϕ = march_ray(StoreSamples(S), nothing, pc, j, met, θo, case, Val(N), Val(M))
    @inbounds resid_t[j] = rt
    @inbounds resid_ϕ[j] = rϕ
end

# K2 fused mode: the consumer's accumulator per ray goes to `out[j]`
@kernel function fused_march_kernel!(out, pc, resid_t, resid_ϕ, met::Krang.Kerr, θo, case, f, acc0, ::Val{N}, ::Val{M}, offset) where {N,M}
    j0 = @index(Global, Linear)
    j = j0 + offset
    acc, rt, rϕ = march_ray(f, acc0, pc, j, met, θo, case, Val(N), Val(M))
    @inbounds out[j] = acc
    @inbounds resid_t[j] = rt
    @inbounds resid_ϕ[j] = rϕ
end

"""
    quadrature_march!(S, pc, resid_t, resid_ϕ, ranges, met, θo, Val(N), Val(M); workgroup = 128)

Fill `S` with (t̃, r, θ, φ, ν_r, ν_θ, ok) from the recurrence plus anchored quadrature, one
kernel launch per root case, writing each ray's largest anchor residual into `resid_t`,
`resid_ϕ`.
"""
function quadrature_march!(S::GeodesicSamples, pc::PixelConstants, resid_t, resid_ϕ, ranges, met::Krang.Kerr, θo,
                           ::Val{N}, ::Val{M}; workgroup::Integer = 128) where {N,M}
    npixels(S) == npixels(pc) || throw(DimensionMismatch("sample and pixel counts differ"))
    nsamples(S) == N || throw(DimensionMismatch("sample storage has $(nsamples(S)) columns, kernel is compiled for $N"))
    M >= 1 || throw(ArgumentError("re-anchoring interval must be positive"))
    backend = KA.get_backend(S.t)
    for (case, rng) in ((Case2(), ranges.case2), (Case3(), ranges.case3), (Case4(), ranges.case4))
        isempty(rng) && continue
        quadrature_march_kernel!(backend, workgroup)(S, pc, resid_t, resid_ϕ, met, θo, case, Val(N), Val(M), first(rng) - 1;
                                                     ndrange = length(rng))
    end
    return S
end

"""
    fused_march!(f, out, acc0, pc, resid_t, resid_ϕ, ranges, met, θo, Val(N), Val(M); workgroup = 128)

Run the marcher with the consumer `f` folded over every ray's samples (see [`march_ray`](@ref))
and write each ray's final accumulator to `out` (sorted order); nothing is stored per sample.
"""
function fused_march!(f, out, acc0, pc::PixelConstants, resid_t, resid_ϕ, ranges, met::Krang.Kerr, θo,
                      ::Val{N}, ::Val{M}; workgroup::Integer = 128) where {N,M}
    length(out) == npixels(pc) || throw(DimensionMismatch("output and pixel counts differ"))
    backend = KA.get_backend(out)
    for (case, rng) in ((Case2(), ranges.case2), (Case3(), ranges.case3), (Case4(), ranges.case4))
        isempty(rng) && continue
        fused_march_kernel!(backend, workgroup)(out, pc, resid_t, resid_ϕ, met, θo, case, f, acc0, Val(N), Val(M), first(rng) - 1;
                                                ndrange = length(rng))
    end
    return out
end
