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
#  * the horizon (plunging rays, 1/Δ → 1/(r − r₊)): subtract Q (dr/dτ)(r₊/(r(r − r₊)))/s₊ with
#    Q the coefficient of 1/(r − r₊) at the horizon and s₊ = dr/dτ there; its integral is
#    (Q/s₊) Δ ln((r − r₊)/r);
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
    rp::T          # outer horizon
    sp::T          # dr/dτ at the horizon on the inbound leg
    Qt::T          # coefficient of 1/(r − r₊) in dt/dτ at the horizon
    Qϕ::T          # same for dφ/dτ
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
    Qϕ = a * (2 * rp - a * λ) / (rp - rmn)
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
    return QuadratureConstants(a, λ, η, rp, sp, Qt, Qϕ, rm.fo, r4, scattering, C, n,
                               sqrt(max(one(T) - n, zero(T))), inv(sqrt(one(T) - μ)), pm.tempfac, pm.offset, K2)
end

"Antiderivative of the large-r subtraction |dr/dτ|(1 + 2/r) along a monotone leg."
@inline F_large_r(r) = r + 2 * log(r)

"Antiderivative of the horizon subtraction (per unit Q)."
@inline F_horizon(qc::QuadratureConstants, r) = log((r - qc.rp) / r) / qc.sp

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
    smooth_integrands(qc, r, θ, s, sn, cn, dn) -> (g_t, g_φ)

The Simpson integrands: the Mino-time rates minus the three singular pieces. `s = dr/dτ`
(signed); `sn, cn, dn` are the polar Jacobi functions at the closed form's own parameter.
"""
@inline function smooth_integrands(qc::QuadratureConstants{T}, r, θ, s, sn, cn, dn) where {T}
    a = qc.a
    λ = qc.λ
    Δ = r^2 - 2 * r + a^2
    s2 = sin(θ)^2
    ft = (r^2 + a^2) * (r^2 + a^2 - a * λ) / Δ + a * (λ - a * s2)
    fϕ = a * (2 * r - a * λ) / Δ
    w = s * qc.rp / (r * (r - qc.rp)) / qc.sp
    gt = ft - abs(s) * (1 + 2 / r) - qc.Qt * w
    gpol = λ * qc.C * (1 - qc.gc * dn) / (1 - qc.n * sn * sn)
    gϕ = fϕ - qc.Qϕ * w + gpol
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
    θ, νθ = polar_angle(pm, xθ)
    νr = radial_sign(rm, τ)
    s = (νr ? -one(T) : one(T)) * sqrt(max(Krang.r_potential(Krang.Kerr(qc.a), qc.η, qc.λ, r), zero(T)))
    sn = xθ.sn / (pm.scale * xθ.dn)
    cn = xθ.cn / xθ.dn
    dn = inv(xθ.dn)
    gt, gϕ = smooth_integrands(qc, r, θ, s, sn, cn, dn)
    return RayPoint(r, θ, νr, νθ, s, sn, cn, gt, gϕ)
end

# direct (non-recurrence) evaluation of a ray point, for the turning-point split
@inline function ray_point_direct(qc::QuadratureConstants, rm::RadialMarcher, pm::PolarMarcher, τ)
    xr = jacobi_state(radial_argument(rm, τ), rm.k)
    xθ = jacobi_state(polar_argument(pm, τ), pm.μ)
    return ray_point(qc, rm, pm, xr, xθ, τ)
end

@inline simpson3(fa, fm, fb, h) = h / 6 * (fa + 4 * fm + fb)

"""
    interval_increment(qc, pa, pm_, pb, τa, τb) -> (Δt̃, Δφ)

Increment over one monotone (in r) interval from the points at its ends and midpoint: the
Simpson panel of the smooth integrands plus the closed-form pieces.
"""
@inline function interval_increment(qc::QuadratureConstants, pa::RayPoint, pm_::RayPoint, pb::RayPoint, τa, τb)
    h = τb - τa
    It = simpson3(pa.gt, pm_.gt, pb.gt, h) + abs(F_large_r(pb.r) - F_large_r(pa.r)) +
         qc.Qt * (F_horizon(qc, pb.r) - F_horizon(qc, pa.r))
    Iϕ = simpson3(pa.gϕ, pm_.gϕ, pb.gϕ, h) + qc.Qϕ * (F_horizon(qc, pb.r) - F_horizon(qc, pa.r)) +
         qc.λ * qc.C * qc.f * qc.gc * (polar_A(qc, τb, pb.sn, pb.cn) - polar_A(qc, τa, pa.sn, pa.cn))
    return It, Iϕ
end

# The kernel: like recurrence_march_kernel! but marching at half the sample spacing and
# accumulating t̃, φ. `resid_t`, `resid_ϕ`: per-pixel largest anchor residuals (error estimates).
@kernel function quadrature_march_kernel!(S, pc, resid_t, resid_ϕ, met::Krang.Kerr, θo, case, ::Val{N}, ::Val{M}, offset) where {N,M}
    j0 = @index(Global, Linear)
    j = j0 + offset
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
    # first sample: states and Krang's t̃, φ (the first anchor)
    xr = jacobi_state(radial_argument(rm, Δτ), rm.k)
    xθ = jacobi_state(polar_argument(pm, Δτ), pm.μ)
    pa = ray_point(qc, rm, pm, xr, xθ, Δτ)
    d1 = direct_sample(pix, Δτ)
    t = d1.t
    ϕ = d1.ϕ
    rt = zero(T)
    rϕ = zero(T)
    store_sample!(S, j, 1, GeodesicSample(t, pa.r, pa.θ, ϕ, pa.νr, pa.νθ, (Δτ <= τ_valid) & krang_visible(pix, pa.θ)))
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
            pt = ray_point_direct(qc, rm, pm, qc.fo)
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
        store_sample!(S, j, k, GeodesicSample(t, pb.r, pb.θ, ϕ, pb.νr, pb.νθ, (τb <= τ_valid) & krang_visible(pix, pb.θ)))
        pa = pb
    end
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
