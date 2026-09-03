# Motion mode C (addendum §6.2): splat centres advected by their own fluid velocity. The centre's
# trajectory is integrated on the host in Boyer–Lindquist coordinates from the splat's ZAMO
# 3-velocity (dxⁱ/dt = uⁱ/uᵗ with u^μ the BL 4-velocity of ũ at the current position), stored as
# knots at a grid of coordinate times, and interpolated linearly at each sample's emission time
# inside the kernels. The pattern rotation and the temporal envelope of the splat still apply on
# top of the knots (both are usually switched off in this mode).

"BL coordinates (r, θ, φ) of a quasi-Cartesian Kerr–Schild point (the inverse of `quasi_cartesian_kerr_schild`)."
function boyer_lindquist(met::Krang.Kerr{T}, x, y, z) where {T}
    r = sqrt(x * x + y * y + z * z)
    θ = acos(clamp(z / r, -one(T), one(T)))
    a = met.spin
    temp = sqrt(one(T) - a^2)
    rp = one(T) + temp; rm = one(T) - temp
    ϕks = atan(y, x)
    ϕ = ϕks - a / (2 * temp) * log(abs((r - rp + eps(T)) / (r - rm + eps(T)))) + atan(a, r)
    return r, θ, ϕ
end

"Coordinate velocity (dr/dt, dθ/dt, dφ/dt) of a fluid element with ZAMO 3-velocity ũ at (r, θ)."
function coordinate_velocity(met::Krang.Kerr{T}, r, θ, ũ::SVector{3}) where {T}
    γ = sqrt(1 + ũ[1]^2 + ũ[2]^2 + ũ[3]^2)
    u = Krang.jac_bl_u_zamo_d(met, r, θ) * SVector(γ, ũ[1], ũ[2], ũ[3])
    return SVector(u[2] / u[1], u[3] / u[1], u[4] / u[1])
end

"""
    trajectory_knots(params, met, times; substeps = 4) -> Array{T,3}

Centres of every polarized splat at the coordinate `times` (a knot grid), integrated from the
centre at t₀ (row `:t0`) with the splat's ZAMO velocity (rows `:u1, :u2, :u3`) by RK4 in BL
coordinates with `substeps` steps per knot interval, forward and backward from t₀. Returns a
3 × nknots × nsplat array of quasi-Cartesian Kerr–Schild positions.
"""
function trajectory_knots(params::AbstractMatrix{T}, met::Krang.Kerr{T}, times::AbstractVector; substeps::Integer = 4) where {T}
    n = size(params, 2)
    knots = Array{T}(undef, 3, length(times), n)
    for i in 1:n
        ũ = SVector(params[18, i], params[19, i], params[20, i])
        t0 = params[11, i]
        x0 = SVector(params[1, i], params[2, i], params[3, i])
        q0 = SVector(boyer_lindquist(met, x0...)...)
        f(q) = coordinate_velocity(met, q[1], q[2], ũ)
        function integrate(q, t_from, t_to)
            m = max(1, ceil(Int, abs(t_to - t_from) / (abs(times[end] - times[1]) / max(length(times) - 1, 1)) * substeps))
            h = (t_to - t_from) / m
            for _ in 1:m
                k1 = f(q); k2 = f(q + h / 2 * k1); k3 = f(q + h / 2 * k2); k4 = f(q + h * k3)
                q = q + h / 6 * (k1 + 2k2 + 2k3 + k4)
            end
            return q
        end
        # march from t₀ outward in both directions through the sorted knot times
        order = sortperm(collect(times))
        kt0 = searchsortedfirst(times[order], t0)
        q = q0; tprev = t0
        for k in kt0:length(times)
            q = integrate(q, tprev, times[order[k]]); tprev = times[order[k]]
            knots[:, order[k], i] .= quasi_cartesian_kerr_schild(met, q[1], q[2], q[3])
        end
        q = q0; tprev = t0
        for k in kt0-1:-1:1
            q = integrate(q, tprev, times[order[k]]); tprev = times[order[k]]
            knots[:, order[k], i] .= quasi_cartesian_kerr_schild(met, q[1], q[2], q[3])
        end
    end
    return knots
end

"""
    KnotSplats(params, t_obs, knots, times)

Polarized splats whose centres follow trajectory knots (`trajectory_knots`): at a sample's
emission time the centre is interpolated linearly between the knots and replaces the
`:x, :y, :z` rows; everything else is as for `PolarizedSplats`. Outside the knot times the
centre is clamped to the end knots, so the knots must cover every emission time of a movie:
from t_obs minus the largest lookback of a ray that can see the splat (lensed paths reach it
tens of M before the direct one) to t_obs plus the largest negative lookback.
"""
struct KnotSplats{P,V,K,W}
    params::P
    t_obs::V
    knots::K        # 3 × nknots × nsplat
    times::W        # nknots, increasing
end
Adapt.@adapt_structure KnotSplats
KnotSplats(params::AbstractMatrix{T}, t_obs::Real, knots, times) where {T} =
    KnotSplats(params, fill!(similar(params, 1), T(t_obs)), knots, times)

Transfer.nelements(m::KnotSplats) = size(m.params, 2)

@inline function knot_centre(m::KnotSplats, i, t)
    ts = m.times
    n = length(ts)
    @inbounds begin
        if t <= ts[1]
            return SVector(m.knots[1, 1, i], m.knots[2, 1, i], m.knots[3, 1, i])
        elseif t >= ts[n]
            return SVector(m.knots[1, n, i], m.knots[2, n, i], m.knots[3, n, i])
        end
        k = 1
        while k < n - 1 && ts[k + 1] <= t
            k += 1
        end
        w = (t - ts[k]) / (ts[k + 1] - ts[k])
        return SVector((1 - w) * m.knots[1, k, i] + w * m.knots[1, k + 1, i],
                       (1 - w) * m.knots[2, k, i] + w * m.knots[2, k + 1, i],
                       (1 - w) * m.knots[3, k, i] + w * m.knots[3, k + 1, i])
    end
end

@inline function Transfer.element(m::KnotSplats, i, pix, s::GeodesicSample{T}, ν_obs) where {T}
    p = m.params
    met = Krang.metric(pix)
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    t = @inbounds(m.t_obs[1]) - s.t
    c = knot_centre(m, i, t)
    @inbounds begin
        d = SVector(x - c[1], y - c[2], z - c[3])
        R = quaternion_rotation(p[7, i], p[8, i], p[9, i], p[10, i])
        u = R' * d
        q2 = (u[1] * exp(-p[4, i]))^2 + (u[2] * exp(-p[5, i]))^2 + (u[3] * exp(-p[6, i]))^2
        τ2 = ((t - p[11, i]) * exp(-p[12, i]))^2
        G = exp(-(q2 + τ2) / 2)
        G > T(WEIGHT_CUTOFF) || return zero(StokesCoefficients{T}), LocalFrame(one(T), zero(T), zero(T))
        ne = exp(p[13, i]) * G
        Θe = exp(p[14, i])
        Bmag = exp(p[15, i])
        sθ, cθ = sincos(p[16, i]); sϕ, cϕ = sincos(p[17, i])
        B = SVector(Bmag * sθ * cϕ, Bmag * sθ * sϕ, Bmag * cθ)
        ũ = SVector(p[18, i], p[19, i], p[20, i])
    end
    fr = local_frame(pix, s, ũ, B)
    νf = ν_obs / fr.g
    θB = acos(clamp(fr.cosθB, -one(T), one(T)))
    return thermal_synchrotron(ne, Θe, Bmag, νf, θB), fr
end
