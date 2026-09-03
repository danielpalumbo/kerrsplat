# One-zone polarized splats (addendum §1): the Gaussian and temporal envelope of the thin splats,
# carrying a thermal electron population (density scaled by the Gaussian weight, uniform
# temperature), a uniform magnetic field given in the splat's fluid frame, and a ZAMO 3-velocity,
# rendered through `Transfer.RadiativeTransport` as overlapping fluid elements.

"""
    POLARIZED_SPLAT_PARAMS

Row layout of a polarized splat parameter column: the geometry and temporal envelope of
[`SPLAT_PARAMS`](@ref) (`:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :t0, :logw`) followed by
`:logne` (ln of the peak electron density [cm⁻³]), `:logTe` (ln Θe), `:logB` (ln |B| [G]),
`:thB, :phB` (direction of B in the fluid frame, polar and azimuthal angles about the ZAMO axes
r̂, φ̂, −θ̂), `:u1, :u2, :u3` (the spatial ZAMO components γβ⃗ of the fluid 4-velocity) and
`:omega` (pattern angular velocity of the centre about the spin axis, as for [`SPLAT_PARAMS`](@ref)).
"""
const POLARIZED_SPLAT_PARAMS = (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :t0, :logw,
                                :logne, :logTe, :logB, :thB, :phB, :u1, :u2, :u3, :omega)
const NPOLARIZEDPARAMS = length(POLARIZED_SPLAT_PARAMS)

"Gaussian × temporal-envelope weight (unit peak) of splat `i` at time `t` and position `(x, y, z)`."
@inline function splat_weight(p, i, t, x, y, z)
    @inbounds begin
        u = pattern_offset(p, i, t, x, y, z, 21)
        q2 = (u[1] * exp(-p[4, i]))^2 + (u[2] * exp(-p[5, i]))^2 + (u[3] * exp(-p[6, i]))^2
        τ2 = ((t - p[11, i]) * exp(-p[12, i]))^2
        return exp(-(q2 + τ2) / 2)
    end
end

"""
    PolarizedSplats(params, t_obs)

Model for [`Transfer.RadiativeTransport`](@ref): `params` is the `NPOLARIZEDPARAMS × nsplat`
matrix (on the backend of the cache), `t_obs` the observation time as a scalar or a one-element
array on the same backend. Each splat is one fluid element: at a sample it contributes the
thermal synchrotron coefficients of density e^{logne} × (Gaussian weight at the emission event),
temperature e^{logTe} and field e^{logB} along (thB, phB), seen from the frame of velocity
(u1, u2, u3); splats with weight below `WEIGHT_CUTOFF` at the sample are skipped.
"""
struct PolarizedSplats{P,V}
    params::P
    t_obs::V
end
Adapt.@adapt_structure PolarizedSplats
PolarizedSplats(params::AbstractMatrix{T}, t_obs::Real) where {T} =
    PolarizedSplats(params, fill!(similar(params, 1), T(t_obs)))

const WEIGHT_CUTOFF = 1e-12

Transfer.nelements(m::PolarizedSplats) = size(m.params, 2)

@inline function Transfer.element(m::PolarizedSplats, i, pix, s::GeodesicSample{T}, ν_obs) where {T}
    p = m.params
    met = Krang.metric(pix)
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    t = @inbounds(m.t_obs[1]) - s.t
    G = splat_weight(p, i, t, x, y, z)
    G > T(WEIGHT_CUTOFF) || return zero(StokesCoefficients{T}), LocalFrame(one(T), zero(T), zero(T))
    @inbounds begin
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

"""
    polarized_image!(out, cache, params, t_obs, ν_obs, L)

Full Stokes transport of the polarized splats `params` through a regenerated `GeodesicCache`
at observation time `t_obs` (M units) and frequency `ν_obs` [Hz], with the length unit `L` [cm]
(see `Transfer.gravitational_radius`). `out` is a vector of `RadiativeState` in sorted pixel
order; `observed_stokes.(out, ν_obs)` gives (I, Q, U, V) in cgs. Returns `out`.
"""
function polarized_image!(out, cache::GeodesicCache, params, t_obs, ν_obs, L)
    fused_march!(RadiativeTransport(PolarizedSplats(params, t_obs), ν_obs, L), out, cache)
    return out
end

"Screen-shaped array of Stokes 4-vectors (allocating)."
function polarized_image(cache::GeodesicCache{T}, params, t_obs, ν_obs, L) where {T}
    out = KA.allocate(cache.backend, RadiativeState{T}, npixels(cache))
    polarized_image!(out, cache, params, t_obs, ν_obs, L)
    return map(st -> observed_stokes(st, ν_obs), to_screen(cache, out))
end

"""
    polarized_cube(cache, params, times, νs, L)

Stokes movie cube over observation times and frequencies sharing one geodesic cache: an array of
Stokes 4-vectors of size (nα, nβ, length(times), length(νs)) on the host.
"""
function polarized_cube(cache::GeodesicCache{T}, params, times, νs, L) where {T}
    out = KA.allocate(cache.backend, RadiativeState{T}, npixels(cache))
    frame(t, ν) = (polarized_image!(out, cache, params, t, ν, L); Array(map(st -> observed_stokes(st, ν), to_screen(cache, out))))
    first = frame(times[1], νs[1])
    cube = Array{SVector{4,T}}(undef, size(first)..., length(times), length(νs))
    cube[:, :, 1, 1] = first
    for (l, ν) in enumerate(νs), (k, t) in enumerate(times)
        (k == 1 && l == 1) && continue
        cube[:, :, k, l] = frame(t, ν)
    end
    return cube
end

"Flux density per pixel [Jy] from an intensity or Stokes image for a pixel side `Δα` (M), length unit `L` and distance `D` (cm)."
flux_density(img, Δα, L, D) = img .* (Transfer.pixel_solid_angle(Δα, L, D) / Transfer.JY)
