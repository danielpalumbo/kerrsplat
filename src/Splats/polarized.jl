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
        u = pattern_offset(p, i, t, x, y, z, size(p, 1))     # the pattern rate is the last row of every layout
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
const SUPPORT_RADIUS2 = -2 * log(WEIGHT_CUTOFF)    # (7.43 σ)²: beyond it the weight is below the cutoff for any orientation

Transfer.nelements(m::PolarizedSplats) = size(m.params, 2)

"""
    outside_support(p, i, t, x, y, z) -> Bool

Cheap culling test: whether the point lies beyond the bounding sphere of splat `i` at time `t`
(radius √(2 ln(1/WEIGHT_CUTOFF)) times the largest scale about the rotated centre), where the
Gaussian weight is below `WEIGHT_CUTOFF` for every orientation. Costs one `sincos` and a few
multiplications instead of the quaternion rotation and the exponential of `splat_weight`.
"""
@inline function outside_support(p, i, t, x, y, z)
    @inbounds begin
        φ = p[end, i] * (t - p[11, i])
        sφ, cφ = sincos(φ)
        cx = p[1, i] * cφ - p[2, i] * sφ
        cy = p[1, i] * sφ + p[2, i] * cφ
        d2 = (x - cx)^2 + (y - cy)^2 + (z - p[3, i])^2
        smax = max(p[4, i], p[5, i], p[6, i])
        return d2 > SUPPORT_RADIUS2 * exp(2 * smax)
    end
end
@inline function Transfer.element(m::PolarizedSplats, i, pix, s::GeodesicSample{T}, ν_obs) where {T}
    p = m.params
    met = Krang.metric(pix)
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    t = @inbounds(m.t_obs[1]) - s.t
    outside_support(p, i, t, x, y, z) && return zero(StokesCoefficients{T}), LocalFrame(one(T), zero(T), zero(T))
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

"""
    fields(params, t, x, y, z) -> (ne, Θe, B)

Field-level view of a set of polarized splats at an event (addendum §5.2.3): the total electron
density Σ nₑ,ₖ Gₖ and the density-weighted mean temperature and field strength of the parcels
present there (zero temperature and field where there is no density).
"""
function fields(p::AbstractMatrix{T}, t, x, y, z) where {T}
    ne = zero(T); wΘ = zero(T); wB = zero(T)
    for i in 1:size(p, 2)
        n = exp(p[13, i]) * splat_weight(p, i, t, x, y, z)
        ne += n; wΘ += n * exp(p[14, i]); wB += n * exp(p[15, i])
    end
    return ne > 0 ? (ne, wΘ / ne, wB / ne) : (zero(T), zero(T), zero(T))
end

"""
    field_grid(params, t, xs, ys, zs) -> (ne, Θe, B)

`fields` on a voxel grid, as three arrays of size (length(xs), length(ys), length(zs)).
"""
function field_grid(p::AbstractMatrix{T}, t, xs, ys, zs) where {T}
    ne = Array{T}(undef, length(xs), length(ys), length(zs)); Θ = similar(ne); B = similar(ne)
    for (k, z) in enumerate(zs), (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        ne[i, j, k], Θ[i, j, k], B[i, j, k] = fields(p, t, x, y, z)
    end
    return ne, Θ, B
end

"""
    recovery_metrics(p_fit, p_true, t, xs, ys, zs) -> NamedTuple

Voxel-grid comparison of the fields of a fitted splat set with the truth (the self-consistency
test of plan §7.5 item 7): the peak signal-to-noise ratio of the density, PSNR = 10 log₁₀(max nₑ²/MSE),
and the density-weighted relative errors of temperature and field strength.
"""
function recovery_metrics(p_fit, p_true, t, xs, ys, zs)
    nf, Θf, Bf = field_grid(p_fit, t, xs, ys, zs)
    nt, Θt, Bt = field_grid(p_true, t, xs, ys, zs)
    mse = sum(abs2, nf .- nt) / length(nt)
    psnr = 10 * log10(maximum(nt)^2 / mse)
    w = nt ./ sum(nt)
    return (psnr_density = psnr, rel_density = sqrt(mse) / maximum(nt),
            temperature = sum(w .* abs.(Θf .- Θt) ./ max.(Θt, eps())), field = sum(w .* abs.(Bf .- Bt) ./ max.(Bt, eps())))
end
