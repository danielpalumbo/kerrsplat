# Non-thermal populations as splat sets of their own (addendum §1, two-population parcels): the
# same geometry, envelope, field, velocity and pattern-rotation rows as `PolarizedSplats`, with the
# two temperature-like rows reinterpreted, so that a thermal set and a non-thermal set combine
# through `Transfer.CompositeModel` without changing the thermal layout.

"""
    POWERLAW_SPLAT_PARAMS

Layout of a power-law splat column: the geometry and envelope rows of [`POLARIZED_SPLAT_PARAMS`](@ref),
then `:logne`, `:p` (the index of n(γ) ∝ γ^(−p)), `:loggmin` (ln γmin; γmax is a fixed
property of the set), `:logB, :thB, :phB, :u1, :u2, :u3, :omega` (22 rows).
"""
const POWERLAW_SPLAT_PARAMS = (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :t0, :logw,
                               :logne, :p, :loggmin, :logB, :thB, :phB, :u1, :u2, :u3, :omega)
const NPOWERLAWPARAMS = length(POWERLAW_SPLAT_PARAMS)

"""
    KAPPA_SPLAT_PARAMS

Layout of a κ splat column: `(:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :t0, :logw, :logne,
:kappa, :logw_kappa, :logB, :thB, :phB, :u1, :u2, :u3, :omega)` (22 rows); the hypergeometric
factor of each splat's (κ, w) is computed on the host by the `KappaSplats` constructor.
"""
const KAPPA_SPLAT_PARAMS = (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :t0, :logw,
                            :logne, :kappa, :logw_kappa, :logB, :thB, :phB, :u1, :u2, :u3, :omega)
const NKAPPAPARAMS = length(KAPPA_SPLAT_PARAMS)

"""
    PowerLawSplats(params, t_obs; γmax = 1e5)

Splats of power-law electrons (Pandya+ 2016 fits, Jones & O'Dell rotativities): density
e^{logne} × Gaussian weight, index `p`, γmin = e^{loggmin}, and the common `γmax`.
"""
struct PowerLawSplats{P,V,T}
    params::P
    t_obs::V
    γmax::T
end
Adapt.@adapt_structure PowerLawSplats
PowerLawSplats(params::AbstractMatrix{T}, t_obs::Real; γmax = 1e5) where {T} =
    PowerLawSplats(params, fill!(similar(params, 1), T(t_obs)), T(γmax))

"""
    KappaSplats(params, t_obs)

Splats of κ-distributed electrons (Pandya+ 2016 and Marszewski+ 2021 fits): index `kappa` and
width w = e^{logw_kappa}; the hypergeometric factors are precomputed on the host (they depend on
κ and w only, and are held fixed during a fit).
"""
struct KappaSplats{P,V,H}
    params::P
    t_obs::V
    hyp::H
end
Adapt.@adapt_structure KappaSplats
function KappaSplats(params::AbstractMatrix{T}, t_obs::Real) where {T}
    hp = Array(params)
    hyp = [Transfer.kappa_hypergeometric(hp[14, i], exp(hp[15, i])) for i in 1:size(hp, 2)]
    return KappaSplats(params, fill!(similar(params, 1), T(t_obs)), adapt_like(params, hyp))
end
adapt_like(::Array, x) = x
adapt_like(ref::AbstractArray, x) = (y = similar(ref, eltype(x), length(x)); copyto!(y, x); y)

Transfer.nelements(m::PowerLawSplats) = size(m.params, 2)
Transfer.nelements(m::KappaSplats) = size(m.params, 2)

# shared geometry: weight, field, velocity and frame of splat i at a sample (or nothing when culled)
@inline function _splat_geometry(p, t_obs, i, pix, s::GeodesicSample{T}) where {T}
    met = Krang.metric(pix)
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    t = @inbounds(t_obs[1]) - s.t
    outside_support(p, i, t, x, y, z) && return zero(T), zero(SVector{3,T}), zero(SVector{3,T}), zero(T)
    G = splat_weight(p, i, t, x, y, z)
    G > T(WEIGHT_CUTOFF) || return zero(T), zero(SVector{3,T}), zero(SVector{3,T}), zero(T)
    @inbounds begin
        Bmag = exp(p[16, i])
        sθ, cθ = sincos(p[17, i]); sϕ, cϕ = sincos(p[18, i])
        B = SVector(Bmag * sθ * cϕ, Bmag * sθ * sϕ, Bmag * cθ)
        ũ = SVector(p[19, i], p[20, i], p[21, i])
    end
    return G, B, ũ, Bmag
end

@inline function Transfer.element(m::PowerLawSplats, i, pix, s::GeodesicSample{T}, ν_obs) where {T}
    G, B, ũ, Bmag = _splat_geometry(m.params, m.t_obs, i, pix, s)
    G > 0 || return zero(StokesCoefficients{T}), LocalFrame(one(T), zero(T), zero(T))
    fr = local_frame(pix, s, ũ, B)
    νf = ν_obs / fr.g
    θB = acos(clamp(fr.cosθB, -one(T), one(T)))
    @inbounds ne, pidx, γmin = exp(m.params[13, i]) * G, m.params[14, i], exp(m.params[15, i])
    return powerlaw_synchrotron(ne, pidx, γmin, m.γmax, Bmag, νf, θB), fr
end

@inline function Transfer.element(m::KappaSplats, i, pix, s::GeodesicSample{T}, ν_obs) where {T}
    G, B, ũ, Bmag = _splat_geometry(m.params, m.t_obs, i, pix, s)
    G > 0 || return zero(StokesCoefficients{T}), LocalFrame(one(T), zero(T), zero(T))
    fr = local_frame(pix, s, ũ, B)
    νf = ν_obs / fr.g
    θB = acos(clamp(fr.cosθB, -one(T), one(T)))
    @inbounds ne, κ, w, hyp = exp(m.params[13, i]) * G, m.params[14, i], exp(m.params[15, i]), m.hyp[i]
    return kappa_synchrotron(ne, κ, w, Bmag, νf, θB, hyp), fr
end
