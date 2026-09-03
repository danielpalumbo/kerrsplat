# Fused-march consumers that integrate the transfer equation along the cached rays.
#
# Units. Krang's momenta have E = 1 and the geodesics are parameterized by Mino time τ with the
# affine parameter dλ = Σ dτ in units of M. For a black hole of mass M at distance D, with
# L = GM/c² [cm], the invariant transfer equation along a ray reads
#     d(I/ν³)/dλ = (L/ν_obs) (j/ν² − να · I/ν³),
# where j, α, ρ are evaluated in the fluid frame at ν = ν_obs/g (g = 1/(−p·u)); it reduces to
# dI/ds = j − αI at infinity. So every interval of Mino length Δτ at (r, θ) is a transfer step of
# length Δ = (L/ν_obs) Σ Δτ over the invariants (j/ν², να, νρ), and the observed intensity is
# I_obs = ν_obs³ × (accumulated invariant) in erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹. A pixel of angular size
# ΔαL/D on a side subtends (ΔαL/D)² sr.

"Length unit L = GM/c² in cm for a mass in solar masses."
gravitational_radius(M_solar) = GNEWT * M_solar * MSUN / (CL * CL)

"""
    UnpolarizedState(τ, I)

Front-to-back accumulator for Stokes I only: the optical depth `τ` between the observer and the
current interval and the invariant intensity `I` (in units of ν_obs⁻³ × cgs) reaching the observer.
"""
struct UnpolarizedState{T}
    τ::T
    I::T
end
Base.zero(::Type{UnpolarizedState{T}}) where {T} = UnpolarizedState(zero(T), zero(T))

"""
    unpolarized_step(st::UnpolarizedState, j, α, Δ)

Advance the accumulator over an interval of length `Δ` with constant invariant emissivity `j` and
absorptivity `α` (the interval lies beyond everything accumulated so far).
"""
@inline function unpolarized_step(st::UnpolarizedState{T}, j, α, Δ) where {T}
    τi = α * Δ
    E = j * Δ * phi(τi)                      # ∫₀^Δ e^{−α u} du · j without cancellation
    return UnpolarizedState(st.τ + τi, st.I + exp(-st.τ) * E)
end

"""
    UnpolarizedTransport(model, ν_obs, L)

Fused-march consumer integrating Stokes I along each ray for a plasma `model`, an observed
frequency `ν_obs` [Hz] and the length unit `L` [cm]. The model provides
`unpolarized_coefficients(model, pix, sample, ν_obs) -> (j, α, g)`: the fluid-frame emissivity
and absorptivity [cgs] at the fluid-frame frequency ν_obs/g and the redshift g. Each sample is
treated as an interval of Mino length Δτ centered on it; samples inside the horizon or flagged
invalid are skipped.
"""
struct UnpolarizedTransport{M,T}
    model::M
    ν_obs::T
    L::T
end
Adapt.@adapt_structure UnpolarizedTransport

@inline function (c::UnpolarizedTransport)(acc::UnpolarizedState{T}, j, k, s::GeodesicSample, Δτ, pix) where {T}
    met = Krang.metric(pix)
    (s.ok && s.r > Krang.horizon(met) * (1 + T(1e-3))) || return acc
    jν, αν, g = unpolarized_coefficients(c.model, pix, s, c.ν_obs)
    Σ = s.r * s.r + met.spin^2 * cos(s.θ)^2
    Δ = c.L / c.ν_obs * Σ * Δτ
    jinv = jν * g * g / (c.ν_obs * c.ν_obs)     # j/ν² at ν = ν_obs/g
    αinv = αν * c.ν_obs / g                      # ν α
    return unpolarized_step(acc, jinv, αinv, Δ)
end

"Model hook; see `UnpolarizedTransport`."
function unpolarized_coefficients end

"Observed specific intensity [erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹] from an accumulator at ν_obs."
observed_intensity(st::UnpolarizedState, ν_obs) = st.I * ν_obs^3

"""
    pixel_solid_angle(Δα, L, D)

Solid angle [sr] of a square pixel of side `Δα` (in M) for the length unit `L` [cm] and distance
`D` [cm].
"""
pixel_solid_angle(Δα, L, D) = (Δα * L / D)^2

# ---- polarized transport ----------------------------------------------------------------------
"""
    RadiativeTransport(model, ν_obs, L)

Fused-march consumer integrating the full Stokes vector along each ray. The plasma `model` is a
collection of fluid elements that may overlap (the one-zone splats of the plan's addendum): it
provides `nelements(model)` and `element(model, i, pix, sample, ν_obs) -> (c, frame)`, the
fluid-frame coefficients `c::StokesCoefficients` of element `i` at its own frequency ν_obs/g
together with its `frame::LocalFrame` (redshift, pitch angle, screen angle χ). The consumer caps
the polarization fractions as ipole does, forms the invariants, rotates every element's
coefficients into the screen basis by 2χ, sums them (the coefficients of superposed populations
add), and takes one exact constant-coefficient step per sample interval, composed front to back
in a `RadiativeState`. Samples inside the horizon or flagged invalid are skipped.
"""
struct RadiativeTransport{M,T}
    model::M
    ν_obs::T
    L::T
end
Adapt.@adapt_structure RadiativeTransport

"Number of fluid elements of a model; see `RadiativeTransport`."
function nelements end
"Coefficients and frame of one fluid element at a sample; see `RadiativeTransport`."
function element end

@inline function (c::RadiativeTransport)(acc::RadiativeState{T}, j, k, s::GeodesicSample, Δτ, pix) where {T}
    met = Krang.metric(pix)
    (s.ok && s.r > Krang.horizon(met) * (1 + T(1e-3))) || return acc
    j4 = zero(SVector{4,T}); α4 = zero(SVector{4,T}); ρ3 = zero(SVector{3,T})
    active = false
    for i in 1:nelements(c.model)
        cf, fr = element(c.model, i, pix, s, c.ν_obs)
        (cf.jI > 0 || cf.αI > 0 || cf.ρQ != 0 || cf.ρV != 0) || continue
        cinv = invariants(cap_polarization(cf), c.ν_obs / fr.g)
        jj, aa, rr = rotate_to_screen(cinv, fr.χ)
        j4 += jj; α4 += aa; ρ3 += rr
        active = true
    end
    active || return acc
    Σ = s.r * s.r + met.spin^2 * cos(s.θ)^2
    Δ = c.L / c.ν_obs * Σ * Δτ
    O, E = transfer_step(j4, α4, ρ3, Δ)
    return advance(acc, O, E)
end

"Observed Stokes vector (I, Q, U, V) [erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹] from an accumulator at ν_obs."
observed_stokes(st::RadiativeState, ν_obs) = st.S * ν_obs^3
