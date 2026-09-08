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
    nmax::Int32
    slab::T
end
Adapt.@adapt_structure RadiativeTransport
function RadiativeTransport(model, ν_obs::Real, L::Real; nmax::Integer = -1, slab::Real = 0)
    T = promote_type(typeof(ν_obs), typeof(L))
    return RadiativeTransport(model, T(ν_obs), T(L), Int32(nmax), T(slab))
end

"Number of fluid elements of a model; see `RadiativeTransport`."
function nelements end
"Coefficients and frame of one fluid element at a sample; see `RadiativeTransport`."
function element end

@inline function (c::RadiativeTransport)(acc::RadiativeState{T}, j, k, s::GeodesicSample, Δτ, pix) where {T}
    met = Krang.metric(pix)
    (s.ok && s.r > Krang.horizon(met) * (1 + T(1e-3))) || return acc
    return transfer_sample(c, acc, s, Δτ, pix)
end

"""
Fused-march consumer with the half-orbit counter (accumulators of type [`WindingState`](@ref)):
the counter advances at every valid sample and, when the transport has `nmax ≥ 0`, samples
beyond the `nmax`-th passage contribute neither emission nor absorption (the ray is truncated).
"""
@inline function (c::RadiativeTransport)(acc::WindingState{T}, j, k, s::GeodesicSample, Δτ, pix) where {T}
    met = Krang.metric(pix)
    (s.ok && s.r > Krang.horizon(met) * (1 + T(1e-3))) || return acc
    w = wind(acc, s.r * cos(s.θ), c.slab)
    (c.nmax >= 0 && w.n > c.nmax) && return w
    return WindingState(transfer_sample(c, w.state, s, Δτ, pix), w.zprev, w.n, w.inside, w.crossed)
end

@inline function transfer_sample(c::RadiativeTransport, acc::RadiativeState{T}, s::GeodesicSample, Δτ, pix) where {T}
    met = Krang.metric(pix)
    j4, α4, ρ3, active = accumulate_elements(c, s, pix, static_elements(c.model))
    active || return acc
    Σ = s.r * s.r + met.spin^2 * cos(s.θ)^2
    Δ = c.L / c.ν_obs * Σ * Δτ
    O, E = transfer_step(j4, α4, ρ3, Δ)
    return advance(acc, O, E)
end

"""
    sample_step(c::RadiativeTransport, s, Δτ, pix) -> (O, E, active)

The step operator and emission of one sample, what `transfer_sample` composes into the
accumulator; a skipped sample (invalid, inside the horizon, or with no active element) returns
the identity, zero and `active = false`.
"""
@inline function sample_step(c::RadiativeTransport, s::GeodesicSample, Δτ, pix)
    T = typeof(c.ν_obs)
    met = Krang.metric(pix)
    if s.ok && s.r > Krang.horizon(met) * (1 + T(1e-3))
        j4, α4, ρ3, active = accumulate_elements(c, s, pix, static_elements(c.model))
        if active
            Σ = s.r * s.r + met.spin^2 * cos(s.θ)^2
            Δ = c.L / c.ν_obs * Σ * Δτ
            O, E = transfer_step(j4, α4, ρ3, Δ)
            return O, E, true
        end
    end
    return identity4(T), zero(SVector{4,T}), false
end

"Tag of the forward-mode duals through the step operator."
struct StepTag end

"An `SVector` of duals with unit partials, `x[i]` seeded in partial `i` (`N` partials in all, offset by `off`)."
@inline function seed_duals(x::SVector{M,T}, ::Type{Tag}, ::Val{N}, off::Int) where {M,T,Tag,N}
    return SVector(ntuple(i -> ForwardDiff.Dual{Tag}(x[i], ForwardDiff.Partials(ntuple(m -> m == i + off ? one(T) : zero(T), Val(N)))), Val(M)))
end

"""
    sample_adjoint(j4, α4, ρ3, Δ, w, R) -> (O, E, j̄, ᾱ, ρ̄)

The step of one sample together with the adjoints of its coefficients, for the loss
`wᵀ (O R + E)` in which `w` is the adjoint of the Stokes vector in front of the sample and `R`
the Stokes vector arriving from behind it (the tail): `j̄ = Ejᵀ w` exactly, since `E = Ej j`, and
`ᾱ`, `ρ̄` are the partials of `wᵀ O R + wᵀ Ej j` by forward-mode duals through
[`step_operator`](@ref) (seven partials). `O` and `E` are the plain values.
"""
@inline function sample_adjoint(j4::SVector{4,T}, α4::SVector{4,T}, ρ3::SVector{3,T}, Δ, w::SVector{4,T}, R::SVector{4,T}) where {T}
    αd = seed_duals(α4, StepTag, Val(7), 0)
    ρd = seed_duals(ρ3, StepTag, Val(7), 4)
    Od, Ejd = step_operator(αd, ρd, Δ)
    q = dot(w, Od * R) + dot(w, Ejd * j4)
    O = map(ForwardDiff.value, Od)
    Ej = map(ForwardDiff.value, Ejd)
    pq = ForwardDiff.partials(q)
    return O, Ej * j4, Ej' * w, SVector(pq[1], pq[2], pq[3], pq[4]), SVector(pq[5], pq[6], pq[7])
end

"One element's screen-basis coefficients at a sample (zero when it does not emit or absorb there)."
@inline function element_coefficients(c::RadiativeTransport, i, s::GeodesicSample, pix)
    cf, fr = element(c.model, i, pix, s, c.ν_obs)
    return screen_coefficients(cf, fr, c.ν_obs)
end

"""
    screen_coefficients(cf, fr, ν_obs) -> (j4, α4, ρ3, active)

The invariant screen-basis coefficients of one element from its fluid-frame coefficients `cf`
and frame `fr` (polarization capped, invariants formed at ν_obs/g, rotated by 2χ), and whether
it emits, absorbs or rotates at all; typed by `cf`, so duals in the element's parameters
propagate.
"""
@inline function screen_coefficients(cf::StokesCoefficients{T}, fr::LocalFrame, ν_obs) where {T}
    (cf.jI > 0 || cf.αI > 0 || cf.ρQ != 0 || cf.ρV != 0) || return zero(SVector{4,T}), zero(SVector{4,T}), zero(SVector{3,T}), false
    cinv = invariants(cap_polarization(cf), ν_obs / fr.g)
    jj, aa, rr = rotate_to_screen(cinv, fr.χ)
    return jj, aa, rr, true
end

"""
    static_elements(model) -> nothing | Val{N}

Models whose element count is known at compile time return `Val(N)`, and the sum over elements
in `transfer_sample` is then unrolled; the default is a loop. A loop inside a function Enzyme
differentiates on the GPU keeps its per-iteration cache in device malloc, which is thirty times
slower than the stack, so the adjoint kernels wrap their model in [`StaticCount`](@ref).
"""
static_elements(model) = nothing

@inline function accumulate_elements(c::RadiativeTransport, s::GeodesicSample, pix, ::Nothing)
    T = typeof(c.ν_obs)
    j4 = zero(SVector{4,T}); α4 = zero(SVector{4,T}); ρ3 = zero(SVector{3,T})
    active = false
    for i in 1:nelements(c.model)
        jj, aa, rr, on = element_coefficients(c, i, s, pix)
        j4 += jj; α4 += aa; ρ3 += rr
        active |= on
    end
    return j4, α4, ρ3, active
end

@inline function accumulate_elements(c::RadiativeTransport, s::GeodesicSample, pix, ::Val{NS}) where {NS}
    # ntuple with a Val length and the tuple reductions are unrolled by the compiler into straight-line
    # code with concrete types (a recursion on the index is not inferable, a loop is cached in device
    # malloc, and a captured Type in the closure is a dynamic dispatch on the device)
    terms = ntuple(i -> element_coefficients(c, i, s, pix), Val(NS))
    j4 = mapreduce(t -> t[1], +, terms)
    α4 = mapreduce(t -> t[2], +, terms)
    ρ3 = mapreduce(t -> t[3], +, terms)
    active = mapreduce(t -> t[4], |, terms)
    return j4, α4, ρ3, active
end

"""
    StaticCount(model, Val(N))

`model` with its element count fixed at compile time (see [`static_elements`](@ref)); the
adjoint kernels use it so that the sum over splats is unrolled.
"""
struct StaticCount{NS,M}
    model::M
    StaticCount(model::M, ::Val{NS}) where {NS,M} = new{NS,M}(model)
end
Adapt.adapt_structure(to, m::StaticCount{NS}) where {NS} = StaticCount(Adapt.adapt(to, m.model), Val(NS))
nelements(::StaticCount{NS}) where {NS} = NS
static_elements(::StaticCount{NS}) where {NS} = Val(NS)
@inline element(m::StaticCount, i, pix, s, ν_obs) = element(m.model, i, pix, s, ν_obs)

"Observed Stokes vector (I, Q, U, V) [erg s⁻¹ cm⁻² Hz⁻¹ sr⁻¹] from an accumulator at ν_obs."
observed_stokes(st::RadiativeState, ν_obs) = st.S * ν_obs^3
observed_stokes(w::WindingState, ν_obs) = observed_stokes(w.state, ν_obs)

# ---- composite models ------------------------------------------------------------------------------
"""
    CompositeModel(a, b)

The union of two `RadiativeTransport` models: its elements are those of `a` followed by those of
`b` (the coefficients of overlapping elements add, so a background flow and a set of splats, or
two splat sets with different populations, combine without any further work).
"""
struct CompositeModel{A,B}
    a::A
    b::B
end
Adapt.@adapt_structure CompositeModel
nelements(m::CompositeModel) = nelements(m.a) + nelements(m.b)
@inline function element(m::CompositeModel, i, pix, s, ν_obs)
    na = nelements(m.a)
    return i <= na ? element(m.a, i, pix, s, ν_obs) : element(m.b, i - na, pix, s, ν_obs)
end
export CompositeModel
