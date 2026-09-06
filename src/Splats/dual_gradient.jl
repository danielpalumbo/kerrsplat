# The dual sweep: polarized gradients on the device without a reverse-mode tape.
#
# Along one ray the observed (invariant) Stokes vector is T₁ with the tail recurrence
#     T_{N+1} = 0,   T_k = E_k + O_k T_{k+1}
# (T_k is the Stokes vector arriving at the observer side of sample k from everything behind it;
# O_k, E_k the step operator and emission of sample k, identity and zero for a skipped sample),
# and for a loss l = w₁ · T₁ the adjoints of the step of sample k are
#     ∂l/∂O_k = w_k T_{k+1}ᵀ,   ∂l/∂E_k = w_k,   with  w_{k+1} = O_kᵀ w_k
# (w_k is the adjoint of the Stokes vector in front of sample k, propagated by the transposed
# operators from the observer inward). So a backward pass over the stored samples that keeps
# the tails T_k, followed by a forward pass that carries w_k, gives the adjoint of every sample
# from four-vectors alone; the derivatives of the sample's own step with respect to the
# coefficients, and of the coefficients with respect to the splat parameters, come from
# forward-mode duals evaluated sample by sample (no tape, no device malloc):
#   • the step operator with seven partials (α⃗, ρ⃗); the emission is linear in j so j̄ = Ejᵀ w;
#   • each splat's coefficients with seven partials in its fluid rows (Θe, B, the field angles,
#     the velocity); every coefficient is proportional to the density, so the thirteen geometric
#     rows and ln nₑ enter through the scalar c̄ · c and the gradient of ln G (thirteen cheap
#     partials through the Gaussian weight alone).

"Column of a parameter matrix indexed like the matrix (`p[row, i]`, `size(p, 1)`), so that the splat weight runs unchanged on a column of duals."
struct ParamColumn{NP,T}
    v::SVector{NP,T}
end
Base.@propagate_inbounds Base.getindex(c::ParamColumn, row::Integer, i::Integer) = c.v[row]
Base.size(::ParamColumn{NP}, d::Integer) where {NP} = d == 1 ? NP : 1
Base.lastindex(::ParamColumn{NP}, d::Integer) where {NP} = d == 1 ? NP : 1

struct WeightTag end
struct FluidTag end

"Partial index of a geometric row in the weight duals (rows 1–12 and the pattern rate; 0 for the fluid rows)."
@inline _weight_partial(r) = r <= 12 ? r : (r == NPOLARIZEDPARAMS ? 13 : 0)

"""
    element_adjoint!(grad, m::PolarizedSplats, i, j, pix, s, ν_obs, j̄, ᾱ, ρ̄)

Accumulate into `grad[j, :, i]` the contribution of sample `s` of ray `j` to ∂l/∂(splat i)
given the adjoints `j̄, ᾱ, ρ̄` of the sample's summed screen-basis coefficients: the fluid rows
by forward-mode duals through `splat_coefficients` and `screen_coefficients`, the geometric
rows and ln nₑ through the density scaling (see the header of this file).
"""
@inline function element_adjoint!(grad, m::PolarizedSplats, i, j, pix, s::GeodesicSample{T}, ν_obs, j̄::SVector{4,T}, ᾱ::SVector{4,T}, ρ̄::SVector{3,T}) where {T}
    p = m.params
    met = Krang.metric(pix)
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    t = @inbounds(m.t_obs[1]) - s.t
    outside_support(p, i, t, x, y, z) && return nothing
    col = ParamColumn(SVector(ntuple(r -> ForwardDiff.Dual{WeightTag}(@inbounds(p[r, i]), ForwardDiff.Partials(ntuple(q -> q == _weight_partial(r) ? one(T) : zero(T), Val(13)))), Val(NPOLARIZEDPARAMS))))
    Gd = splat_weight(col, 1, t, x, y, z)
    G = ForwardDiff.value(Gd)
    G > T(WEIGHT_CUTOFF) || return nothing
    dlnG = ForwardDiff.partials(Gd) / G
    @inbounds fl = seed_duals(SVector(p[14, i], p[15, i], p[16, i], p[17, i], p[18, i], p[19, i], p[20, i]), FluidTag, Val(7), 0)
    ne = exp(@inbounds p[13, i]) * G
    cf, fr = splat_coefficients(pix, s, ν_obs, ne, fl[1], fl[2], fl[3], fl[4], fl[5], fl[6], fl[7])
    jj, aa, rr, on = screen_coefficients(cf, fr, ν_obs)
    on || return nothing
    q = dot(j̄, jj) + dot(ᾱ, aa) + dot(ρ̄, rr)
    sc = ForwardDiff.value(q)
    pq = ForwardDiff.partials(q)
    @inbounds begin
        for r in 1:12
            grad[j, r, i] += sc * dlnG[r]
        end
        grad[j, 13, i] += sc
        for r in 1:7
            grad[j, 13 + r, i] += pq[r]
        end
        grad[j, NPOLARIZEDPARAMS, i] += sc * dlnG[13]
    end
    return nothing
end

# the backward pass: tails[j, k] is the invariant Stokes vector arriving from samples k…N (tails[j, 1] the image)
@kernel function polarized_tails_kernel!(tails, params, tvec, S, pc, met::Krang.Kerr, θo, ν, L, ::Val{N}) where {N}
    j = @index(Global, Linear)
    pix = build_pixel(pc, j, met, θo)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    c = RadiativeTransport(PolarizedSplats(params, tvec), ν, L)
    R = zero(SVector{4,typeof(ν)})
    @inbounds tails[j, N + 1] = R
    for k in N:-1:1
        O, E, on = Transfer.sample_step(c, _stored_sample(S, j, k), Δτ, pix)
        on && (R = E + O * R)
        @inbounds tails[j, k] = R
    end
end

# the forward pass: the adjoint w of the Stokes vector in front of each sample, the sample's
# adjoint from w and its tail, and the splat parameters' share of it into grad[j, :, :]
@kernel function polarized_dual_kernel!(grad, dstokes, tails, params, tvec, S, pc, met::Krang.Kerr, θo, ν, L, ::Val{N}) where {N}
    j = @index(Global, Linear)
    T = typeof(ν)
    pix = build_pixel(pc, j, met, θo)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    m = PolarizedSplats(params, tvec)
    c = RadiativeTransport(m, ν, L)
    hor = Krang.horizon(met) * (1 + T(1e-3))
    @inbounds w = dstokes[j] * ν^3
    for k in 1:N
        s = _stored_sample(S, j, k)
        (s.ok && s.r > hor) || continue
        j4, α4, ρ3, active = Transfer.accumulate_elements(c, s, pix, nothing)
        active || continue
        Σ = s.r * s.r + met.spin^2 * cos(s.θ)^2
        Δ = L / ν * Σ * Δτ
        @inbounds R = tails[j, k + 1]
        O, E, j̄, ᾱ, ρ̄ = Transfer.sample_adjoint(j4, α4, ρ3, Δ, w, R)
        for i in 1:nelements(m)
            element_adjoint!(grad, m, i, j, pix, s, ν, j̄, ᾱ, ρ̄)
        end
        w = O' * w
    end
end

"""
    polarized_tails!(tails, cache, params, t_obs, ν_obs, L) -> tails

The backward pass of the dual sweep over the samples stored in `cache`: `tails[j, k]` is the
invariant Stokes vector arriving at the observer side of sample `k` of ray `j` from the samples
behind it (`tails[j, N + 1] = 0`; `tails[j, 1]` is the image, see [`tail_image`](@ref)). `tails`
is `npix × (N + 1)` of `SVector{4}` on the backend.
"""
function polarized_tails!(tails, cache::GeodesicCache{T,N}, params, t_obs, ν_obs, L) where {T,N}
    backend = cache.backend
    nsamples(cache.samples) == N || throw(ArgumentError("the cache holds no stored samples: build it with store_samples = true and a storing marcher"))
    size(tails) == (npixels(cache), N + 1) || throw(ArgumentError("tails must be npix × (N + 1)"))
    prepare_backend!(backend)
    tvec = KA.allocate(backend, T, 1); fill!(tvec, T(t_obs))
    polarized_tails_kernel!(backend, 64)(tails, params, tvec, cache.samples, cache.consts, Krang.Kerr(cache.spin), cache.θo, T(ν_obs), T(L), Val(N); ndrange = npixels(cache))
    KA.synchronize(backend)
    return tails
end

"The observed Stokes vectors (sorted pixel order, cgs) from the tails of [`polarized_tails!`](@ref)."
tail_image(tails, ν_obs) = map(R -> R * ν_obs^3, tails[:, 1])

"""
    polarized_dual_sweep!(dparams, dstokes, tails, cache, params, t_obs, ν_obs, L) -> dparams

The forward pass of the dual sweep: with the `tails` of [`polarized_tails!`](@ref) and
`dstokes` the adjoint of the observed Stokes vectors (sorted pixel order, ∂l/∂(I, Q, U, V) in
cgs), carry the adjoint of the Stokes vector in front of each sample along every ray, form the
adjoints of the sample's step by forward-mode duals through the step operator, and accumulate
∂l/∂params into `dparams` (per-ray partial sums on the backend, reduced at the end, so that the
result does not depend on the order of the threads).
"""
function polarized_dual_sweep!(dparams, dstokes::AbstractVector{SVector{4,T}}, tails, cache::GeodesicCache{T,N}, params, t_obs, ν_obs, L) where {T,N}
    backend = cache.backend
    npix = npixels(cache)
    size(tails) == (npix, N + 1) || throw(ArgumentError("tails must be npix × (N + 1)"))
    prepare_backend!(backend)
    tvec = KA.allocate(backend, T, 1); fill!(tvec, T(t_obs))
    grad = KA.allocate(backend, T, npix, size(params, 1), size(params, 2)); fill!(grad, zero(T))
    polarized_dual_kernel!(backend, 64)(grad, dstokes, tails, params, tvec, cache.samples, cache.consts, Krang.Kerr(cache.spin), cache.θo, T(ν_obs), T(L), Val(N); ndrange = npix)
    KA.synchronize(backend)
    dparams .+= reshape(sum(grad; dims = 1), size(params))
    return dparams
end
