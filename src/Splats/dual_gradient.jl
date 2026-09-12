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
@inline element_adjoint!(grad, m::PolarizedSplats, i, j, pix, s::GeodesicSample, ν_obs, j̄, ᾱ, ρ̄) = element_adjoint!(grad, m, i, i, j, pix, s, ν_obs, j̄, ᾱ, ρ̄)
"The subset form: parcel `ids[slot]` of the ray's list, its share written into slot `slot` of `grad[j, :, :]`."
@inline element_adjoint!(grad, m::RaySubset, slot, j, pix, s::GeodesicSample, ν_obs, j̄, ᾱ, ρ̄) = element_adjoint!(grad, m.model, @inbounds(m.ids[slot]), slot, j, pix, s, ν_obs, j̄, ᾱ, ρ̄)
@inline function element_adjoint!(grad, m::PolarizedSplats, i, slot, j, pix, s::GeodesicSample{T}, ν_obs, j̄::SVector{4,T}, ᾱ::SVector{4,T}, ρ̄::SVector{3,T}) where {T}
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
            grad[j, r, slot] += sc * dlnG[r]
        end
        grad[j, 13, slot] += sc
        for r in 1:7
            grad[j, 13 + r, slot] += pq[r]
        end
        grad[j, NPOLARIZEDPARAMS, slot] += sc * dlnG[13]
    end
    return nothing
end

# the half-orbit cutoff: kstop[j] is the first sample of ray j beyond its nmax-th passage through the slab
# (N + 1 when the ray is never truncated); the counter advances exactly as in the WindingState consumer,
# and since it never decreases, every sample from kstop on is skipped by the truncated transport
@kernel function winding_cutoff_kernel!(kstop, S, pc, met::Krang.Kerr, θo, slab, nmax, ::Val{N}) where {N}
    j = @index(Global, Linear)
    T = typeof(slab)
    hor = Krang.horizon(met) * (1 + T(1e-3))
    w = zero(WindingState{T})
    stop = N + 1
    for k in 1:N
        s = _stored_sample(S, j, k)
        (s.ok && s.r > hor) || continue
        w = wind(w, s.r * cos(s.θ), slab)
        if w.n > nmax
            stop = k
            break
        end
    end
    @inbounds kstop[j] = stop
end

"""
    winding_cutoff!(kstop, cache, slab, nmax) -> kstop

For every ray the index of its first sample beyond the `nmax`-th passage through the slab
|z| < `slab` (`N + 1` when there is none; see `Transfer.WindingState`), from the samples stored
in `cache`: the truncated transport skips that sample and every later one, so the dual sweep
needs only this cutoff. `kstop` is a vector of `Int` of length npix on the backend.
"""
function winding_cutoff!(kstop, cache::GeodesicCache{T,N}, slab, nmax) where {T,N}
    backend = cache.backend
    winding_cutoff_kernel!(backend, 64)(kstop, cache.samples, cache.consts, Krang.Kerr(cache.spin), cache.θo, T(slab), Int32(nmax), Val(N); ndrange = npixels(cache))
    KA.synchronize(backend)
    return kstop
end

"The per-ray cutoffs of a truncation (`nmax ≥ 0`), or `N + 1` everywhere for the full rays."
function _cutoffs(cache::GeodesicCache{T,N}, nmax, slab) where {T,N}
    kstop = KA.allocate(cache.backend, Int, npixels(cache))
    if nmax >= 0
        winding_cutoff!(kstop, cache, slab, nmax)
    else
        fill!(kstop, N + 1)
    end
    return kstop
end

# the backward pass: tails[j, k] is the invariant Stokes vector arriving from samples k…N (tails[j, 1] the image);
# samples from the cutoff on contribute nothing
@kernel function polarized_tails_kernel!(tails, kstop, params, tvec, lists, S, pc, met::Krang.Kerr, θo, ν, L, ::Val{N}) where {N}
    j = @index(Global, Linear)
    pix = build_pixel(pc, j, met, θo)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    c = RadiativeTransport(Transfer.ray_model(PolarizedSplats(params, tvec, lists), j), ν, L)
    R = zero(SVector{4,typeof(ν)})
    @inbounds stop = kstop[j]
    @inbounds tails[j, N + 1] = R
    for k in N:-1:1
        if k < stop
            O, E, on = Transfer.sample_step(c, _stored_sample(S, j, k), Δτ, pix)
            on && (R = E + O * R)
        end
        @inbounds tails[j, k] = R
    end
end

# the forward pass: the adjoint w of the Stokes vector in front of each sample, the sample's
# adjoint from w and its tail, and the splat parameters' share of it into grad[j, :, :]
@kernel function polarized_dual_kernel!(grad, dstokes, tails, kstop, params, tvec, lists, S, pc, met::Krang.Kerr, θo, ν, L, ::Val{N}) where {N}
    j = @index(Global, Linear)
    T = typeof(ν)
    pix = build_pixel(pc, j, met, θo)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    m = Transfer.ray_model(PolarizedSplats(params, tvec, lists), j)
    c = RadiativeTransport(m, ν, L)
    hor = Krang.horizon(met) * (1 + T(1e-3))
    @inbounds w = dstokes[j] * ν^3
    @inbounds stop = kstop[j]
    for k in 1:min(stop - 1, N)
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
    polarized_tails!(tails, cache, params, t_obs, ν_obs, L; nmax = -1, slab = 0, lists = nothing) -> tails

The backward pass of the dual sweep over the samples stored in `cache`: `tails[j, k]` is the
invariant Stokes vector arriving at the observer side of sample `k` of ray `j` from the samples
behind it (`tails[j, N + 1] = 0`; `tails[j, 1]` is the image, see [`tail_image`](@ref)). `tails`
is `npix × (N + 1)` of `SVector{4}` on the backend. With `nmax ≥ 0` the rays are truncated
after their `nmax`-th passage through the slab |z| < `slab`, as `polarized_image!` does
(`winding_cutoff!`). With `lists` (`ray_lists`) every ray loops over its own parcels.
"""
function polarized_tails!(tails, cache::GeodesicCache{T,N}, params, t_obs, ν_obs, L; nmax = -1, slab = 0, lists = nothing) where {T,N}
    backend = cache.backend
    nsamples(cache.samples) == N || throw(ArgumentError("the cache holds no stored samples: build it with store_samples = true and a storing marcher"))
    size(tails) == (npixels(cache), N + 1) || throw(ArgumentError("tails must be npix × (N + 1)"))
    prepare_backend!(backend)
    tvec = KA.allocate(backend, T, 1); fill!(tvec, T(t_obs))
    kstop = _cutoffs(cache, nmax, slab)
    polarized_tails_kernel!(backend, 64)(tails, kstop, params, tvec, lists, cache.samples, cache.consts, Krang.Kerr(cache.spin), cache.θo, T(ν_obs), T(L), Val(N); ndrange = npixels(cache))
    KA.synchronize(backend)
    return tails
end

"The observed Stokes vectors (sorted pixel order, cgs) from the tails of [`polarized_tails!`](@ref)."
tail_image(tails, ν_obs) = map(R -> R * ν_obs^3, tails[:, 1])

"""
    polarized_dual_sweep!(dparams, dstokes, tails, cache, params, t_obs, ν_obs, L; nmax = -1, slab = 0, lists = nothing) -> dparams

The forward pass of the dual sweep: with the `tails` of [`polarized_tails!`](@ref) and
`dstokes` the adjoint of the observed Stokes vectors (sorted pixel order, ∂l/∂(I, Q, U, V) in
cgs), carry the adjoint of the Stokes vector in front of each sample along every ray, form the
adjoints of the sample's step by forward-mode duals through the step operator, and accumulate
∂l/∂params into `dparams` (per-ray partial sums on the backend, reduced at the end, so that the
result does not depend on the order of the threads). `nmax`, `slab` and `lists` must match
the tails; with `lists` the per-ray sums go into the slots of the ray's list (`capacity`
slots per ray instead of one per parcel) and are gathered parcel by parcel in ray order.
"""
function polarized_dual_sweep!(dparams, dstokes::AbstractVector{SVector{4,T}}, tails, cache::GeodesicCache{T,N}, params, t_obs, ν_obs, L; nmax = -1, slab = 0, lists = nothing) where {T,N}
    backend = cache.backend
    npix = npixels(cache)
    size(tails) == (npix, N + 1) || throw(ArgumentError("tails must be npix × (N + 1)"))
    prepare_backend!(backend)
    tvec = KA.allocate(backend, T, 1); fill!(tvec, T(t_obs))
    kstop = _cutoffs(cache, nmax, slab)
    slots = lists === nothing ? size(params, 2) : capacity(lists)
    grad = KA.allocate(backend, T, npix, size(params, 1), slots); fill!(grad, zero(T))
    polarized_dual_kernel!(backend, 64)(grad, dstokes, tails, kstop, params, tvec, lists, cache.samples, cache.consts, Krang.Kerr(cache.spin), cache.θo, T(ν_obs), T(L), Val(N); ndrange = npix)
    KA.synchronize(backend)
    if lists === nothing
        dparams .+= reshape(sum(grad; dims = 1), size(params))
    else
        gather_slots!(dparams, grad, lists, backend)
    end
    return dparams
end

# the per-ray slot sums of every parcel, in ascending ray order (deterministic): a CSR by parcel of the
# (ray, slot) pairs, built on the host from the lists, and one thread per (row, parcel) on the backend
@kernel function gather_slots_kernel!(dparams, @Const(grad), @Const(ptr), @Const(rays), @Const(slots))
    r, i = @index(Global, NTuple)
    acc = zero(eltype(dparams))
    @inbounds for q in ptr[i]+1:ptr[i+1]
        acc += grad[rays[q], r, slots[q]]
    end
    @inbounds dparams[r, i] += acc
end
function gather_slots!(dparams, grad, lists::RayLists, backend)
    ids = Array(lists.ids); count = Array(lists.count)
    nrows, nsplat = size(dparams)
    ptr = zeros(Int32, nsplat + 1)
    for j in eachindex(count), s in 1:count[j]
        ptr[ids[s, j] + 1] += 1
    end
    cumsum!(ptr, ptr)
    pos = copy(ptr); total = Int(ptr[end])
    rays = Vector{Int32}(undef, total); slotsv = Vector{Int32}(undef, total)
    for j in eachindex(count), s in 1:count[j]
        i = ids[s, j]; pos[i] += 1
        rays[pos[i]] = j; slotsv[pos[i]] = s
    end
    dptr = KA.allocate(backend, Int32, nsplat + 1); copyto!(dptr, ptr)
    drays = KA.allocate(backend, Int32, max(total, 1)); dslots = KA.allocate(backend, Int32, max(total, 1))
    total > 0 && (copyto!(drays, rays); copyto!(dslots, slotsv))
    gather_slots_kernel!(backend, 64)(dparams, grad, dptr, drays, dslots; ndrange = (nrows, nsplat))
    KA.synchronize(backend)
    return dparams
end

# the per-ray parcel lists: for every ray the parcels whose support one of its samples enters. One kernel marks the hits
# (samples outer, so each sample's position is formed once; a byte per parcel and ray), the counts are column sums, and a
# second kernel writes each ray's list in ascending parcel order into capacity = maximum(count) slots: nothing overflows
# and no per-thread scratch is needed
@kernel function ray_hits_kernel!(hits, kstop, params, tvec, S, pc, met::Krang.Kerr, θo, ::Val{N}) where {N}
    j = @index(Global, Linear)
    T = eltype(tvec)
    pix = build_pixel(pc, j, met, θo)
    hor = Krang.horizon(met) * (1 + T(1e-3))
    @inbounds stop = kstop[j]
    @inbounds t_obs = tvec[1]
    nsplat = size(params, 2)
    for k in 1:N
        k < stop || break
        s = _stored_sample(S, j, k)
        (s.ok && s.r > hor) || continue
        x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
        t = t_obs - s.t
        for i in 1:nsplat
            @inbounds hits[i, j] != 0x00 && continue
            @inbounds outside_support(params, i, t, x, y, z) || (hits[i, j] = 0x01)
        end
    end
end
@kernel function ray_fill_kernel!(ids, @Const(hits), @Const(count))
    j = @index(Global, Linear)
    n = Int32(0)
    @inbounds for i in 1:size(hits, 1)
        if hits[i, j] != 0x00
            n += Int32(1)
            ids[n, j] = Int32(i)
        end
    end
end

"""
    ray_lists(cache, params, t_obs; nmax = -1, slab = 0) -> RayLists

For every stored ray the parcels whose bounding sphere (`outside_support`) one of its samples
enters at the observation time `t_obs` (samples beyond the half-orbit cutoff excluded as in
the transport), in ascending parcel order: the parcels the ray can see, a few out of
thousands. The transport over the lists is the same sum as over all parcels, because a parcel
outside its support contributes exactly zero. A hit byte per parcel and ray on the backend,
the counts as column sums, and the lists filled into `capacity = maximum(count)` slots.
"""
function ray_lists(cache::GeodesicCache{T,N}, params, t_obs; nmax = -1, slab = 0) where {T,N}
    backend = cache.backend
    npix = npixels(cache)
    nsamples(cache.samples) == N || throw(ArgumentError("the cache holds no stored samples: build it with store_samples = true and a storing marcher"))
    prepare_backend!(backend)
    tvec = KA.allocate(backend, T, 1); fill!(tvec, T(t_obs))
    kstop = _cutoffs(cache, nmax, slab)
    hits = KA.allocate(backend, UInt8, size(params, 2), npix); fill!(hits, 0x00)
    ray_hits_kernel!(backend, 64)(hits, kstop, params, tvec, cache.samples, cache.consts, Krang.Kerr(cache.spin), cache.θo, Val(N); ndrange = npix)
    KA.synchronize(backend)
    count = KA.allocate(backend, Int32, npix)
    copyto!(count, Int32.(vec(sum(Int32.(Array(hits)); dims = 1))))
    cap = max(Int(maximum(count)), 1)
    ids = KA.allocate(backend, Int32, cap, npix); fill!(ids, Int32(0))
    ray_fill_kernel!(backend, 64)(ids, hits, count; ndrange = npix)
    KA.synchronize(backend)
    return RayLists(ids, count)
end
