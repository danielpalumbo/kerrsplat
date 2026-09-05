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
r̂, φ̂, −θ̂), `:u1, :u2, :u3` (the spatial ZAMO components γβ⃗ of the fluid 4-velocity along the
same axes: `u3` is the vertical component) and
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
        sφ, cφ = sincos_pair(φ)
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
        sθ, cθ = sincos_pair(p[16, i]); sϕ, cϕ = sincos_pair(p[17, i])
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
function polarized_image!(out, cache::GeodesicCache, params, t_obs, ν_obs, L; nmax = -1, slab = 0)
    fused_march!(RadiativeTransport(PolarizedSplats(params, t_obs), ν_obs, L; nmax, slab), out, cache)
    return out
end

"Accumulator type for the polarized transport: `WindingState` when the ray is truncated by its half-orbit count, `RadiativeState` otherwise."
accumulator_type(::Type{T}, nmax) where {T} = nmax >= 0 ? WindingState{T} : RadiativeState{T}

"""
Screen-shaped array of Stokes 4-vectors (allocating). With `nmax ≥ 0` the rays are truncated
after their `nmax`-th passage through the slab |z| < `slab` (see `Transfer.WindingState`), so
the image holds the sub-images of order 0…nmax; the order-n sub-image alone is the difference of
the images with `nmax = n` and `nmax = n − 1`.
"""
function polarized_image(cache::GeodesicCache{T}, params, t_obs, ν_obs, L; nmax = -1, slab = 0) where {T}
    out = KA.allocate(cache.backend, accumulator_type(T, nmax), npixels(cache))
    fill!(out, zero(eltype(out)))
    polarized_image!(out, cache, params, t_obs, ν_obs, L; nmax, slab)
    return map(st -> observed_stokes(st, ν_obs), to_screen(cache, out))
end

"""
    polarized_cube(cache, params, times, νs, L)

Stokes movie cube over observation times and frequencies sharing one geodesic cache: an array of
Stokes 4-vectors of size (nα, nβ, length(times), length(νs)) on the host.
"""
function polarized_cube(cache::GeodesicCache{T}, params, times, νs, L; nmax = -1, slab = 0) where {T}
    out = KA.allocate(cache.backend, accumulator_type(T, nmax), npixels(cache))
    frame(t, ν) = (fill!(out, zero(eltype(out))); polarized_image!(out, cache, params, t, ν, L; nmax, slab); Array(map(st -> observed_stokes(st, ν), to_screen(cache, out))))
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

# ---- gradients inside the GPU kernel over stored samples ---------------------------------------
"""
    thin_gradient!(dparams, dout, cache, params, t_obs) -> dparams

∂(dout · image)/∂params for the thin splats by Enzyme reverse mode *inside* the fused kernel, one
ray per thread, over the samples stored in `cache` (a cache built with `store_samples = true` and
regenerated with the `Direct` or `Recurrence` marcher). The geodesic march stays outside the
differentiated region: Enzyme compiles the march's special functions (Jacobi elliptic
amplitudes, inverse trigonometric functions) through their checked host implementations when it
differentiates a CUDA kernel, which the device cannot run, while the consumer's own arithmetic
differentiates exactly. The adjoint seed `dout` is consumed. Works on the CPU backend and on
CUDA (gate: `test_stored_gradient`, host Enzyme gradient to 1e-12).
"""
function thin_gradient!(dparams, dout, cache::GeodesicCache{T,N}, params, t_obs) where {T,N}
    backend = cache.backend
    nsamples(cache.samples) == N || throw(ArgumentError("the cache holds no stored samples: build it with store_samples = true and a storing marcher"))
    prepare_backend!(backend; stack_bytes = ENZYME_STACK_BYTES, heap_bytes = ENZYME_HEAP_BYTES)   # the reverse tape: per-thread stack, then device malloc
    tvec = KA.allocate(backend, T, 1); fill!(tvec, T(t_obs))
    dtvec = KA.allocate(backend, T, 1); fill!(dtvec, zero(T))
    out = KA.allocate(backend, T, npixels(cache))
    met = Krang.Kerr(cache.spin)
    if backend isa CPU   # compile on one work item first (concurrent Enzyme compilation on several tasks deadlocks)
        dp = similar(dparams); fill!(dp, zero(T)); dt = similar(dtvec); fill!(dt, zero(T)); dw = copy(dout)
        stored_adjoint_kernel!(backend, 1)(out, dw, dp, params, tvec, dt, cache.samples, cache.consts, met, cache.θo, Val(N); ndrange = 1)
        KA.synchronize(backend)
    end
    stored_adjoint_kernel!(backend, 64)(out, dout, dparams, params, tvec, dtvec, cache.samples, cache.consts, met, cache.θo, Val(N); ndrange = npixels(cache))
    KA.synchronize(backend)
    return dparams
end

@inline function stored_thin_value!(out, params, tvec, S, pc, j, met, θo, ::Val{N}) where {N}
    pix = build_pixel(pc, j, met, θo)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    c = ThinRenderer(params, tvec)
    acc = zero(eltype(out))
    for k in 1:N
        @inbounds f = S.flags[j, k]
        ok = (f & 0x01) != 0x00; νr = (f & 0x02) != 0x00; νθ = (f & 0x04) != 0x00
        @inbounds s = GeodesicSample(S.t[j, k], S.r[j, k], S.θ[j, k], S.ϕ[j, k], νr, νθ, ok)
        acc = c(acc, j, k, s, Δτ, pix)
    end
    @inbounds out[j] = acc
    return nothing
end

@kernel function stored_adjoint_kernel!(out, dout, dparams, params, tvec, dtvec, S, pc, met::Krang.Kerr, θo, ::Val{N}) where {N}
    j = @index(Global, Linear)
    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(stored_thin_value!), Enzyme.Const,
                             Enzyme.Duplicated(out, dout), Enzyme.Duplicated(params, dparams), Enzyme.Duplicated(tvec, dtvec),
                             Enzyme.Const(S), Enzyme.Const(pc), Enzyme.Const(j), Enzyme.Const(met), Enzyme.Const(θo), Enzyme.Const(Val(N)))
end

# ---- the polarized gradient inside the GPU kernel: a chunked reverse sweep -------------------------
"""
    chunk_size(N; kmax = 1) -> Int

Samples per chunk of the polarized reverse sweep: the largest divisor of `N` not above `kmax`.
One sample per chunk is the fastest on the GPU (the reverse pass of one polarized sample costs
about 26 ms of thread time on the 2080 SUPER, two samples per chunk 94 ms, and four or more no
longer compile: ptxas runs out of memory on the unrolled code), so `kmax = 1` is the default;
the per-thread stack holds the tape of up to eight.
"""
function chunk_size(N::Integer; kmax::Integer = 1)
    for k in min(kmax, 8, N):-1:1
        N % k == 0 && return k
    end
    return 1
end

@inline function _stored_sample(S, j, k)
    @inbounds f = S.flags[j, k]
    ok = (f & 0x01) != 0x00; νr = (f & 0x02) != 0x00; νθ = (f & 0x04) != 0x00
    @inbounds return GeodesicSample(S.t[j, k], S.r[j, k], S.θ[j, k], S.ϕ[j, k], νr, νθ, ok)
end

# forward pass over stored samples: the radiative state at every chunk boundary (states[j, c + 1] after chunk c)
@kernel function polarized_chunks_forward!(states, params, tvec, S, pc, met::Krang.Kerr, θo, ν, L, ::Val{N}, ::Val{K}) where {N,K}
    j = @index(Global, Linear)
    pix = build_pixel(pc, j, met, θo)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    c = RadiativeTransport(PolarizedSplats(params, tvec), ν, L)
    acc = zero(eltype(states))
    @inbounds states[j, 1] = acc
    for k in 1:N
        acc = c(acc, j, k, _stored_sample(S, j, k), Δτ, pix)
        k % K == 0 && (@inbounds states[j, k ÷ K + 1] = acc)
    end
end

# one chunk of K samples starting after sample (c − 1)K, from `states[j, c]` to `states[j, c + 1]`: one
# straight-line method per chunk size (a loop in the differentiated device function keeps Enzyme's
# per-iteration cache in device malloc, and the differentiated kernel then runs thirty times slower)
for K in 1:8
    body = [:(acc = tr(acc, j, k0 + $i, _stored_sample(S, j, k0 + $i), Δτ, pix)) for i in 1:K]
    @eval @inline function _fold_chunk(acc, tr, S, j, k0, Δτ, pix, ::Val{$K})
        $(body...)
        return acc
    end
end

@inline function _polarized_chunk!(states, params, tvec, S, pc, j, met, θo, ν, L, c, ::Val{N}, ::Val{K}, ::Val{NS}) where {N,K,NS}
    pix = build_pixel(pc, j, met, θo)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    tr = RadiativeTransport(StaticCount(PolarizedSplats(params, tvec), Val(NS)), ν, L)
    @inbounds acc = states[j, c]
    acc = _fold_chunk(acc, tr, S, j, (c - 1) * K, Δτ, pix, Val(K))
    @inbounds states[j, c + 1] = acc
    return nothing
end

@kernel function polarized_chunk_adjoint!(states, dstates, params, dparams, tvec, dtvec, S, pc, met::Krang.Kerr, θo, ν, L, c, ::Val{N}, ::Val{K}, ::Val{NS}) where {N,K,NS}
    j = @index(Global, Linear)
    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(_polarized_chunk!), Enzyme.Const,
                             Enzyme.Duplicated(states, dstates), Enzyme.Duplicated(params, dparams), Enzyme.Duplicated(tvec, dtvec),
                             Enzyme.Const(S), Enzyme.Const(pc), Enzyme.Const(j), Enzyme.Const(met), Enzyme.Const(θo),
                             Enzyme.Const(ν), Enzyme.Const(L), Enzyme.Const(c), Enzyme.Const(Val(N)), Enzyme.Const(Val(K)), Enzyme.Const(Val(NS)))
end

"The zero of the adjoint space of a radiative state (`zero(RadiativeState)` is the compositing identity, P = 1, not a zero adjoint)."
@inline zero_adjoint(::Type{RadiativeState{T}}) where {T} = RadiativeState(zero(SMatrix{4,4,T}), zero(SVector{4,T}))

"""
    polarized_forward_states!(states, cache, params, t_obs, ν_obs, L, ::Val{K}) -> states

The forward pass of the chunked sweep over the samples stored in `cache`: the radiative state
of every ray at each chunk boundary (`states[j, c + 1]` after chunk `c`; `states[j, 1]` is the
compositing identity), `K` samples per chunk. `states` is `npix × (N ÷ K + 1)` on the backend.
"""
function polarized_forward_states!(states, cache::GeodesicCache{T,N}, params, t_obs, ν_obs, L, ::Val{K}) where {T,N,K}
    backend = cache.backend
    nsamples(cache.samples) == N || throw(ArgumentError("the cache holds no stored samples: build it with store_samples = true and a storing marcher"))
    N % K == 0 || throw(ArgumentError("the chunk size must divide the sample count"))
    tvec = KA.allocate(backend, T, 1); fill!(tvec, T(t_obs))
    polarized_chunks_forward!(backend, 64)(states, params, tvec, cache.samples, cache.consts, Krang.Kerr(cache.spin), cache.θo, T(ν_obs), T(L), Val(N), Val(K); ndrange = npixels(cache))
    KA.synchronize(backend)
    return states
end

"""
    polarized_reverse_sweep!(dparams, dstokes, states, cache, params, t_obs, ν_obs, L, ::Val{K}) -> dparams

The reverse sweep: with the boundary `states` of the forward pass and `dstokes` the adjoint of
the observed Stokes vectors (sorted pixel order, ∂loss/∂(I, Q, U, V) in cgs), differentiate
the chunks from the last to the first, carrying the adjoint of the incoming state along, and
accumulate ∂loss/∂params into `dparams`.
"""
function polarized_reverse_sweep!(dparams, dstokes::AbstractVector{SVector{4,T}}, states, cache::GeodesicCache{T,N}, params, t_obs, ν_obs, L, ::Val{K}) where {T,N,K}
    backend = cache.backend
    prepare_backend!(backend; stack_bytes = ENZYME_STACK_BYTES, heap_bytes = ENZYME_HEAP_BYTES)
    C = N ÷ K
    npix = npixels(cache)
    met = Krang.Kerr(cache.spin)
    tvec = KA.allocate(backend, T, 1); fill!(tvec, T(t_obs))
    dtvec = KA.allocate(backend, T, 1); fill!(dtvec, zero(T))
    ν = T(ν_obs); Lc = T(L)
    # the shadow of the boundary states: column c + 1 holds the adjoint of the state after chunk c (the seed of
    # chunk c, consumed by its reverse pass, which accumulates the adjoint of its incoming state into column c)
    dstates = similar(states)
    fill!(dstates, zero_adjoint(RadiativeState{T}))
    copyto!(view(dstates, :, C + 1), map(d -> RadiativeState(zero(SMatrix{4,4,T}), d .* ν^3), dstokes))
    NSv = Val(size(params, 2))                          # the splat count is a compile-time constant of the adjoint kernel (one compile per count)
    if backend isa CPU   # compile on one work item first (concurrent Enzyme compilation on several tasks deadlocks)
        dp = similar(dparams); fill!(dp, zero(T)); dt = similar(dtvec); fill!(dt, zero(T)); ds = copy(dstates); st = copy(states)
        polarized_chunk_adjoint!(backend, 1)(st, ds, params, dp, tvec, dt, cache.samples, cache.consts, met, cache.θo, ν, Lc, C, Val(N), Val(K), NSv; ndrange = 1)
        KA.synchronize(backend)
    end
    for c in C:-1:1
        polarized_chunk_adjoint!(backend, 64)(states, dstates, params, dparams, tvec, dtvec, cache.samples, cache.consts, met, cache.θo, ν, Lc, c, Val(N), Val(K), NSv; ndrange = npix)
    end
    KA.synchronize(backend)
    return dparams
end

"""
    polarized_gradient!(dparams, dstokes, cache, params, t_obs, ν_obs, L; kmax = 1) -> (dparams, image)

Gradient of Σⱼ dstokes[j] · observed_stokes(ray j) with respect to the polarized splat parameters,
by Enzyme reverse mode inside the kernel over the samples stored in `cache`, one ray per thread,
as a chunked reverse sweep: a forward kernel folds every ray over all `N` samples and keeps the
radiative state at each chunk boundary (`chunk_size(N; kmax)` samples per chunk, at most the
eight a 64 KB per-thread stack can tape), then the chunks are differentiated from the last to
the first with the adjoint of the incoming state carried along (the seed of chunk c is the
adjoint of its outgoing state, which the reverse pass consumes). `dstokes` is a vector of
`SVector{4}` in sorted pixel order (∂loss/∂(I, Q, U, V) in cgs); the accumulated `dparams` and
the image (observed Stokes vectors, sorted order) are returned. Gate: `test_polarized_gradient`
(host Enzyme gradient of the same loss, CPU and CUDA).
"""
function polarized_gradient!(dparams, dstokes::AbstractVector{SVector{4,T}}, cache::GeodesicCache{T,N}, params, t_obs, ν_obs, L; kmax::Integer = 1) where {T,N}
    K = chunk_size(N; kmax)
    states = KA.allocate(cache.backend, RadiativeState{T}, npixels(cache), N ÷ K + 1)
    polarized_forward_states!(states, cache, params, t_obs, ν_obs, L, Val(K))
    image = map(st -> observed_stokes(st, T(ν_obs)), states[:, end])
    polarized_reverse_sweep!(dparams, dstokes, states, cache, params, t_obs, ν_obs, L, Val(K))
    return dparams, image
end
