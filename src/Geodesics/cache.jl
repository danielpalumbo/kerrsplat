# The geodesic cache and regenerate! (plan §9 step 2).

"""
    prepare_backend!(backend; stack_bytes = cuda_stack_bytes(T))

One-time device configuration needed before Krang code runs in kernels. On CUDA this raises
the per-thread stack limit: Krang's Float64 radial integrals overflow the default 1 KB stack,
which shows up as an illegal memory access (plan §2, item 3). No-op elsewhere.
"""
prepare_backend!(::KA.Backend; kwargs...) = nothing

const CUDA_STACK_BYTES = 4096

"""
    cuda_stack_bytes(T)

Per-thread stack the kernels need for scalar type `T`: 4 KB for Float64 (measured: K1 and
both K2 marchers up to 1000 samples), 16 KB for two-partial ForwardDiff duals (12 KB fails),
scaled up for wider duals.
"""
cuda_stack_bytes(::Type{T}) where {T} = sizeof(T) <= 8 ? CUDA_STACK_BYTES : 16384 * cld(sizeof(T), 24)

function prepare_backend!(::CUDA.CUDABackend; stack_bytes::Integer = CUDA_STACK_BYTES)
    if CUDA.limit(CUDA.LIMIT_STACK_SIZE) < stack_bytes
        CUDA.limit!(CUDA.LIMIT_STACK_SIZE, stack_bytes)
    end
    return nothing
end

"""
    Direct()
    Recurrence(M = 64)   (a `Recurrence{M}`)
    Fused(M = 64)        (a `Fused{M}`)

Marcher choice for [`regenerate!`](@ref). `Direct` evaluates Krang's closed forms at every
sample (the reference; slow). `Recurrence{M}` advances r and θ by the Jacobi addition theorems
and t̃, φ by quadrature of the Mino-time rates, re-anchoring everything to the closed forms
every `M` samples (plan §4; fewer near the critical curve). The largest anchor residual of
each ray is available in `cache.residual_t`, `cache.residual_ϕ` (sorted order). `Fused{M}` is
the same marcher run later, inside the consumer, by [`fused_march!`](@ref): `regenerate!`
then only rebuilds the per-pixel constants and stores no samples (plan §4 "fused mode", §7).
"""
struct Direct end
struct Recurrence{M} end
struct Fused{M} end
Recurrence(M::Integer = 64) = Recurrence{Int(M)}()
Fused(M::Integer = 64) = Fused{Int(M)}()

anchor_interval(::Recurrence{M}) where {M} = M
anchor_interval(::Fused{M}) where {M} = M

"""
    GeodesicCache(backend, camera, Val(N); store_samples = true)
    GeodesicCache(backend, camera, N; store_samples = true)

Device-resident geodesic data for one camera and `N` samples per ray. With
`store_samples = false` no sample storage is allocated (fused mode only). Owns:

- `αs, βs`: the camera's pixel coordinates in screen order;
- `perm` (device) / `perm_host`: `perm[j]` is the screen index of the pixel in sorted slot `j`;
  `ranges` gives the slot ranges of the three root cases (see [`case_permutation`](@ref));
- `consts::PixelConstants`: K1 output, sorted order;
- `samples::GeodesicSamples`: K2 output, `(npix, N)`, sorted order;
- `residual_t`, `residual_ϕ`: per ray (sorted order), the largest |quadrature − Krang| at the
  re-anchoring samples of the last `Recurrence` run (zero after a `Direct` run);
- `spin`, `θo`: the spacetime the cache currently holds (`NaN` until the first `regenerate!`);
- `generation`: number of `regenerate!` calls so far.

Nothing is computed by the constructor; call [`regenerate!`](@ref).
"""
mutable struct GeodesicCache{T,N,B<:KA.Backend,VT<:AbstractVector{T},VI<:AbstractVector{Int},
                             VC<:AbstractVector{Int8},PC<:PixelConstants{T},S<:GeodesicSamples{T}}
    const backend::B
    const nval::Val{N}
    const αs::VT
    const βs::VT
    screen_size::Tuple{Int,Int}
    const numreals_screen::VC
    const perm::VI
    perm_host::Vector{Int}
    ranges::NamedTuple{(:case2, :case3, :case4),NTuple{3,UnitRange{Int}}}
    const consts::PC
    const samples::S
    const residual_t::VT
    const residual_ϕ::VT
    spin::T
    θo::T
    generation::Int
    marcher::Union{Direct,Recurrence,Fused}
end

function GeodesicCache(backend::KA.Backend, camera::Camera{T}, nval::Val{N}; store_samples::Bool = true) where {T,N}
    N >= 1 || throw(ArgumentError("need at least one sample per ray"))
    npix = npixels(camera)
    αs = KA.allocate(backend, T, npix)
    βs = KA.allocate(backend, T, npix)
    copyto!(αs, camera.αs)
    copyto!(βs, camera.βs)
    numreals_screen = KA.allocate(backend, Int8, npix)
    perm = KA.allocate(backend, Int, npix)
    consts = PixelConstants{T}(backend, npix)
    samples = GeodesicSamples{T}(backend, store_samples ? npix : 0, N)
    residual_t = KA.allocate(backend, T, npix)
    residual_ϕ = KA.allocate(backend, T, npix)
    fill!(residual_t, zero(T))
    fill!(residual_ϕ, zero(T))
    empty = (case2 = 1:0, case3 = 1:0, case4 = 1:0)
    return GeodesicCache(backend, nval, αs, βs, size(camera), numreals_screen, perm,
                         collect(1:npix), empty, consts, samples, residual_t, residual_ϕ, T(NaN), T(NaN), 0, Direct())
end
GeodesicCache(backend::KA.Backend, camera::Camera, N::Integer; kwargs...) = GeodesicCache(backend, camera, Val(Int(N)); kwargs...)

"Whether the cache holds per-sample storage (false for fused-only caches)."
has_samples(c::GeodesicCache) = npixels(c.samples) == npixels(c)

npixels(c::GeodesicCache) = length(c.αs)
nsamples(::GeodesicCache{T,N}) where {T,N} = N
Base.eltype(::GeodesicCache{T}) where {T} = T
Base.size(c::GeodesicCache) = c.screen_size
KA.get_backend(c::GeodesicCache) = c.backend

function Base.show(io::IO, c::GeodesicCache{T,N}) where {T,N}
    print(io, "GeodesicCache{$T}: ", npixels(c), " pixels ", c.screen_size, " × ", N,
          has_samples(c) ? " samples on " : " samples (not stored) on ", nameof(typeof(c.backend)), " (", c.marcher, "); a = ", c.spin, ", θo = ", c.θo,
          "; cases (4,2,0 real roots): ", length(c.ranges.case2), "/", length(c.ranges.case3),
          "/", length(c.ranges.case4))
end

"""
    regenerate!(cache, a, θo; marcher = Direct(), workgroup = 256)
    regenerate!(cache, a, θo, camera; marcher = Direct(), workgroup = 256)

Rebuild everything in `cache` for spin `a` and observer inclination `θo` (radians), optionally
with a new camera of the same pixel count: K0 (root cases, screen order) → case sort (host)
→ K1 (per-pixel constants, sorted order) → K2 (stored samples, by `marcher`; see
[`Direct`](@ref) and [`Recurrence`](@ref)). Synchronizes the backend before returning.
Returns `cache`.
"""
function regenerate!(cache::GeodesicCache{T,N}, spin::Real, θo::Real; marcher::Union{Direct,Recurrence,Fused} = Direct(),
                     workgroup::Integer = 256) where {T,N}
    marcher isa Fused || has_samples(cache) ||
        throw(ArgumentError("this cache was built without sample storage; use marcher = Fused(M) and fused_march!"))
    a = T(spin)
    θ = T(θo)
    backend = cache.backend
    prepare_backend!(backend; stack_bytes = cuda_stack_bytes(T))
    met = Krang.Kerr(a)
    npix = npixels(cache)

    root_case_kernel!(backend, workgroup)(cache.numreals_screen, met, θ, cache.αs, cache.βs; ndrange = npix)
    KA.synchronize(backend)
    perm_host, ranges = case_permutation(Array(cache.numreals_screen))
    copyto!(cache.perm, perm_host)
    cache.perm_host = perm_host
    cache.ranges = ranges

    pixel_constants_kernel!(backend, workgroup)(cache.consts, met, θ, cache.αs, cache.βs, cache.perm; ndrange = npix)
    march!(marcher, cache, ranges, met, θ)
    KA.synchronize(backend)

    cache.spin = a
    cache.θo = θ
    cache.generation += 1
    cache.marcher = marcher
    return cache
end

function march!(::Direct, cache::GeodesicCache, ranges, met, θo)
    fill!(cache.residual_t, zero(eltype(cache)))
    fill!(cache.residual_ϕ, zero(eltype(cache)))
    return direct_march!(cache.samples, cache.consts, met, θo, cache.nval)
end
march!(m::Recurrence, cache::GeodesicCache, ranges, met, θo) =
    quadrature_march!(cache.samples, cache.consts, cache.residual_t, cache.residual_ϕ, ranges, met, θo,
                      cache.nval, Val(anchor_interval(m)))
march!(::Fused, cache::GeodesicCache, ranges, met, θo) = nothing

"""
    fused_march!(f, out, cache; acc0 = zero(eltype(out)), workgroup = 128)

Run the marcher of `cache` (a `Fused(M)` cache after `regenerate!`, or any regenerated cache
with `M` taken from its marcher) with the consumer `f` folded over each ray's samples,
`acc = f(acc, j, k, sample, Δτ, pix)` starting from `acc0`, and write each ray's final
accumulator to `out` (sorted order; `unsort`/`to_screen` map it to the screen). The anchor
residuals of this run replace `cache.residual_t`, `cache.residual_ϕ`. Synchronizes the backend.
"""
function fused_march!(f, out, cache::GeodesicCache{T,N}; acc0 = zero(eltype(out)), workgroup::Integer = 128) where {T,N}
    cache.generation >= 1 || throw(ArgumentError("regenerate! the cache first"))
    M = cache.marcher isa Direct ? 64 : anchor_interval(cache.marcher)
    fused_march!(f, out, acc0, cache.consts, cache.residual_t, cache.residual_ϕ, cache.ranges, Krang.Kerr(cache.spin), cache.θo,
                 cache.nval, Val(M); workgroup = workgroup)
    KA.synchronize(cache.backend)
    return out
end

"""
    tiles(camera, ntiles) -> Vector{Tuple{Camera, UnitRange{Int}}}

Split a camera into `ntiles` cameras of consecutive pixels (screen order) with the index range
each covers, so that a screen too large for stored samples can be regenerated tile by tile
with a smaller cache and the per-tile results written back into the full screen (plan §7).
"""
function tiles(camera::Camera, ntiles::Integer)
    npix = npixels(camera)
    bounds = round.(Int, range(0, npix, length = ntiles + 1))
    return [(Camera(camera.αs[bounds[i]+1:bounds[i+1]], camera.βs[bounds[i]+1:bounds[i+1]]), bounds[i]+1:bounds[i+1])
            for i in 1:ntiles if bounds[i+1] > bounds[i]]
end

function regenerate!(cache::GeodesicCache, spin::Real, θo::Real, camera::Camera; kwargs...)
    npixels(camera) == npixels(cache) ||
        throw(DimensionMismatch("camera has $(npixels(camera)) pixels, cache was built for $(npixels(cache))"))
    copyto!(cache.αs, camera.αs)
    copyto!(cache.βs, camera.βs)
    cache.screen_size = size(camera)
    return regenerate!(cache, spin, θo; kwargs...)
end

# Scatter sorted-order data back to screen order.
@kernel function unsort_kernel!(dst, @Const(src), @Const(perm))
    j, k = @index(Global, NTuple)
    @inbounds dst[perm[j], k] = src[j, k]
end

"""
    unsort(cache, x)

Map a vector (`npix`) or matrix (`npix × M`) stored in sorted-pixel order back to screen order.
Runs on the backend that holds `x`: on the cache's own backend with the device permutation, or
on the host (for arrays obtained through [`host`](@ref)) with `cache.perm_host`. Returns a new
array of the same type.
"""
function unsort(cache::GeodesicCache, x::AbstractVecOrMat)
    size(x, 1) == npixels(cache) || throw(DimensionMismatch("first dimension must be the pixel count"))
    src = reshape(x, size(x, 1), size(x, 2))
    dst = similar(src)
    backend = KA.get_backend(x)
    if backend == cache.backend
        perm = cache.perm
    elseif backend isa KA.CPU
        perm = cache.perm_host
    else
        throw(ArgumentError("array lives on $(nameof(typeof(backend))), cache on $(nameof(typeof(cache.backend)))"))
    end
    unsort_kernel!(backend, (128, 1))(dst, src, perm; ndrange = size(src))
    KA.synchronize(backend)
    return x isa AbstractVector ? vec(dst) : dst
end

"""
    to_screen(cache, x)

[`unsort`](@ref) followed by a reshape to the screen shape, `(nα, nβ[, M])`.
"""
function to_screen(cache::GeodesicCache, x::AbstractVecOrMat)
    y = unsort(cache, x)
    return x isa AbstractVector ? reshape(y, cache.screen_size) : reshape(y, cache.screen_size..., size(x, 2))
end
