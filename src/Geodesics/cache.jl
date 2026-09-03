# The geodesic cache and regenerate! (plan §9 step 2).

"""
    prepare_backend!(backend)

One-time device configuration needed before Krang code runs in kernels. On CUDA this raises
the per-thread stack limit: Krang's Float64 radial integrals overflow the default 1 KB stack,
which shows up as an illegal memory access (plan §2, item 3). No-op elsewhere.
"""
prepare_backend!(::KA.Backend) = nothing

const CUDA_STACK_BYTES = 4096

function prepare_backend!(::CUDA.CUDABackend; stack_bytes::Integer = CUDA_STACK_BYTES)
    if CUDA.limit(CUDA.LIMIT_STACK_SIZE) < stack_bytes
        CUDA.limit!(CUDA.LIMIT_STACK_SIZE, stack_bytes)
    end
    return nothing
end

"""
    GeodesicCache(backend, camera, Val(N))
    GeodesicCache(backend, camera, N)

Device-resident geodesic data for one camera and `N` samples per ray. Owns:

- `αs, βs`: the camera's pixel coordinates in screen order;
- `perm` (device) / `perm_host`: `perm[j]` is the screen index of the pixel in sorted slot `j`;
  `ranges` gives the slot ranges of the three root cases (see [`case_permutation`](@ref));
- `consts::PixelConstants`: K1 output, sorted order;
- `samples::GeodesicSamples`: K2 output, `(npix, N)`, sorted order;
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
    spin::T
    θo::T
    generation::Int
end

function GeodesicCache(backend::KA.Backend, camera::Camera{T}, nval::Val{N}) where {T,N}
    N >= 1 || throw(ArgumentError("need at least one sample per ray"))
    npix = npixels(camera)
    αs = KA.allocate(backend, T, npix)
    βs = KA.allocate(backend, T, npix)
    copyto!(αs, camera.αs)
    copyto!(βs, camera.βs)
    numreals_screen = KA.allocate(backend, Int8, npix)
    perm = KA.allocate(backend, Int, npix)
    consts = PixelConstants{T}(backend, npix)
    samples = GeodesicSamples{T}(backend, npix, N)
    empty = (case2 = 1:0, case3 = 1:0, case4 = 1:0)
    return GeodesicCache(backend, nval, αs, βs, size(camera), numreals_screen, perm,
                         collect(1:npix), empty, consts, samples, T(NaN), T(NaN), 0)
end
GeodesicCache(backend::KA.Backend, camera::Camera, N::Integer) = GeodesicCache(backend, camera, Val(Int(N)))

npixels(c::GeodesicCache) = length(c.αs)
nsamples(::GeodesicCache{T,N}) where {T,N} = N
Base.eltype(::GeodesicCache{T}) where {T} = T
Base.size(c::GeodesicCache) = c.screen_size
KA.get_backend(c::GeodesicCache) = c.backend

function Base.show(io::IO, c::GeodesicCache{T,N}) where {T,N}
    print(io, "GeodesicCache{$T}: ", npixels(c), " pixels ", c.screen_size, " × ", N,
          " samples on ", nameof(typeof(c.backend)), "; a = ", c.spin, ", θo = ", c.θo,
          "; cases (4,2,0 real roots): ", length(c.ranges.case2), "/", length(c.ranges.case3),
          "/", length(c.ranges.case4))
end

"""
    regenerate!(cache, a, θo; workgroup = 256)
    regenerate!(cache, a, θo, camera; workgroup = 256)

Rebuild everything in `cache` for spin `a` and observer inclination `θo` (radians), optionally
with a new camera of the same pixel count: K0 (root cases, screen order) → case sort (host)
→ K1 (per-pixel constants, sorted order) → K2 (stored samples). Synchronizes the backend
before returning. Returns `cache`.
"""
function regenerate!(cache::GeodesicCache{T,N}, spin::Real, θo::Real; workgroup::Integer = 256) where {T,N}
    a = T(spin)
    θ = T(θo)
    backend = cache.backend
    prepare_backend!(backend)
    met = Krang.Kerr(a)
    npix = npixels(cache)

    root_case_kernel!(backend, workgroup)(cache.numreals_screen, met, θ, cache.αs, cache.βs; ndrange = npix)
    KA.synchronize(backend)
    perm_host, ranges = case_permutation(Array(cache.numreals_screen))
    copyto!(cache.perm, perm_host)
    cache.perm_host = perm_host
    cache.ranges = ranges

    pixel_constants_kernel!(backend, workgroup)(cache.consts, met, θ, cache.αs, cache.βs, cache.perm; ndrange = npix)
    direct_march!(cache.samples, cache.consts, met, θ, cache.nval)
    KA.synchronize(backend)

    cache.spin = a
    cache.θo = θ
    cache.generation += 1
    return cache
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
