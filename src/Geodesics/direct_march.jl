# K2, stored mode, reference implementation: Krang's direct per-sample evaluation.

const SAMPLE_OK = 0x01
const SAMPLE_NUR = 0x02
const SAMPLE_NUTH = 0x04

"""
    pack_flags(ok, νr, νθ) -> UInt8
    unpack_flags(flags) -> (; ok, νr, νθ)

Per-sample flags stored in one byte: bit 0 `ok` (Krang produced a valid point), bit 1 `νr`
(radial momentum sign, true = r increasing along the ray), bit 2 `νθ` (polar momentum sign).
"""
@inline pack_flags(ok::Bool, νr::Bool, νθ::Bool) =
    UInt8(ok) | (UInt8(νr) << 1) | (UInt8(νθ) << 2)
@inline unpack_flags(f::UInt8) =
    (ok = (f & SAMPLE_OK) != 0x00, νr = (f & SAMPLE_NUR) != 0x00, νθ = (f & SAMPLE_NUTH) != 0x00)

"""
    GeodesicSample{T}

One point on a ray: regularized Boyer–Lindquist time `t` (Krang's `emission_time_regularized`,
so that arrival-time *differences* are exact), radius `r`, polar angle `θ`, unwrapped azimuth
`ϕ`, the momentum signs `νr`, `νθ` (true = coordinate increasing), and `ok`. When `ok` is
false Krang returns zero coordinates; consumers must discard the sample.
"""
struct GeodesicSample{T}
    t::T
    r::T
    θ::T
    ϕ::T
    νr::Bool
    νθ::Bool
    ok::Bool
end

"""
    mino_step(τ_total, Val(N))

Mino-time spacing of the sample grid. A ray with total Mino time `τ_total` carries `N`
samples at τ_k = k Δτ, k = 1…N, with Δτ = τ_total / (N + 1): uniform, and never on either end
point (the observer at τ = 0; radial infinity or the horizon at τ_total). Uniform spacing is
what the addition-theorem recurrence (plan §4) requires.
"""
@inline mino_step(τ_total, ::Val{N}) where {N} = τ_total / (N + 1)

"""
    mino_times(τ_total, Val(N))

The sample grid of [`mino_step`](@ref) as a vector (host convenience).
"""
mino_times(τ_total, ::Val{N}) where {N} = [k * mino_step(τ_total, Val(N)) for k in 1:N]

"""
    direct_sample(pix, τ) -> GeodesicSample

Reference per-sample evaluation: Krang's closed-form `emission_coordinates` at Mino time `τ`.
"""
@inline function direct_sample(pix::Krang.AbstractPixel, τ)
    t, r, θ, ϕ, νr, νθ, ok = Krang.emission_coordinates(pix, τ)
    return GeodesicSample(t, r, θ, ϕ, νr, νθ, ok)
end

"""
    GeodesicSamples{T}

Stored samples for every ray: matrices of size `(npix, N)` in sorted-pixel order, so that a
warp of consecutive rays reading sample `k` touches contiguous memory. Fields `t, r, θ, ϕ`
(see [`GeodesicSample`](@ref)) and `flags` (see [`pack_flags`](@ref)). 33 bytes per sample.
"""
struct GeodesicSamples{T,MT<:AbstractMatrix{T},MF<:AbstractMatrix{UInt8}}
    t::MT
    r::MT
    θ::MT
    ϕ::MT
    flags::MF
end
Adapt.@adapt_structure GeodesicSamples

"""
    GeodesicSamples{T}(backend, npix, N)

Allocate (uninitialized) sample storage for `npix` rays with `N` samples each.
"""
function GeodesicSamples{T}(backend::KA.Backend, npix::Integer, N::Integer) where {T}
    m() = KA.allocate(backend, T, npix, N)
    return GeodesicSamples(m(), m(), m(), m(), KA.allocate(backend, UInt8, npix, N))
end

npixels(S::GeodesicSamples) = size(S.t, 1)
nsamples(S::GeodesicSamples) = size(S.t, 2)
Base.eltype(::GeodesicSamples{T}) where {T} = T
Base.sizeof(S::GeodesicSamples) = sizeof(S.t) + sizeof(S.r) + sizeof(S.θ) + sizeof(S.ϕ) + sizeof(S.flags)

"""
    S[j, k] -> GeodesicSample

Host-side accessor (scalar indexing; use on `host(S)` for device storage).
"""
Base.@propagate_inbounds function Base.getindex(S::GeodesicSamples, j::Integer, k::Integer)
    f = unpack_flags(S.flags[j, k])
    return GeodesicSample(S.t[j, k], S.r[j, k], S.θ[j, k], S.ϕ[j, k], f.νr, f.νθ, f.ok)
end

@inline function store_sample!(S, j, k, s::GeodesicSample)
    @inbounds begin
        S.t[j, k] = s.t
        S.r[j, k] = s.r
        S.θ[j, k] = s.θ
        S.ϕ[j, k] = s.ϕ
        S.flags[j, k] = pack_flags(s.ok, s.νr, s.νθ)
    end
    return nothing
end

# K2 (stored mode, direct evaluation): one thread per ray, static trip count.
@kernel function direct_march_kernel!(S, pc, met::Krang.Kerr, θo, ::Val{N}) where {N}
    j = @index(Global, Linear)
    pix = build_pixel(pc, j, met, θo)
    Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
    for k in 1:N
        store_sample!(S, j, k, direct_sample(pix, k * Δτ))
    end
end

"""
    direct_march!(S, pc, met, θo, Val(N); workgroup = 128)

Fill `S` with Krang's direct evaluation at the sample grid of every ray in `pc`.
"""
function direct_march!(S::GeodesicSamples, pc::PixelConstants, met::Krang.Kerr, θo, ::Val{N};
                       workgroup::Integer = 128) where {N}
    npixels(S) == npixels(pc) || throw(DimensionMismatch("sample and pixel counts differ"))
    nsamples(S) == N || throw(DimensionMismatch("sample storage has $(nsamples(S)) columns, kernel is compiled for $N"))
    backend = KA.get_backend(S.t)
    direct_march_kernel!(backend, workgroup)(S, pc, met, θo, Val(N); ndrange = npixels(S))
    return S
end
