# The visibility likelihood of a frame on the backend: the direct Fourier transform of the frame's image onto the
# scans' baselines, the χ², and the adjoint transform of the weighted residuals that seeds the frame's dual sweep,
# as KernelAbstractions kernels over the device image (sorted pixel order, as the tails kernel leaves it). The host
# transform (`visibilities`, `visibility_seed!`) had become the largest share of a time-resolved iteration: ninety
# frames at a second each against twenty seconds of sweeps.

"""
    FrameScans(u, v, vis, σ, tap)

The visibility scans of one frame concatenated on the backend: baselines [wavelengths], observed
Stokes visibilities [Jy], their noise per Stokes parameter, and the scattering taper of each
baseline (one without a kernel), all in the scalar type of the fit.
"""
struct FrameScans{U,W,S}
    u::U
    v::U
    vis::W
    σ::S
    tap::U
end

"The scans of one frame (all `VisibilityData`) as `FrameScans` on `backend` in the type `T`."
function frame_scans(backend, ::Type{T}, scans) where {T}
    u = T[]; v = T[]; vis = SVector{4,Complex{T}}[]; σ = SVector{4,T}[]; tap = T[]
    for s in scans
        d = s.data
        d isa VisibilityData || throw(ArgumentError("the backend likelihood takes visibility scans"))
        append!(u, T.(d.u)); append!(v, T.(d.v)); append!(vis, SVector{4,Complex{T}}.(d.vis))
        append!(σ, [SVector{4,T}(noise(d.σ, k)) for k in eachindex(d.u)])
        one4 = SVector{4,Complex{T}}(1, 1, 1, 1)
        append!(tap, [s.kernel === nothing ? one(T) : T(real(taper(s.kernel, [one4], [d.u[k]], [d.v[k]])[1][1])) for k in eachindex(d.u)])
    end
    dev(x) = (y = KernelAbstractions.allocate(backend, eltype(x), length(x)); copyto!(y, x); y)
    return FrameScans(dev(u), dev(v), dev(vis), dev(σ), dev(tap))
end

"The sky offsets (l toward the east, m toward the north) of sorted pixel `j` from the screen index `perm[j]`."
@inline function _pixel_offsets(perm, j, nα, nβ, psize::T) where {T}
    @inbounds p = perm[j]
    i = (p - 1) % nα + 1
    jj = (p - 1) ÷ nα + 1
    return -(T(i) - T(nα + 1) / 2) * psize, (T(jj) - T(nβ + 1) / 2) * psize     # east is −x
end

# V_k = tap_k Σ_j I_j scale e^{2πi (u_k l_j + v_k m_j)}: one work-item per baseline
@kernel function vis_kernel!(V, @Const(image), c, @Const(perm), nα, nβ, psize, scale, @Const(u), @Const(v), @Const(tap))
    k = @index(Global, Linear)
    T = typeof(psize)
    E = eltype(eltype(image))                    # the image's scalar type: T, or a dual carrying a directional derivative
    acc = zero(SVector{4,Complex{E}})
    @inbounds uk = u[k]; @inbounds vk = v[k]
    for j in 1:size(image, 1)
        l, m = _pixel_offsets(perm, j, nα, nβ, psize)
        s, co = sincos(2 * T(π) * (uk * l + vk * m))
        @inbounds acc += image[j, c] .* (scale * Complex(co, s))
    end
    @inbounds V[k] = acc * tap[k]
end

# seed_j = Σ_k Re[w_k conj(∂V_k/∂I_j)], ∂V_k/∂I_j = tap_k scale e^{iφ}: one work-item per pixel
@kernel function seed_kernel!(seed, c, @Const(w), @Const(perm), nα, nβ, psize, scale, @Const(u), @Const(v), @Const(tap))
    j = @index(Global, Linear)
    T = typeof(psize)
    l, m = _pixel_offsets(perm, j, nα, nβ, psize)
    acc = zero(SVector{4,T})
    for k in 1:length(u)
        @inbounds s, co = sincos(2 * T(π) * (u[k] * l + v[k] * m))
        e = Complex(co, -s)
        @inbounds wk = w[k] .* (scale * tap[k])
        acc += SVector(real(wk[1] * e), real(wk[2] * e), real(wk[3] * e), real(wk[4] * e))
    end
    @inbounds seed[j, c] = acc
end

"""
    frame_chi2_seed!(seed, c, images, fs::FrameScans, cache, Δα, L, D) -> χ²

The χ² of frame `c` of the device images (npix × nframes, sorted order) against its scans, and
the adjoint of the weighted residuals written into column `c` of `seed` (the frame's dual-sweep
seed), on the cache's backend.
"""
function frame_chi2_seed!(seed, c, images, fs::FrameScans, cache::GeodesicCache{T}, Δα, L, D) where {T}
    backend = cache.backend
    nα, nβ = cache.screen_size
    psize = T(Δα * L / D)
    scale = psize^2 / T(Transfer.JY)
    V = similar(fs.vis)
    vis_kernel!(backend, 64)(V, images, c, cache.perm, nα, nβ, psize, scale, fs.u, fs.v, fs.tap; ndrange = length(V))
    KernelAbstractions.synchronize(backend)
    r = map((x, d, σ) -> (x .- d) ./ σ, V, fs.vis, fs.σ)          # elementwise per Stokes parameter (a broadcast `./` of two SVectors is a solve)
    χ = sum(x -> sum(abs2, x), r)
    w = map((x, σ) -> 2 .* x ./ σ, r, fs.σ)
    seed_kernel!(backend, 64)(seed, c, w, cache.perm, nα, nβ, psize, scale, fs.u, fs.v, fs.tap; ndrange = size(images, 1))
    KernelAbstractions.synchronize(backend)
    return T(χ)
end

export FrameScans, frame_scans, frame_chi2_seed!
