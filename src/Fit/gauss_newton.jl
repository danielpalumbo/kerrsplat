# A matrix-free Gauss–Newton (Levenberg–Marquardt) polish of the splat parameters against time-resolved visibility
# scans, on the backend: the Jacobian J of the residual vector r(q) (every scan's weighted real and imaginary
# residuals of the four Stokes visibilities) is never formed. J·v comes from one tails pass with the parameters as
# one-partial duals seeded along v (the transport of `polarized_tails!` runs on duals) followed by the frame's
# device transform on the dual image; Jᵀw comes from the adjoint sweep seeded by the adjoint transform of w
# (`seed_kernel!` with weights w/σ); conjugate gradients on (JᵀJ + λ diag) p = −Jᵀr give the step, and λ moves with
# the outcome of each step as in Levenberg–Marquardt. Adam on the over-complete basis stalls a decade above the
# floor of a campaign's χ² (docs/notes/2026-09-14_ngeht_triband.md); the quadratic model is the tool for that last
# decade.

struct GNTag end

"""
    ScanBands(bands, cache, L; nmax, slab, cull) -> the residual machinery of a set of `BandScans` on the cache's backend

Prepares, once, the per-frame device scans (`frame_scans`) and the sweep passes of every band.
"""
struct ScanBands{B,F,P}
    bands::B                     # the BandScans
    frames::Vector{F}            # per band, the frame times
    fsd::Vector{Dict{Float64,P}} # per band, the FrameScans of every frame time
end

function ScanBands(bands::AbstractVector{<:BandScans}, cache::GeodesicCache{T}) where {T}
    frames = [frame_times(b.tr) for b in bands]
    fsd = [Dict(Float64(t) => frame_scans(cache.backend, T, [s for s in b.tr.scans if s.time == t]) for t in frame_times(b.tr)) for b in bands]
    return ScanBands(bands, frames, fsd)
end

"The number of residual entries of a frame's scans (real and imaginary parts of the four Stokes visibilities)."
_nres(fs::FrameScans) = 8 * length(fs.u)

"""
    residuals!(r, sb::ScanBands, cache, params, L; nmax, slab, cull, batch_frames) -> χ²

The weighted residuals of every band, frame and scan into the device vector `r` (as laid out by
`residual_layout`), from the parameters on the backend; returns the χ² = ‖r‖².
"""
function residuals!(r, sb::ScanBands, cache::GeodesicCache{T,N}, params, L; nmax = -1, slab = 0, cull::Bool = size(params, 2) > 16, batch_frames::Integer = 4) where {T,N}
    backend = cache.backend
    dummy = KernelAbstractions.allocate(backend, T, size(params))
    forward, _ = sweep_passes(dummy, cache, params, L; method = :dual, nmax, slab, cull)
    off = 0
    for (bi, b) in enumerate(sb.bands)
        times = sb.frames[bi]
        for chunk in Iterators.partition(times, max(Int(batch_frames), 1))
            ts = collect(chunk)
            images = forward(ts, b.ν; device = true)
            for (c, t) in enumerate(ts)
                fs = sb.fsd[bi][Float64(t)]
                n = _nres(fs)
                _frame_residuals!(view(r, off + 1:off + n), c, images, fs, cache, b.Δα, L, b.D)
                off += n
            end
        end
    end
    return sum(abs2, r)
end

"The residual vector's length over all bands and frames."
residual_length(sb::ScanBands) = sum(sum(_nres(fs) for fs in values(d)) for d in sb.fsd)

# the frame's weighted residuals (Re, Im of the four Stokes parameters per baseline) into a device view
function _frame_residuals!(rv, c, images, fs::FrameScans, cache::GeodesicCache{T}, Δα, L, D) where {T}
    backend = cache.backend
    nα, nβ = cache.screen_size
    psize = T(Δα * L / D)
    scale = psize^2 / T(Transfer.JY)
    E = eltype(eltype(images))                                   # T, or a dual for J·v
    V = KernelAbstractions.allocate(backend, SVector{4,Complex{E}}, length(fs.u))
    vis_kernel!(backend, 64)(V, images, c, cache.perm, nα, nβ, psize, scale, fs.u, fs.v, fs.tap; ndrange = length(V))
    KernelAbstractions.synchronize(backend)
    res = map((x, d, σ) -> (x .- d) ./ σ, V, fs.vis, fs.σ)     # SVector{4,Complex{E}} per baseline
    flat = map(x -> SVector(real(x[1]), imag(x[1]), real(x[2]), imag(x[2]), real(x[3]), imag(x[3]), real(x[4]), imag(x[4])), res)
    copyto!(rv, reinterpret(E, flat))
    return rv
end

"""
    jvp!(out, sb, cache, params, v, L; ...) -> out

J·v in residual space: the parameters as one-partial duals seeded along `v` through the tails
pass and the frame transforms, the partials of the residuals into `out`.
"""
function jvp!(out, sb::ScanBands, cache::GeodesicCache{T,N}, params, v, L; nmax = -1, slab = 0, cull::Bool = size(params, 2) > 16, batch_frames::Integer = 4) where {T,N}
    backend = cache.backend
    D = ForwardDiff.Dual{GNTag,T,1}
    ph = Array(params); vh = Array(v)
    pd = KernelAbstractions.allocate(backend, D, size(params))
    copyto!(pd, [ForwardDiff.Dual{GNTag}(ph[i, j], vh[i, j]) for i in axes(ph, 1), j in axes(ph, 2)])
    npix = npixels(cache)
    off = 0
    for (bi, b) in enumerate(sb.bands)
        times = sb.frames[bi]
        for chunk in Iterators.partition(times, max(Int(batch_frames), 1))
            ts = collect(T, chunk)
            lists = cull ? Splats.ray_lists(cache, params, ts; nmax, slab) : nothing          # the lists from the values
            tails = KernelAbstractions.allocate(backend, SVector{4,D}, npix, N + 1, length(ts))
            Splats.polarized_tails!(tails, cache, pd, ts, T(b.ν), T(L); nmax, slab, lists)
            images = Splats.tail_image(tails, T(b.ν))                                        # npix × nf of SVector{4,D}
            for (c, t) in enumerate(ts)
                fs = sb.fsd[bi][Float64(t)]
                n = _nres(fs)
                rd = KernelAbstractions.allocate(backend, D, n)
                _frame_residuals!(rd, c, images, fs, cache, b.Δα, L, b.D)
                copyto!(view(out, off + 1:off + n), map(x -> ForwardDiff.partials(x, 1), rd))
                off += n
            end
        end
    end
    return out
end

"""
    jtvp!(out, sb, cache, params, w, L; ...) -> out

Jᵀw in parameter space: every frame's dual sweep seeded by the adjoint transform of its slice of
`w` (residual space), accumulated into `out` (a device matrix of the parameters' shape).
"""
function jtvp!(out, sb::ScanBands, cache::GeodesicCache{T,N}, params, w, L; nmax = -1, slab = 0, cull::Bool = size(params, 2) > 16, batch_frames::Integer = 4) where {T,N}
    backend = cache.backend
    fill!(out, zero(T))
    forward, reverse! = sweep_passes(out, cache, params, L; method = :dual, nmax, slab, cull)
    npix = npixels(cache)
    nα, nβ = cache.screen_size
    off = 0
    for (bi, b) in enumerate(sb.bands)
        times = sb.frames[bi]
        psize = T(b.Δα * L / b.D)
        scale = psize^2 / T(Transfer.JY)
        for chunk in Iterators.partition(times, max(Int(batch_frames), 1))
            ts = collect(chunk)
            forward(ts, b.ν; device = true)                          # the tails of the chunk, for the reverse pass
            seeds = KernelAbstractions.allocate(backend, SVector{4,T}, npix, length(ts)); fill!(seeds, zero(SVector{4,T}))
            for (c, t) in enumerate(ts)
                fs = sb.fsd[bi][Float64(t)]
                n = _nres(fs)
                wf = reinterpret(SVector{8,T}, view(w, off + 1:off + n))                         # per baseline: Re, Im of the four Stokes
                wc = map((x, σ) -> SVector(Complex(x[1], x[2]), Complex(x[3], x[4]), Complex(x[5], x[6]), Complex(x[7], x[8])) ./ σ, wf, fs.σ)
                seed_kernel!(backend, 64)(seeds, c, wc, cache.perm, nα, nβ, psize, scale, fs.u, fs.v, fs.tap; ndrange = npix)
                KernelAbstractions.synchronize(backend)
                off += n
            end
            length(ts) > 1 ? reverse!(seeds, ts, b.ν) : reverse!(vec(seeds), ts[1], b.ν)
        end
    end
    return out
end

"""
    polish_timeresolved!(params, bands, cache, L; iterations = 5, cg_iterations = 20, λ = 1e-2, free = trues(size(params)), nmax, slab, callback) -> (params, history)

Levenberg–Marquardt steps on the splat parameters against the bands' visibility scans, each step
from conjugate gradients on the matrix-free normal equations (J·v by the dual tails pass, Jᵀw by
the adjoint sweep); λ divides by three on an accepted step and multiplies by ten on a rejected
one. `params` lives on the backend (its shape is kept: no hygiene here). Returns the parameters
and the χ² after every step.
"""
function polish_timeresolved!(params, bands::AbstractVector{<:BandScans}, cache::GeodesicCache{T,N}, L; iterations::Integer = 5, cg_iterations::Integer = 20, λ::Real = 1e-2,
                              free = trues(size(params)), nmax = -1, slab = 0, batch_frames::Integer = 4, callback = nothing) where {T,N}
    backend = cache.backend
    sb = ScanBands(bands, cache)
    cull = size(params, 2) > 16
    m = residual_length(sb)
    mask = KernelAbstractions.allocate(backend, T, size(params)); copyto!(mask, T.(free))
    r = KernelAbstractions.allocate(backend, T, m)
    χ = residuals!(r, sb, cache, params, L; nmax, slab, cull, batch_frames)
    history = T[χ]
    g = KernelAbstractions.allocate(backend, T, size(params))
    damping = T(λ)
    for it in 1:iterations
        jtvp!(g, sb, cache, params, r, L; nmax, slab, cull, batch_frames); g .*= mask         # Jᵀr
        # the diagonal scale of the normal matrix from a probe: diag(JᵀJ) ≈ |Jᵀ(J e)| is not available cheaply; use the
        # gradient's scale per row as the Marquardt diagonal (a fixed scaling of the parameters)
        dscale = map(x -> max(abs(x), eps(T)), g)
        # conjugate gradients on (JᵀJ + damping·diag) p = −g, matrix-free
        p = KernelAbstractions.allocate(backend, T, size(params)); fill!(p, zero(T))
        res = -g; d = copy(res); rr = sum(abs2, res)
        Jd = KernelAbstractions.allocate(backend, T, m); Ad = similar(g)
        for k in 1:cg_iterations
            jvp!(Jd, sb, cache, params, d .* mask, L; nmax, slab, cull, batch_frames)
            jtvp!(Ad, sb, cache, params, Jd, L; nmax, slab, cull, batch_frames)
            Ad .= (Ad .+ damping .* dscale .* d) .* mask
            α = rr / max(sum(d .* Ad), eps(T))
            p .+= α .* d
            res .-= α .* Ad
            rr_new = sum(abs2, res)
            rr_new <= T(1e-6)^2 * sum(abs2, g) && break
            d .= res .+ (rr_new / rr) .* d
            rr = rr_new
        end
        trial = params .+ p
        rt = similar(r)
        χt = residuals!(rt, sb, cache, trial, L; nmax, slab, cull, batch_frames)
        if χt < χ
            copyto!(params, trial); copyto!(r, rt); χ = χt
            damping = max(damping / 3, T(1e-8))
        else
            damping *= 10
        end
        push!(history, χ)
        callback === nothing || callback(it, params, χ, damping)
    end
    return params, history
end

export ScanBands, residuals!, residual_length, jvp!, jtvp!, polish_timeresolved!
