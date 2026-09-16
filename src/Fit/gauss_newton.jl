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
    normal_diagonal!(dg, sb, cache, params, L; probes = 8, rng, mask, ...) -> dg

A Hutchinson estimate of the diagonal of the normal matrix JᵀJ: the mean over `probes`
Rademacher vectors z of z ∘ Jᵀ(J z), each probe one tails pass on duals and one adjoint sweep
(the exact diagonal would take one pass per column). The estimate carries the off-diagonal
couplings as noise, so it is taken in absolute value and floored at 1e-6 of its largest entry.
It is the Jacobi preconditioner of the conjugate gradients and the Marquardt diagonal.
"""
function normal_diagonal!(dg, sb::ScanBands, cache::GeodesicCache{T,N}, params, L; probes::Integer = 8, rng = Random.default_rng(), mask = nothing,
                          nmax = -1, slab = 0, cull::Bool = size(params, 2) > 16, batch_frames::Integer = 4) where {T,N}
    backend = cache.backend
    fill!(dg, zero(T))
    z = KernelAbstractions.allocate(backend, T, size(params)); Jz = KernelAbstractions.allocate(backend, T, residual_length(sb)); Az = similar(dg)
    for _ in 1:probes
        copyto!(z, T.(rand(rng, (-1, 1), size(params))))
        mask === nothing || (z .*= mask)
        jvp!(Jz, sb, cache, params, z, L; nmax, slab, cull, batch_frames)
        jtvp!(Az, sb, cache, params, Jz, L; nmax, slab, cull, batch_frames)
        dg .+= z .* Az
    end
    dg .= abs.(dg) ./ T(max(probes, 1))
    floor = T(1e-6) * maximum(dg)
    dg .= max.(dg, floor)
    return dg
end

"""
    polish_timeresolved!(params, bands, cache, L; iterations = 5, cg_iterations = 20, λ = 1e-2, probes = 8, probe_every = 4, free = trues(size(params)), nmax, slab, callback) -> (params, history)

Levenberg–Marquardt steps on the splat parameters against the bands' visibility scans, each step
from preconditioned conjugate gradients on the matrix-free normal equations
(JᵀJ + λ D) p = −Jᵀr, with J·v by the dual tails pass and Jᵀw by the adjoint sweep. D is the
Hutchinson estimate of diag(JᵀJ) from `probes` probes (`normal_diagonal!`; with `probes = 0`
the gradient's magnitude per row, a cruder scale), and the Jacobi preconditioner is its
inverse: the rows (positions in M, logarithms, angles, rates in rad/M) span decades of scale,
and unpreconditioned iterations resolve only the stiffest directions; the estimate is refreshed
every `probe_every` steps. The iterations stop on a non-positive curvature estimate (a null
direction of the over-complete basis in single precision, whose step would be unbounded: the
first preconditioned run of the triband state took such a step, the residual five times its
start, the χ² up) or when the residual has not improved for ten iterations. λ follows the gain ratio
ρ = (actual decrease)/(decrease of the quadratic model): ÷3 when ρ > 0.75, ×2 when ρ < 0.25,
×10 and the step rejected when χ² rises. `params` lives on the backend (its shape is kept: no
hygiene here). Returns the parameters and the χ² after every step; the callback receives
`(it, params, χ, λ, info)` with the step's gain ratio, the model's predicted decrease and the
conjugate gradients' final relative residual.
"""
function polish_timeresolved!(params, bands::AbstractVector{<:BandScans}, cache::GeodesicCache{T,N}, L; iterations::Integer = 5, cg_iterations::Integer = 20, λ::Real = 1e-2, probes::Integer = 8, probe_every::Integer = 4,
                              free = trues(size(params)), nmax = -1, slab = 0, batch_frames::Integer = 4, callback = nothing, rng = Random.default_rng()) where {T,N}
    backend = cache.backend
    sb = ScanBands(bands, cache)
    cull = size(params, 2) > 16
    m = residual_length(sb)
    mask = KernelAbstractions.allocate(backend, T, size(params)); copyto!(mask, T.(free))
    r = KernelAbstractions.allocate(backend, T, m)
    χ = residuals!(r, sb, cache, params, L; nmax, slab, cull, batch_frames)
    history = T[χ]
    g = KernelAbstractions.allocate(backend, T, size(params)); dscale = similar(g); Ad32 = similar(g); Jd = KernelAbstractions.allocate(backend, T, m)
    F = Float64                                                  # the conjugate-gradient recurrences in Float64; the products in T
    damping = F(λ)
    for it in 1:iterations
        jtvp!(g, sb, cache, params, r, L; nmax, slab, cull, batch_frames); g .*= mask         # Jᵀr
        if probes > 0
            (it == 1 || (it - 1) % max(probe_every, 1) == 0) && normal_diagonal!(dscale, sb, cache, params, L; probes, rng, mask, nmax, slab, cull, batch_frames)
        else
            dscale .= max.(abs.(g), eps(T))
        end
        # preconditioned conjugate gradients on (JᵀJ + damping·D) p = −g, matrix-free, M = (D (1 + damping))⁻¹; stopped on a
        # non-positive curvature estimate (a null direction of the over-complete basis in single precision, whose step would be
        # unbounded) or when the residual has not improved on its best for ten iterations
        D64 = F.(dscale); Minv = 1 ./ (D64 .* (1 + damping)); mask64 = F.(mask)
        p = KernelAbstractions.allocate(backend, F, size(params)); fill!(p, zero(F))
        res = -F.(g); z = Minv .* res; d = copy(z); rz = sum(res .* z)
        g2 = sum(abs2, res); rel = one(F); best = rel; since = 0
        Ad = similar(res)
        for k in 1:cg_iterations
            jvp!(Jd, sb, cache, params, T.(d .* mask64), L; nmax, slab, cull, batch_frames)
            jtvp!(Ad32, sb, cache, params, Jd, L; nmax, slab, cull, batch_frames)
            Ad .= (F.(Ad32) .+ damping .* D64 .* d) .* mask64
            dAd = sum(d .* Ad)
            dAd > 0 || break
            α = rz / dAd
            p .+= α .* d
            res .-= α .* Ad
            rel = sqrt(sum(abs2, res) / g2)
            rel <= 1e-6 && break
            if rel < best
                best = rel; since = 0
            else
                since += 1; since >= 10 && break
            end
            z .= Minv .* res
            rz_new = sum(res .* z)
            d .= z .+ (rz_new / rz) .* d
            rz = rz_new
        end
        p32 = T.(p)
        jvp!(Jd, sb, cache, params, p32, L; nmax, slab, cull, batch_frames)               # the quadratic model's decrease: ‖r‖² − ‖r + Jp‖²
        predicted = F(χ) - F(sum(abs2, r .+ Jd))
        trial = params .+ p32
        rt = similar(r)
        χt = residuals!(rt, sb, cache, trial, L; nmax, slab, cull, batch_frames)
        gain = (F(χ) - F(χt)) / max(predicted, eps(F))
        if χt < χ
            copyto!(params, trial); copyto!(r, rt); χ = χt
            damping = gain > 0.75 ? max(damping / 3, 1e-8) : gain < 0.25 ? damping * 2 : damping
        else
            damping *= 10
        end
        push!(history, χ)
        callback === nothing || callback(it, params, χ, damping, (gain = gain, predicted = predicted, cg_residual = rel))
    end
    return params, history
end

"""
    jacobian!(J, sb, cache, params, L; columns = eachindex(params), chunk = Val(8), ...) -> J

The Jacobian of the residual vector with respect to the parameters at the linear indices
`columns`, into the host matrix `J` (residuals × columns): `chunk` columns per tails pass, the
parameters as duals with that many partials, the parcel lists built once per frame chunk. A
problem of a thousand-odd unknowns is small enough for the explicit matrix (the residual count
times the columns in Float32: a few gigabytes of host memory for a campaign), and the exact
damped step it allows resolves every direction where conjugate gradients on the matrix-free
normal equations resolve a few dozen.
"""
function jacobian!(J::AbstractMatrix, sb::ScanBands, cache::GeodesicCache{T,N}, params, L; columns = eachindex(params), chunk::Val{C} = Val(8),
                   nmax = -1, slab = 0, cull::Bool = size(params, 2) > 16, batch_frames::Integer = 4) where {T,N,C}
    backend = cache.backend
    size(J) == (residual_length(sb), length(columns)) || throw(ArgumentError("J must be $(residual_length(sb)) × $(length(columns))"))
    D = ForwardDiff.Dual{GNTag,T,C}
    ph = Array(params)
    npix = npixels(cache)
    groups = collect(Iterators.partition(eachindex(columns), C))
    duals = map(groups) do grp                                  # the dual parameter matrix of each column group, on the backend
        pdh = D.(ph)
        for (c, ci) in enumerate(grp)
            k = columns[ci]
            pdh[k] = D(ph[k], ForwardDiff.Partials(ntuple(i -> i == c ? one(T) : zero(T), Val(C))))
        end
        pd = KernelAbstractions.allocate(backend, D, size(params)); copyto!(pd, pdh)
        pd
    end
    off = 0
    for (bi, b) in enumerate(sb.bands)
        times = sb.frames[bi]
        for fchunk in Iterators.partition(times, max(Int(batch_frames), 1))
            ts = collect(T, fchunk)
            lists = cull ? Splats.ray_lists(cache, params, ts; nmax, slab) : nothing
            tails = KernelAbstractions.allocate(backend, SVector{4,D}, npix, N + 1, length(ts))
            frames = [(sb.fsd[bi][Float64(t)], _nres(sb.fsd[bi][Float64(t)])) for t in ts]
            ntot = sum(last, frames)
            rd = KernelAbstractions.allocate(backend, D, ntot)
            for (gi, grp) in enumerate(groups)
                Splats.polarized_tails!(tails, cache, duals[gi], ts, T(b.ν), T(L); nmax, slab, lists)
                images = Splats.tail_image(tails, T(b.ν))
                o = 0
                for (c, (fs, n)) in enumerate(frames)
                    _frame_residuals!(view(rd, o + 1:o + n), c, images, fs, cache, b.Δα, L, b.D)
                    o += n
                end
                rdh = Array(rd)
                for (c, ci) in enumerate(grp)
                    @inbounds for k in 1:ntot
                        J[off + k, ci] = ForwardDiff.partials(rdh[k], c)
                    end
                end
            end
            off += ntot
        end
    end
    return J
end

"""
    polish_dense!(params, bands, cache, L; iterations = 5, λ = 1e-2, free = trues(size(params)), chunk = Val(8), tries = 6, nmax, slab, callback) -> (params, history, normal)

Levenberg–Marquardt on the explicit Jacobian (`jacobian!`): each iteration forms JᵀJ and Jᵀr
in Float64 on the host and solves (JᵀJ + λ diag(JᵀJ)) p = −Jᵀr by Cholesky, trying up to
`tries` dampings (×10 each rejection) on the same Jacobian before giving up on the step; λ
follows the gain ratio as in `polish_timeresolved!`, the Marquardt diagonal is floored at
`diag_floor` of its largest entry, and the iterations end once λ exceeds `λ_max`. Returns the parameters, the χ² after every
iteration and the last normal matrix (its inverse is the Laplace covariance of the free
parameters). The callback receives `(it, params, χ, λ, info)` with the gain ratio, the model's
predicted decrease, the number of dampings tried and the seconds the Jacobian took.
"""
function polish_dense!(params, bands::AbstractVector{<:BandScans}, cache::GeodesicCache{T,N}, L; iterations::Integer = 5, λ::Real = 1e-2, free = trues(size(params)), chunk::Val{C} = Val(8),
                       tries::Integer = 6, diag_floor::Real = 1e-6, λ_max::Real = 1e6, nmax = -1, slab = 0, batch_frames::Integer = 4, callback = nothing) where {T,N,C}
    backend = cache.backend
    sb = ScanBands(bands, cache)
    cull = size(params, 2) > 16
    columns = findall(vec(collect(free)))
    m = residual_length(sb); n = length(columns)
    J = Matrix{T}(undef, m, n)
    r = KernelAbstractions.allocate(backend, T, m)
    χ = residuals!(r, sb, cache, params, L; nmax, slab, cull, batch_frames)
    history = T[χ]
    damping = Float64(λ)
    A = zeros(n, n)
    for it in 1:iterations
        tj = @elapsed jacobian!(J, sb, cache, params, L; columns, chunk, nmax, slab, cull, batch_frames)
        rh = Float64.(Array(r))
        fill!(A, 0.0); g = zeros(n)
        for rows in Iterators.partition(1:m, 65536)                  # JᵀJ and Jᵀr accumulated in Float64 over row blocks
            Jb = Float64.(view(J, rows, :))
            mul!(A, Jb', Jb, 1.0, 1.0)
            mul!(g, Jb', rh[rows], 1.0, 1.0)
        end
        # the Marquardt diagonal floored at `diag_floor` of its largest entry: a column the residuals barely see (a parcel
        # of no flux, a rate of a parcel at rest) has a diagonal of single-precision noise, and the damped solve would
        # send it anywhere (the first dense run of the triband state: the trial χ² of 1e10 at a damping of 1e4)
        dA = diag(A)
        dg = max.(dA, diag_floor * maximum(dA))
        ph = Array(params)
        gain = NaN; predicted = NaN; tried = 0; accepted = false; pmax = NaN
        for _ in 1:tries
            tried += 1
            p = -(cholesky(Symmetric(A + damping * Diagonal(dg))) \ g)
            predicted = -(2 * dot(g, p) + dot(p, A * p))            # ‖r‖² − ‖r + Jp‖²
            pmax = maximum(abs, p)
            th = copy(ph); th[columns] .+= T.(p)
            trial = KernelAbstractions.allocate(backend, T, size(params)); copyto!(trial, th)
            rt = similar(r)
            χt = residuals!(rt, sb, cache, trial, L; nmax, slab, cull, batch_frames)
            gain = (Float64(χ) - Float64(χt)) / max(predicted, eps())
            if χt < χ
                copyto!(params, trial); copyto!(r, rt); χ = χt; accepted = true
                damping = gain > 0.75 ? max(damping / 3, 1e-8) : gain < 0.25 ? damping * 2 : damping
                break
            else
                damping *= 10
                damping > λ_max && break
            end
        end
        push!(history, χ)
        callback === nothing || callback(it, params, χ, damping, (gain = gain, predicted = predicted, tries = tried, accepted = accepted, jacobian_seconds = tj, max_step = pmax,
                                                                diag_range = (minimum(dA), maximum(dA))))
        damping > λ_max && break                                   # no damping makes a step: the quadratic model has nothing left
    end
    return params, history, A
end

export ScanBands, residuals!, residual_length, jvp!, jtvp!, normal_diagonal!, polish_timeresolved!, jacobian!, polish_dense!
