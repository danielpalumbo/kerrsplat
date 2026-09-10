# The time-resolved visibility likelihood: an observation whose scans see different frames of the
# slow-light movie (the plan's Phase 5 "eventual EHT application"; what Sgr A* needs, where the
# source changes within a night). Each scan carries its frame time and its own data over the
# scan's baselines; scans with the same time share one rendered frame. The χ² is the sum over
# scans of the static visibility or closure χ² of that frame, and its gradient on the device is
# the sum over frames of `image_loss_gradient!` (one dual sweep per frame). The synthetic-data
# generator lends a real array's coverage to a model movie (the self-fit to synthetic VLBI data
# of the plan's gate 7, in the data domain).

"""
    ScanData(time, data, kernel = nothing)

One scan: its frame time `time` (the movie's units, M), its data over the scan's baselines
(a `VisibilityData`, a `ClosureData` or an `ObservedScan`), and the scattering kernel its
model visibilities go through (`ScatteringKernel`, or `nothing`).
"""
struct ScanData{T,D,K}
    time::T
    data::D
    kernel::K
end
ScanData(time, data) = ScanData(time, data, nothing)

"""
    TimeResolved(scans)

An observation at one frequency as a vector of [`ScanData`](@ref); scans with equal `time`
share a frame ([`frame_times`](@ref)).
"""
struct TimeResolved{S<:ScanData}
    scans::Vector{S}
end
Base.length(tr::TimeResolved) = length(tr.scans)
"The distinct frame times of the scans, sorted."
frame_times(tr::TimeResolved) = unique(sort([s.time for s in tr.scans]))

"The χ² of one screen image (Stokes matrix in cgs) against one scan."
scan_loss(image, Δα, L, D, s::ScanData{<:Any,<:ClosureData}) = chi2_closures(image, Δα, L, D, s.data; kernel = s.kernel)
scan_loss(image, Δα, L, D, s::ScanData{<:Any,<:VisibilityData}) = chi2_visibilities(image, Δα, L, D, s.data; kernel = s.kernel)

"Number of χ² terms: two real values per Stokes parameter and baseline for visibilities, one per closure quantity."
ndata(s::ScanData{<:Any,<:VisibilityData}) = 8 * length(s.data.u)
ndata(s::ScanData{<:Any,<:ClosureData}) = length(s.data.phases) + length(s.data.logamps)
ndata(tr::TimeResolved) = sum(ndata, tr.scans; init = 0)

"""
    chi2_timeresolved(params, tr, cache, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing, priors = nothing, instrument = nothing, image_prior = nothing)

The χ² of the splat movie against a time-resolved observation: every distinct frame time is
rendered once (slow light, `polarized_image!`, the pixels integrated by `binning` if given,
pixel size `Δα` in M, distance `D` in cm) and compared with the scans at that time, through
the `instrument` for `ObservedScan`s. `image_prior(img)`, if given, returns scaled residuals
of every rendered frame (e.g. a total-flux prior `img -> [(total_flux(img, Δα, L, D) − F)/σ]`),
added to the χ² as their squared norm: the amplitude anchor of an instrument fit, whose gains'
log-amplitudes are otherwise held only by their priors. Enzyme on the host differentiates it
(the reference of `timeresolved_gradient!`).
"""
function chi2_timeresolved(params::AbstractMatrix{T}, tr::TimeResolved, cache::GeodesicCache{T}, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing, priors = nothing, instrument = nothing, image_prior = nothing) where {T}
    total = nmax >= 0 ? _tr_frames(Vector{WindingState{T}}(undef, npixels(cache)), params, tr, cache, L, Δα, D, ν, nmax, slab, binning, instrument, image_prior) :
                        _tr_frames(Vector{RadiativeState{T}}(undef, npixels(cache)), params, tr, cache, L, Δα, D, ν, nmax, slab, binning, instrument, image_prior)
    return total + penalty(params, priors) + _instrument_penalty(instrument)
end

"The squared norm of an image prior's residuals on a frame (zero without one); `image_prior(img)` returns a vector of scaled residuals, e.g. `img -> [(flux(img) − F)/σ]`."
_image_penalty(::Nothing, img) = zero(real(eltype(first(img))))
_image_penalty(image_prior, img) = sum(abs2, image_prior(img); init = zero(real(eltype(first(img)))))
_image_residuals(::Nothing, img, ::Type{S}) where {S} = S[]
_image_residuals(image_prior, img, ::Type{S}) where {S} = S.(image_prior(img))

function _tr_frames(out::AbstractVector, params::AbstractMatrix{T}, tr::TimeResolved, cache::GeodesicCache{T}, L, Δα, D, ν, nmax, slab, binning, instrument, image_prior) where {T}
    total = zero(T)
    for t in frame_times(tr)
        fill!(out, zero(eltype(out)))
        polarized_image!(out, cache, params, t, ν, L; nmax, slab)
        img = pixel_stokes(to_screen(cache, out), ν, binning)
        for s in tr.scans
            s.time == t || continue
            total += scan_loss(img, Δα, L, D, s, instrument)
        end
        total += _image_penalty(image_prior, img)
    end
    return total
end

"""
    timeresolved_gradient!(dparams, params, tr, cache, L, Δα, D, ν; method = :dual, kmax = 1, nmax = -1, slab = 0, binning = nothing, priors = nothing) -> χ²

`chi2_timeresolved` and its gradient with respect to the splat parameters on the backend of
`cache` (stored samples): one `image_loss_gradient!` per distinct frame time, the loss of a
frame being the sum over its scans; the priors' penalty and gradient are added on the host.
`params` and `dparams` live on the backend; `dparams` accumulates.
"""
function timeresolved_gradient!(dparams, params, tr::TimeResolved, cache::GeodesicCache{T,N}, L, Δα, D, ν; method::Symbol = :dual, kmax = 1, nmax = -1, slab = 0, binning = nothing, priors = nothing,
                                instrument = nothing, dinstrument = nothing, image_prior = nothing) where {T,N}
    total = zero(T)
    for t in frame_times(tr)
        scans = [s for s in tr.scans if s.time == t]
        loss(img) = sum(scan_loss(img, Δα, L, D, s, instrument) for s in scans) + _image_penalty(image_prior, img)
        frame = Ref{Any}(nothing)
        total += image_loss_gradient!(dparams, loss, cache, params, t, ν, L; method, kmax, nmax, slab, binning, on_image = img -> (frame[] = img))
        if dinstrument !== nothing
            total_i, gg, gd = instrument_gradient(frame[], Δα, L, D, scans, instrument)
            dinstrument[1] .+= gg; dinstrument[2] .+= gd
        end
    end
    if priors !== nothing
        value, g = _enzyme_valgrad(x -> penalty(x, priors), Array(params))
        gd = similar(dparams); copyto!(gd, g)
        dparams .+= gd
        total += value
    end
    if instrument !== nothing
        total += _instrument_penalty(instrument)
        if dinstrument !== nothing
            inst, gains, dterms = instrument
            dinstrument[1] .+= ForwardDiff.gradient(g -> penalty_instrument(inst, g, dterms), gains)
            dinstrument[2] .+= ForwardDiff.gradient(d -> penalty_instrument(inst, gains, d), dterms)
        end
    end
    return total
end

"""
    instrument_gradient(image, Δα, L, D, scans, instrument) -> (χ², dgains, dterms)

The χ² of the scans of one frame through the instrument and its gradient with respect to the
gains and d-terms (ForwardDiff on the host, the model visibilities of the frame fixed): the
instrument's share of a joint step, next to the sky's share from the dual sweep.
"""
function instrument_gradient(image, Δα, L, D, scans, instrument::Tuple)
    inst, gains, dterms = instrument
    parts = [(s.data.obs, s.data.rows, scan_model(image, Δα, L, D, s)) for s in scans if s.data isa ObservedScan]
    f(g, d) = sum(chi2_instrument(model, o, rows, inst, g, d) for (o, rows, model) in parts; init = zero(promote_type(eltype(g), eltype(d))))
    value = f(gains, dterms)
    return value, ForwardDiff.gradient(g -> f(g, dterms), gains), ForwardDiff.gradient(d -> f(gains, d), dterms)
end

"The scaled residuals of one scan against one screen image (real and imaginary parts per Stokes parameter and baseline; wrapped closure phases and log closure amplitudes), whose squared norm is `scan_loss`."
function scan_residuals(image, Δα, L, D, s::ScanData{<:Any,<:VisibilityData})
    model = taper(s.kernel, visibilities(image, Δα, L, D, s.data.u, s.data.v), s.data.u, s.data.v)
    S = real(eltype(first(model)))
    res = S[]
    for k in eachindex(model)
        r = (model[k] .- s.data.vis[k]) ./ noise(s.data.σ, k)
        append!(res, real.(r)); append!(res, imag.(r))
    end
    return res
end
function scan_residuals(image, Δα, L, D, s::ScanData{<:Any,<:ClosureData})
    d = s.data
    model = taper(s.kernel, visibilities(image, Δα, L, D, d.u, d.v), d.u, d.v)
    S = real(eltype(first(model)))
    res = S[]
    if !isempty(d.triangles)
        cp = closure_phases(model, d.triangles)
        for k in eachindex(cp)
            push!(res, rem(cp[k] - d.phases[k], 2 * oftype(cp[k], π), RoundNearest) / d.σ_phase[k])
        end
    end
    if !isempty(d.quadrangles)
        la = log_closure_amplitudes(model, d.quadrangles)
        for k in eachindex(la)
            push!(res, (la[k] - d.logamps[k]) / d.σ_logamp[k])
        end
    end
    return res
end

"""
    timeresolved_residuals(params, tr, cache, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing) -> Vector

The scaled residuals of the splat movie against a time-resolved observation, scan by scan
(`scan_residuals`), whose squared norm is `chi2_timeresolved` without the priors; generic in the
element type of `params`, so `polish!(params, q -> timeresolved_residuals(q, ...))` runs the
Levenberg–Marquardt polish and gives the Laplace covariance for VLBI data.
"""
function timeresolved_residuals(params::AbstractMatrix{S}, tr::TimeResolved, cache::GeodesicCache, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing, instrument = nothing, image_prior = nothing) where {S}
    out = Vector{accumulator_type(S, nmax)}(undef, npixels(cache))
    R = instrument === nothing ? S : promote_type(S, eltype(instrument[2]), eltype(instrument[3]))
    res = R[]
    for t in frame_times(tr)
        fill!(out, zero(eltype(out)))
        polarized_image!(out, cache, params, S(t), S(ν), S(L); nmax, slab)
        img = pixel_stokes(to_screen(cache, out), S(ν), binning)
        for s in tr.scans
            s.time == t || continue
            append!(res, scan_residuals(img, Δα, L, D, s, instrument))
        end
        append!(res, _image_residuals(image_prior, img, R))
    end
    append!(res, _instrument_prior_residuals(instrument, R))
    return res
end

"""
    closure_scans(obs::Observation, times; snr = 3) -> TimeResolved

The closure phases and log closure amplitudes of a real observation, scan by scan, as
`ScanData(times[k], ClosureData)` over the rows of scan k (`scan_index`; the triangles and
quadrangles of `scan_triangles`/`scan_quadrangles` re-indexed into the scan's rows), the
uncertainties propagated from the Stokes I noise as the real-data path does and closure
quantities with a leg below `snr` dropped: the gain-free stage of a time-resolved fit
(`observed_scans` gives the products through the instrument for the self-calibrated stage).
"""
function closure_scans(obs::Observation{T}, times::AbstractVector; snr = 3, kernel = nothing) where {T}
    scans = scan_index(obs); nscans = maximum(scans)
    length(times) == nscans || throw(DimensionMismatch("$(length(times)) times for $nscans scans"))
    tri = scan_triangles(obs); quad = scan_quadrangles(obs)
    phases = closure_phases(obs.vis, tri); logamps = log_closure_amplitudes(obs.vis, quad)
    relnoise(k) = obs.σ[abs(k)][1] / abs(obs.vis[abs(k)][1])
    σ_phase = [sqrt(sum(relnoise(k)^2 for k in t)) for t in tri]
    σ_logamp = [sqrt(sum(relnoise(k)^2 for k in q)) for q in quad]
    function scan(k)
        rows = findall(==(k), scans)
        local_index = Dict(r => i for (i, r) in enumerate(rows))
        remap(idx) = sign(idx) * local_index[abs(idx)]
        kt = [i for i in eachindex(tri) if scans[abs(tri[i][1])] == k && σ_phase[i] < 1 / snr]
        kq = [i for i in eachindex(quad) if scans[abs(quad[i][1])] == k && σ_logamp[i] < 1 / snr]
        data = ClosureData(obs.u[rows], obs.v[rows], NTuple{3,Int}[remap.(tri[i]) for i in kt], phases[kt], σ_phase[kt], NTuple{4,Int}[remap.(quad[i]) for i in kq], logamps[kq], σ_logamp[kq])
        return ScanData(T(times[k]), data, kernel)
    end
    return TimeResolved([scan(k) for k in 1:nscans])                     # one concrete element type for all scans
end

"Total flux density (Jy) of a screen image of Stokes vectors (cgs) with pixel side `Δα` (M), length unit `L` and distance `D` (cm): the zero-spacing visibility."
total_flux(img, Δα, L, D) = real(visibilities(img, Δα, L, D, [zero(Δα)], [zero(Δα)])[1][1])

"""
    scan_times(obs::Observation, M_solar; t_ref = nothing) -> Vector

The frame time (in units of GM/c³) of every scan of an observation (`scan_index`), from the
scans' mean UT hours: `(t − t_ref) × 3600 s / (GM/c³)`, with `t_ref` the first scan's mean time
unless given (hours). GM/c³ = `gravitational_radius(M_solar)` / c: 20.5 s for Sgr A*
(4.15 × 10⁶ M⊙), so a night of 8 hours spans 1400 M; 9 hours for M87. The input of
`observed_scans(obs, scan_times(obs, M))` for real time-resolved data.
"""
function scan_times(obs::Observation{T}, M_solar; t_ref = nothing) where {T}
    scans = scan_index(obs); nscans = maximum(scans)
    means = [sum(obs.time[scans .== k]) / count(==(k), scans) for k in 1:nscans]
    t0 = t_ref === nothing ? minimum(means) : T(t_ref)
    tM = gravitational_radius(M_solar) / Transfer.CL                          # seconds per M
    return [(m - t0) * 3600 / tM for m in means]
end

"""
    ScanCoverage(time, u, v, s1, s2)

The baselines of one scan of an array (`u`, `v` in wavelengths, station indices `s1`, `s2`)
and the frame time `time` (M) that a synthetic source is given for it; see [`coverage`](@ref)
and [`synthetic_scans`](@ref).
"""
struct ScanCoverage{T}
    time::T
    u::Vector{T}
    v::Vector{T}
    s1::Vector{Int}
    s2::Vector{Int}
end

"""
    coverage(obs::Observation, times; uvmin = 0) -> Vector{ScanCoverage}

A real observation's coverage, scan by scan (`scan_index`), with the frame time `times[k]`
assigned to scan k: the array's (u, v) sampling lent to a synthetic movie whose time axis is
chosen freely. `uvmin` (wavelengths) drops shorter baselines.
"""
function coverage(obs::Observation{T}, times::AbstractVector; uvmin = 0.0) where {T}
    scans = scan_index(obs); nscans = maximum(scans)
    length(times) == nscans || throw(DimensionMismatch("$(length(times)) times for $nscans scans"))
    out = ScanCoverage{T}[]
    for k in 1:nscans
        rows = findall(r -> scans[r] == k && hypot(obs.u[r], obs.v[r]) >= uvmin, eachindex(scans))
        isempty(rows) && continue
        push!(out, ScanCoverage(T(times[k]), obs.u[rows], obs.v[rows], obs.s1[rows], obs.s2[rows]))
    end
    return out
end

"""
    synthetic_scans(cache, params, L, Δα, D, ν, cov; noise = 0.01, closures = true, rng = Random.default_rng(), nmax = -1, slab = 0, binning = nothing) -> TimeResolved

Synthetic data of a splat movie on the coverage `cov` (a vector of [`ScanCoverage`](@ref)):
the frame at each scan's time is rendered, its model visibilities on the scan's baselines get
complex Gaussian noise of standard deviation `noise` × the frame's total flux density (per
real and imaginary part, every Stokes parameter and baseline), and each scan becomes a
`VisibilityData`, or with `closures = true` a `ClosureData` with the closure phases of all
triangles and the log closure amplitudes of all quadrangles of the scan, their uncertainties
propagated from the visibility noise as the real-data path does (closures with a leg below 3σ
dropped).
"""
function synthetic_scans(cache::GeodesicCache{T}, params, L, Δα, D, ν, cov::AbstractVector{<:ScanCoverage}; noise = 0.01, closures::Bool = true,
                         rng = Random.default_rng(), nmax = -1, slab = 0, binning = nothing, kernel = nothing) where {T}
    out = Vector{accumulator_type(T, nmax)}(undef, npixels(cache))
    frames = Dict{T,Matrix{SVector{4,T}}}()
    for t in unique(c.time for c in cov)
        fill!(out, zero(eltype(out)))
        polarized_image!(out, cache, params, t, ν, L; nmax, slab)
        frames[t] = pixel_stokes(to_screen(cache, out), ν, binning)
    end
    function scan(c)
        img = frames[c.time]
        vis = taper(kernel, visibilities(img, Δα, L, D, c.u, c.v), c.u, c.v)
        flux = real(visibilities(img, Δα, L, D, [zero(T)], [zero(T)])[1][1])
        σ = T(noise) * flux
        noisy = [vis[k] .+ σ .* SVector{4}(complex.(randn(rng, 4), randn(rng, 4))) for k in eachindex(vis)]
        if closures
            n = length(c.u)
            o = Observation{T}(zeros(T, n), zeros(T, n), c.s1, c.s2, String[], c.u, c.v, noisy, fill(SVector(σ, σ, σ, σ), n), T(ν), zero(T), zero(T), zero(T), 0, "synthetic")
            tri = scan_triangles(o); quad = scan_quadrangles(o)
            phases = closure_phases(noisy, tri)
            σ_phase = [sqrt(sum((σ / abs(noisy[abs(k)][1]))^2 for k in t)) for t in tri]
            logamps = log_closure_amplitudes(noisy, quad)
            σ_logamp = [sqrt(sum((σ / abs(noisy[abs(k)][1]))^2 for k in q)) for q in quad]
            keep_t = σ_phase .< 1; keep_q = σ_logamp .< 1
            return ScanData(c.time, ClosureData(c.u, c.v, tri[keep_t], phases[keep_t], σ_phase[keep_t], quad[keep_q], logamps[keep_q], σ_logamp[keep_q]), kernel)
        else
            return ScanData(c.time, VisibilityData(c.u, c.v, noisy, SVector(σ, σ, σ, σ)), kernel)
        end
    end
    return TimeResolved([scan(c) for c in cov])
end

export ScanData, TimeResolved, frame_times, scan_loss, scan_residuals, ndata, chi2_timeresolved, timeresolved_gradient!, timeresolved_residuals, ScanCoverage, coverage, synthetic_scans, ObservedScan, observed_scans, instrument_gradient, total_flux, scan_times, closure_scans

# ---- scans compared through the instrument model (self-calibration) --------------------------------
"""
    ObservedScan(obs, rows)

The rows of an observation that form one scan, compared with the model through the instrument
(`InstrumentModel`, gains and d-terms): the data of a `ScanData` for self-calibration. The
instrument and its parameters are shared by all scans and passed alongside
(`instrument = (model, gains, dterms)`).
"""
struct ObservedScan{O<:Observation}
    obs::O
    rows::Vector{Int}
end
ndata(s::ScanData{<:Any,<:ObservedScan}) = sum(count(isfinite, s.data.obs.σ_coh[r]) for r in s.data.rows; init = 0) * 2

"""
    observed_scans(obs::Observation, times; segmentation = ScanSeg()) -> TimeResolved

The scans of an observation (`scan_index`) as `ObservedScan`s with the frame time `times[k]`
(M) of scan k, for the time-resolved self-calibrated likelihood.
"""
function observed_scans(obs::Observation{T}, times::AbstractVector; kernel = nothing) where {T}
    scans = scan_index(obs); nscans = maximum(scans)
    length(times) == nscans || throw(DimensionMismatch("$(length(times)) times for $nscans scans"))
    return TimeResolved([ScanData(T(times[k]), ObservedScan(obs, findall(==(k), scans)), kernel) for k in 1:nscans])
end

"The model visibilities of an `ObservedScan` from one screen image, through the scan's scattering kernel."
function scan_model(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan})
    o = s.data.obs; rows = s.data.rows
    return taper(s.kernel, visibilities(image, Δα, L, D, o.u[rows], o.v[rows]), o.u[rows], o.v[rows])
end

# the losses of scans that need no instrument ignore it
scan_loss(image, Δα, L, D, s::ScanData, ::Nothing) = scan_loss(image, Δα, L, D, s)
scan_residuals(image, Δα, L, D, s::ScanData, ::Nothing) = scan_residuals(image, Δα, L, D, s)
function scan_loss(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan}, instrument::Tuple)
    return chi2_instrument(scan_model(image, Δα, L, D, s), s.data.obs, s.data.rows, instrument[1], instrument[2], instrument[3])
end
function scan_residuals(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan}, instrument::Tuple)
    return instrument_residuals(scan_model(image, Δα, L, D, s), s.data.obs, s.data.rows, instrument[1], instrument[2], instrument[3])
end
scan_loss(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan}) = throw(ArgumentError("an ObservedScan needs the instrument: pass instrument = (model, gains, dterms)"))
scan_residuals(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan}) = throw(ArgumentError("an ObservedScan needs the instrument: pass instrument = (model, gains, dterms)"))

"The instrument's prior penalty, zero without an instrument."
_instrument_penalty(::Nothing) = 0.0
_instrument_penalty(instrument::Tuple) = penalty_instrument(instrument[1], instrument[2], instrument[3])
_instrument_prior_residuals(::Nothing, ::Type{S}) where {S} = S[]
_instrument_prior_residuals(instrument::Tuple, ::Type{S}) where {S} = S.(instrument_prior_residuals(instrument[1], instrument[2], instrument[3]))

"""
    selfcal!(sky, gains, dterms, tr, cache, L, Δα, D, ν; inst, masks, free = trues(size(sky)), iterations = 300, η = 0.005, η_end = η / 10,
             η_inst = 0.01, steps = nothing, nmax = -1, slab = 0, binning = nothing, priors = nothing, method = :dual, callback = nothing)
        -> (sky, gains, dterms, history)

Joint self-calibration by Adam (Comrade's joint sky-and-instrument posterior, here its mode):
every iteration takes the χ² of the time-resolved observation through the instrument and its
gradients, the sky's from the dual sweep on the backend of `cache` and the instrument's on the
host (`timeresolved_gradient!` with `dinstrument`), and updates the free sky entries with the
cosine-decayed step `η` and the free instrument entries (`masks = free_mask(inst, obs)`) with
`η_inst`; `steps` scales the sky's step row by row (`step_scale`). The sky matrix stays on the
host. `history` holds the χ² (with the priors) at the start of every iteration.
"""
function selfcal!(sky::AbstractMatrix{T}, gains::AbstractMatrix{T}, dterms::AbstractMatrix{T}, tr::TimeResolved, cache::GeodesicCache{T}, L, Δα, D, ν;
                  inst::InstrumentModel, masks::Tuple, free = trues(size(sky)), iterations::Integer = 300, η = 0.005, η_end = η / 10, η_inst = 0.01,
                  steps = nothing, nmax = -1, slab = 0, binning = nothing, priors = nothing, method::Symbol = :dual, callback = nothing, image_prior = nothing) where {T}
    gm, dm = masks
    backend = cache.backend
    scale = steps === nothing ? nothing : step_scale(sky, steps)
    opt = Optimisers.setup(Optimisers.Adam(η), sky)
    optg = Optimisers.setup(Optimisers.Adam(η_inst), gains)
    optd = Optimisers.setup(Optimisers.Adam(η_inst), dterms)
    history = T[]
    for it in 1:iterations
        ηt = η_end + (η - η_end) * (1 + cos(π * (it - 1) / max(iterations - 1, 1))) / 2
        Optimisers.adjust!(opt, ηt)
        pdev = KernelAbstractions.allocate(backend, T, size(sky)); copyto!(pdev, sky)
        gdev = KernelAbstractions.allocate(backend, T, size(sky)); fill!(gdev, zero(T))
        dg = zeros(T, size(gains)); dd = zeros(T, size(dterms))
        χ = timeresolved_gradient!(gdev, pdev, tr, cache, L, Δα, D, ν; method, nmax, slab, binning, priors, instrument = (inst, gains, dterms), dinstrument = (dg, dd), image_prior)
        push!(history, χ)
        old = scale === nothing ? nothing : copy(sky)
        opt, sky = Optimisers.update!(opt, sky, Array(gdev) .* T.(free))
        Fit._scaled_update!(sky, old, scale)
        optg, gains = Optimisers.update!(optg, gains, dg .* T.(gm))
        optd, dterms = Optimisers.update!(optd, dterms, dd .* T.(dm))
        callback === nothing || callback(it, sky, gains, dterms, χ)
    end
    return sky, gains, dterms, history
end

export selfcal!

"""
    scan_models(params, tr, cache, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing) -> Vector

The model visibilities (a Stokes `SVector{4}` per row) of every `ObservedScan` of `tr` from
the splat movie, one render per frame (`nothing` for scans of other kinds): the sky's side of
the products, held while the instrument alone is solved (`calibrate!`).
"""
function scan_models(params::AbstractMatrix{S}, tr::TimeResolved, cache::GeodesicCache, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing) where {S}
    out = Vector{accumulator_type(S, nmax)}(undef, npixels(cache))
    models = Vector{Any}(nothing, length(tr.scans))
    for t in frame_times(tr)
        fill!(out, zero(eltype(out)))
        polarized_image!(out, cache, params, S(t), S(ν), S(L); nmax, slab)
        img = pixel_stokes(to_screen(cache, out), S(ν), binning)
        for (k, s) in enumerate(tr.scans)
            (s.time == t && s.data isa ObservedScan) || continue
            models[k] = scan_model(img, Δα, L, D, s)
        end
    end
    return models
end

"""
    reference_phases!(gains, tr, models, inst; dterms = zero d-terms) -> gains

Starting phases for an instrument solve, scan by scan: the gain phases of the segment's
stations chained from its reference station through the parallel-hand products, each
station's R phase from the phase of the observed over the predicted RR on a baseline to a
station already set (`gp = ±arg(RR_obs / RR_pred)`, the sign by which end it is) and its
L/R phase ratio from LL the same way (LL alone sets the R phase when RR is missing). The
amplitudes and the d-terms stay as they are and enter the prediction. Phase-only
self-calibration from unit gains is multimodal (a wrong phase is best "fitted" by a vanishing
amplitude), so a solve from zero phases strands; this is the closed-form start every
self-calibration uses, and `calibrate!` applies it by default.
"""
function reference_phases!(gains::AbstractMatrix{T}, tr::TimeResolved, models, inst::InstrumentModel; dterms = zero_instrument(inst)[2]) where {T}
    for (k, s) in enumerate(tr.scans)
        s.data isa ObservedScan || continue
        o = s.data.obs; rows = s.data.rows; model = models[k]
        for g in unique(inst.seg[rows])
            done = falses(nstations(inst)); done[inst.ref[g]] = true
            for st in 1:nstations(inst)
                c = gain_column(inst, st, g)
                gains[2, c] = zero(T); inst.polarized && (gains[4, c] = zero(T))
            end
            changed = true
            while changed
                changed = false
                for (j, r) in enumerate(rows)
                    inst.seg[r] == g || continue
                    s1 = o.s1[r]; s2 = o.s2[r]
                    (done[s1] ⊻ done[s2]) || continue
                    J1 = station_jones(inst, gains, dterms, s1, g, inst.φ1[r]); J2 = station_jones(inst, gains, dterms, s2, g, inst.φ2[r])
                    V = products(apply_jones(coherency(model[j]), J1, J2))
                    unknown = done[s1] ? s2 : s1; sign = done[s1] ? -one(T) : one(T)
                    c = gain_column(inst, unknown, g)
                    δR = isfinite(o.σ_coh[r][1]) && abs(V[1]) > 0 ? T(angle(o.coh[r][1] / V[1])) : T(NaN)
                    δL = isfinite(o.σ_coh[r][2]) && abs(V[2]) > 0 ? T(angle(o.coh[r][2] / V[2])) : T(NaN)
                    (isfinite(δR) || isfinite(δL)) || continue
                    if isfinite(δR)
                        gains[2, c] = sign * δR
                        inst.polarized && isfinite(δL) && (gains[4, c] = sign * (δL - δR))
                    else
                        gains[2, c] = sign * δL
                    end
                    done[unknown] = true; changed = true
                end
            end
        end
    end
    return gains
end

"""
    calibrate!(gains, dterms, tr, models, inst; masks, iterations = 10, chunk = 12, λ = 1e-3, phases = true, phase_first = true)
        -> (gains, dterms, history, covariance)

The instrument alone: Levenberg–Marquardt over the free gain and d-term entries (`masks`,
`free_mask`) with the sky's model visibilities `models` (`scan_models`) held. The residuals
are the products' through the Jones chain (`instrument_residuals`) scan by scan and the
instrument's priors: the same residuals as the joint polish with the sky frozen, without a
render per Jacobian column. This is the self-calibration step of every imaging pipeline
(Comrade's instrument-only solve, ehtim's `self_cal`) between sky updates. `phases` starts
the gain phases from the reference baselines (`reference_phases!`) and `phase_first` solves
the phases alone (amplitudes and d-terms held) for `iterations` before the full solve, both
against the multimodality of the phase problem. Returns the matrices updated in place, the
χ² history (both solves, in order) and the Laplace covariance of the packed entries of the
full solve.
"""
function calibrate!(gains::AbstractMatrix{T}, dterms::AbstractMatrix{T}, tr::TimeResolved, models, inst::InstrumentModel; masks::Tuple,
                    iterations::Integer = 10, chunk::Integer = 12, λ::Real = 1e-3, phases::Bool = true, phase_first::Bool = true) where {T}
    gm, dm = masks
    nosky = zeros(T, 0, 0); nofree = falses(0, 0)
    function solve(gmask, dmask)
        function residuals(x::AbstractVector{S}) where {S}
            _, g, d = unpack(nosky, nofree, gains, gmask, dterms, dmask, x)
            res = S[]
            for (k, s) in enumerate(tr.scans)
                s.data isa ObservedScan || continue
                append!(res, instrument_residuals(models[k], s.data.obs, s.data.rows, inst, g, d))
            end
            append!(res, S.(instrument_prior_residuals(inst, g, d)))
            return res
        end
        xs0 = pack(nosky, nofree, gains, gmask, dterms, dmask)
        xs, hs, cs = levenberg_marquardt!(xs0, residuals; iterations, chunk, λ)
        unpack!(nosky, nofree, gains, gmask, dterms, dmask, xs)
        return hs, cs
    end
    # (the solver's locals are named apart from this scope's: a closure's assignment to a name of the enclosing
    # function assigns the enclosing variable)
    phases && reference_phases!(gains, tr, models, inst; dterms)
    history = T[]
    if phase_first
        pm = copy(gm); pm[1, :] .= false; inst.polarized && (pm[3, :] .= false)
        h1, _ = solve(pm, falses(size(dterms)))
        append!(history, h1)
    end
    h2, covariance = solve(gm, dm)
    append!(history, h2)
    return gains, dterms, history, covariance
end

export scan_models, scan_model, calibrate!, reference_phases!
