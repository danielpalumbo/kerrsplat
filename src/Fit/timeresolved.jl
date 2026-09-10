# The time-resolved visibility likelihood: an observation whose scans see different frames of the
# slow-light movie (the plan's Phase 5 "eventual EHT application"; what Sgr A* needs, where the
# source changes within a night). Each scan carries its frame time and its own data over the
# scan's baselines; scans with the same time share one rendered frame. The χ² is the sum over
# scans of the static visibility or closure χ² of that frame, and its gradient on the device is
# the sum over frames of `image_loss_gradient!` (one dual sweep per frame). The synthetic-data
# generator lends a real array's coverage to a model movie (the self-fit to synthetic VLBI data
# of the plan's gate 7, in the data domain).

"""
    ScanData(time, data)

One scan: its frame time `time` (the movie's units, M) and its data over the scan's
baselines, a `VisibilityData` or a `ClosureData`.
"""
struct ScanData{T,D}
    time::T
    data::D
end

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
scan_loss(image, Δα, L, D, s::ScanData{<:Any,<:ClosureData}) = chi2_closures(image, Δα, L, D, s.data)
scan_loss(image, Δα, L, D, s::ScanData{<:Any,<:VisibilityData}) = chi2_visibilities(image, Δα, L, D, s.data)

"Number of χ² terms: two real values per Stokes parameter and baseline for visibilities, one per closure quantity."
ndata(s::ScanData{<:Any,<:VisibilityData}) = 8 * length(s.data.u)
ndata(s::ScanData{<:Any,<:ClosureData}) = length(s.data.phases) + length(s.data.logamps)
ndata(tr::TimeResolved) = sum(ndata, tr.scans; init = 0)

"""
    chi2_timeresolved(params, tr, cache, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing, priors = nothing)

The χ² of the splat movie against a time-resolved observation: every distinct frame time is
rendered once (slow light, `polarized_image!`, the pixels integrated by `binning` if given,
pixel size `Δα` in M, distance `D` in cm) and compared with the scans at that time. Enzyme on
the host differentiates it (the reference of `timeresolved_gradient!`).
"""
function chi2_timeresolved(params::AbstractMatrix{T}, tr::TimeResolved, cache::GeodesicCache{T}, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing, priors = nothing, instrument = nothing) where {T}
    total = nmax >= 0 ? _tr_frames(Vector{WindingState{T}}(undef, npixels(cache)), params, tr, cache, L, Δα, D, ν, nmax, slab, binning, instrument) :
                        _tr_frames(Vector{RadiativeState{T}}(undef, npixels(cache)), params, tr, cache, L, Δα, D, ν, nmax, slab, binning, instrument)
    return total + penalty(params, priors) + _instrument_penalty(instrument)
end

function _tr_frames(out::AbstractVector, params::AbstractMatrix{T}, tr::TimeResolved, cache::GeodesicCache{T}, L, Δα, D, ν, nmax, slab, binning, instrument) where {T}
    total = zero(T)
    for t in frame_times(tr)
        fill!(out, zero(eltype(out)))
        polarized_image!(out, cache, params, t, ν, L; nmax, slab)
        img = pixel_stokes(to_screen(cache, out), ν, binning)
        for s in tr.scans
            s.time == t || continue
            total += scan_loss(img, Δα, L, D, s, instrument)
        end
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
                                instrument = nothing, dinstrument = nothing) where {T,N}
    total = zero(T)
    for t in frame_times(tr)
        scans = [s for s in tr.scans if s.time == t]
        loss(img) = sum(scan_loss(img, Δα, L, D, s, instrument) for s in scans)
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
    parts = [(s.data.obs, s.data.rows, visibilities(image, Δα, L, D, s.data.obs.u[s.data.rows], s.data.obs.v[s.data.rows])) for s in scans if s.data isa ObservedScan]
    f(g, d) = sum(chi2_instrument(model, o, rows, inst, g, d) for (o, rows, model) in parts; init = zero(promote_type(eltype(g), eltype(d))))
    value = f(gains, dterms)
    return value, ForwardDiff.gradient(g -> f(g, dterms), gains), ForwardDiff.gradient(d -> f(gains, d), dterms)
end

"The scaled residuals of one scan against one screen image (real and imaginary parts per Stokes parameter and baseline; wrapped closure phases and log closure amplitudes), whose squared norm is `scan_loss`."
function scan_residuals(image, Δα, L, D, s::ScanData{<:Any,<:VisibilityData})
    model = visibilities(image, Δα, L, D, s.data.u, s.data.v)
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
    model = visibilities(image, Δα, L, D, d.u, d.v)
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
function timeresolved_residuals(params::AbstractMatrix{S}, tr::TimeResolved, cache::GeodesicCache, L, Δα, D, ν; nmax = -1, slab = 0, binning = nothing, instrument = nothing) where {S}
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
    end
    append!(res, _instrument_prior_residuals(instrument, R))
    return res
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
                         rng = Random.default_rng(), nmax = -1, slab = 0, binning = nothing) where {T}
    out = Vector{accumulator_type(T, nmax)}(undef, npixels(cache))
    frames = Dict{T,Matrix{SVector{4,T}}}()
    for t in unique(c.time for c in cov)
        fill!(out, zero(eltype(out)))
        polarized_image!(out, cache, params, t, ν, L; nmax, slab)
        frames[t] = pixel_stokes(to_screen(cache, out), ν, binning)
    end
    function scan(c)
        img = frames[c.time]
        vis = visibilities(img, Δα, L, D, c.u, c.v)
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
            return ScanData(c.time, ClosureData(c.u, c.v, tri[keep_t], phases[keep_t], σ_phase[keep_t], quad[keep_q], logamps[keep_q], σ_logamp[keep_q]))
        else
            return ScanData(c.time, VisibilityData(c.u, c.v, noisy, SVector(σ, σ, σ, σ)))
        end
    end
    return TimeResolved([scan(c) for c in cov])
end

export ScanData, TimeResolved, frame_times, scan_loss, scan_residuals, ndata, chi2_timeresolved, timeresolved_gradient!, timeresolved_residuals, ScanCoverage, coverage, synthetic_scans, ObservedScan, observed_scans, instrument_gradient

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
function observed_scans(obs::Observation{T}, times::AbstractVector) where {T}
    scans = scan_index(obs); nscans = maximum(scans)
    length(times) == nscans || throw(DimensionMismatch("$(length(times)) times for $nscans scans"))
    return TimeResolved([ScanData(T(times[k]), ObservedScan(obs, findall(==(k), scans))) for k in 1:nscans])
end

# the losses of scans that need no instrument ignore it
scan_loss(image, Δα, L, D, s::ScanData, ::Nothing) = scan_loss(image, Δα, L, D, s)
scan_residuals(image, Δα, L, D, s::ScanData, ::Nothing) = scan_residuals(image, Δα, L, D, s)
function scan_loss(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan}, instrument::Tuple)
    o = s.data.obs; rows = s.data.rows
    model = visibilities(image, Δα, L, D, o.u[rows], o.v[rows])
    return chi2_instrument(model, o, rows, instrument[1], instrument[2], instrument[3])
end
function scan_residuals(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan}, instrument::Tuple)
    o = s.data.obs; rows = s.data.rows
    model = visibilities(image, Δα, L, D, o.u[rows], o.v[rows])
    return instrument_residuals(model, o, rows, instrument[1], instrument[2], instrument[3])
end
scan_loss(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan}) = throw(ArgumentError("an ObservedScan needs the instrument: pass instrument = (model, gains, dterms)"))
scan_residuals(image, Δα, L, D, s::ScanData{<:Any,<:ObservedScan}) = throw(ArgumentError("an ObservedScan needs the instrument: pass instrument = (model, gains, dterms)"))

"The instrument's prior penalty, zero without an instrument."
_instrument_penalty(::Nothing) = 0.0
_instrument_penalty(instrument::Tuple) = penalty_instrument(instrument[1], instrument[2], instrument[3])
_instrument_prior_residuals(::Nothing, ::Type{S}) where {S} = S[]
_instrument_prior_residuals(instrument::Tuple, ::Type{S}) where {S} = S.(instrument_prior_residuals(instrument[1], instrument[2], instrument[3]))
