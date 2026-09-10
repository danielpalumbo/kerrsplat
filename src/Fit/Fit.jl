"""
    KerrSplat.Fit

Likelihood and fitting loop for Stokes movies (Phase 4 of the plan, first pieces): a data
container with per-Stokes noise and a mask, the χ² of the polarized splat model over a minibatch
of frames and frequencies, and an Adam loop on Enzyme reverse-mode gradients with staged
unfreezing through a parameter mask. The forward model runs on the backend of the geodesic cache;
the gradient currently runs on the CPU backend (Enzyme through CUDA kernels is open).
"""
module Fit

using ..Geodesics
using ..Transfer
using ..Splats
using Enzyme
using Optimisers
using StaticArrays
using KernelAbstractions

const KA = KernelAbstractions

export StokesMovie, chi2, fit!, freeze, step_scale

"""
    StokesMovie(data, times, νs, σ; mask = trues(size(data)))

Observed Stokes cube `data` (array of Stokes 4-vectors of size (nα, nβ, nt, nν), cgs intensity),
its observation times (M) and frequencies (Hz), the noise per Stokes parameter `σ` (a Stokes
4-vector, or an array of them matching `data`), and a boolean mask of the pixels that enter χ².
"""
struct StokesMovie{T,A,S,M}
    data::A
    times::Vector{T}
    νs::Vector{T}
    σ::S
    mask::M
end
StokesMovie(data::AbstractArray{SVector{4,T},4}, times, νs, σ; mask = trues(size(data))) where {T} =
    StokesMovie(data, collect(T, times), collect(T, νs), σ, mask)

@inline noise(σ::SVector{4}, k) = σ
@inline noise(σ::AbstractArray, k) = σ[k]

"""
    chi2(params, movie, cache, L; frames = eachindex(movie.times), freqs = eachindex(movie.νs))

Σ ((model − data)/σ)² over the selected frames and frequencies of the movie (a minibatch), with
the polarized splat model rendered through `cache` (already regenerated) and the length unit `L`.
Written as a plain function of `params` so that Enzyme can differentiate it.
"""
function chi2(params, movie::StokesMovie{T}, cache::GeodesicCache{T}, L; frames = eachindex(movie.times), freqs = eachindex(movie.νs), priors = nothing, nmax = -1, slab = 0, binning = nothing) where {T}
    # the accumulator type is chosen in a branch so that each path is type-stable for Enzyme
    total = nmax >= 0 ? _chi2_frames(Vector{WindingState{T}}(undef, npixels(cache)), params, movie, cache, L, frames, freqs, nmax, slab, binning) :
                        _chi2_frames(Vector{RadiativeState{T}}(undef, npixels(cache)), params, movie, cache, L, frames, freqs, nmax, slab, binning)
    return total + penalty(params, priors)
end

"The observed Stokes vectors of an image in screen order, integrated over the pixels of a `Binning` when one is given."
@inline pixel_stokes(img, ν, ::Nothing) = map(st -> observed_stokes(st, ν), img)
@inline pixel_stokes(img, ν, binning::Binning) = bin(binning, map(st -> observed_stokes(st, ν), vec(img)))

function _chi2_frames(out::AbstractVector, params, movie::StokesMovie{T}, cache::GeodesicCache{T}, L, frames, freqs, nmax, slab, binning) where {T}
    nα, nβ = size(movie.data, 1), size(movie.data, 2)
    total = zero(T)
    for l in freqs, k in frames
        ν = movie.νs[l]
        fill!(out, zero(eltype(out)))
        polarized_image!(out, cache, params, movie.times[k], ν, L; nmax, slab)
        stokes = pixel_stokes(to_screen(cache, out), ν, binning)
        for j in 1:nβ, i in 1:nα
            idx = CartesianIndex(i, j, k, l)
            movie.mask[idx] || continue
            r = (stokes[i, j] - movie.data[idx]) ./ noise(movie.σ, idx)
            total += r[1] * r[1] + r[2] * r[2] + r[3] * r[3] + r[4] * r[4]
        end
    end
    return total
end

"Boolean mask over a parameter matrix that frees the given rows (by index or name) of all splats."
function freeze(params::AbstractMatrix, rows)
    free = falses(size(params))
    for r in rows
        i = r isa Symbol ? findfirst(==(r), POLARIZED_SPLAT_PARAMS) : r
        i === nothing && throw(ArgumentError("unknown parameter $r"))
        free[i, :] .= true
    end
    return free
end

"""
    step_scale(params, steps) -> Matrix{T}

Per-entry multipliers of the Adam step: ones everywhere except the rows named in `steps`
(pairs or a named tuple of row name or index => factor, e.g. `(; omega = 0.05)`), which get
their factor. Adam's update is invariant to the gradient's scale, so a row that must move more
slowly than the others (the pattern rates, in rad/M, next to positions in M) needs a smaller
step, not a smaller gradient: the fitting loops apply the scaled step as
`old + scale * (new - old)`, which is Adam with a per-entry learning rate since its state does
not depend on the step.
"""
function step_scale(params::AbstractMatrix{T}, steps) where {T}
    scale = ones(T, size(params))
    steps === nothing && return scale
    for (r, f) in pairs(steps)
        i = r isa Symbol ? findfirst(==(r), POLARIZED_SPLAT_PARAMS) : r
        i === nothing && throw(ArgumentError("unknown parameter $r"))
        scale[i, :] .= T(f)
    end
    return scale
end

"Adam's update `new` of `old`, taken with the per-entry step multipliers `scale` (`nothing`: as is)."
_scaled_update!(new, old, scale) = scale === nothing ? new : (new .= old .+ scale .* (new .- old))

"""
    fit!(params, movie, cache, L; free = trues(size(params)), iterations = 100, η = 0.02,
         batch = nothing, rng = Random.default_rng(), callback = nothing) -> history

Adam on the Enzyme reverse-mode gradient of `chi2`, updating only the parameters where `free` is
true (staged unfreezing: call repeatedly with different masks). `batch = (nframes, nfreqs)`
draws a random minibatch of frames and frequencies at every iteration; otherwise every frame and
frequency enters each step. Returns the χ² history (full data at the start and the end, the
minibatch value in between).
"""
function fit!(params::AbstractMatrix{T}, movie::StokesMovie{T}, cache::GeodesicCache{T}, L; free = trues(size(params)),
              iterations::Integer = 100, η = 0.02, batch = nothing, rng = Random.default_rng(), callback = nothing, binning = nothing) where {T}
    mask = T.(free)
    opt = Optimisers.setup(Optimisers.Adam(η), params)
    history = T[chi2(params, movie, cache, L; binning)]
    for it in 1:iterations
        frames = batch === nothing ? eachindex(movie.times) : sort(randperm(rng, length(movie.times))[1:min(batch[1], length(movie.times))])
        freqs = batch === nothing ? eachindex(movie.νs) : sort(randperm(rng, length(movie.νs))[1:min(batch[2], length(movie.νs))])
        f(q) = chi2(q, movie, cache, L; frames, freqs, binning)
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(f), params)[1] .* mask
        opt, params = Optimisers.update!(opt, params, g)
        push!(history, f(params))
        callback === nothing || callback(it, params, history[end])
    end
    push!(history, chi2(params, movie, cache, L))
    return history
end

using Random: randperm
import Random

# ---- partition hygiene: prune, densify, merge (addendum §5.2.3, 3DGS practice) --------------------
"Row index of a polarized splat parameter by name."
prow(name::Symbol) = findfirst(==(name), POLARIZED_SPLAT_PARAMS)

"""
    prune(params; fraction = 1e-4) -> (params, kept)

Drop splats whose density × volume proxy, e^{logne + s₁ + s₂ + s₃}, is below `fraction` of the
largest. Returns the reduced matrix and the indices kept.
"""
function prune(params::AbstractMatrix; fraction = 1e-4)
    w = [exp(params[13, i] + params[4, i] + params[5, i] + params[6, i]) for i in 1:size(params, 2)]
    kept = findall(>=(fraction * maximum(w)), w)
    return params[:, kept], kept
end

"""
    densify(params, grad; threshold, scale_factor = 1.6) -> params

Split every splat whose position-gradient norm (rows x, y, z of `grad`) exceeds `threshold` into
two: the children sit at ±½ of the largest principal scale along that axis, with all scales
divided by `scale_factor` (the usual Gaussian-splatting split) and the density set so that the
total density × volume of the pair equals the parent's, keeping the other parameters. Returns
the enlarged matrix.
"""
function densify(params::AbstractMatrix{T}, grad::AbstractMatrix; threshold, scale_factor = 1.6) where {T}
    cols = Vector{Vector{T}}()
    for i in 1:size(params, 2)
        g = sqrt(grad[1, i]^2 + grad[2, i]^2 + grad[3, i]^2)
        col = params[:, i]
        if g > threshold
            k = argmax(col[4:6])
            R = Splats.quaternion_rotation(col[7], col[8], col[9], col[10])
            axis = R[:, k] * exp(col[3 + k]) / 2
            for sgn in (-1, 1)
                c = copy(col)
                c[1:3] .+= sgn * axis
                c[4:6] .-= log(scale_factor)
                c[13] += 3 * log(scale_factor) - log(2)
                push!(cols, c)
            end
        else
            push!(cols, col)
        end
    end
    return reduce(hcat, cols)
end

"""
    merge(params; position_tol = 0.2, shape_tol = 0.1) -> (params, groups)

Merge splats whose centres lie within `position_tol` (M) and whose log-scales agree within
`shape_tol`: the merged splat carries the summed density (exact for co-located identical parcels,
since the transfer coefficients add) and the density-weighted mean of the other parameters.
"""
function merge(params::AbstractMatrix{T}; position_tol = 0.2, shape_tol = 0.1) where {T}
    n = size(params, 2)
    assigned = zeros(Int, n)
    groups = Vector{Vector{Int}}()
    for i in 1:n
        assigned[i] == 0 || continue
        g = [i]; assigned[i] = length(groups) + 1
        for j in i+1:n
            assigned[j] == 0 || continue
            dpos = sqrt(sum(abs2, params[1:3, i] .- params[1:3, j]))
            dshape = maximum(abs.(params[4:6, i] .- params[4:6, j]))
            if dpos <= position_tol && dshape <= shape_tol
                push!(g, j); assigned[j] = assigned[i]
            end
        end
        push!(groups, g)
    end
    out = Matrix{T}(undef, size(params, 1), length(groups))
    for (k, g) in enumerate(groups)
        w = exp.(params[13, g]); wsum = sum(w)
        out[:, k] = sum(params[:, g] .* (w ./ wsum)', dims = 2)
        out[13, k] = log(wsum)
    end
    return out, groups
end

export prune, densify, merge

# ---- Fisher audit (addendum §5.3) ---------------------------------------------------------------
using ForwardDiff
using LinearAlgebra

"""
    fisher(params, movie, cache, L; free = trues(size(params)), chunk = 8) -> (F, J, idx)

Fisher information matrix F = Jᵀ Σ⁻¹ J of the polarized splat model for the free parameters
(the Jacobian J of every masked data value with respect to them by ForwardDiff duals through
the whole pipeline), with the noise of the movie. Returns F, J and the linear indices of the
free parameters. The eigen-decomposition of F ranks the identifiable parameter combinations:
small eigenvalues are the degeneracies of the data set (e.g. nₑ–B–Θe at one frequency).
"""
function fisher(params::AbstractMatrix{T}, movie::StokesMovie{T}, cache::GeodesicCache{T}, L; free = trues(size(params)), chunk = 8, nmax = -1, slab = 0, binning = nothing) where {T}
    idx = findall(vec(free))
    function model(x::AbstractVector{S}) where {S}
        q = S.(params)
        for (k, i) in enumerate(idx)
            q[i] = x[k]
        end
        return model_values(q, movie, cache, L; nmax, slab, binning)
    end
    x0 = params[idx]
    J = ForwardDiff.jacobian(model, x0, ForwardDiff.JacobianConfig(model, x0, ForwardDiff.Chunk{min(chunk, length(x0))}()))
    return J' * J, J, idx
end

"""
    model_values(params, movie, cache, L; nmax, slab, binning) -> Vector

The model's Stokes values over the movie's masked pixels, frames and frequencies, divided by
the noise (the order `data_values` uses): the residual vector is `model_values .- data_values`
and χ² its squared norm. Generic in the element type of `params` (ForwardDiff duals).
"""
function model_values(q::AbstractMatrix{S}, movie::StokesMovie, cache::GeodesicCache, L; nmax = -1, slab = 0, binning = nothing) where {S}
    out = Vector{accumulator_type(S, nmax)}(undef, npixels(cache))
    vals = S[]
    for l in eachindex(movie.νs), k in eachindex(movie.times)
        ν = movie.νs[l]
        fill!(out, zero(eltype(out)))
        polarized_image!(out, cache, q, S(movie.times[k]), S(ν), S(L); nmax, slab)
        stokes = pixel_stokes(to_screen(cache, out), S(ν), binning)
        for j in 1:size(movie.data, 2), i in 1:size(movie.data, 1)
            c = CartesianIndex(i, j, k, l)
            movie.mask[c] || continue
            append!(vals, stokes[i, j] ./ noise(movie.σ, c))
        end
    end
    return vals
end

"The movie's data over its masked pixels, frames and frequencies, divided by the noise (the order of `model_values`)."
function data_values(movie::StokesMovie{T}) where {T}
    vals = T[]
    for l in eachindex(movie.νs), k in eachindex(movie.times), j in 1:size(movie.data, 2), i in 1:size(movie.data, 1)
        c = CartesianIndex(i, j, k, l)
        movie.mask[c] || continue
        append!(vals, movie.data[c] ./ noise(movie.σ, c))
    end
    return vals
end

"""
    audit(F, names; nshow = 5) -> (values, vectors)

Eigenvalues of the Fisher matrix in ascending order with the best-constrained parameter
combinations printed as `@info`: the relative uncertainty of a combination is 1/√λ, so the
smallest eigenvalues name the degeneracies.
"""
function audit(F::AbstractMatrix, names; nshow = 5)
    e = eigen(Symmetric(Matrix(F)))
    for k in 1:min(nshow, length(e.values))
        v = e.vectors[:, k]
        top = sortperm(abs.(v); rev = true)[1:min(4, length(v))]
        @info "Fisher eigenvalue $(k): λ = $(e.values[k]) (σ of the combination $(1 / sqrt(max(e.values[k], eps()))) ) dominated by " * join(("$(names[i]) ($(round(v[i], digits = 2)))" for i in top), ", ")
    end
    return e.values, e.vectors
end

export fisher, audit, model_values, data_values

# ---- schedules: staged unfreezing, annealing, frequency curriculum, hygiene ----------------------
"""
    Stage(; free = nothing, iterations = 100, η = 0.02, η_end = η, batch = nothing, freqs = nothing, steps = nothing, label = "")

One stage of a fitting schedule: the parameter rows to free (names or indices; `nothing` frees
all), the number of Adam iterations, the learning rate annealed from `η` to `η_end` (cosine), the
minibatch `(nframes, nfreqs)` or `nothing` for full batches, the frequency indices of the
movie that enter (`nothing` for all): the frequency curriculum of the addendum starts with the
optically and Faraday thin channels and adds the thick ones later; and `steps`, per-row
multipliers of the step (`step_scale`) for rows that must move more slowly than the rest.
"""
Base.@kwdef struct Stage
    free::Any = nothing
    iterations::Int = 100
    η::Float64 = 0.02
    η_end::Float64 = η
    batch::Any = nothing
    freqs::Any = nothing
    steps::Any = nothing
    label::String = ""
end

"""
    Hygiene(; every = 0, prune_fraction = 1e-4, densify_threshold = Inf, merge_position = 0.2, merge_shape = 0.1, max_splats = typemax(Int))

Partition hygiene applied every `every` iterations of a stage (0 disables): `prune` of splats
below `prune_fraction` of the largest density × volume, `merge` of co-located parcels, and
`densify` of splats whose position-gradient norm exceeds `densify_threshold` while the count
stays below `max_splats`. Every pass re-creates the optimizer state.
"""
Base.@kwdef struct Hygiene
    every::Int = 0
    prune_fraction::Float64 = 1e-4
    densify_threshold::Float64 = Inf
    merge_position::Float64 = 0.2
    merge_shape::Float64 = 0.1
    max_splats::Int = typemax(Int)
end

"""
    fit!(params, movie, cache, L, stages; hygiene = Hygiene(), rng = Random.default_rng(), callback = nothing, priors = nothing,
         nmax = -1, slab = 0, gradient = :enzyme, binning = nothing) -> (params, history, events)

Run the stages in order on a parameter matrix (returned, since hygiene can change its size);
`binning` integrates the cache's points over pixels (`Geodesics.Binning`), and the movie then
holds the binned pixels.
`history` holds the χ² at the start of every iteration (the point where the gradient was taken,
on the stage's minibatch and frequency subset) and `events` the hygiene passes as
`(stage, iteration, nsplats_before, nsplats_after)`. `gradient = :enzyme` differentiates
`chi2` with Enzyme on the host (any cache); `gradient = :dual` takes χ² and gradient from
`chi2_gradient!` on the backend of `cache`, which must hold stored samples (the dual sweep; the
priors' penalty is added on the host). The parameters stay on the host in both cases (hygiene
works there); the dual path copies them to the backend every iteration.
"""
function fit!(params::AbstractMatrix{T}, movie::StokesMovie{T}, cache::GeodesicCache{T}, L, stages::AbstractVector{Stage};
              hygiene::Hygiene = Hygiene(), rng = Random.default_rng(), callback = nothing, priors = nothing, nmax = -1, slab = 0,
              gradient::Symbol = :enzyme, binning = nothing) where {T}
    gradient in (:enzyme, :dual) || throw(ArgumentError("gradient must be :enzyme or :dual, got $gradient"))
    function batches(st)
        freqs_all = st.freqs === nothing ? collect(eachindex(movie.νs)) : collect(st.freqs)
        frames = st.batch === nothing ? collect(eachindex(movie.times)) : sort(randperm(rng, length(movie.times))[1:min(st.batch[1], length(movie.times))])
        freqs = st.batch === nothing ? freqs_all : sort(freqs_all[randperm(rng, length(freqs_all))[1:min(st.batch[2], length(freqs_all))]])
        return frames, freqs
    end
    make_valgrad = if gradient == :enzyme
        (st, it) -> begin
            frames, freqs = batches(st)
            q -> _enzyme_valgrad(x -> chi2(x, movie, cache, L; frames, freqs, priors, nmax, slab, binning), q)
        end
    else
        backend = cache.backend
        (st, it) -> begin
            frames, freqs = batches(st)
            q -> _dual_valgrad(q, movie, cache, L, frames, freqs, priors, nmax, slab, binning, backend)
        end
    end
    return _fit_loop!(params, make_valgrad, stages; hygiene, callback)
end

"Loss and gradient at `q` by Enzyme reverse mode on the host, in one call."
function _enzyme_valgrad(f, q)
    r = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.ReverseWithPrimal), Enzyme.Const(f), q)
    return r.val, r.derivs[1]
end

"χ² (with the priors' penalty) and its gradient at the host parameters `q` by `chi2_gradient!` on `backend`."
function _dual_valgrad(q::AbstractMatrix{T}, movie, cache, L, frames, freqs, priors, nmax, slab, binning, backend) where {T}
    pdev = KernelAbstractions.allocate(backend, T, size(q)); copyto!(pdev, q)
    gdev = KernelAbstractions.allocate(backend, T, size(q)); fill!(gdev, zero(T))
    χ = chi2_gradient!(gdev, pdev, movie, cache, L; frames, freqs, nmax, slab, binning)
    g = Array(gdev)
    if priors !== nothing
        value, gp = _enzyme_valgrad(x -> penalty(x, priors), q)
        χ += value; g .+= gp
    end
    return χ, g
end

"""
    fit!(params, loss, stages; hygiene = Hygiene(), callback = nothing, gradient = nothing)

The same staged schedule, annealing and partition hygiene for an arbitrary differentiable loss
`loss(params)` (a visibility or closure χ² with gains, a composite of several data sets, …):
`Stage.batch` and `Stage.freqs` are ignored (the loss decides what it evaluates). `gradient`,
if given, is called as `gradient(params)` in place of Enzyme and may return either the gradient
or a `(value, gradient)` tuple (e.g. the GPU `image_loss_gradient!` wrapped to return both);
with the gradient alone the loss is evaluated separately for `history`. Returns
`(params, history, events)` like the movie form.
"""
function fit!(params::AbstractMatrix{T}, loss, stages::AbstractVector{Stage}; hygiene::Hygiene = Hygiene(), callback = nothing, gradient = nothing) where {T}
    valgrad = gradient === nothing ? (q -> _enzyme_valgrad(loss, q)) : (q -> (r = gradient(q); r isa Tuple ? r : (loss(q), r)))
    return _fit_loop!(params, (st, it) -> valgrad, stages; hygiene, callback)
end

"""
    sweep_passes(dparams, cache, params, L; method = :dual, kmax = 1, nmax = -1, slab = 0) -> (forward, reverse!)

The two passes of an in-kernel polarized gradient as closures over their work arrays:
`forward(t, ν)` returns the model image (observed Stokes vectors, sorted pixel order, on the
host) and leaves the state the reverse pass needs; `reverse!(dstokes, t, ν)` accumulates
∂(dstokes · image)/∂params into `dparams`. `method = :dual` is the dual sweep
(`Splats.polarized_tails!` and `Splats.polarized_dual_sweep!`), `:enzyme` the chunked Enzyme
reverse sweep (`kmax` samples per chunk). `nmax`, `slab` truncate the rays by their half-orbit
count as in `polarized_image!` (dual sweep only).
"""
function sweep_passes(dparams, cache::GeodesicCache{T,N}, params, L; method::Symbol = :dual, kmax = 1, nmax = -1, slab = 0) where {T,N}
    npix = npixels(cache)
    if method == :dual
        tails = KernelAbstractions.allocate(cache.backend, SVector{4,T}, npix, N + 1)
        forward = (t, ν) -> (Splats.polarized_tails!(tails, cache, params, t, ν, L; nmax, slab); Array(Splats.tail_image(tails, T(ν))))
        reverse! = (dstokes, t, ν) -> Splats.polarized_dual_sweep!(dparams, dstokes, tails, cache, params, t, ν, L; nmax, slab)
        return forward, reverse!
    elseif method == :enzyme
        nmax < 0 || throw(ArgumentError("the Enzyme sweep has no half-orbit truncation; use method = :dual"))
        K = Splats.chunk_size(N; kmax)
        states = KernelAbstractions.allocate(cache.backend, RadiativeState{T}, npix, N ÷ K + 1)
        forward = (t, ν) -> (Splats.polarized_forward_states!(states, cache, params, t, ν, L, Val(K)); Array(map(st -> observed_stokes(st, T(ν)), states[:, end])))
        reverse! = (dstokes, t, ν) -> Splats.polarized_reverse_sweep!(dparams, dstokes, states, cache, params, t, ν, L, Val(K))
        return forward, reverse!
    else
        throw(ArgumentError("method must be :dual or :enzyme, got $method"))
    end
end

"""
    chi2_gradient!(dparams, params, movie, cache, L; frames, freqs, method = :dual, kmax = 1, nmax = -1, slab = 0, binning = nothing) -> χ²

The movie χ² and its gradient with respect to the splat parameters, evaluated on the backend of
`cache` (a cache with stored samples) inside the kernel (the dual sweep by default, or the
chunked Enzyme reverse sweep with `method = :enzyme`, see `Splats.polarized_gradient!`): for
every frame and frequency the forward pass gives the model image, the residual seeds the
reverse pass, and ∂χ²/∂params accumulates into `dparams` (on the backend). Priors are not
included; `nmax`, `slab` truncate the rays and `binning` integrates the points over pixels as
in `chi2`. The CPU backend and CUDA give the gradient of `chi2` (gates `test_chi2_gradient`,
`test_binning`).
"""
function chi2_gradient!(dparams, params, movie::StokesMovie{T}, cache::GeodesicCache{T,N}, L; frames = eachindex(movie.times), freqs = eachindex(movie.νs), method::Symbol = :dual, kmax = 1, nmax = -1, slab = 0, binning = nothing) where {T,N}
    forward, reverse! = sweep_passes(dparams, cache, params, L; method, kmax, nmax, slab)
    npix = npixels(cache)
    dstokes = KernelAbstractions.allocate(cache.backend, SVector{4,T}, npix)
    seed = Vector{SVector{4,T}}(undef, npix)
    perm = cache.perm_host                                  # screen index of every sorted pixel
    nα = size(movie.data, 1)
    total = zero(T)
    for l in freqs, k in frames
        ν = movie.νs[l]
        image = forward(movie.times[k], ν)
        if binning === nothing
            for j in 1:npix
                i = perm[j]
                idx = CartesianIndex((i - 1) % nα + 1, (i - 1) ÷ nα + 1, k, l)
                if movie.mask[idx]
                    σ = noise(movie.σ, idx)
                    r = (image[j] - movie.data[idx]) ./ σ
                    total += sum(abs2, r)
                    seed[j] = 2 .* r ./ σ
                else
                    seed[j] = zero(SVector{4,T})
                end
            end
        else
            # the residual of every binned pixel, and its adjoint shared equally by the pixel's points
            screen = Vector{SVector{4,T}}(undef, npix)
            for j in 1:npix
                screen[perm[j]] = image[j]
            end
            stokes = bin(binning, screen)
            pseed = Vector{SVector{4,T}}(undef, Geodesics.npixels(binning))
            for q in 1:Geodesics.npixels(binning)
                idx = CartesianIndex((q - 1) % nα + 1, (q - 1) ÷ nα + 1, k, l)
                if movie.mask[idx]
                    σ = noise(movie.σ, idx)
                    r = (stokes[q] - movie.data[idx]) ./ σ
                    total += sum(abs2, r)
                    pseed[q] = 2 .* r ./ σ ./ binning.count[q]
                else
                    pseed[q] = zero(SVector{4,T})
                end
            end
            for j in 1:npix
                seed[j] = pseed[binning.pixel[perm[j]]]
            end
        end
        copyto!(dstokes, seed)
        reverse!(dstokes, movie.times[k], ν)
    end
    return total
end

"""
    image_loss_gradient!(dparams, loss, cache, params, t_obs, ν_obs, L; method = :dual, kmax = 1, nmax = -1, slab = 0, binning = nothing) -> value

Gradient of an arbitrary differentiable function of one model image with respect to the splat
parameters, on the backend of `cache` (stored samples): the forward pass gives the image
(screen-shaped matrix of Stokes vectors in cgs, on the host), Enzyme on the host differentiates
`loss(image)` with respect to the image (a small problem: four numbers per pixel), and the
reverse pass on the backend (the dual sweep, or the chunked Enzyme sweep with
`method = :enzyme`) turns that seed into ∂loss/∂params, accumulated into `dparams`. This is
how the visibility, closure and self-calibration χ² of a frame get a GPU gradient without
differentiating the Fourier transform on the device. With a `binning` the loss receives the
image integrated over its pixels; `on_image`, if given, is called with that image (the frame
the loss saw, for a caller's own use, e.g. the instrument's gradient). Returns the loss value.
"""

"The screen image as Stokes vectors from its plain-number form, binned over pixels when a `Binning` is given."
_pixel_image(x, nα, nβ, ::Nothing) = reshape([SVector(x[1, i], x[2, i], x[3, i], x[4, i]) for i in 1:nα*nβ], nα, nβ)
_pixel_image(x, nα, nβ, binning::Binning) = bin(binning, [SVector(x[1, i], x[2, i], x[3, i], x[4, i]) for i in 1:nα*nβ])
function image_loss_gradient!(dparams, loss, cache::GeodesicCache{T,N}, params, t_obs, ν_obs, L; method::Symbol = :dual, kmax = 1, nmax = -1, slab = 0, binning = nothing, on_image = nothing) where {T,N}
    forward, reverse! = sweep_passes(dparams, cache, params, L; method, kmax, nmax, slab)
    npix = npixels(cache)
    sorted = forward(t_obs, ν_obs)
    perm = cache.perm_host
    nα, nβ = cache.screen_size
    screen = Matrix{T}(undef, 4, nα * nβ)                   # the image as plain numbers for Enzyme, screen order
    for j in 1:npix, s in 1:4
        screen[s, perm[j]] = sorted[j][s]
    end
    g(x) = loss(_pixel_image(x, nα, nβ, binning))
    on_image === nothing || on_image(_pixel_image(screen, nα, nβ, binning))
    value = g(screen)
    dscreen = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(g), screen)[1]
    seed = [SVector(dscreen[1, perm[j]], dscreen[2, perm[j]], dscreen[3, perm[j]], dscreen[4, perm[j]]) for j in 1:npix]
    dstokes = KernelAbstractions.allocate(cache.backend, SVector{4,T}, npix); copyto!(dstokes, seed)
    reverse!(dstokes, t_obs, ν_obs)
    return value
end

export chi2_gradient!, image_loss_gradient!, sweep_passes

# `make_valgrad(stage, iteration)` returns a function of the parameters giving the loss and its gradient
function _fit_loop!(params::AbstractMatrix{T}, make_valgrad, stages::AbstractVector{Stage}; hygiene::Hygiene, callback) where {T}
    history = T[]
    events = Tuple{Int,Int,Int,Int}[]
    for (si, st) in enumerate(stages)
        free = st.free === nothing ? trues(size(params)) : freeze(params, st.free)
        mask = T.(free)
        scale = st.steps === nothing ? nothing : step_scale(params, st.steps)
        opt = Optimisers.setup(Optimisers.Adam(st.η), params)
        for it in 1:st.iterations
            η = st.η_end + (st.η - st.η_end) * (1 + cos(π * (it - 1) / max(st.iterations - 1, 1))) / 2
            Optimisers.adjust!(opt, η)
            value, g = make_valgrad(st, it)(params)
            if hygiene.every > 0 && it % hygiene.every == 0
                before = size(params, 2)
                q, kept = prune(params; fraction = hygiene.prune_fraction)
                gk = g[:, kept]
                q, groups = merge(q; position_tol = hygiene.merge_position, shape_tol = hygiene.merge_shape)
                gm = reduce(hcat, (sum(gk[:, grp], dims = 2) for grp in groups))
                if size(q, 2) < hygiene.max_splats
                    q = densify(q, gm; threshold = hygiene.densify_threshold)
                end
                if size(q) != size(params) || q != params
                    params = q
                    push!(events, (si, it, before, size(params, 2)))
                    free = st.free === nothing ? trues(size(params)) : freeze(params, st.free)
                    mask = T.(free)
                    scale = st.steps === nothing ? nothing : step_scale(params, st.steps)
                    opt = Optimisers.setup(Optimisers.Adam(η), params)
                    continue
                end
            end
            old = scale === nothing ? nothing : copy(params)
            opt, params = Optimisers.update!(opt, params, g .* mask)
            _scaled_update!(params, old, scale)
            push!(history, value)
            callback === nothing || callback(si, it, params, value)
        end
    end
    return params, history, events
end

export Stage, Hygiene

# ---- priors and hierarchical shrinkage (addendum §6.4) ------------------------------------------
"""
    Prior(; rows, μ = nothing, σ, shrink = false)

Gaussian penalty on parameter `rows` (names or indices) added to χ²: with `μ` a value per row
(or a vector), Σ ((p − μ)/σ)² over the splats; with `shrink = true` the centre is the population
mean over the splats instead (hierarchical shrinkage: per-splat values are drawn around a fitted
population mean with spread `σ`, so that σ → ∞ recovers full independence and no flow model is
imposed, only shared statistics if the data like them).
"""
Base.@kwdef struct Prior
    rows::Any
    μ::Any = nothing
    σ::Any
    shrink::Bool = false
end

"The prior penalty of a parameter matrix (a plain function for Enzyme)."
function penalty(params::AbstractMatrix{T}, prior::Prior) where {T}
    total = zero(T)
    n = size(params, 2)
    for (k, r) in enumerate(prior.rows)
        i = r isa Symbol ? findfirst(==(r), POLARIZED_SPLAT_PARAMS) : r
        σ = prior.σ isa Number ? prior.σ : prior.σ[k]
        if prior.shrink
            m = zero(T)
            for j in 1:n
                m += params[i, j]
            end
            m /= n
            for j in 1:n
                total += ((params[i, j] - m) / σ)^2
            end
        else
            μ = prior.μ isa Number ? prior.μ : prior.μ[k]
            for j in 1:n
                total += ((params[i, j] - μ) / σ)^2
            end
        end
    end
    return total
end
penalty(params, priors::AbstractVector) = sum(penalty(params, pr) for pr in priors; init = zero(eltype(params)))
penalty(params, ::Nothing) = zero(eltype(params))

"""
    PatternPrior(met, σ)

Motion mode C of the addendum (§3) as a soft constraint: the pattern angular velocity of every
splat (its last row) is pulled toward the azimuthal coordinate velocity dφ/dt of its own fluid,
evaluated from the ZAMO 3-velocity (the three rows before the last) at the splat's centre in the
spacetime `met`, with the penalty Σ_k ((ω_k − Ω_k)/σ)². σ → ∞ recovers mode B (free pattern
motion); σ → 0 ties the pattern to the flow. Works for every splat layout that ends with the
rows (u1, u2, u3, ω).
"""
struct PatternPrior{M,S}
    met::M
    σ::S
end

"Azimuthal coordinate velocity dφ/dt of the fluid of splat `i` at its centre."
@inline function fluid_pattern_rate(params::AbstractMatrix, i, met)
    nrow = size(params, 1)
    @inbounds begin
        r, θ, _ = Splats.boyer_lindquist(met, params[1, i], params[2, i], params[3, i])
        ũ = SVector(params[nrow - 3, i], params[nrow - 2, i], params[nrow - 1, i])
    end
    return Splats.coordinate_velocity(met, r, θ, ũ)[3]
end

function penalty(params::AbstractMatrix{T}, prior::PatternPrior) where {T}
    total = zero(T)
    nrow = size(params, 1)
    for i in 1:size(params, 2)
        Ω = fluid_pattern_rate(params, i, prior.met)
        total += ((params[nrow, i] - Ω) / prior.σ)^2
    end
    return total
end

"""
    prior_residuals(params, priors) -> Vector

The priors as residuals `(value − centre)/σ`, so that `sum(abs2, prior_residuals(...))` equals
`penalty(params, priors)`: a `Prior` contributes one residual per row and splat (about the
splats' mean for `shrink = true`), a `PatternPrior` one per splat. Used by the
Levenberg–Marquardt polish, where the priors join the data residuals.
"""
prior_residuals(params::AbstractMatrix{T}, ::Nothing) where {T} = T[]
prior_residuals(params::AbstractMatrix, priors::AbstractVector) = reduce(vcat, (prior_residuals(params, pr) for pr in priors); init = eltype(params)[])
function prior_residuals(params::AbstractMatrix{T}, prior::Prior) where {T}
    res = T[]
    n = size(params, 2)
    for (k, r) in enumerate(prior.rows)
        i = r isa Symbol ? findfirst(==(r), POLARIZED_SPLAT_PARAMS) : r
        σ = prior.σ isa Number ? prior.σ : prior.σ[k]
        centre = prior.shrink ? sum(params[i, j] for j in 1:n) / n : (prior.μ isa Number ? prior.μ : prior.μ[k])
        for j in 1:n
            push!(res, (params[i, j] - centre) / σ)
        end
    end
    return res
end
function prior_residuals(params::AbstractMatrix{T}, prior::PatternPrior) where {T}
    nrow = size(params, 1)
    return T[(params[nrow, i] - fluid_pattern_rate(params, i, prior.met)) / prior.σ for i in 1:size(params, 2)]
end

export Prior, PatternPrior, fluid_pattern_rate, penalty, prior_residuals

include("fits.jl")
include("spacetime.jl")
include("polish.jl")
include("visibilities.jl")
include("scattering.jl")
include("uvfits.jl")
include("instrument.jl")
include("feedrotation.jl")
include("timeresolved.jl")

end
