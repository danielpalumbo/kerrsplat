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

export StokesMovie, chi2, fit!, freeze

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
function chi2(params, movie::StokesMovie{T}, cache::GeodesicCache{T}, L; frames = eachindex(movie.times), freqs = eachindex(movie.νs), priors = nothing) where {T}
    out = Vector{RadiativeState{T}}(undef, npixels(cache))
    nα, nβ = size(movie.data, 1), size(movie.data, 2)
    total = zero(T)
    for l in freqs, k in frames
        ν = movie.νs[l]
        fill!(out, zero(RadiativeState{T}))
        polarized_image!(out, cache, params, movie.times[k], ν, L)
        img = to_screen(cache, out)
        for j in 1:nβ, i in 1:nα
            idx = CartesianIndex(i, j, k, l)
            movie.mask[idx] || continue
            r = (observed_stokes(img[i, j], ν) - movie.data[idx]) ./ noise(movie.σ, idx)
            total += r[1] * r[1] + r[2] * r[2] + r[3] * r[3] + r[4] * r[4]
        end
    end
    return total + penalty(params, priors)
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
    fit!(params, movie, cache, L; free = trues(size(params)), iterations = 100, η = 0.02,
         batch = nothing, rng = Random.default_rng(), callback = nothing) -> history

Adam on the Enzyme reverse-mode gradient of `chi2`, updating only the parameters where `free` is
true (staged unfreezing: call repeatedly with different masks). `batch = (nframes, nfreqs)`
draws a random minibatch of frames and frequencies at every iteration; otherwise every frame and
frequency enters each step. Returns the χ² history (full data at the start and the end, the
minibatch value in between).
"""
function fit!(params::AbstractMatrix{T}, movie::StokesMovie{T}, cache::GeodesicCache{T}, L; free = trues(size(params)),
              iterations::Integer = 100, η = 0.02, batch = nothing, rng = Random.default_rng(), callback = nothing) where {T}
    mask = T.(free)
    opt = Optimisers.setup(Optimisers.Adam(η), params)
    history = T[chi2(params, movie, cache, L)]
    for it in 1:iterations
        frames = batch === nothing ? eachindex(movie.times) : sort(randperm(rng, length(movie.times))[1:min(batch[1], length(movie.times))])
        freqs = batch === nothing ? eachindex(movie.νs) : sort(randperm(rng, length(movie.νs))[1:min(batch[2], length(movie.νs))])
        f(q) = chi2(q, movie, cache, L; frames, freqs)
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
function fisher(params::AbstractMatrix{T}, movie::StokesMovie{T}, cache::GeodesicCache{T}, L; free = trues(size(params)), chunk = 8) where {T}
    idx = findall(vec(free))
    function model(x::AbstractVector{S}) where {S}
        q = S.(params)
        for (k, i) in enumerate(idx)
            q[i] = x[k]
        end
        out = Vector{RadiativeState{S}}(undef, npixels(cache))
        vals = S[]
        for l in eachindex(movie.νs), k in eachindex(movie.times)
            ν = movie.νs[l]
            fill!(out, zero(RadiativeState{S}))
            polarized_image!(out, cache, q, S(movie.times[k]), S(ν), S(L))
            img = to_screen(cache, out)
            for j in 1:size(movie.data, 2), i in 1:size(movie.data, 1)
                c = CartesianIndex(i, j, k, l)
                movie.mask[c] || continue
                st = observed_stokes(img[i, j], S(ν)) ./ noise(movie.σ, c)
                append!(vals, st)
            end
        end
        return vals
    end
    x0 = params[idx]
    J = ForwardDiff.jacobian(model, x0, ForwardDiff.JacobianConfig(model, x0, ForwardDiff.Chunk{min(chunk, length(x0))}()))
    return J' * J, J, idx
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

export fisher, audit

# ---- schedules: staged unfreezing, annealing, frequency curriculum, hygiene ----------------------
"""
    Stage(; free = nothing, iterations = 100, η = 0.02, η_end = η, batch = nothing, freqs = nothing, label = "")

One stage of a fitting schedule: the parameter rows to free (names or indices; `nothing` frees
all), the number of Adam iterations, the learning rate annealed from `η` to `η_end` (cosine), the
minibatch `(nframes, nfreqs)` or `nothing` for full batches, and the frequency indices of the
movie that enter (`nothing` for all): the frequency curriculum of the addendum starts with the
optically and Faraday thin channels and adds the thick ones later.
"""
Base.@kwdef struct Stage
    free::Any = nothing
    iterations::Int = 100
    η::Float64 = 0.02
    η_end::Float64 = η
    batch::Any = nothing
    freqs::Any = nothing
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
    fit!(params, movie, cache, L, stages; hygiene = Hygiene(), rng = Random.default_rng(), callback = nothing, priors = nothing)
        -> (params, history, events)

Run the stages in order on a parameter matrix (returned, since hygiene can change its size).
`history` holds the χ² after every iteration (the stage's minibatch and frequency subset) and
`events` the hygiene passes as `(stage, iteration, nsplats_before, nsplats_after)`.
"""
function fit!(params::AbstractMatrix{T}, movie::StokesMovie{T}, cache::GeodesicCache{T}, L, stages::AbstractVector{Stage};
              hygiene::Hygiene = Hygiene(), rng = Random.default_rng(), callback = nothing, priors = nothing) where {T}
    function make_loss(st, it)
        freqs_all = st.freqs === nothing ? collect(eachindex(movie.νs)) : collect(st.freqs)
        frames = st.batch === nothing ? collect(eachindex(movie.times)) : sort(randperm(rng, length(movie.times))[1:min(st.batch[1], length(movie.times))])
        freqs = st.batch === nothing ? freqs_all : sort(freqs_all[randperm(rng, length(freqs_all))[1:min(st.batch[2], length(freqs_all))]])
        return q -> chi2(q, movie, cache, L; frames, freqs, priors)
    end
    return _fit_loop!(params, make_loss, stages; hygiene, callback)
end

"""
    fit!(params, loss, stages; hygiene = Hygiene(), callback = nothing)

The same staged schedule, annealing and partition hygiene for an arbitrary differentiable loss
`loss(params)` (a visibility or closure χ² with gains, a composite of several data sets, …):
`Stage.batch` and `Stage.freqs` are ignored (the loss decides what it evaluates). Returns
`(params, history, events)` like the movie form.
"""
function fit!(params::AbstractMatrix{T}, loss, stages::AbstractVector{Stage}; hygiene::Hygiene = Hygiene(), callback = nothing) where {T}
    return _fit_loop!(params, (st, it) -> loss, stages; hygiene, callback)
end

function _fit_loop!(params::AbstractMatrix{T}, make_loss, stages::AbstractVector{Stage}; hygiene::Hygiene, callback) where {T}
    history = T[]
    events = Tuple{Int,Int,Int,Int}[]
    for (si, st) in enumerate(stages)
        free = st.free === nothing ? trues(size(params)) : freeze(params, st.free)
        mask = T.(free)
        opt = Optimisers.setup(Optimisers.Adam(st.η), params)
        for it in 1:st.iterations
            η = st.η_end + (st.η - st.η_end) * (1 + cos(π * (it - 1) / max(st.iterations - 1, 1))) / 2
            Optimisers.adjust!(opt, η)
            f = make_loss(st, it)
            g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(f), params)[1]
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
                    opt = Optimisers.setup(Optimisers.Adam(η), params)
                    continue
                end
            end
            opt, params = Optimisers.update!(opt, params, g .* mask)
            push!(history, f(params))
            callback === nothing || callback(si, it, params, history[end])
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

export Prior, PatternPrior, fluid_pattern_rate, penalty

include("fits.jl")
include("spacetime.jl")
include("visibilities.jl")
include("uvfits.jl")

end
