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
function chi2(params, movie::StokesMovie{T}, cache::GeodesicCache{T}, L; frames = eachindex(movie.times), freqs = eachindex(movie.νs)) where {T}
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

end
