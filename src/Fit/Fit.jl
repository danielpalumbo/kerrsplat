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

end
