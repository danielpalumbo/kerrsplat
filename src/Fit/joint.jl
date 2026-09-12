# Joint fitting of the spacetime and the splats (plan §7.1 regime b, §7.6 Phase 5; Daniel,
# 2026-09-11: fitting the spacetime parameters in parallel with the splats is crucial). Mixed
# mode: the many splat parameters take their gradient from the dual sweep over Float64 samples
# stored at the current spacetime, the few spacetime parameters take theirs from forward-mode
# duals through a fused march and the transport (a Dual{3} cache regenerated every step), and
# one Adam iteration updates both blocks with their own steps. The spacetime block is
# x = (a, θo, ln L): spin, inclination and the mass through the length unit L = GM/c², which is
# how mass enters data in M units (for data in physical units the camera scale and the frame
# times scale with it too; that path is the real-data one, later).

"""
    spacetime_chi2(cache, params, movie, L; frames, freqs, nmax = -1, slab = 0, binning = nothing) -> χ²

The movie χ² rendered on `cache`, generic in the cache's element type (duals in the spacetime
propagate through the march and the transport to the value) and in the backend (each frame is
rendered on the backend and compared on the host). No priors.
"""
function spacetime_chi2(cache::GeodesicCache{S}, params, movie::StokesMovie, L; frames = eachindex(movie.times), freqs = eachindex(movie.νs), nmax = -1, slab = 0, binning = nothing) where {S}
    out = KernelAbstractions.allocate(cache.backend, SVector{4,S}, npixels(cache))
    nα, nβ = size(movie.data, 1), size(movie.data, 2)
    total = zero(S)
    for l in freqs, k in frames
        ν = movie.νs[l]
        render_frame!(out, cache, params, movie.times[k], ν, L; nmax, slab)
        stokes = _screen_stokes(cache, out, binning)
        for j in 1:nβ, i in 1:nα
            idx = CartesianIndex(i, j, k, l)
            movie.mask[idx] || continue
            r = (stokes[i, j] - movie.data[idx]) ./ noise(movie.σ, idx)
            total += r[1] * r[1] + r[2] * r[2] + r[3] * r[3] + r[4] * r[4]
        end
    end
    return total
end

"""
    render_frame!(out, cache, params, t, ν, L; nmax = -1, slab = 0)

One frame's Stokes vectors (sorted pixel order, cgs) on the cache's backend: through the tails
kernel over the stored samples when the cache has them (one march per cache, the transport
per frame), through the fused march otherwise. Generic in the cache's element type, so a
dual-typed cache gives dual images either way; the stored form makes the spacetime block's
Jacobian cost one dual march per visit instead of one per frame.
"""
function render_frame!(out, cache::GeodesicCache{S,N}, params, t, ν, L; nmax = -1, slab = 0) where {S,N}
    if Geodesics.has_samples(cache)
        tails = KernelAbstractions.allocate(cache.backend, SVector{4,S}, npixels(cache), N + 1)
        Splats.polarized_tails!(tails, cache, params, t, ν, L; nmax, slab)
        copyto!(out, Splats.tail_image(tails, S(ν)))
    else
        acc = KernelAbstractions.allocate(cache.backend, accumulator_type(S, nmax), npixels(cache))
        fill!(acc, zero(eltype(acc)))
        polarized_image!(acc, cache, params, S(t), S(ν), S(L); nmax, slab)
        copyto!(out, map(st -> observed_stokes(st, S(ν)), acc))
    end
    return out
end

struct SpacetimeTag end

"""
    spacetime_valgrad(loss, x, camera, backend; N, stored = true) -> (value, gradient)

The value and the gradient of `loss(cache, L)` with respect to the spacetime block
`x = (a, θo, ln L)` by forward-mode duals: a `GeodesicCache` of the `camera` with `N` samples
per ray and three-partial dual scalars is built on `backend` and regenerated with the fused
marcher at the dual spin and inclination, and `loss` renders on it with the dual `L` (e.g.
`(cache, L) -> spacetime_chi2(cache, q, movie, L)` with `q` the splat matrix converted to the
cache's element type on the backend, see `spacetime_movie_loss`). Three partials cost about
four forward passes. With `stored` (the default) the dual cache stores its samples, so the
march runs once and every frame renders through the tails kernel (`render_frame!`); without
it each frame re-marches.
"""
function spacetime_valgrad(loss, x::AbstractVector{T}, camera::Geodesics.Camera, backend; N::Integer, stored::Bool = true) where {T}
    S = ForwardDiff.Dual{SpacetimeTag,T,3}
    xd = SVector{3,S}(ntuple(i -> S(x[i], ForwardDiff.Partials(ntuple(q -> q == i ? one(T) : zero(T), Val(3)))), Val(3)))
    cache = GeodesicCache(backend, Geodesics.Camera(S.(camera.αs), S.(camera.βs), camera.size), Val(Int(N)); store_samples = stored)
    regenerate!(cache, xd[1], xd[2]; marcher = stored ? Recurrence(64) : Fused(64))
    v = loss(cache, exp(xd[3]))
    g = ForwardDiff.partials(v)
    return ForwardDiff.value(v), T[g[1], g[2], g[3]]
end

"Observed Stokes vectors (sorted order, on the backend) to the screen on the host, integrated over the pixels of a `Binning` when one is given."
_screen_stokes(cache, out, ::Nothing) = Array(to_screen(cache, out))
_screen_stokes(cache, out, binning::Binning) = bin(binning, vec(Array(to_screen(cache, out))))

"The movie loss of a splat matrix for `spacetime_valgrad`: `params` converted to the cache's element type on its backend at every call."
function spacetime_movie_loss(params::AbstractMatrix, movie::StokesMovie; frames = eachindex(movie.times), freqs = eachindex(movie.νs), nmax = -1, slab = 0, binning = nothing)
    return function (cache::GeodesicCache{S}, L) where {S}
        q = KernelAbstractions.allocate(cache.backend, S, size(params)); copyto!(q, S.(Array(params)))
        return spacetime_chi2(cache, q, movie, L; frames, freqs, nmax, slab, binning)
    end
end

"""
    spacetime_movie_residuals(cache, params, movie, L; frames, freqs, nmax = -1, slab = 0, binning = nothing) -> Vector

The scaled residuals (model − data)/σ of the movie rendered on `cache`, generic in the cache's
element type and backend like `spacetime_chi2` (whose value is their squared norm).
"""
function spacetime_movie_residuals(cache::GeodesicCache{S}, params, movie::StokesMovie, L; frames = eachindex(movie.times), freqs = eachindex(movie.νs), nmax = -1, slab = 0, binning = nothing) where {S}
    out = KernelAbstractions.allocate(cache.backend, SVector{4,S}, npixels(cache))
    nα, nβ = size(movie.data, 1), size(movie.data, 2)
    res = S[]
    for l in freqs, k in frames
        ν = movie.νs[l]
        render_frame!(out, cache, params, movie.times[k], ν, L; nmax, slab)
        stokes = _screen_stokes(cache, out, binning)
        for j in 1:nβ, i in 1:nα
            idx = CartesianIndex(i, j, k, l)
            movie.mask[idx] || continue
            append!(res, (stokes[i, j] - movie.data[idx]) ./ noise(movie.σ, idx))
        end
    end
    return res
end

"""
    spacetime_jacobian(residuals, x, camera, backend; N, stored = true) -> (r, J)

The residual vector of `residuals(cache, L)` and its Jacobian with respect to the spacetime
block `x` (length 2, `(a, θo)`, or 3 with `ln L`) by one forward-mode dual pass through the
fused march and the transport, as `spacetime_valgrad` does for a scalar loss.
"""
function spacetime_jacobian(residuals, x::AbstractVector{T}, camera::Geodesics.Camera, backend; N::Integer, stored::Bool = true) where {T}
    n = length(x)
    n in (2, 3) || throw(ArgumentError("the spacetime block is (a, θo) or (a, θo, ln L)"))
    S = ForwardDiff.Dual{SpacetimeTag,T,n}
    xd = [S(x[i], ForwardDiff.Partials(ntuple(q -> q == i ? one(T) : zero(T), n))) for i in 1:n]
    cache = GeodesicCache(backend, Geodesics.Camera(S.(camera.αs), S.(camera.βs), camera.size), Val(Int(N)); store_samples = stored)
    regenerate!(cache, xd[1], xd[2]; marcher = stored ? Recurrence(64) : Fused(64))
    Ld = n == 3 ? exp(xd[3]) : S(T(NaN))
    rd = residuals(cache, Ld)
    r = ForwardDiff.value.(rd)
    J = Matrix{T}(undef, length(rd), n)
    for k in eachindex(rd), i in 1:n
        J[k, i] = ForwardDiff.partials(rd[k], i)
    end
    return r, J
end

"""
    fit_joint!(params, x, movie, cache, camera; L = NaN, iterations = 100, η = 0.02, η_end = η / 10, warmup = 0, every = 1, inner = 3, λ = 1e-2,
               free = trues(size(params)), steps = nothing, priors = nothing, spacetime_priors = nothing,
               bounds = ((-0.998, 0.998), (0.01, π - 0.01), (-Inf, Inf)), nmax = -1, slab = 0, binning = nothing,
               frames = eachindex(movie.times), freqs = eachindex(movie.νs), callback = nothing)
        -> (params, x, history, accepted)

Joint fit of the splats and the spacetime block `x` (`[a, θo]` with the length unit `L` given,
or `[a, θo, ln L]` with the mass through the length unit) to a Stokes movie, mixed-mode and
block-wise within one loop. Every
iteration regenerates the Float64 samples of `cache` (a stored-sample cache on the backend the
fit runs on; `camera` is its camera) at the current spin and inclination and takes one Adam step
on the free splat entries from the dual sweep (`chi2_gradient!`; `priors` added on the host;
`steps` the per-row multipliers of `step_scale`; the step cosine-decayed from `η` to `η_end`).
After the first `warmup` iterations, every `every`-th iteration also takes `inner`
Levenberg–Marquardt steps on the spacetime with the splats held: the residual Jacobian with
respect to `x` by forward duals through a fused march (`spacetime_jacobian`, a few marches'
worth of work independent of the pixel count), the damped Gauss–Newton step clipped to
`bounds`, accepted when the χ² at the new spacetime (one Float64 fused pass) falls, the damping
divided by 3 on acceptance and multiplied by 10 otherwise within a visit, and reset to `λ` at
the next (the splats move between visits, so a damping grown large by rejections at one visit
would only freeze the spacetime at the next: the first GPU self-fit stalled that way after
thirty iterations with every later step accepted and microscopic). Gauss–Newton is the right optimizer
for a block of two or three parameters whose curvature comes for free from the duals, where a
gradient step of any fixed size either stalls or overshoots. The defaults, no warmup and three
inner steps, come from the schedule experiment of `docs/notes/2026-09-11_joint_spacetime.md`:
the spacetime has to move before the splats adapt to the wrong one (a warmup of five
iterations left the inclination 6° off where no warmup recovered it to 0.4°), and one inner
step per iteration got only halfway. `spacetime_priors = (μ, σ)` adds
Gaussian penalties on `x` as residuals of the spacetime block. `history` holds `(χ², x...)` at
the start of every iteration, the χ² of the stored-sample pass; `accepted` counts the
spacetime steps taken.

Mass through `ln L` alone is degenerate with the densities where the emission is optically
thin (both scale the emissivity along the path); it is identifiable through absorption and
Faraday depth, and, for data in physical units, through the angular and time scales, which
this movie form in M units does not carry. Fit `[a, θo]` on M-unit movies.
"""
function fit_joint!(params::AbstractMatrix{T}, x0::AbstractVector{T}, movie::StokesMovie{T}, cache::GeodesicCache{T,N}, camera::Geodesics.Camera;
                    L::Real = NaN, iterations::Integer = 100, η = 0.02, η_end = η / 10, warmup::Integer = 0, every::Integer = 1, inner::Integer = 3, λ::Real = 1e-2,
                    free = trues(size(params)), steps = nothing, priors = nothing, spacetime_priors = nothing,
                    bounds = ((-0.998, 0.998), (0.01, π - 0.01), (-Inf, Inf)), nmax = -1, slab = 0, binning = nothing,
                    frames = eachindex(movie.times), freqs = eachindex(movie.νs), callback = nothing) where {T,N}
    backend = cache.backend
    x = collect(T, x0)
    n = length(x)
    n in (2, 3) || throw(ArgumentError("the spacetime block is (a, θo) or (a, θo, ln L)"))
    n == 3 || isfinite(L) || throw(ArgumentError("give the length unit L when the mass is not fitted"))
    Lfix = T(L)
    opt = Optimisers.setup(Optimisers.Adam(η), params)
    mask = T.(free)
    scale = steps === nothing ? nothing : step_scale(params, steps)
    damping = T(λ)
    accepted = 0
    history = Vector{T}[]
    function prior_residuals(y)
        spacetime_priors === nothing && return T[]
        μ, σ = spacetime_priors
        return T[(y[i] - μ[i]) / σ[i] for i in 1:n if isfinite(σ[i])]
    end
    for it in 1:iterations
        ηt = η_end + (η - η_end) * (1 + cos(π * (it - 1) / max(iterations - 1, 1))) / 2
        Optimisers.adjust!(opt, ηt)
        regenerate!(cache, x[1], x[2]; marcher = Recurrence(64))
        Lit = n == 3 ? exp(x[3]) : Lfix
        χ, g = _dual_valgrad(params, movie, cache, Lit, frames, freqs, priors, nmax, slab, binning, backend)
        χ += sum(abs2, prior_residuals(x); init = zero(T))
        push!(history, vcat(χ, x))
        if it > warmup && (it - warmup - 1) % every == 0
            χx = χ
            damping = T(λ)                        # the damping restarts every visit: the sky has moved under the spacetime block since the last one
            for _ in 1:inner
                residuals = (c, Lc) -> vcat(spacetime_movie_residuals(c, _on_backend(c, params), movie, n == 3 ? Lc : eltype(c.αs)(Lfix); frames, freqs, nmax, slab, binning), eltype(c.αs).(prior_residuals(x)))
                r, J = spacetime_jacobian(residuals, x, camera, backend; N)
                if spacetime_priors !== nothing
                    μ, σ = spacetime_priors
                    rows = [i for i in 1:n if isfinite(σ[i])]
                    for (k, i) in enumerate(rows)
                        J[end - length(rows) + k, i] = 1 / σ[i]
                    end
                end
                if !(all(isfinite, J) && all(isfinite, r))
                    dump = joinpath(tempdir(), "kerrsplat_nonfinite_$(it).csv")
                    open(dump, "w") do io
                        println(io, join(x, ","))
                        for row in eachrow(params)
                            println(io, join(row, ","))
                        end
                    end
                    @warn "non-finite spacetime residuals or Jacobian; the spacetime step is skipped (state written)" iteration = it x = copy(x) nonfinite_rows = count(k -> !(isfinite(r[k]) && all(isfinite, view(J, k, :))), eachindex(r)) file = dump
                    break
                end
                A = J' * J; gx = J' * r
                d = diag(A); floor = 1e-12 * max(maximum(d), eps(T))
                step = -(A + damping * Diagonal(max.(d, floor))) \ gx
                xn = [clamp(x[i] + step[i], bounds[i]...) for i in 1:n]
                χn = _spacetime_chi2_at(xn, params, movie, camera, backend, N, Lfix, n, frames, freqs, nmax, slab, binning) + sum(abs2, prior_residuals(xn); init = zero(T))
                if χn < χx
                    x = xn; χx = χn
                    damping = max(damping / 3, T(1e-8))
                    accepted += 1
                else
                    damping *= 10
                end
            end
        end
        old = scale === nothing ? nothing : copy(params)
        opt, params = Optimisers.update!(opt, params, g .* mask)
        _scaled_update!(params, old, scale)
        callback === nothing || callback(it, params, x, χ)
    end
    return params, x, history, accepted
end

"`params` as a matrix of the cache's element type on its backend."
function _on_backend(cache::GeodesicCache{S}, params) where {S}
    q = KernelAbstractions.allocate(cache.backend, S, size(params)); copyto!(q, S.(Array(params)))
    return q
end
"The χ² at a trial spacetime by one Float64 fused pass (a non-storing cache regenerated at `xn`)."
function _spacetime_chi2_at(xn, params, movie, camera, backend, N, L, n, frames, freqs, nmax, slab, binning)
    T = eltype(xn)
    c = GeodesicCache(backend, camera, Val(Int(N)); store_samples = true)
    regenerate!(c, xn[1], xn[2]; marcher = Recurrence(64))
    return spacetime_chi2(c, _on_backend(c, params), movie, n == 3 ? exp(xn[3]) : T(L); frames, freqs, nmax, slab, binning)
end

export spacetime_chi2, spacetime_valgrad, spacetime_movie_loss, spacetime_movie_residuals, spacetime_jacobian, fit_joint!
