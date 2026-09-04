# Spacetime-parameter fitting (plan §7.6 Phase 5, addendum §4.3): the few parameters of the
# spacetime and the camera are fitted in forward mode through the geodesic cache, which the
# polarized pipeline supports end to end with ForwardDiff duals (test_spacetime_duals), while the
# many splat parameters stay in reverse mode. Levenberg–Marquardt on the dual Jacobian of the
# residuals with respect to (a, θo), the splat parameters held fixed.

"""
    spacetime_residuals(x, params, movie, camera, L; N) -> Vector

Residuals (model − data)/σ over the whole movie for the spacetime parameters `x = (a, θo)`,
rebuilding the geodesic cache of the `camera` on the CPU backend with `N` samples per ray;
generic in the element type of `x`, so ForwardDiff duals propagate through the cache.
"""
function spacetime_residuals(x::AbstractVector{S}, params, movie::StokesMovie, camera, L; N) where {S}
    cache = GeodesicCache(CPU(), Geodesics.Camera(S.(camera.αs), S.(camera.βs), camera.size), Val(N); store_samples = false)
    regenerate!(cache, x[1], x[2]; marcher = Fused(64))
    q = S.(params)
    out = Vector{RadiativeState{S}}(undef, npixels(cache))
    res = S[]
    for l in eachindex(movie.νs), k in eachindex(movie.times)
        ν = movie.νs[l]
        fill!(out, zero(RadiativeState{S}))
        polarized_image!(out, cache, q, S(movie.times[k]), S(ν), S(L))
        img = to_screen(cache, out)
        for j in 1:size(movie.data, 2), i in 1:size(movie.data, 1)
            c = CartesianIndex(i, j, k, l)
            movie.mask[c] || continue
            append!(res, (observed_stokes(img[i, j], S(ν)) .- movie.data[c]) ./ noise(movie.σ, c))
        end
    end
    return res
end

"""
    fit_spacetime(x0, params, movie, camera, L; N = 100, iterations = 10, λ = 1e-2, bounds = ((-0.998, 0.998), (0.01, π - 0.01)))
        -> (x, χ², history)

Levenberg–Marquardt for the spin and inclination `x = [a, θo]` at fixed splat parameters, with
the Jacobian of the residuals by ForwardDiff duals through the whole pipeline; the step is
clipped to the `bounds`. Returns the parameters, the final χ² and the χ² history.
"""
function fit_spacetime(x0::AbstractVector{T}, params, movie::StokesMovie{T}, camera, L; N::Integer = 100, iterations::Integer = 10, λ = 1e-2,
                       bounds = ((-0.998, 0.998), (0.01, π - 0.01))) where {T}
    x = collect(T, x0)
    f(y) = spacetime_residuals(y, params, movie, camera, L; N)
    r = f(x); χ = sum(abs2, r)
    history = T[χ]
    for it in 1:iterations
        J = ForwardDiff.jacobian(f, x)
        A = J' * J; g = J' * r
        step = -(A + λ * Diagonal(diag(A))) \ g
        xn = [clamp(x[i] + step[i], bounds[i]...) for i in 1:2]
        rn = f(xn); χn = sum(abs2, rn)
        if χn < χ
            x, r, χ = xn, rn, χn
            λ = max(λ / 3, 1e-6)
        else
            λ *= 10
        end
        push!(history, χ)
    end
    return x, χ, history
end

export spacetime_residuals, fit_spacetime
