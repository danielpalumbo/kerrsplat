# Levenberg–Marquardt on the splat parameters from a fitted point: Adam (the staged fits) stops
# in the tail of shallow valleys (nₑ against B along a parcel's emissivity, for instance) with χ²
# still a thousand above the noise floor; a Gauss–Newton step with the Jacobian of the residuals,
# damped à la Marquardt, finishes such descents in a few iterations, and its normal matrix at the
# optimum is the Laplace approximation of the posterior covariance (the plan's §7.4 uncertainty
# item). The Jacobian comes from ForwardDiff duals through the transfer at fixed geodesics, as in
# `fisher`; the priors join the data as residuals.

"""
    polish!(params, movie, cache, L; free = trues(size(params)), iterations = 5, λ = 1e-3, chunk = 8,
            nmax = -1, slab = 0, binning = nothing, priors = nothing) -> (params, history, covariance, idx)

Levenberg–Marquardt on the free entries of `params` (a host matrix, updated in place and
returned) against the movie χ² with the priors: at every iteration the Jacobian of the scaled
residuals (`model_values .- data_values` and `prior_residuals`) is formed by ForwardDiff duals
through the transfer (`chunk` partials at a time, on the CPU cache), the damped normal equations
`(JᵀJ + λ diag JᵀJ) δ = −Jᵀ r` give the step, and the step is kept when χ² decreases (λ then
falls by 3, otherwise grows by 10). `history` holds χ² before every iteration and at the end;
`covariance` is the pseudo-inverse of `JᵀJ` at the final point over the free entries `idx`
(the Laplace covariance, whose diagonal's square root is the marginal 1σ error of each
parameter; exactly singular gauge directions such as a quaternion's norm get zero variance).
"""
function polish!(params::AbstractMatrix{T}, movie::StokesMovie{T}, cache::GeodesicCache{T}, L; free = trues(size(params)), iterations::Integer = 5,
                 λ::Real = 1e-3, chunk::Integer = 8, nmax = -1, slab = 0, binning = nothing, priors = nothing) where {T}
    data = data_values(movie)
    return polish!(params, q -> model_values(q, movie, cache, L; nmax, slab, binning) .- data; free, iterations, λ, chunk, priors)
end

"""
    polish!(params, residuals; free = trues(size(params)), iterations = 5, λ = 1e-3, chunk = 8, priors = nothing) -> (params, history, covariance, idx)

The same Levenberg–Marquardt for any scaled residual vector `residuals(params)` of the full
parameter matrix, generic in its element type (ForwardDiff duals form the Jacobian; the priors
join as residuals): e.g. `q -> timeresolved_residuals(q, tr, cache, L, Δα, D, ν)`.
"""
function polish!(params::AbstractMatrix{T}, residuals_of; free = trues(size(params)), iterations::Integer = 5, λ::Real = 1e-3, chunk::Integer = 8, priors = nothing) where {T}
    idx = findall(vec(free))
    base = copy(params)
    function residuals(x::AbstractVector{S}) where {S}
        q = S.(base)
        for (k, i) in enumerate(idx)
            q[i] = x[k]
        end
        return vcat(residuals_of(q), prior_residuals(q, priors))
    end
    x, history, covariance = levenberg_marquardt!(params[idx], residuals; iterations, λ, chunk)
    params[idx] .= x
    return params, history, covariance, idx
end

"""
    levenberg_marquardt!(x, residuals; iterations = 5, λ = 1e-3, chunk = 8) -> (x, history, covariance)

The Levenberg–Marquardt core of `polish!` on a plain parameter vector `x` and a scaled residual
function `residuals(x)` generic in its element type: the Jacobian by ForwardDiff duals (`chunk`
partials at a time), damped Gauss–Newton steps kept when the squared norm falls (λ then falls
by 3, otherwise grows by 10), and the pseudo-inverse of JᵀJ at the end (the Laplace
covariance; exactly singular gauge directions get zero variance).
"""
function levenberg_marquardt!(x::AbstractVector{T}, residuals; iterations::Integer = 5, λ::Real = 1e-3, chunk::Integer = 8) where {T}
    x = copy(x)
    cfg = ForwardDiff.JacobianConfig(residuals, x, ForwardDiff.Chunk{min(chunk, length(x))}())
    r = residuals(x); χ = sum(abs2, r)
    history = T[χ]
    J = ForwardDiff.jacobian(residuals, x, cfg)
    damping = T(λ)
    for it in 1:iterations
        A = Symmetric(J' * J); g = J' * r
        # Marquardt's scaling with a floor: a parameter the residuals do not see (a zero column of J) would otherwise
        # leave the damped system singular; the floor keeps its step at zero
        d = diag(A); floor = 1e-12 * max(maximum(d), eps(T))
        step = -(A + damping * Diagonal(max.(d, floor))) \ g
        xn = x .+ step
        rn = residuals(xn); χn = sum(abs2, rn)
        if χn < χ
            x, r, χ = xn, rn, χn
            damping = max(damping / 3, T(1e-8))
            J = ForwardDiff.jacobian(residuals, x, cfg)
        else
            damping *= 10
        end
        push!(history, χ)
    end
    covariance = pinv(Matrix(Symmetric(J' * J)); rtol = 1e-12)
    return x, history, covariance
end

"""
    pack(sky, free, gains, gm, dterms, dm) -> Vector
    unpack!(sky, free, gains, gm, dterms, dm, x)

The joint parameter vector of a self-calibration (the free sky entries, then the free gain
entries, then the free d-term entries; column-major order within each block) and its inverse.
"""
pack(sky, free, gains, gm, dterms, dm) = vcat(sky[findall(vec(free))], gains[findall(vec(gm))], dterms[findall(vec(dm))])
function unpack!(sky, free, gains, gm, dterms, dm, x)
    i1 = findall(vec(free)); i2 = findall(vec(gm)); i3 = findall(vec(dm))
    sky[i1] .= x[1:length(i1)]; gains[i2] .= x[length(i1)+1:length(i1)+length(i2)]; dterms[i3] .= x[length(i1)+length(i2)+1:end]
    return sky, gains, dterms
end
"The joint parameters as new matrices of the element type of `x` (for duals)."
function unpack(sky, free, gains, gm, dterms, dm, x::AbstractVector{S}) where {S}
    return unpack!(S.(sky), free, S.(gains), gm, S.(dterms), dm, x)
end

export polish!, levenberg_marquardt!, pack, unpack, unpack!
