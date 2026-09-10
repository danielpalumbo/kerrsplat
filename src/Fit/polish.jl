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
    idx = findall(vec(free))
    data = data_values(movie)
    base = copy(params)
    function residuals(x::AbstractVector{S}) where {S}
        q = S.(base)
        for (k, i) in enumerate(idx)
            q[i] = x[k]
        end
        return vcat(model_values(q, movie, cache, L; nmax, slab, binning) .- data, prior_residuals(q, priors))
    end
    x = params[idx]
    cfg = ForwardDiff.JacobianConfig(residuals, x, ForwardDiff.Chunk{min(chunk, length(x))}())
    r = residuals(x); χ = sum(abs2, r)
    history = T[χ]
    J = ForwardDiff.jacobian(residuals, x, cfg)
    damping = T(λ)
    for it in 1:iterations
        A = Symmetric(J' * J); g = J' * r
        step = -(A + damping * Diagonal(diag(A))) \ g
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
    params[idx] .= x
    # the pseudo-inverse: JᵀJ is exactly singular along every parcel's quaternion-norm direction (a gauge, the
    # rotation is invariant), and those directions get zero variance rather than poisoning the inverse
    covariance = pinv(Matrix(Symmetric(J' * J)); rtol = 1e-12)
    return params, history, covariance, idx
end

export polish!
