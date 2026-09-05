# Jacobi elliptic functions advanced by their addition theorems (plan §4).

"""
    JacobiState{T}

The triple (sn, cn, dn)(u | μ) at one argument `u`, for a parameter `μ` that the caller keeps.
"""
struct JacobiState{T}
    sn::T
    cn::T
    dn::T
end

"""
    jacobi_state(u, μ) -> JacobiState

Direct evaluation of (sn, cn, dn)(u | μ) through JacobiElliptic's amplitude (one amplitude
computation; the sine and cosine are taken separately rather than through `sincos`, whose CUDA
intrinsic Enzyme cannot differentiate inside a kernel).
"""
@inline function jacobi_state(u, μ)
    φ = JacobiElliptic.CarlsonAlg.am(u, μ)
    s = sin(φ); c = cos(φ)
    return JacobiState(s, c, sqrt(muladd(-μ, s * s, one(s))))
end

"""
    jacobi_step(x, Δ, μ) -> JacobiState

Advance `x = (sn, cn, dn)(u | μ)` to argument `u + δ` given `Δ = (sn, cn, dn)(δ | μ)`, by the
addition theorems

    sn(u+δ) = (sn u cn δ dn δ + sn δ cn u dn u) / (1 − μ sn²u sn²δ)
    cn(u+δ) = (cn u cn δ − sn u dn u sn δ dn δ) / (1 − μ sn²u sn²δ)
    dn(u+δ) = (dn u dn δ − μ sn u cn u sn δ cn δ) / (1 − μ sn²u sn²δ)

Fifteen multiplications and one division; no transcendental function.
"""
@inline function jacobi_step(x::JacobiState{T}, Δ::JacobiState{T}, μ) where {T}
    den = one(T) - μ * (x.sn * x.sn) * (Δ.sn * Δ.sn)
    sn = (x.sn * Δ.cn * Δ.dn + Δ.sn * x.cn * x.dn) / den
    cn = (x.cn * Δ.cn - x.sn * x.dn * Δ.sn * Δ.dn) / den
    dn = (x.dn * Δ.dn - μ * x.sn * x.cn * Δ.sn * Δ.cn) / den
    return JacobiState(sn, cn, dn)
end

"""
    jacobi_step_constants(δ, μ) -> JacobiState

`(sn, cn, dn)(δ | μ)` for use as the `Δ` of [`jacobi_step`](@ref). `δ` may be negative (sn is
odd, cn and dn are even).
"""
@inline function jacobi_step_constants(δ, μ)
    x = jacobi_state(abs(δ), μ)
    return JacobiState(copysign(x.sn, δ), x.cn, x.dn)
end

"""
    negative_parameter_transform(k) -> (μ, scale)

For a negative Jacobi parameter `k < 0`, the parameter `μ = −k/(1 − k) ∈ (0, 1)` and the
argument scale `√(1 − k)` of the imaginary-modulus transformation (DLMF 22.17.4–6):

    sn(u | k) = sn(s u | μ) / (s dn(s u | μ)),   cn(u | k) = cn(s u | μ) / dn(s u | μ),
    dn(u | k) = 1 / dn(s u | μ),                  s = √(1 − k).

The recurrence is run at `μ`, where the addition theorems are numerically benign for any `k < 0`.
"""
@inline function negative_parameter_transform(k::T) where {T}
    return -k / (one(T) - k), sqrt(one(T) - k)
end
