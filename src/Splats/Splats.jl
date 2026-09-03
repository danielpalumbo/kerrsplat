"""
    KerrSplat.Splats

Gaussian splats of plasma in the Kerr spacetime (main plan §7.2, addendum §1) and their
optically-thin rendering through the geodesic layer (Phase 1 of the roadmap: unpolarized,
frequency-independent emissivity, ZAMO-frame emitters, slow light).

A splat is a Gaussian in quasi-Cartesian Kerr–Schild coordinates with a smooth temporal
envelope:

    j(t, x) = A · exp(−½ (x − μ)ᵀ Σ⁻¹ (x − μ)) · exp(−½ ((t − t₀)/w)²),
    Σ = R(q) diag(e^{2s}) R(q)ᵀ,

parameterized by the 13 unconstrained numbers (μ ∈ ℝ³, s ∈ ℝ³, q ∈ ℝ⁴ normalized on use,
t₀, ln w, ln A) of one column of a parameter matrix (see [`SPLAT_PARAMS`](@ref)), so that
every parameter can move freely under gradient descent. Coordinates and times are in units
of GM/c² and GM/c³ (addendum §4.2).

The renderer is a consumer for [`Geodesics.fused_march!`](@ref): at every sample of a ray it
evaluates all splats at the emission event (t_obs − t̃, r, θ, φ) and accumulates the
optically-thin intensity ∫ g² j dλ with dλ = Σ_BL dτ (affine parameter from Mino time) and the
ZAMO redshift factor g from the analytic photon momentum.
"""
module Splats

using ..Geodesics
using Krang
using StaticArrays
using KernelAbstractions

const KA = KernelAbstractions

export SPLAT_PARAMS, NSPLATPARAMS, splat_emissivity, ThinRenderer, render!, render

"""
    SPLAT_PARAMS

Row layout of a splat parameter column: `(:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :t0, :logw, :logA)`.
"""
const SPLAT_PARAMS = (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :t0, :logw, :logA)
const NSPLATPARAMS = length(SPLAT_PARAMS)

"Rotation matrix of a (not necessarily normalized) quaternion (w, x, y, z)."
@inline function quaternion_rotation(q1, q2, q3, q4)
    n = sqrt(q1 * q1 + q2 * q2 + q3 * q3 + q4 * q4)
    w, x, y, z = q1 / n, q2 / n, q3 / n, q4 / n
    return @SMatrix [1-2(y*y+z*z)  2(x*y-w*z)    2(x*z+w*y);
                     2(x*y+w*z)    1-2(x*x+z*z)  2(y*z-w*x);
                     2(x*z-w*y)    2(y*z+w*x)    1-2(x*x+y*y)]
end

"""
    splat_emissivity(p, i, t, x, y, z)

Emissivity of splat `i` (column `i` of the parameter matrix `p`) at coordinate time `t` and
quasi-Cartesian position `(x, y, z)`.
"""
@inline function splat_emissivity(p, i, t, x, y, z)
    @inbounds begin
        d = SVector(x - p[1, i], y - p[2, i], z - p[3, i])
        R = quaternion_rotation(p[7, i], p[8, i], p[9, i], p[10, i])
        u = R' * d                                   # in the splat's principal frame
        q2 = (u[1] * exp(-p[4, i]))^2 + (u[2] * exp(-p[5, i]))^2 + (u[3] * exp(-p[6, i]))^2
        τ2 = ((t - p[11, i]) * exp(-p[12, i]))^2
        return exp(p[13, i] - (q2 + τ2) / 2)
    end
end

"""
    ThinRenderer(params, t_obs)

Fused-march consumer for the optically-thin image: `params` is the `NSPLATPARAMS × nsplat`
parameter matrix (on the backend of the cache), `t_obs` the observation time. Per sample the
accumulator gains g² · Σ_i j_i(t_obs − t̃, x) · Σ_BL · Δτ; invalid samples and samples inside
r_h(1 + 1e-3) contribute nothing.
"""
struct ThinRenderer{P,T}
    params::P
    t_obs::T
end

@inline function (c::ThinRenderer)(acc, j, k, s::GeodesicSample{T}, Δτ, pix) where {T}
    met = Krang.metric(pix)
    rh = Krang.horizon(met)
    (s.ok && (rh * (1 + T(1e-3)) < s.r < T(1e3))) || return acc
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    t = c.t_obs - s.t
    jtot = zero(T)
    for i in 1:size(c.params, 2)
        jtot += splat_emissivity(c.params, i, t, x, y, z)
    end
    pbl = Krang.p_bl_d(met, s.r, s.θ, Krang.η(pix), Krang.λ(pix), s.νr, s.νθ)
    pzamo = Krang.jac_zamo_u_bl_d(met, s.r, s.θ) * (Krang.metric_uu(met, s.r, s.θ) * pbl)
    g = inv(pzamo[1])
    Σ = s.r * s.r + met.spin^2 * cos(s.θ)^2
    return acc + g * g * jtot * Σ * Δτ
end

"""
    render!(out, cache, params, t_obs)

Optically-thin image of the splats `params` at observation time `t_obs` through a regenerated
`GeodesicCache` (any marcher; `Fused` needs no sample storage). `out` is a vector of `npixels`
in sorted order (use `to_screen(cache, out)` for the screen); returns `out`.
"""
function render!(out, cache::GeodesicCache, params, t_obs)
    fused_march!(ThinRenderer(params, t_obs), out, cache)
    return out
end

"Screen-shaped optically-thin image (allocating)."
function render(cache::GeodesicCache{T}, params, t_obs) where {T}
    out = KA.allocate(cache.backend, T, npixels(cache))
    render!(out, cache, params, t_obs)
    return to_screen(cache, out)
end

end
