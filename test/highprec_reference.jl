# High-precision (BigFloat) reference for r(τ) and θ(τ) on a Krang pixel.
#
# Krang cannot simply be run in BigFloat: it forms constants such as T(√3/2), T(1/3) and T(π/2)
# through Float64, and its root-classification tolerance is eps(T). This file re-evaluates the
# same closed forms (Gralla & Lupsasca; Krang's `_rs_case*` and `_θs`) in BigFloat from the
# exact screen inputs: the quartic roots are Krang's Float64 roots polished by Newton's method
# in Complex{BigFloat}, and everything downstream (elliptic parameter, radial antiderivative at
# infinity, Jacobi functions, polar constants) is BigFloat. It is a check of the *numerics* of
# the Float64 paths (recurrence and direct), not of the formulas themselves.

using Krang
using JacobiElliptic
using KerrSplat.Geodesics

const HP_BITS = 256
setprecision(BigFloat, HP_BITS)

# JacobiElliptic restricts its internal square root to Float32/Float64/duals; extend it for the
# reference (tests only).
JacobiElliptic.CarlsonAlg._sqrt(x::BigFloat) = sqrt(x)

"Newton-polish Krang's Float64 quartic roots to BigFloat precision."
function polished_roots(a, η, λ, roots64)
    a, η, λ = big(a), big(η), big(λ)
    c2 = a^2 - η - λ^2
    c1 = 2 * (η + (λ - a)^2)
    c0 = -a^2 * η
    p(r) = ((r * r + c2) * r + c1) * r + c0
    dp(r) = (4 * r * r + 2 * c2) * r + c1
    return map(roots64) do z0
        z = Complex{BigFloat}(z0)
        for _ in 1:80
            δ = p(z) / dp(z)
            z -= δ
            abs(δ) < eps(BigFloat) * max(abs(z), one(BigFloat)) && break
        end
        z
    end
end

"""
    HighPrecisionRay(a, α, β, θo, pix64)

BigFloat constants of one ray: polished roots, radial case, elliptic parameter, I0_inf, and the
polar constants. `pix64` is Krang's Float64 pixel (its root ordering and case are kept).
"""
struct HighPrecisionRay
    case::Int
    a::BigFloat
    roots::NTuple{4,Complex{BigFloat}}
    k::BigFloat
    c::BigFloat        # radial argument X = c (fo − τ)
    fo::BigFloat
    rfun::Function     # r from (sn, cn) of X
    # polar
    vortical::Bool
    kθ::BigFloat       # (negative) polar parameter of the closed form
    tempfac::BigFloat
    offset::BigFloat
    up::BigFloat
    um::BigFloat
    signθ::BigFloat
end

function HighPrecisionRay(a, α, β, θo, pix64)
    η64, λ64 = pix64.η, pix64.λ
    roots = polished_roots(a, η64, λ64, pix64.roots)
    case = num_real_roots_big(pix64.roots)
    A, B_ = big(0), big(0)
    if case == 2
        r1, r2, r3, r4 = real.(roots)
        r31, r32, r41, r42 = r3 - r1, r3 - r2, r4 - r1, r4 - r2
        k = r32 * r41 / (r31 * r42)
        c = sqrt(r31 * r42) / 2
        fo = 2 / sqrt(r31 * r42) * JacobiElliptic.F(asin(sqrt(r31 / r41)), k)
        rfun = (sn, cn) -> (s2 = r41 * sn * sn; (r31 * r4 - r3 * s2) / (r31 - s2))
    elseif case == 3
        r1, r2 = real(roots[1]), real(roots[2])
        r21 = r2 - r1
        A = abs(roots[3] - roots[2])
        B_ = abs(roots[3] - roots[1])
        k = ((A + B_)^2 - r21^2) / (4 * A * B_)
        c = sqrt(A * B_)
        fo = JacobiElliptic.F(acos(clamp((A - B_) / (A + B_), -1, 1)), k) / sqrt(A * B_)
        rfun = (sn, cn) -> (-A * r1 + B_ * r2 + (A * r1 + B_ * r2) * cn) / (-A + B_ + (A + B_) * cn)
    else
        r2, r4 = roots[2], roots[4]
        a1, a2, b1, b2 = abs(imag(r4)), abs(imag(r2)), real(r4), real(r2)
        C = sqrt((a1 - a2)^2 + (b1 - b2)^2)
        D = sqrt((a1 + a2)^2 + (b1 - b2)^2)
        k = 4 * C * D / (C + D)^2
        c = (C + D) / 2
        go = sqrt(max((4 * a2^2 - (C - D)^2) / ((C + D)^2 - 4 * a2^2), big(0)))
        fo = 2 / (C + D) * JacobiElliptic.F(big(π) / 2 + atan(go), k)
        rfun = (sn, cn) -> (sc = sn / cn; -(a2 * (go - sc) / (1 + go * sc) + b1))
    end
    # polar constants from the exact screen inputs
    ab, αb, βb, θob = big(a), big(α), big(β), big(θo)
    η = (αb^2 - ab^2) * cos(θob)^2 + βb^2
    λ = -αb * sin(θob)
    Δθ = (1 - (η + λ^2) / ab^2) / 2
    dsc = sqrt(Δθ^2 + η / ab^2)
    up = Δθ + dsc
    um = Δθ - dsc
    m = up / um
    tempfac = 1 / sqrt(abs(um * ab^2))
    vortical = η < 0
    signβ = big(sign(β))
    if vortical
        kθ = 1 - m
        Gθo = tempfac * JacobiElliptic.F(asin(sqrt(clamp((cos(θob)^2 - um) / (up - um), 0, 1))), kθ)
        σ = θob > big(π) / 2 ? big(-1) : big(1)
        offset = σ * signβ * Gθo
        signθ = σ
    else
        kθ = m
        Gθo = tempfac * JacobiElliptic.F(asin(clamp(cos(θob) / sqrt(up), -1, 1)), kθ)
        offset = signβ * Gθo
        signθ = signβ
    end
    return HighPrecisionRay(case, ab, roots, k, c, fo, rfun, vortical, kθ, tempfac, offset, up, um, signθ)
end

num_real_roots_big(roots64) = (n = sum(Krang._isreal2, roots64); n == 4 ? 2 : n == 2 ? 3 : 4)

"r and θ at Mino time τ (given as the Float64 the kernel used), in BigFloat."
function highprec_coordinates(ray::HighPrecisionRay, τ64::Float64)
    τ = big(τ64)
    X = ray.c * (ray.fo - τ)
    sn = JacobiElliptic.sn(X, ray.k)
    cn = JacobiElliptic.cn(X, ray.k)
    r = ray.rfun(sn, cn)
    Xθ = (τ + ray.offset) / ray.tempfac
    snθ = JacobiElliptic.sn(Xθ, ray.kθ)
    if ray.vortical
        cosθ = ray.signθ * sqrt(ray.um + (ray.up - ray.um) * snθ^2)
    else
        cosθ = ray.signθ * sqrt(ray.up) * snθ
    end
    θ = acos(clamp(cosθ, -1, 1))
    return r, θ
end
