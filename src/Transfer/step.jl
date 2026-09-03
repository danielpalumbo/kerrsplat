# The analytic constant-coefficient step of polarized radiative transfer.
#
# Along a ray parameter s the Stokes vector S = (I, Q, U, V) obeys dS/ds = j − K S with
#     K = [αI αQ αU αV; αQ αI ρV −ρU; αU −ρV αI ρQ; αV ρU −ρQ αI]
# (the sign convention of Landi Degl'Innocenti & Landi Degl'Innocenti 1985 and ipole). For
# constant j and K over a step of length Δ the exact solution is S(Δ) = O S(0) + E with
#     O = exp(−KΔ),   E = ∫₀^Δ exp(−Ku) du · j.
# Writing K = αI 1 + K', the traceless part K' has eigenvalues ±Λ₁ and ±iΛ₂ with
#     Λ₁,₂² = ±½(α⃗² − ρ⃗²) + √(¼(α⃗² − ρ⃗²)² + (α⃗·ρ⃗)²),
# so by Cayley–Hamilton both operators are cubic polynomials in K' whose coefficients are
# combinations of e^{−αIu} cosh Λ₁u, cos Λ₂u, sinh Λ₁u/Λ₁, sin Λ₂u/Λ₂ and their integrals. The
# combinations are evaluated in forms without cancellation: cosh − 1 and 1 − cos through squared
# half-angle functions, sinh x/x − 1 and 1 − sin x/x by series for small x, the emission integrals
# by series in the small arguments and by closed forms otherwise, and the hyperbolic terms at large
# optical depth through the separate exponentials e^{−(αI∓Λ₁)Δ}, as in ipole. The remaining
# denominators Λ₁² + Λ₂² are removable: every numerator vanishes with them.

"""
    transfer_step(j, α, ρ, Δ) -> (O, E)

Exact solution operator `O = exp(−KΔ)` (4×4) and emission vector `E = ∫₀^Δ exp(−Ku) du j` for the
constant coefficients `j = (jI, jQ, jU, jV)`, `α = (αI, αQ, αU, αV)`, `ρ = (ρQ, ρU, ρV)` over a step
of length `Δ` (all in the same units, e.g. invariants and the affine length). The Stokes vector after
the step is `O * S + E`.
"""
@inline function transfer_step(j::SVector{4,T}, α::SVector{4,T}, ρ::SVector{3,T}, Δ) where {T}
    αI = α[1]
    αQ, αU, αV = α[2], α[3], α[4]
    ρQ, ρU, ρV = ρ[1], ρ[2], ρ[3]
    α2 = αQ * αQ + αU * αU + αV * αV
    ρ2 = ρQ * ρQ + ρU * ρU + ρV * ρV
    a = αI * Δ
    if (α2 + ρ2) * Δ^2 < T(1e-200)         # no polarized transfer, or negligible and underflowing (ρ ~ 1e-170 from
        e = exp(-a)                        # density tails): the unpolarized operator with the full emission vector
        return SMatrix{4,4,T}(e * I4), (Δ * phi(a)) * j
    end
    K1 = @SMatrix [zero(T) αQ αU αV; αQ zero(T) ρV -ρU; αU -ρV zero(T) ρQ; αV ρU -ρQ zero(T)]
    K2 = K1 * K1
    K3 = K2 * K1
    αρ = αQ * ρQ + αU * ρU + αV * ρV
    h = (α2 - ρ2) / 2
    root = sqrt(h * h + αρ * αρ)
    if h >= 0                              # the larger of Λ₁², Λ₂² from the stable expression, the smaller from their product
        Λ1sq = h + root
        Λ2sq = Λ1sq > 0 ? αρ * αρ / Λ1sq : zero(T)
    else
        Λ2sq = root - h
        Λ1sq = Λ2sq > 0 ? αρ * αρ / Λ2sq : zero(T)
    end
    Λ1 = sqrt(Λ1sq)
    Λ2 = sqrt(Λ2sq)
    b1 = Λ1 * Δ
    b2 = Λ2 * Δ
    Θ = Λ1sq + Λ2sq
    # K' nilpotent (α⃗² = ρ⃗², α⃗ ⟂ ρ⃗, Θ = 0): the cubic with the b → 0 limits is exact; it stays accurate to
    # ΘΔ²(1 + |K'Δ|²) when Θ is merely tiny, which also covers Θ underflowing while K' does not.
    if Θ * Δ^2 * (1 + (α2 + ρ2) * Δ^2) < T(1e-17)
        e = exp(-a)
        O = e * (SMatrix{4,4,T}(I4) - Δ * K1 + (Δ^2 / 2) * K2 - (Δ^3 / 6) * K3)
        M = moments(a)
        E = Δ * (M[1] * j - Δ * M[2] * (K1 * j) + Δ^2 * M[3] / 2 * (K2 * j) - Δ^3 * M[4] / 6 * (K3 * j))
        return O, E
    end
    # ---- O = e^{−a} [c0 1 + c2 K'² − Δ sinh(b1)/b1 M3 − Δ sin(b2)/b2 M2] --------------------
    # with c0 = (Λ2² cosh b1 + Λ1² cos b2)/Θ, c2 = (cosh b1 − cos b2)/Θ and the bounded matrices
    # M2 = (Λ1² K' − K'³)/Θ, M3 = (Λ2² K' + K'³)/Θ (Landi Degl'Innocenti's form): the odd powers of
    # K' enter only through M2, M3, whose products with the trigonometric factors stay O(1) even for
    # thousands of radians of rotation per step, where the plain cubic in K' would cancel large terms.
    M2 = (Λ1sq * K1 - K3) / Θ
    M3 = (Λ2sq * K1 + K3) / Θ
    ecosh, esinh_b, ecoshm1 = damped_hyperbolics(a, b1)   # e^{−a}cosh b1, e^{−a}sinh(b1)/b1, e^{−a}(cosh b1 − 1)
    e = exp(-a)
    c0 = (Λ2sq * ecosh + Λ1sq * e * cos(b2)) / Θ
    c2 = (ecoshm1 + e * versin(b2)) / Θ
    O = c0 * SMatrix{4,4,T}(I4) + c2 * K2 - (Δ * esinh_b) * M3 - (Δ * e * sinc(b2)) * M2
    # ---- E = [C0 1 + C2 K'² − Ish M3 − Is M2] j: the same combinations of the integrals -----------
    # Ich = ∫₀^Δ e^{−αIu} cosh Λ1u du = Δ ∫₀¹ e^{−at} cosh b1t dt, Ish = ∫ e^{−αIu} sinh(Λ1u)/Λ1 du, etc.
    Ich = Δ * (phi(a) + int_coshm1(a, b1))
    Ic = Δ * int_cos(a, b2)
    Ish = Δ^2 * (moment1(a) + int_sinhc_excess(a, b1))
    Is = Δ^2 * int_sinc(a, b2)
    C0 = (Λ2sq * Ich + Λ1sq * Ic) / Θ
    C2 = Δ * (int_coshm1(a, b1) + int_versin(a, b2)) / Θ
    E = C0 * j + C2 * (K2 * j) - Ish * (M3 * j) - Is * (M2 * j)
    return O, E
end

const I4 = SMatrix{4,4,Float64}(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)

# ---- scalar building blocks -------------------------------------------------------------------
"φ(a) = (1 − e^{−a})/a = ∫₀¹ e^{−at} dt."
@inline phi(a) = a == 0 ? one(a) : -expm1(-a) / a

"1 − cos b, without cancellation."
@inline versin(b) = 2 * sin(b / 2)^2

"sinh(b)/b."
@inline sinhc(b) = b == 0 ? one(b) : sinh(b) / b

"sin(b)/b."
@inline sinc(b) = b == 0 ? one(b) : sin(b) / b

"1 − sin(b)/b ≥ 0, by series for small b."
@inline function sinc_deficit(b)
    if abs(b) < 0.1
        b2 = b * b
        return b2 / 6 * (1 - b2 / 20 * (1 - b2 / 42 * (1 - b2 / 72 * (1 - b2 / 110))))
    else
        return 1 - sin(b) / b
    end
end

"sinh(b)/b − 1 ≥ 0, by series for small b."
@inline function sinhc_excess(b)
    if abs(b) < 0.1
        b2 = b * b
        return b2 / 6 * (1 + b2 / 20 * (1 + b2 / 42 * (1 + b2 / 72 * (1 + b2 / 110))))
    else
        return sinh(b) / b - 1
    end
end

"e^{−a}(sinh(b)/b − 1) with the large-b form through the separate exponentials."
@inline function e_sinhc_excess(a, b)
    if b < 1
        return exp(-a) * sinhc_excess(b)
    else
        return (exp(-(a - b)) - exp(-(a + b))) / (2b) - exp(-a)
    end
end

"e^{−a}cosh b, e^{−a}sinh(b)/b and e^{−a}(cosh b − 1), stable for large a ≥ b."
@inline function damped_hyperbolics(a, b)
    if b < 1
        e = exp(-a)
        return e * cosh(b), e * sinhc(b), e * 2 * sinh(b / 2)^2
    else
        em = exp(-(a - b))
        ep = exp(-(a + b))
        e = exp(-a)
        return (em + ep) / 2, (em - ep) / (2b), (em - 2e + ep) / 2
    end
end

"Mₙ(a) = ∫₀¹ tⁿ e^{−at} dt for n = 0…9, by series for |a| < 1 and by the closed-form recurrence otherwise."
@inline function moments(a::T) where {T}
    if abs(a) < 1
        # Mₙ = Σ_m (−a)^m / (m! (n + m + 1)); 24 terms reach 1e-24 for |a| < 1
        M0 = M1 = M2 = M3 = M4 = M5 = M6 = M7 = M8 = M9 = zero(T)
        term = one(T)
        for m in 0:23
            M0 += term / (m + 1); M1 += term / (m + 2); M2 += term / (m + 3); M3 += term / (m + 4); M4 += term / (m + 5)
            M5 += term / (m + 6); M6 += term / (m + 7); M7 += term / (m + 8); M8 += term / (m + 9); M9 += term / (m + 10)
            term *= -a / (m + 1)
        end
        return (M0, M1, M2, M3, M4, M5, M6, M7, M8, M9)
    else
        e = exp(-a)
        M0 = (1 - e) / a
        M1 = (M0 - e) / a
        M2 = (2M1 - e) / a
        M3 = (3M2 - e) / a
        M4 = (4M3 - e) / a
        M5 = (5M4 - e) / a
        M6 = (6M5 - e) / a
        M7 = (7M6 - e) / a
        M8 = (8M7 - e) / a
        M9 = (9M8 - e) / a
        return (M0, M1, M2, M3, M4, M5, M6, M7, M8, M9)
    end
end

@inline moment1(a) = moments(a)[2]

"∫₀¹ e^{−at}(cosh(bt) − 1) dt."
@inline function int_coshm1(a, b)
    if b < 0.1
        M = moments(a)
        b2 = b * b
        return b2 * (M[3] / 2 + b2 * (M[5] / 24 + b2 * (M[7] / 720 + b2 * M[9] / 40320)))
    else
        return (phi(a - b) + phi(a + b)) / 2 - phi(a)
    end
end

"∫₀¹ e^{−at}(1 − cos(bt)) dt."
@inline function int_versin(a, b)
    if b < 0.1
        M = moments(a)
        b2 = b * b
        return b2 * (M[3] / 2 - b2 * (M[5] / 24 - b2 * (M[7] / 720 - b2 * M[9] / 40320)))
    else
        return phi(a) - int_cos(a, b)
    end
end

"∫₀¹ e^{−at} cos(bt) dt."
@inline function int_cos(a, b)
    if b < 0.1
        return phi(a) - int_versin(a, b)
    else
        s, c = sincos(b)
        return (a - exp(-a) * (a * c - b * s)) / (a * a + b * b)
    end
end

"∫₀¹ e^{−at} sin(bt)/b dt."
@inline function int_sinc(a, b)
    if b < 0.1
        return moment1(a) - int_sinc_deficit(a, b)
    else
        s, c = sincos(b)
        return (1 - exp(-a) * (a * s / b + c)) / (a * a + b * b)
    end
end

"∫₀¹ e^{−at}(t − sin(bt)/b) dt."
@inline function int_sinc_deficit(a, b)
    if b < 0.1
        M = moments(a)
        b2 = b * b
        return b2 * (M[4] / 6 - b2 * (M[6] / 120 - b2 * (M[8] / 5040 - b2 * M[10] / 362880)))
    else
        return moment1(a) - int_sinc(a, b)
    end
end

"∫₀¹ e^{−at}(sinh(bt)/b − t) dt."
@inline function int_sinhc_excess(a, b)
    if b < 0.1
        M = moments(a)
        b2 = b * b
        return b2 * (M[4] / 6 + b2 * (M[6] / 120 + b2 * (M[8] / 5040 + b2 * M[10] / 362880)))
    else
        return (phi(a - b) - phi(a + b)) / (2b) - moment1(a)
    end
end

# ---- compositing along a ray ------------------------------------------------------------------
"""
    RadiativeState(P, S)

Front-to-back accumulator for a ray marched from the observer outward: `P` is the product of the
step operators of the intervals already passed (closest to the observer first) and `S` the Stokes
vector reaching the observer from them. A new interval with operator `O` and emission `E` (further
from the observer than everything accumulated so far) contributes `S += P E`, `P = P O`.
"""
struct RadiativeState{T}
    P::SMatrix{4,4,T,16}
    S::SVector{4,T}
end
Base.zero(::Type{RadiativeState{T}}) where {T} = RadiativeState(SMatrix{4,4,T}(I4), zero(SVector{4,T}))
@inline advance(st::RadiativeState, O, E) = RadiativeState(st.P * O, st.S + st.P * E)

"""
    rotate_to_screen(c::StokesCoefficients, χ) -> (j, α, ρ)

Coefficients of the field-aligned basis rotated by the angle `χ` into a common (screen) Stokes
basis: (Q, U) rotate by 2χ, I and V are unchanged. Returns the 4-vectors j, α and the 3-vector ρ
expected by `transfer_step`.
"""
@inline function rotate_to_screen(c::StokesCoefficients{T}, χ) where {T}
    s2, c2 = sincos(2χ)
    j = SVector(c.jI, c.jQ * c2, c.jQ * s2, c.jV)
    α = SVector(c.αI, c.αQ * c2, c.αQ * s2, c.αV)
    ρ = SVector(c.ρQ * c2, c.ρQ * s2, c.ρV)
    return j, α, ρ
end
