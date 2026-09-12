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
the step is `O * S + E`. The step is linear in `j`: `E = Ej * j` with the emission matrix of
[`step_operator`](@ref), which holds the whole dependence on the coefficients.
"""
@inline function transfer_step(j::SVector{4,T}, α::SVector{4,T}, ρ::SVector{3,T}, Δ) where {T}
    O, Ej = step_operator(α, ρ, Δ)
    return O, Ej * j
end

"""
    step_operator(α, ρ, Δ) -> (O, Ej)

The solution operator `O = exp(−KΔ)` and the emission matrix `Ej = ∫₀^Δ exp(−Ku) du` of the
constant-coefficient step (see [`transfer_step`](@ref): `E = Ej * j`). Written for any real
number type, so that forward-mode duals in `α`, `ρ` give the derivatives of both matrices
(the dual sweep of `Splats.polarized_gradient!`); the removable singularities are guarded on
the value of their argument so that duals pass through them. Derivatives are exact except on
the codimension-two degeneracy Λ₁ = Λ₂ = 0 (|α⃗| = |ρ⃗| and α⃗ ⟂ ρ⃗): at exact nilpotency the
cubic branch's derivative misses the quartic terms (an O((K′Δ)⁴) relative error), and near it
the derivative of the closed forms loses accuracy as eps/(ΘΔ²) while their value does not
(the same holds for any differentiation of these formulae, Enzyme included).
"""
@inline function step_operator(α::SVector{4,T}, ρ::SVector{3,T}, Δ) where {T}
    αI = α[1]
    αQ, αU, αV = α[2], α[3], α[4]
    ρQ, ρU, ρV = ρ[1], ρ[2], ρ[3]
    α2 = αQ * αQ + αU * αU + αV * αV
    ρ2 = ρQ * ρQ + ρU * ρU + ρV * ρV
    a = αI * Δ
    K1 = @SMatrix [zero(T) αQ αU αV; αQ zero(T) ρV -ρU; αU -ρV zero(T) ρQ; αV ρU -ρQ zero(T)]
    if (α2 + ρ2) * Δ^2 < T(1e-200)         # no polarized transfer, or negligible and underflowing (ρ ~ 1e-170 from
        e = exp(-a)                        # density tails): the unpolarized operator to first order in K' (its value is
        return e * (identity4(T) - Δ * K1), Δ * (phi(a) * identity4(T) - (Δ * moment1(a)) * K1)   # unpolarized, its derivative continuous)
    end
    K2 = K1 * K1
    K3 = K2 * K1
    αρ = αQ * ρQ + αU * ρU + αV * ρV
    h = (α2 - ρ2) / 2
    root = safe_sqrt(h * h + αρ * αρ)
    if h >= 0                              # the larger of Λ₁², Λ₂² from the stable expression, the smaller from their product
        Λ1sq = h + root
        Λ2sq = _value(Λ1sq) > 0 ? αρ * αρ / Λ1sq : zero(T)
    else
        Λ2sq = root - h
        Λ1sq = _value(Λ2sq) > 0 ? αρ * αρ / Λ2sq : zero(T)
    end
    Λ1 = safe_sqrt(Λ1sq)
    Λ2 = safe_sqrt(Λ2sq)
    b1 = Λ1 * Δ
    b2 = Λ2 * Δ
    Θ = Λ1sq + Λ2sq
    # K' nilpotent (α⃗² = ρ⃗², α⃗ ⟂ ρ⃗, Θ = 0): the cubic with the b → 0 limits is exact; it stays accurate to
    # ΘΔ²(1 + |K'Δ|²) when Θ is merely tiny, which also covers Θ underflowing while K' does not.
    if Θ * Δ^2 * (1 + (α2 + ρ2) * Δ^2) < T(1e-17)
        e = exp(-a)
        O = e * (identity4(T) - Δ * K1 + (Δ^2 / 2) * K2 - (Δ^3 / 6) * K3)
        M = moments(a)
        Ej = Δ * (M[1] * identity4(T) - (Δ * M[2]) * K1 + (Δ^2 * M[3] / 2) * K2 - (Δ^3 * M[4] / 6) * K3)
        return O, Ej
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
    O = c0 * identity4(T) + c2 * K2 - (Δ * esinh_b) * M3 - (Δ * e * sinc(b2)) * M2
    # ---- Ej = C0 1 + C2 K'² − Ish M3 − Is M2 (E = Ej j): the same combinations of the integrals ----
    # Ich = ∫₀^Δ e^{−αIu} cosh Λ1u du = Δ ∫₀¹ e^{−at} cosh b1t dt, Ish = ∫ e^{−αIu} sinh(Λ1u)/Λ1 du, etc.
    Ich = Δ * (phi(a) + int_coshm1(a, b1))
    Ic = Δ * int_cos(a, b2)
    Ish = Δ^2 * (moment1(a) + int_sinhc_excess(a, b1))
    Is = Δ^2 * int_sinc(a, b2)
    C0 = (Λ2sq * Ich + Λ1sq * Ic) / Θ
    C2 = Δ * (int_coshm1(a, b1) + int_versin(a, b2)) / Θ
    Ej = C0 * identity4(T) + C2 * K2 - Ish * M3 - Is * M2
    return O, Ej
end

"The 4×4 identity built in place: a global constant matrix referenced from a CUDA kernel is materialized as a constant table, which Enzyme's device reverse pass cannot handle."
@inline identity4(::Type{T}) where {T} = SMatrix{4,4,T}(one(T), zero(T), zero(T), zero(T), zero(T), one(T), zero(T), zero(T), zero(T), zero(T), one(T), zero(T), zero(T), zero(T), zero(T), one(T))

# ---- scalar building blocks -------------------------------------------------------------------
# The removable singularities below are guarded on the *value* of the argument: for a
# forward-mode dual `x == 0` also asks for zero partials (ForwardDiff ≥ 1), and the unguarded
# branch would then evaluate 0/0 at an argument whose value is zero.
@inline _value(x::Real) = x
@inline _value(x::ForwardDiff.Dual) = ForwardDiff.value(x)
"Whether the value of `x` is zero (its partials, if any, aside)."
@inline vanishes(x) = iszero(_value(x))
"Square root with a finite derivative at zero: the step depends on Λ₁, Λ₂ only through even functions, whose derivatives vanish there."
@inline safe_sqrt(x) = _value(x) > 0 ? sqrt(x) : zero(x)

"""
    safe_acos(x)

`acos` of a cosine that may have rounded past ±1: the value at the clamped argument, and for a
dual the derivative −1/√(1 − x²) at the clamped value, zero when that value is at the boundary
(the cusp of the pitch angle where the field lies along the ray, a set of measure zero). A
plain `acos(clamp(x, -1, 1))` on a dual gives NaN partials there: the clamp returns the
boundary with zero partials and the acos rule multiplies them by −∞.
"""
@inline safe_acos(x) = acos(clamp(x, -one(x), one(x)))
@inline function safe_acos(d::ForwardDiff.Dual{T}) where {T}
    v = clamp(ForwardDiff.value(d), -one(ForwardDiff.value(d)), one(ForwardDiff.value(d)))
    s = one(v) - v * v
    dv = s > 0 ? -inv(sqrt(s)) : zero(v)
    return ForwardDiff.Dual{T}(acos(v), dv * ForwardDiff.partials(d))
end

"φ(a) = (1 − e^{−a})/a = ∫₀¹ e^{−at} dt (to first order at a = 0, so that a dual's derivative there is −1/2)."
@inline phi(a) = vanishes(a) ? one(a) - a / 2 : -expm1(-a) / a

"1 − cos b, without cancellation."
@inline versin(b) = 2 * sin(b / 2)^2

"sinh(b)/b."
@inline sinhc(b) = vanishes(b) ? one(b) : sinh(b) / b

"sin(b)/b."
@inline sinc(b) = vanishes(b) ? one(b) : sin(b) / b

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
        # Mₙ = Σ_m (−a)^m / (m! (n + m + 1)); 24 terms reach 1e-24 for |a| < 1, written as Horner chains with
        # literal coefficients (a loop here would put Enzyme's per-iteration cache in device malloc)
        M0 = @muladd_chain(a, 1.0, -0.5, 0.16666666666666666, -0.041666666666666664, 0.008333333333333333, -0.001388888888888889, 0.0001984126984126984, -2.48015873015873e-05, 2.7557319223985893e-06, -2.755731922398589e-07, 2.505210838544172e-08, -2.08767569878681e-09, 1.6059043836821613e-10, -1.1470745597729725e-11, 7.647163731819816e-13, -4.779477332387385e-14, 2.8114572543455206e-15, -1.5619206968586225e-16, 8.22063524662433e-18, -4.110317623312165e-19, 1.9572941063391263e-20, -8.896791392450574e-22, 3.868170170630684e-23, -1.6117375710961184e-24)
        M1 = @muladd_chain(a, 0.5, -0.3333333333333333, 0.125, -0.03333333333333333, 0.006944444444444444, -0.0011904761904761906, 0.00017361111111111112, -2.2045855379188714e-05, 2.48015873015873e-06, -2.505210838544172e-07, 2.296443268665491e-08, -1.9270852604185937e-09, 1.4911969277048643e-10, -1.0706029224547743e-11, 7.169215998581078e-13, -4.498331606952833e-14, 2.6552651846596585e-15, -1.4797143443923793e-16, 7.809603484293113e-18, -3.9145882126782523e-19, 1.8683261924146203e-20, -8.509974375387505e-22, 3.706996413521072e-23, -1.5472680682522736e-24)
        M2 = @muladd_chain(a, 0.3333333333333333, -0.25, 0.1, -0.027777777777777776, 0.005952380952380952, -0.0010416666666666667, 0.00015432098765432098, -1.984126984126984e-05, 2.2546897546897547e-06, -2.296443268665491e-07, 2.1197937864604532e-08, -1.789436313245837e-09, 1.3917837991912066e-10, -1.0036902398013508e-11, 6.74749741042925e-13, -4.2484242954554536e-14, 2.515514385467045e-15, -1.4057286271727604e-16, 7.437717604088679e-18, -3.7366523848292405e-19, 1.787094618831376e-20, -8.1553921097463585e-22, 3.5587165569802293e-23, -1.4877577579348785e-24)
        M3 = @muladd_chain(a, 0.25, -0.2, 0.08333333333333333, -0.023809523809523808, 0.005208333333333333, -0.000925925925925926, 0.0001388888888888889, -1.8037518037518038e-05, 2.066798941798942e-06, -2.119793786460453e-07, 1.968379944570421e-08, -1.6701405590294479e-09, 1.3047973117417563e-10, -9.44649637460095e-12, 6.372636443183181e-13, -4.024823016747272e-14, 2.3897386661936928e-15, -1.3387891687359624e-16, 7.099639531175557e-18, -3.574189237662752e-19, 1.7126323430467353e-20, -7.829176425356505e-22, 3.4218428432502203e-23, -1.4326556187521052e-24)
        M4 = @muladd_chain(a, 0.2, -0.16666666666666666, 0.07142857142857142, -0.020833333333333332, 0.004629629629629629, -0.0008333333333333334, 0.00012626262626262626, -1.6534391534391536e-05, 1.907814407814408e-06, -1.9683799445704207e-07, 1.8371546149323926e-08, -1.5657567740901075e-09, 1.2280445286981235e-10, -8.921691020456453e-12, 6.037234525120908e-13, -3.8235818659099085e-14, 2.275941586851136e-15, -1.2779351156116003e-16, 6.7909595515592285e-18, -3.4252646860934706e-19, 1.644127049324866e-20, -7.528054255150485e-22, 3.295107923129842e-23, -1.3814893466538157e-24)
        M5 = @muladd_chain(a, 0.16666666666666666, -0.14285714285714285, 0.0625, -0.018518518518518517, 0.004166666666666667, -0.0007575757575757576, 0.00011574074074074075, -1.5262515262515263e-05, 1.7715419501133788e-06, -1.8371546149323928e-07, 1.722332451499118e-08, -1.4736534344377482e-09, 1.1598198326593389e-10, -8.45212833516927e-12, 5.735372798864862e-13, -3.6415065389618177e-14, 2.1724896965397208e-15, -1.2223727192806612e-16, 6.508002903577594e-18, -3.288254098649732e-19, 1.5808913935816018e-20, -7.249237430885653e-22, 3.1774254973037764e-23, -1.333851782976098e-24)
        M6 = @muladd_chain(a, 0.14285714285714285, -0.125, 0.05555555555555555, -0.016666666666666666, 0.003787878787878788, -0.0006944444444444445, 0.00010683760683760684, -1.417233560090703e-05, 1.6534391534391535e-06, -1.7223324514991183e-07, 1.621018777881523e-08, -1.3917837991912067e-09, 1.0987766835720052e-10, -8.029521918410807e-12, 5.462259808442726e-13, -3.475983514463553e-14, 2.078033622777124e-15, -1.171440522643967e-16, 6.2476827874344904e-18, -3.161782787163204e-19, 1.522339860485987e-20, -6.990336094068308e-22, 3.067859100845025e-23, -1.2893900568768947e-24)
        M7 = @muladd_chain(a, 0.125, -0.1111111111111111, 0.05, -0.015151515151515152, 0.003472222222222222, -0.000641025641025641, 9.92063492063492e-05, -1.3227513227513228e-05, 1.5500992063492063e-06, -1.621018777881523e-07, 1.5309621791103273e-08, -1.3185320202864062e-09, 1.0438378493934049e-10, -7.647163731819816e-12, 5.21397527169533e-13, -3.3248537964433987e-14, 1.991448888494744e-15, -1.1245829017382084e-16, 6.007387295610087e-18, -3.044679720971974e-19, 1.4679705797543445e-20, -6.749290021859056e-22, 2.965597130816858e-23, -1.2477968292357046e-24)
        M8 = @muladd_chain(a, 0.1111111111111111, -0.1, 0.045454545454545456, -0.013888888888888888, 0.003205128205128205, -0.0005952380952380953, 9.259259259259259e-05, -1.240079365079365e-05, 1.4589169000933706e-06, -1.5309621791103272e-07, 1.4503852223150469e-08, -1.252605419272086e-09, 9.941312851365762e-11, -7.299565380373461e-12, 4.987280694665098e-13, -3.1863182215915904e-14, 1.9117909329549543e-15, -1.0813297132098157e-16, 5.7848914698467505e-18, -2.935941159508689e-19, 1.4173509045904017e-20, -6.524313687797087e-22, 2.86993270724212e-23, -1.2088031783220888e-24)
        M9 = @muladd_chain(a, 0.1, -0.09090909090909091, 0.041666666666666664, -0.01282051282051282, 0.002976190476190476, -0.0005555555555555556, 8.680555555555556e-05, -1.1671335200746965e-05, 1.3778659611992946e-06, -1.4503852223150468e-07, 1.3778659611992945e-08, -1.1929575421638914e-09, 9.4894349944855e-11, -6.982192972531137e-12, 4.779477332387385e-13, -3.058865492727927e-14, 1.8382605124566867e-15, -1.0412804645724151e-16, 5.578288203066509e-18, -2.8347018091808033e-19, 1.3701058744373883e-20, -6.313851955932665e-22, 2.7802473101408044e-23, -1.1721727789789952e-24)
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
# The four trigonometric integrals below are written without calling one another (each regime of
# each function is spelled out): mutually recursive helpers cannot be differentiated by Enzyme
# inside a CUDA kernel, where the reverse pass of a recursive call needs a dynamic tape.
"Series of ∫₀¹ e^{−at}(1 − cos bt) dt for small b."
@inline function _int_versin_series(a, b)
    M = moments(a)
    b2 = b * b
    return b2 * (M[3] / 2 - b2 * (M[5] / 24 - b2 * (M[7] / 720 - b2 * M[9] / 40320)))
end
"Closed form of ∫₀¹ e^{−at} cos(bt) dt for b not small."
@inline function _int_cos_closed(a, b)
    s, c = sincos_pair(b)
    return (a - exp(-a) * (a * c - b * s)) / (a * a + b * b)
end
"Series of ∫₀¹ e^{−at}(t − sin(bt)/b) dt for small b."
@inline function _int_sinc_deficit_series(a, b)
    M = moments(a)
    b2 = b * b
    return b2 * (M[4] / 6 - b2 * (M[6] / 120 - b2 * (M[8] / 5040 - b2 * M[10] / 362880)))
end
"Closed form of ∫₀¹ e^{−at} sin(bt)/b dt for b not small."
@inline function _int_sinc_closed(a, b)
    s, c = sincos_pair(b)
    return (1 - exp(-a) * (a * s / b + c)) / (a * a + b * b)
end

"∫₀¹ e^{−at}(1 − cos bt) dt."
@inline int_versin(a, b) = b < 0.1 ? _int_versin_series(a, b) : phi(a) - _int_cos_closed(a, b)

"∫₀¹ e^{−at} cos(bt) dt."
@inline int_cos(a, b) = b < 0.1 ? phi(a) - _int_versin_series(a, b) : _int_cos_closed(a, b)

"∫₀¹ e^{−at} sin(bt)/b dt."
@inline int_sinc(a, b) = b < 0.1 ? moment1(a) - _int_sinc_deficit_series(a, b) : _int_sinc_closed(a, b)

"∫₀¹ e^{−at}(t − sin(bt)/b) dt."
@inline int_sinc_deficit(a, b) = b < 0.1 ? _int_sinc_deficit_series(a, b) : moment1(a) - _int_sinc_closed(a, b)

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
Base.zero(::Type{RadiativeState{T}}) where {T} = RadiativeState(identity4(T), zero(SVector{4,T}))
@inline advance(st::RadiativeState, O, E) = RadiativeState(st.P * O, st.S + st.P * E)

"""
    WindingState{T}

A [`RadiativeState`](@ref) together with the half-orbit count of the ray: `n` is the number of
completed passages of the ray through the slab |z| < h about the midplane, each containing an
equatorial crossing (for h = 0 the number of equatorial crossings), counted from the observer
inward, so that emission at the current sample belongs to the sub-image of order `n`
(Johnson et al. 2020: n = 0 direct, n = 1 the first lensed ring, …); `zprev`, `inside` and
`crossed` are the counter's memory (`zprev` is NaN before the first sample). A consumer with
`nmax ≥ 0` truncates the ray once `n > nmax`, so that the sub-image of order n is the
difference of the images truncated at n and n − 1 (exactly: the segments in front are the
same, so their transmission is the same).
"""
struct WindingState{T}
    state::RadiativeState{T}
    zprev::T
    n::Int32
    inside::Bool
    crossed::Bool
end
Base.zero(::Type{WindingState{T}}) where {T} = WindingState(zero(RadiativeState{T}), T(NaN), Int32(0), false, false)

"""
    wind(w::WindingState, z, h) -> WindingState

Advance the passage counter of `w` to a sample at height `z` (the slab half-thickness `h`): a
passage is entered when |z| drops below h, marked as crossed when z changes sign inside it, and
counted when the ray leaves the slab after a crossing; a crossing that skips the slab in one
step (or any crossing when h = 0) is counted at once. The radiative state is unchanged.
"""
@inline function wind(w::WindingState{T}, z, h) where {T}
    zp = w.zprev
    isnan(zp) && return WindingState(w.state, T(z), w.n, abs(z) < h, false)
    inside_now = abs(z) < h
    crossing = zp * z < 0
    n = w.n; inside = w.inside; crossed = w.crossed
    if inside || inside_now
        crossed |= crossing
        if inside && !inside_now                     # leaving the slab
            crossed && (n += Int32(1))
            crossed = false
        elseif !inside && inside_now
            crossed = crossing                       # entering (a crossing on the entering step counts for this passage)
        end
        inside = inside_now
    elseif crossing
        n += Int32(1)                                # skipped through the slab (or h = 0)
    end
    return WindingState(w.state, T(z), n, inside, crossed)
end

"""
    rotate_to_screen(c::StokesCoefficients, χ) -> (j, α, ρ)

Coefficients of the field-aligned basis rotated by the angle `χ` into a common (screen) Stokes
basis: (Q, U) rotate by 2χ, I and V are unchanged. Returns the 4-vectors j, α and the 3-vector ρ
expected by `transfer_step`.
"""
@inline function rotate_to_screen(c::StokesCoefficients{T}, χ) where {T}
    s2, c2 = sincos_pair(2χ)
    j = SVector(c.jI, c.jQ * c2, c.jQ * s2, c.jV)
    α = SVector(c.αI, c.αQ * c2, c.αQ * s2, c.αV)
    ρ = SVector(c.ρQ * c2, c.ρQ * s2, c.ρV)
    return j, α, ρ
end
