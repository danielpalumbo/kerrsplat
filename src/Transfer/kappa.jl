# κ-distribution synchrotron coefficients: emissivities and absorptivities from the Pandya et al.
# (2016) fits as shipped with ipole (symphony kappa_fits.c, pure κ, ipole's signs), rotativities
# from the Marszewski et al. (2021) fits (their eqs. 51–54, κ = 3.5, 4, 4.5, 5) interpolated
# linearly in κ as ipole does inside [3.5, 5]. The absorptivities carry a hypergeometric factor
# ₂F₁(κ − 1/3, κ + 1, κ + 2/3, −κw) that depends only on the population (κ, w): it is evaluated
# on the host by `kappa_hypergeometric` and passed in, so that the coefficients themselves stay
# kernel-safe (Γ from Bessels.jl).
#
# Note: ipole's kappa_V zeroes the Stokes V emissivity through a guard `Nhigh < SMALL²` on a
# quantity that is always negative, so its κ j_V is identically zero; here the fit is applied as
# written (guard on |Nhigh|), which is what symphony's numerical evaluation supports.

using HypergeometricFunctions: _₂F₁

"The hypergeometric factor ₂F₁(κ − 1/3, κ + 1, κ + 2/3, −κ w) of the κ absorptivity fits (host side)."
kappa_hypergeometric(κ, w) = _₂F₁(κ - 1 / 3, κ + 1, κ + 2 / 3, -κ * w)

"""
    kappa_synchrotron(ne, κ, w, B, ν, θ, hyp) -> StokesCoefficients

Coefficients of a relativistic κ distribution of index `κ` and width `w` (Θe-like), with
`hyp = kappa_hypergeometric(κ, w)`; ρ_Q and ρ_V follow the Marszewski et al. (2021) fits for
κ ∈ [3.5, 5] (clamped outside).
"""
@inline function kappa_synchrotron(ne, κ, w, B, ν, θ, hyp)
    T = typeof(float(ne * κ * w * B * ν * θ))
    sinθ, cosθ = sincos_pair(θ)
    νc = EE * B / (2 * T(π) * ME * CL)
    νw = (w * κ)^2 * νc * sinθ
    X = ν / νw
    small = T(1e-40)
    # ---- emissivities (kappa_I, kappa_Q, kappa_V) ------------------------------------------------
    pref = ne * EE^2 * νc * sinθ / CL
    g43 = Bessels.gamma(κ - 4 / T(3)); g2 = Bessels.gamma(κ - 2)
    gq = Bessels.gamma(κ / 4 - 1 / T(3)); gp = Bessels.gamma(κ / 4 + 4 / T(3))
    NlowI = 4 * T(π) * g43 / (T(3)^(7 / 3) * g2)
    NhighI = (1 / T(4)) * T(3)^((κ - 1) / 2) * (κ - 2) * (κ - 1) * gq * gp + small
    xI = 3 * κ^(-3 / T(2))
    jI = pref * NlowI * X^(1 / 3) * (1 + X^(xI * (3κ - 4) / 6) * (NlowI / NhighI)^xI)^(-1 / xI)
    NlowQ = -(1 / T(2)) * 4 * T(π) * g43 / (T(3)^(7 / 3) * g2)
    NhighQ = -((4 / T(5))^2 + κ / 50) * (1 / T(4)) * T(3)^((κ - 1) / 2) * (κ - 2) * (κ - 1) * gq * gp + small
    xQ = (37 / T(10)) * κ^(-8 / T(5))
    jQ = -(pref * NlowQ * X^(1 / 3) * (1 + X^(xQ * (3κ - 4) / 6) * (NlowQ / NhighQ)^xQ)^(-1 / xQ))
    NlowV = -(3 / T(4))^2 * (sinθ^(-12 / T(5)) - 1)^(12 / T(25)) * (κ^(-66 / T(125)) / w) * X^(-7 / T(20)) * 4 * T(π) * g43 / (T(3)^(7 / 3) * g2)
    NhighV = -(7 / T(8))^2 * (sinθ^(-5 / T(2)) - 1)^(11 / T(25)) * (κ^(-11 / T(25)) / w) * X^(-1 / T(2)) * (1 / T(4)) * T(3)^((κ - 1) / 2) * (κ - 2) * (κ - 1) * gq * gp - small
    xV = 3 * κ^(-3 / T(2))
    jV = abs(NhighV) < small * small ? zero(T) :
         -(pref * NlowV * X^(1 / 3) * (1 + X^(xV * (3κ - 4) / 6) * (NlowV / NhighV)^xV)^(-1 / xV)) * sign(cosθ)
    # ---- absorptivities (kappa_I_abs, kappa_Q_abs, kappa_V_abs) ------------------------------------
    apref = ne * EE / (B * sinθ)
    g53 = Bessels.gamma(5 / T(3)); gk2 = Bessels.gamma(2 + κ / 2)
    cκ = (κ - 2) * (κ - 1) * κ
    NlowA = T(3)^(1 / 6) * (10 / T(41)) * (2 * T(π))^2 / (w * κ)^(16 / T(3) - κ) * cκ / (3κ - 1) * g53 * hyp
    NhighA = 2 * T(π)^(5 / 2) / 3 * cκ / (w * κ)^5 * (2 * gk2 / (2 + κ) - 1) * ((3 / κ)^(19 / T(4)) + 3 / T(5)) + small
    xA = (-7 / T(4) + 8 * κ / 5)^(-43 / T(50))
    aI = apref * NlowA * X^(-5 / T(3)) * (1 + X^(xA * (3κ - 1) / 6) * (NlowA / NhighA)^xA)^(-1 / xA)
    NlowQA = -(25 / T(48)) * T(3)^(1 / 6) * (10 / T(41)) * (2 * T(π))^2 / (w * κ)^(16 / T(3) - κ) * cκ / (3κ - 1) * g53 * hyp
    NhighQA = -(T(21)^2 * κ^(-144 / T(25)) + 11 / T(20)) * 2 * T(π)^(5 / 2) / 3 * cκ / (w * κ)^5 * (2 * gk2 / (2 + κ) - 1) + small
    xQA = (7 / T(5)) * κ^(-23 / T(20))
    aQ = -(apref * NlowQA * X^(-5 / T(3)) * (1 + X^(xQA * (3κ - 1) / 6) * (NlowQA / NhighQA)^xQA)^(-1 / xQA))
    NlowVA = -(77 / (100 * w)) * (sinθ^(-114 / T(50)) - 1)^(223 / T(500)) * X^(-7 / T(20)) * κ^(-7 / T(10)) * T(3)^(1 / 6) * (10 / T(41)) * (2 * T(π))^2 /
             (w * κ)^(16 / T(3) - κ) * cκ / (3κ - 1) * g53 * hyp
    NhighVA = -(143 / T(10) * w^(-116 / T(125))) * (sinθ^(-41 / T(20)) - 1)^(1 / T(2)) * (T(169) * κ^(-8) + 13 / T(2500) * κ - 263 / T(5000) + 47 / (200κ)) *
              X^(-1 / T(2)) * 2 * T(π)^(5 / 2) / 3 * cκ / (w * κ)^5 * (2 * gk2 / (2 + κ) - 1) + small
    xVA = (61 / T(50)) * κ^(-142 / T(125)) + 7 / T(1000)
    aV = -(apref * NlowVA * X^(-5 / T(3)) * (1 + X^(xVA * (3κ - 1) / 6) * (NlowVA / NhighVA)^xVA)^(-1 / xVA)) * sign(cosθ)
    # ---- rotativities (Marszewski+ 2021 eqs. 51–54), linear in κ between the four fits --------------
    ρQ, ρV = kappa_rotativities(ne, κ, w, B, ν, θ)
    return StokesCoefficients(jI, jQ, jV, aI, aQ, aV, ρQ, ρV)
end

# The four κ fits of ρ_Q and ρ_V (Marszewski+ 2021 eqs. 51–54; symphony kappa{35,4,45,5}_rho_{Q,V}).
@inline function kappa_rhoQ_fit(i, w, X)
    sw = sqrt(w); e5 = exp(-5w)
    if i == 1      # κ = 3.5
        return (17w - 3sw + 7sw * e5) * (1 - exp(-X^0.84 / 30) - sin(X / 10) * exp(-3 * X^0.471 / 2))
    elseif i == 2  # κ = 4
        return (46w / 3 - 5sw / 3 + 17sw / 3 * e5) * (1 - exp(-X^0.84 / 18) - sin(X / 6) * exp(-7 * X^0.5 / 4))
    elseif i == 3  # κ = 4.5
        return (14w - 13sw / 8 + 9sw / 2 * e5) * (1 - exp(-X^0.84 / 12) - sin(X / 4) * exp(-2 * X^0.525))
    else           # κ = 5
        return (25w / 2 - sw + 5sw * e5) * (1 - exp(-X^0.84 / 8) - sin(3X / 8) * exp(-9 * X^0.541 / 4))
    end
end
@inline function kappa_rhoV_fit(i, w, X)
    if i == 1
        return (w^2 + 2w + 1) / (25w^2 / 8 + 4w + 1) * (1 - 0.17 * log(1 + 0.447 * X^-0.5))
    elseif i == 2
        return (w^2 + 54w + 50) / (30w^2 / 11 + 134w + 50) * (1 - 0.17 * log(1 + 0.391 * X^-0.5))
    elseif i == 3
        return (w^2 + 43w + 38) / (7w^2 / 3 + 185w / 2 + 38) * (1 - 0.17 * log(1 + 0.348 * X^-0.5))
    else
        return (w + 13 / 14) / (2w + 13 / 14) * (1 - 0.17 * log(1 + 0.313 * X^-0.5))
    end
end

"""
    kappa_rotativities(ne, κ, w, B, ν, θ) -> (ρQ, ρV)

Faraday conversion and rotation of a κ distribution from the Marszewski et al. (2021) fits,
interpolated linearly in κ between the fitted values 3.5, 4, 4.5, 5 (clamped outside).
"""
@inline function kappa_rotativities(ne, κ, w, B, ν, θ)
    T = typeof(float(ne * κ * w * B * ν * θ))
    sinθ, cosθ = sincos_pair(θ)
    νc = EE * B / (2 * T(π) * ME * CL)
    X = ν / ((w * κ)^2 * νc * sinθ)
    κc = clamp(κ, T(3.5), T(5))
    i = κc < 4 ? 1 : (κc < 4.5 ? 2 : 3)
    κlo = T(3.5) + (i - 1) / 2; f = (κc - κlo) * 2
    fQ = (1 - f) * kappa_rhoQ_fit(i, w, X) + f * kappa_rhoQ_fit(i + 1, w, X)
    fV = (1 - f) * kappa_rhoV_fit(i, w, X) + f * kappa_rhoV_fit(i + 1, w, X)
    k0, k1, k2 = besselk012(inv(w))
    ρQ = -(ne * EE^2 * νc^2 * sinθ^2) / (ME * CL * ν^3) * fQ
    ρV = 2 * (ne * EE^2 * νc * cosθ) / (ME * CL * ν^2) * (k0 / (k2 + T(1e-40))) * fV
    return ρQ, ρV
end
