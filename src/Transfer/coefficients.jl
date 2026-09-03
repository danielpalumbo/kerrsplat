# Synchrotron transfer coefficients from fitting formulae, in the fluid frame.
#
# Inputs: electron density ne [cm⁻³], dimensionless temperature Θe, field strength B [G],
# fluid-frame frequency ν [Hz], and the pitch angle θ between the photon direction and the
# field in the fluid frame. Outputs are in cgs: emissivities j_S [erg s⁻¹ cm⁻³ Hz⁻¹ sr⁻¹],
# absorptivities α_S [cm⁻¹] and rotativities ρ_S [cm⁻¹], in the basis described in Transfer.jl.
#
# The formulae are transcribed from the symphony sources distributed with ipole
# (src/symphony/maxwell_juettner_fits.c, power_law_fits.c) including ipole's sign convention;
# validation/symphony/ generates reference tables from those sources and test/test_coefficients.jl
# compares against them.

"""
    StokesCoefficients(jI, jQ, jV, αI, αQ, αV, ρQ, ρV)

Transfer coefficients in the field-aligned Stokes basis (Q axis perpendicular to the projected
field), where the U components vanish.
"""
struct StokesCoefficients{T}
    jI::T
    jQ::T
    jV::T
    αI::T
    αQ::T
    αV::T
    ρQ::T
    ρV::T
end

Base.zero(::Type{StokesCoefficients{T}}) where {T} = StokesCoefficients(zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T))
Base.:+(a::StokesCoefficients, b::StokesCoefficients) =
    StokesCoefficients(a.jI + b.jI, a.jQ + b.jQ, a.jV + b.jV, a.αI + b.αI, a.αQ + b.αQ, a.αV + b.αV, a.ρQ + b.ρQ, a.ρV + b.ρV)
Base.:*(s::Number, c::StokesCoefficients) = StokesCoefficients(s * c.jI, s * c.jQ, s * c.jV, s * c.αI, s * c.αQ, s * c.αV, s * c.ρQ, s * c.ρV)

"""
    planck_invariant(ν, Θe)

The Planck function divided by ν³, B_ν/ν³ = (2h/c²)/(exp(hν/kT) − 1) with kT = Θe mₑc², with the
same small-argument expansion as ipole's `Bnu_inv`.
"""
@inline function planck_invariant(ν, Θe)
    x = HPL * ν / (ME * CL * CL * Θe)
    if x < 2e-3
        return (2 * HPL / (CL * CL)) / (x / 24 * (24 + x * (12 + x * (4 + x))))
    else
        return (2 * HPL / (CL * CL)) / (exp(x) - 1)
    end
end

"The Planck function B_ν for the dimensionless temperature Θe."
@inline planck(ν, Θe) = planck_invariant(ν, Θe) * ν^3

"Modified Bessel functions K₀, K₁, K₂ at x (K₂ by the recurrence K₂ = K₀ + 2K₁/x)."
@inline function besselk012(x)
    k0 = Bessels.besselk0(x)
    k1 = Bessels.besselk1(x)
    return k0, k1, k0 + 2 * k1 / x
end

# Dexter (2016) thermal fits, appendix A: the functions of x = ν/ν_s.
@inline dexter_II(x) = 2.5651 * (1 + 1.92 * x^(-1 / 3) + 0.9977 * x^(-2 / 3)) * exp(-1.8899 * x^(1 / 3))
@inline dexter_IQ(x) = 2.5651 * (1 + 0.93193 * x^(-1 / 3) + 0.499873 * x^(-2 / 3)) * exp(-1.8899 * x^(1 / 3))
@inline dexter_IV(x) = (1.81384 / x + 3.42319 * x^(-2 / 3) + 0.0292545 * x^(-1 / 2) + 2.03773 * x^(-1 / 3)) * exp(-1.8899 * x^(1 / 3))

"""
    thermal_synchrotron(ne, Θe, B, ν, θ; dexter_rhoV = false) -> StokesCoefficients

Thermal (Maxwell–Jüttner) synchrotron coefficients as in ipole's default prescription:
emissivities from Dexter (2016), absorptivities by Kirchhoff's law α_S = j_S/B_ν, Faraday
conversion ρ_Q from Dexter (2016, eqs. B4–B13) and Faraday rotation ρ_V from Shcherbakov (2008)
(or Dexter's fit with `dexter_rhoV = true`). Along the field (sin θ = 0) emission and absorption
vanish and only ρ_V survives, as in ipole.
"""
@inline function thermal_synchrotron(ne, Θe, B, ν, θ; dexter_rhoV::Bool = false)
    T = typeof(float(ne * Θe * B * ν * θ))
    sinθ, cosθ = sincos(θ)
    # rotativities (Dexter 2016 appendix B; ipole maxwell_juettner_rho_Q / rho_V)
    ω0 = EE * B / (ME * CL)
    ωp2 = 4 * T(π) * ne * EE^2 / ME
    k0, k1, k2 = besselk012(inv(Θe))
    x = Θe * sqrt(sqrt(T(2)) * sinθ * (1e3 * ω0 / (2 * T(π) * ν)))
    extraterm = (0.011 * exp(-x / 47.2) - T(2)^(-1 / 3) / T(3)^(23 / 6) * T(π) * 1e4 * (x + 1e-16)^(-8 / 3)) *
                (0.5 + 0.5 * tanh((log(x) - log(T(120))) / 0.1))
    jffunc = 2.011 * exp(-x^1.035 / 4.7) - cos(x / 2) * exp(-x^1.2 / 2.73) - 0.011 * exp(-x / 47.2) + extraterm
    kratio = k2 > 0 ? k1 / k2 : one(T)
    eps11m22 = jffunc * ωp2 * ω0^2 / (2 * T(π) * ν)^4 * (kratio + 6 * Θe) * sinθ^2
    ρQ = 2 * T(π) * ν / (2 * CL) * eps11m22
    if dexter_rhoV && k2 > 0
        fit_factor = (k0 - 0.43793091 * log(1 + 0.00185777 * x^1.50316886)) / k2
    else
        fit_factor = (k2 > 0 ? k0 / k2 : one(T)) * (1 - 0.11 * log(1 + 0.035 * x))
    end
    ρV = 2 * T(π) * ν / CL * (ωp2 * ω0 / (2 * T(π) * ν)^3 * fit_factor * cosθ)
    if !(sinθ > 0)
        return StokesCoefficients(zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), zero(T), ρV)
    end
    # emissivities (Dexter 2016; ipole maxwell_juettner_dexter_*), ipole's sign for Q
    νs = 3 * EE * B * sinθ / (4 * T(π) * ME * CL) * Θe^2 + 1
    xs = ν / νs
    pref = ne * EE^2 * ν / (2 * sqrt(T(3)) * CL * Θe^2)
    jI = pref * dexter_II(xs)
    jQ = pref * dexter_IQ(xs)
    jV = 2 * ne * EE^2 * ν * cosθ / sinθ / (3 * sqrt(T(3)) * CL * Θe^3) * dexter_IV(xs)
    # absorptivities by Kirchhoff's law
    Bν = planck(ν, Θe)
    inv_B = Bν > 0 ? inv(Bν) : zero(T)
    return StokesCoefficients(jI, jQ, jV, jI * inv_B, jQ * inv_B, jV * inv_B, ρQ, ρV)
end

"""
    thermal_synchrotron_pandya(ne, Θe, B, ν, θ) -> (jI, jQ, jV)

Thermal emissivities from the Pandya et al. (2016) fits (symphony `maxwell_juettner_I/Q/V`,
ipole's signs), the alternative to Dexter's fits.
"""
@inline function thermal_synchrotron_pandya(ne, Θe, B, ν, θ)
    T = typeof(float(ne * Θe * B * ν * θ))
    sinθ = sin(θ)
    νc = EE * B / (2 * T(π) * ME * CL)
    νs = (2 / T(9)) * νc * sinθ * Θe^2
    X = ν / νs
    pref = ne * EE^2 * νc / CL
    term1 = sqrt(T(2)) * T(π) / 27 * sinθ
    jI = pref * term1 * (X^(1 / 2) + T(2)^(11 / 12) * X^(1 / 6))^2 * exp(-X^(1 / 3))
    t2 = (7 * Θe^(24 / 25) + 35) / (10 * Θe^(24 / 25) + 75)
    jQ = pref * term1 * (X^(1 / 2) + t2 * T(2)^(11 / 12) * X^(1 / 6))^2 * exp(-X^(1 / 3))
    tv1 = (37 - 87 * sin(θ - 28 / T(25))) / (100 * (Θe + 1))
    tv2 = (1 + (Θe^(3 / 5) / 25 + 7 / T(10)) * X^(9 / 25))^(5 / 3)
    jV = pref * tv1 * tv2 * exp(-X^(1 / 3))
    return jI, jQ, jV
end

"""
    thermal_emissivity_leung(ne, Θe, B, ν, θ)

Unpolarized thermal emissivity j_I of Leung et al. (2011), which ipole uses for unpolarized
transport (symphony `maxwell_juettner_leung_I`).
"""
@inline function thermal_emissivity_leung(ne, Θe, B, ν, θ)
    T = typeof(float(ne * Θe * B * ν * θ))
    sinθ = sin(θ)
    _, _, k2 = besselk012(inv(Θe))
    K2 = max(k2, T(1e-40))
    νc = EE * B / (2 * T(π) * ME * CL)
    νs = (2 / T(9)) * νc * Θe^2 * sinθ
    ν > 1e12 * νs && return zero(T)
    x = ν / νs
    f = (x^(1 / 2) + T(2)^(11 / 12) * x^(1 / 6))^2
    return (sqrt(T(2)) * T(π) * EE^2 * ne * νs / (3 * CL * K2)) * f * exp(-x^(1 / 3))
end

"""
    powerlaw_synchrotron(ne, p, γmin, γmax, B, ν, θ) -> StokesCoefficients

Power-law synchrotron emissivities and absorptivities from the Pandya et al. (2016) fits
(symphony `power_law_*`, ipole's signs), for electrons n(γ) ∝ γ^(−p) on [γmin, γmax]. Symphony
has no power-law rotativities, so ρ_Q = ρ_V = 0 here.
"""
@inline function powerlaw_synchrotron(ne, p, γmin, γmax, B, ν, θ)
    T = typeof(float(ne * p * γmin * γmax * B * ν * θ))
    sinθ, cosθ = sincos(θ)
    νc = EE * B / (2 * T(π) * ME * CL)
    gspan = γmin^(1 - p) - γmax^(1 - p)
    # emissivities
    pref = ne * EE^2 * νc / CL
    jI = pref * (T(3)^(p / 2) * (p - 1) * sinθ) / (2 * (p + 1) * gspan) *
         Bessels.gamma((3p - 1) / 12) * Bessels.gamma((3p + 19) / 12) * (ν / (νc * sinθ))^(-(p - 1) / 2)
    jQ = (p + 1) / (p + 7 / T(3)) * jI
    jV = (171 / T(250)) * p^(49 / 100) * cosθ / sinθ * (ν / (3 * νc * sinθ))^(-1 / 2) * jI
    # absorptivities
    apref = ne * EE^2 / (ν * ME * CL)
    aI = apref * T(3)^((p + 1) / 2) * (p - 1) / (4 * gspan) *
         Bessels.gamma((3p + 2) / 12) * Bessels.gamma((3p + 22) / 12) * (ν / (νc * sinθ))^(-(p + 2) / 2)
    aQ = aI * ((17 / T(500)) * p - 43 / T(1250))^(43 / 500)
    term6 = ((31 / T(10)) * sinθ^(-48 / 25) - 31 / T(10))^(64 / 125)
    aV = aI * ((71 / T(100)) * p + 22 / T(625))^(197 / 500) * term6 * (ν / (νc * sinθ))^(-1 / 2) * sign(cosθ)
    return StokesCoefficients(jI, jQ, jV, aI, aQ, aV, zero(T), zero(T))
end

"""
    cap_polarization(c::StokesCoefficients, fmax = 0.99)

Scale the polarized emissivities and absorptivities so that √(j_Q² + j_V²) ≤ fmax j_I and likewise
for α, as ipole does (`max_pol_frac_e/a`): the analytic transfer step assumes α_P < α_I.
"""
@inline function cap_polarization(c::StokesCoefficients{T}, fmax = T(0.99)) where {T}
    jP = sqrt(c.jQ^2 + c.jV^2)
    fe = (jP > 0 && c.jI < jP / fmax) ? c.jI / jP * fmax : one(T)
    aP = sqrt(c.αQ^2 + c.αV^2)
    fa = (aP > 0 && c.αI < aP / fmax) ? c.αI / aP * fmax : one(T)
    return StokesCoefficients(c.jI, fe * c.jQ, fe * c.jV, c.αI, fa * c.αQ, fa * c.αV, c.ρQ, c.ρV)
end

"""
    invariants(c::StokesCoefficients, ν)

The Lorentz-invariant forms j_S/ν², ν α_S and ν ρ_S used by the transfer step.
"""
@inline function invariants(c::StokesCoefficients, ν)
    iν2 = inv(ν * ν)
    return StokesCoefficients(c.jI * iν2, c.jQ * iν2, c.jV * iν2, ν * c.αI, ν * c.αQ, ν * c.αV, ν * c.ρQ, ν * c.ρV)
end
