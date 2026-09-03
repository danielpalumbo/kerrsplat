# ipole's RIAF model (model/riaf/model.c) as a KerrSplat fluid model: a thick disk with power-law
# density and temperature, a toroidal field in equipartition-like scaling, Keplerian rotation
# outside the ISCO and the conserved-quantity plunge inside, thermal synchrotron with ipole's
# `emission_type 1` (Pandya emissivities, Kirchhoff absorptivities, Dexter ρ_Q, Shcherbakov ρ_V).
# Used by test/test_polarized.jl to compare full Stokes images with ipole's.

using StaticArrays
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Transfer: MP, KBOL, ME, CL, MSUN, PC, GNEWT

struct RIAFModel{T}
    nth0::T
    Te0::T
    disk_h::T
    pow_nth::T
    pow_T::T
    Ne_unit::T
    Te_unit::T
    ε::T
    r_isco::T
    rout::T
    pandya::Bool
end

function isco_radius(a)
    z1 = 1 + cbrt(1 - a^2) * (cbrt(1 + a) + cbrt(1 - a))
    z2 = sqrt(3a^2 + z1^2)
    return 3 + z2 - copysign(sqrt((3 - z1) * (3 + z1 + 2z2)), a)
end

"ipole's example RIAF (model/riaf/example.par)."
function riaf_example(a = 0.9375; pandya = true)
    return RIAFModel(1.0, 1.0, 0.1, -1.1, -0.84, 3e7, 3e11, 0.1, isco_radius(a), 100.0, pandya)
end

Transfer.nelements(::RIAFModel) = 1

@inline function Transfer.element(m::RIAFModel{T}, i, pix, s, ν_obs) where {T}
    met = Krang.metric(pix)
    a = met.spin
    r, θ = s.r, s.θ
    rh = Krang.horizon(met)
    zero_c = zero(StokesCoefficients{T})
    (r > rh + T(0.1) && r < m.rout) || return zero_c, LocalFrame(one(T), zero(T), zero(T))
    sθ, cθ = sincos(θ)
    # density, temperature, field strength (ipole get_model_ne/thetae/b)
    zc = r * cθ; rc = r * sθ
    n = m.nth0 * exp(-zc * zc / 2 / rc / rc / m.disk_h / m.disk_h) * r^m.pow_nth * m.Ne_unit
    Θe = m.Te0 * r^m.pow_T * m.Te_unit * KBOL / (ME * CL * CL)
    bmag = sqrt(8 * T(π) * m.ε * n * MP * CL * CL / 6 / r)
    bmag = bmag == 0 ? T(1e-6) : bmag
    # BL four-velocity (ipole get_model_fourv, keplerian_factor = 1, infall_factor = 0)
    gdd = Krang.metric_dd(met, r, θ)
    guu = Krang.metric_uu(met, r, θ)
    if r < m.r_isco
        ωisco = 1 / (m.r_isco^T(1.5) + a)
        gi = Krang.metric_dd(met, m.r_isco, θ)
        uisco = SVector(one(T), zero(T), zero(T), ωisco)
        nrm = sqrt(-(uisco' * gi * uisco))
        uisco = uisco / nrm
        ud = gi * uisco
        e, l = ud[1], ud[4]
        Kcon = guu[1, 1] * e * e + 2 * guu[1, 4] * e * l + guu[4, 4] * l * l
        urk = -sqrt(max(zero(T), -(1 + Kcon) / guu[2, 2]))
        utmp = guu * SVector(e, urk, zero(T), l)
        ω = utmp[4] / utmp[1]
        ur = utmp[2]
    else
        ω = 1 / (r^T(1.5) + a)
        ur = zero(T)
    end
    K = gdd[1, 1] + 2 * ω * gdd[1, 4] + ω * ω * gdd[4, 4]
    ut = sqrt(max(zero(T), -(1 + ur * ur * gdd[2, 2]) / K))
    u = SVector(ut, ur, zero(T), ω * ut)
    # toroidal field projected orthogonal to u, normalized to bmag (ipole)
    Bc = SVector(zero(T), zero(T), zero(T), one(T))
    BdotU = (gdd * Bc)' * u
    b = Bc + BdotU * u
    b = b * (bmag / sqrt(b' * gdd * b))
    # to the ZAMO and fluid frames
    J = Krang.jac_zamo_u_bl_d(met, r, θ)
    uz = J * u
    ũ = SVector(uz[2], uz[3], uz[4])
    bz = boost_zamo_to_fluid(ũ) * (J * b)
    Bf = SVector(bz[2], bz[3], bz[4])
    α, β = Krang.screen_coordinate(pix)
    fr = local_frame(met, r, θ, Krang.η(pix), Krang.λ(pix), s.νr, s.νθ, α, β, Krang.inclination(pix), ũ, Bf)
    νf = ν_obs / fr.g
    θB = acos(clamp(fr.cosθB, -one(T), one(T)))
    c = thermal_synchrotron(n, Θe, bmag, νf, θB; pandya = m.pandya)
    return c, fr
end
