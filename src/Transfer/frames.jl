# Per-sample geometry for the transfer: the photon momentum in the fluid frame (redshift, pitch
# angle) and the angle by which the local field-aligned Stokes basis appears rotated on the screen
# after parallel transport, from the Walker–Penrose constant (Gelles et al. 2021, PRD 104, 044060;
# the same construction as Krang's `synchrotronPolarization`, restricted to what the transfer needs
# and written for kernels).
#
# Frames. Krang's ZAMO basis has the spatial axes (r̂, φ̂, −θ̂), a right-handed triad, and its
# BL ↔ ZAMO Jacobians are used unchanged. The fluid frame is the boost of the ZAMO frame by the
# fluid's ZAMO 3-velocity, given here as the spatial part ũ = γ β⃗ of the 4-velocity in ZAMO
# components (unconstrained parameters; γ = √(1 + ũ²)). The magnetic field is given in the fluid
# frame. The local Stokes basis is ipole's plasma tetrad: e₁ = k̂ × B̂ normalized (the direction of
# the synchrotron electric vector, perpendicular to the projected field: the Q axis),
# e₂ = k̂ × e₁ (so that (e₁, e₂, k̂) is right-handed, with k̂ the propagation direction).
#
# Screen. e₁ is parallel-transported to the observer through the Walker–Penrose constant and lands
# on the screen along (e_α, e_β). The screen Stokes basis is (north, east) = (+β, −α), Krang's EVPA
# convention (`Krang.evpa`), which with the propagation direction toward the observer forms a
# right-handed triad: with it the Faraday rotation and conversion sense agrees with ipole's, and with
# the opposite choice U flips sign at every frequency and the Faraday-affected Q, U, V at 230 GHz
# disagree (test/test_polarized.jl, RIAF against ipole). So χ = atan(−e_α, e_β) is the position
# angle of the local Q axis on the screen, and the local coefficients rotate into the screen basis
# by 2χ (`rotate_to_screen`).

"""
    boost_zamo_to_fluid(ũ) -> SMatrix{4,4}

Lorentz boost from ZAMO components (t, r̂, φ̂, −θ̂) to the frame of a fluid element whose
4-velocity has ZAMO spatial components `ũ = γβ⃗`; its inverse is `boost_zamo_to_fluid(-ũ)`.
Reproduces Krang's `jac_fluid_u_zamo_d(β, θ, φ)` for ũ = γβ (sin θ cos φ, sin θ sin φ, cos θ).
"""
@inline function boost_zamo_to_fluid(ũ::SVector{3,T}) where {T}
    u2 = ũ[1] * ũ[1] + ũ[2] * ũ[2] + ũ[3] * ũ[3]
    γ = sqrt(1 + u2)
    q = inv(γ + 1)
    return @SMatrix [
        γ -ũ[1] -ũ[2] -ũ[3]
        -ũ[1] 1+ũ[1]*ũ[1]*q ũ[1]*ũ[2]*q ũ[1]*ũ[3]*q
        -ũ[2] ũ[1]*ũ[2]*q 1+ũ[2]*ũ[2]*q ũ[2]*ũ[3]*q
        -ũ[3] ũ[1]*ũ[3]*q ũ[2]*ũ[3]*q 1+ũ[3]*ũ[3]*q
    ]
end

"""
    LocalFrame(g, cosθB, χ)

What the transfer needs of a sample seen from one fluid element: the redshift factor
`g = ν_obs/ν_fluid = 1/(−p·u)`, the cosine of the pitch angle between the photon and the field in
the fluid frame, and the position angle `χ` of the local Q axis on the screen.
"""
struct LocalFrame{T}
    g::T
    cosθB::T
    χ::T
end

"""
    walker_penrose(met, r, θ, p_u, f_u) -> (κ₁, κ₂)

The Walker–Penrose constant κ = κ₁ + iκ₂ of a photon with BL momentum `p_u` and polarization
vector `f_u` (both contravariant, (t, r, θ, ϕ) order), Krang's form of eq. 6 of arXiv:2001.08750.
"""
@inline function walker_penrose(met::Krang.Kerr{T}, r, θ, p_u::SVector{4}, f_u::SVector{4}) where {T}
    a = met.spin
    sθ, cθ = sincos_pair(θ)
    pt, pr, pθ, pϕ = p_u
    ft, fr, fθ, fϕ = f_u
    A = pt * fr - pr * ft + a * sθ * sθ * (pr * fϕ - pϕ * fr)
    B = ((r * r + a * a) * (pϕ * fθ - pθ * fϕ) - a * (pt * fθ - pθ * ft)) * sθ
    return A * r - B * a * cθ, -(A * a * cθ + B * r)
end

"""
    screen_direction(met, κ₁, κ₂, θo, α, β) -> (e_α, e_β)

Direction on the screen of the polarization vector with Walker–Penrose constant κ (unnormalized),
eq. 31 of Gelles et al. (2021).
"""
@inline function screen_direction(met::Krang.Kerr, κ1, κ2, θo, α, β)
    μ = -(α + met.spin * sin(θo))
    return β * κ2 - μ * κ1, β * κ1 + μ * κ2
end

"""
    local_frame(met, r, θ, η, λ, νr, νθ, α, β, θo, ũ, B) -> LocalFrame

Redshift, pitch angle and screen rotation angle for the photon of screen coordinates (α, β)
(constants η, λ, momentum signs νr, νθ) at the BL point (r, θ), seen from a fluid element of
ZAMO 3-velocity `ũ` with fluid-frame magnetic field `B` (any magnitude). Where the photon travels
along the field the Q axis is undefined and χ = 0 is returned (the polarized coefficients vanish
there, and ρ_V acts the same in every basis).
"""
@inline function local_frame(met::Krang.Kerr{T}, r, θ, η, λ, νr::Bool, νθ::Bool, α, β, θo, ũ::SVector{3}, B::SVector{3}) where {T}
    p_d = Krang.p_bl_d(met, r, θ, η, λ, νr, νθ)
    p_u = Krang.metric_uu(met, r, θ) * p_d
    p_zamo = Krang.jac_zamo_u_bl_d(met, r, θ) * p_u
    Λ = boost_zamo_to_fluid(ũ)
    p_f = Λ * p_zamo
    g = inv(p_f[1])
    k = SVector(p_f[2], p_f[3], p_f[4]) * g              # unit propagation direction in the fluid frame
    Bn = sqrt(B[1] * B[1] + B[2] * B[2] + B[3] * B[3])
    Bhat = Bn > 0 ? B / Bn : SVector(zero(T), zero(T), one(T))
    cosθB = k[1] * Bhat[1] + k[2] * Bhat[2] + k[3] * Bhat[3]
    f = SVector(k[2] * Bhat[3] - k[3] * Bhat[2], k[3] * Bhat[1] - k[1] * Bhat[3], k[1] * Bhat[2] - k[2] * Bhat[1])
    fn = sqrt(f[1] * f[1] + f[2] * f[2] + f[3] * f[3])
    fn > 0 || return LocalFrame(g, cosθB, zero(T))
    f_fluid = SVector(zero(T), f[1] / fn, f[2] / fn, f[3] / fn)
    f_zamo = boost_zamo_to_fluid(-ũ) * f_fluid
    f_bl = Krang.jac_bl_u_zamo_d(met, r, θ) * f_zamo
    κ1, κ2 = walker_penrose(met, r, θ, p_u, f_bl)
    eα, eβ = screen_direction(met, κ1, κ2, θo, α, β)
    return LocalFrame(g, cosθB, atan(-eα, eβ))
end

"Convenience for the fused-march consumer: the frame from a pixel and a sample."
@inline function local_frame(pix, s::GeodesicSample, ũ::SVector{3}, B::SVector{3})
    α, β = Krang.screen_coordinate(pix)
    return local_frame(Krang.metric(pix), s.r, s.θ, Krang.η(pix), Krang.λ(pix), s.νr, s.νθ, α, β, Krang.inclination(pix), ũ, B)
end
