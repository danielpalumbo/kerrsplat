# Gate 3 of the plan: the unpolarized GRRT test problems of Gold et al. (2020, ApJ 897, 148),
# section 3, as implemented in ipole's analytic model (model/analytic/model.c, tests/analytic):
# an axisymmetric emitting torus n(r, θ) = ρ_u exp(−½[(r/10)² + (h cos θ)²]) with power-law
# emissivity j = n (ν/ν_p)^(−α) and absorptivity α_ν = A n (ν/ν_p)^(−(2.5+α)), orbiting with the
# normal-observer velocity u_μ ∝ (−1, 0, 0, l), l = l₀ R^(3/2)/(1 + R), R = r sin θ, seen at
# θo = 60°, 230 GHz, 30 M across at 128², for M = 4.063e6 M☉ at 7778 pc. The published total
# fluxes (ipole's test uses a 2% tolerance) are the reference; the geodesics, the transfer
# integration and the unit conversions are all exercised.

using StaticArrays
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Transfer: MSUN, PC, JY

struct GoldModel{T}
    A::T
    α::T
    h::T
    l0::T
    νp::T
    ρu::T
end

"""
Gold et al. models 1–5 (ipole's parameters: spin, A, α, h, l₀). Model 2 is Schwarzschild; Krang's
analytic geodesics return NaN at a = 0 exactly (the polar roots involve η/a², see
docs/notes/upstream_issues.md), so it is run at a = 1e-3, where the samples are finite and the
effect on the flux is of order a.
"""
function gold_model(i)
    p = ((0.9, 0.0, -3.0, 0.0, 0.0), (1e-3, 0.0, -2.0, 0.0, 1.0), (0.9, 0.0, 0.0, 10 / 3, 1.0),
         (0.9, 1e5, 0.0, 10 / 3, 1.0), (0.9, 1e6, 0.0, 100 / 3, 1.0))[i]
    a, A, α, h, l0 = p
    return a, GoldModel(A, α, h, l0, 230e9, 3e-18)
end
const GOLD_PUBLISHED = (1.6465, 1.4360, 0.4418, 0.2710, 0.0255)   # Jy, Gold et al. (2020) Table 2

@inline function Transfer.unpolarized_coefficients(m::GoldModel{T}, pix, s, ν_obs) where {T}
    met = Krang.metric(pix)
    r, θ = s.r, s.θ
    nexp = (r / 10)^2 / 2 + (m.h * cos(θ))^2 / 2
    n = nexp < 200 ? m.ρu * exp(-nexp) : zero(T)
    R = r * sin(θ)
    l = m.l0 / (1 + R) * R^T(1.5)
    guu = Krang.metric_uu(met, r, θ)                                   # BL contravariant metric
    ubar = sqrt(-1 / (guu[1, 1] - 2 * guu[1, 4] * l + guu[4, 4] * l * l))
    ut = ubar * (-guu[1, 1] + guu[1, 4] * l)                           # u^μ from u_μ = ū(−1, 0, 0, l)
    uϕ = ubar * (-guu[1, 4] + guu[4, 4] * l)
    g = inv(ut - Krang.λ(pix) * uϕ)                                    # −p·u with p_μ = (−1, ·, ·, λ)
    νf = ν_obs / g
    j = n * (νf / m.νp)^(-m.α)
    α = m.A * n * (νf / m.νp)^(-(T(2.5) + m.α)) + T(1e-54)
    return j, α, g
end

"Total flux in Jy of a Gold model on the given backend."
function gold_flux(backend, i; res = 128, N = 2000, fov = 30.0, θo = deg2rad(60.0), M_solar = 4.063e6, D_pc = 7778.0)
    a, model = gold_model(i)
    Δα = fov / res
    axis = [-fov / 2 + (k - 0.5) * Δα for k in 1:res]                 # ipole's pixel centres
    αs = [axis[i] for i in 1:res, j in 1:res]; βs = [axis[j] for i in 1:res, j in 1:res]
    camera = Geodesics.Camera(vec(αs), vec(βs), (res, res))
    cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    L = gravitational_radius(M_solar)
    ν = 230e9
    out = KernelAbstractions.allocate(backend, UnpolarizedState{Float64}, npixels(cache))
    fused_march!(UnpolarizedTransport(model, ν, L), out, cache)
    states = Array(to_screen(cache, out))
    I = observed_intensity.(states, ν)
    return sum(I) * pixel_solid_angle(Δα, L, D_pc * PC) / JY, I
end

function test_gold2020(backend; res = 128, N = 2000, tol = 0.02, label = "")
    Geodesics.prepare_backend!(backend)
    @testset "Gold et al. (2020) GRRT test problems ($label)" begin
        for i in 1:5
            F, I = gold_flux(backend, i; res, N)
            err = F / GOLD_PUBLISHED[i] - 1
            @test abs(err) < tol
            @test all(isfinite, I) && all(>=(0), I)
            @info "Gold model $i: F = $(round(F, digits = 4)) Jy, published $(GOLD_PUBLISHED[i]) Jy, relative difference $(round(err, sigdigits = 3)) ($label, $(res)², N = $N)"
        end
    end
end
