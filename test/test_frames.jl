# Gate for the per-sample frames: the boost against Krang's fluid-frame Jacobian, and the redshift
# and the screen polarization angle against Krang's `synchrotronPolarization` (the Walker–Penrose
# transport used in Krang's polarized images) on random rays, samples, velocities and fields; the
# pitch angle against an explicit construction; and the same routine inside a kernel on the backend.

using ForwardDiff
using StaticArrays
using LinearAlgebra
using KerrSplat.Geodesics
using KerrSplat.Transfer

@kernel function frame_kernel!(out, @Const(rs), @Const(θs), @Const(ηs), @Const(λs), @Const(νrs), @Const(νθs), @Const(αs), @Const(βs), θo, a, @Const(us), @Const(Bs))
    i = @index(Global)
    @inbounds begin
        met = Krang.Kerr(a)
        fr = local_frame(met, rs[i], θs[i], ηs[i], λs[i], νrs[i], νθs[i], αs[i], βs[i], θo, SVector(us[i, 1], us[i, 2], us[i, 3]), SVector(Bs[i, 1], Bs[i, 2], Bs[i, 3]))
        out[i, 1] = fr.g; out[i, 2] = fr.cosθB; out[i, 3] = fr.χ
    end
end

function test_frames(backend; N = 400, label = "")
    Geodesics.prepare_backend!(backend)
    @testset "local frames vs Krang ($label)" begin
        rng = Random.MersenneTwister(11)
        for a in (0.0, 0.5, 0.94)
            met = Krang.Kerr(a)
            for _ in 1:20   # boost vs Krang's angle-parameterized Jacobian
                βv = 0.95 * rand(rng); θz = π * rand(rng); φz = 2π * rand(rng)
                γ = inv(sqrt(1 - βv^2))
                ũ = γ * βv * SVector(sin(θz) * cos(φz), sin(θz) * sin(φz), cos(θz))
                @test boost_zamo_to_fluid(ũ) ≈ Krang.jac_fluid_u_zamo_d(met, βv, θz, φz) rtol = 1e-13
                @test boost_zamo_to_fluid(ũ) * boost_zamo_to_fluid(-ũ) ≈ I rtol = 1e-12
            end
        end
        a = 0.94; θo = deg2rad(60.0); met = Krang.Kerr(a); rh = Krang.horizon(met)
        worst_g = 0.0; worst_χ = 0.0; worst_cos = 0.0
        inputs = Float64[]
        n = 0
        while n < N
            α = 12 * (2rand(rng) - 1); β = 12 * (2rand(rng) - 1)
            pix = Krang.SlowLightIntensityPixel(met, α, β, θo)
            τ = rand(rng) * Krang.total_mino_time(pix)
            t, r, θ, ϕ, νr, νθ, ok = Krang.emission_coordinates(pix, τ)
            (ok && isfinite(r) && r > 1.05rh && r < 40) || continue
            βv = 0.9 * rand(rng); θz = π * rand(rng); φz = 2π * rand(rng)
            γ = inv(sqrt(1 - βv^2))
            ũ = γ * βv * SVector(sin(θz) * cos(φz), sin(θz) * sin(φz), cos(θz))
            B = SVector{3}(randn(rng, 3)) * 10.0^(2rand(rng) - 1)
            fr = local_frame(met, r, θ, Krang.η(pix), Krang.λ(pix), νr, νθ, α, β, θo, ũ, B)
            eα, eβ, g_k, _ = Krang.synchrotronPolarization(met, α, β, r, θ, θo, B / norm(B), SVector(βv, θz, φz), νr, νθ)
            χ_k = Krang.evpa(eα, eβ)                 # atan(−e_α, e_β): the screen convention of frames.jl
            dχ = abs(rem2pi(fr.χ - χ_k, RoundNearest))
            worst_g = max(worst_g, abs(fr.g / g_k - 1)); worst_χ = max(worst_χ, dχ)
            # pitch angle from an explicit construction with Krang's Jacobians
            p_u = Krang.metric_uu(met, r, θ) * Krang.p_bl_d(met, r, θ, Krang.η(pix), Krang.λ(pix), νr, νθ)
            p_f = Krang.jac_fluid_u_zamo_d(met, βv, θz, φz) * (Krang.jac_zamo_u_bl_d(met, r, θ) * p_u)
            k = SVector(p_f[2], p_f[3], p_f[4]) / p_f[1]
            worst_cos = max(worst_cos, abs(fr.cosθB - dot(k, B) / norm(B)))
            @test abs(norm(k) - 1) < 1e-10        # the photon is null in the fluid frame
            append!(inputs, (r, θ, Krang.η(pix), Krang.λ(pix), νr, νθ, α, β, ũ..., B..., fr.g, fr.cosθB, fr.χ))
            n += 1
        end
        @test worst_g < 1e-13
        @test worst_χ < 1e-11
        @test worst_cos < 1e-13
        @info "local frames vs Krang.synchrotronPolarization ($N samples): worst errors" g = worst_g χ = worst_χ cosθB = worst_cos
        # the same routine in a kernel on the backend
        M = reshape(inputs, 17, N)
        col(k) = adapt_to(backend, M[k, :])
        out = KernelAbstractions.zeros(backend, Float64, N, 3)
        us = adapt_to(backend, permutedims(M[9:11, :])); Bs = adapt_to(backend, permutedims(M[12:14, :]))
        frame_kernel!(backend, 64)(out, col(1), col(2), col(3), col(4), adapt_to(backend, Vector{Bool}(M[5, :] .!= 0)), adapt_to(backend, Vector{Bool}(M[6, :] .!= 0)), col(7), col(8), θo, a, us, Bs; ndrange = N)
        KernelAbstractions.synchronize(backend)
        O = Array(out)
        @test maximum(abs.(O[:, 1] .- M[15, :]) ./ abs.(M[15, :])) < 1e-13
        @test maximum(abs.(O[:, 2] .- M[16, :])) < 1e-13
        @test maximum(abs.(rem2pi.(O[:, 3] .- M[17, :], RoundNearest))) < 1e-12
    end
end

"""
    test_turning_point_frame()

`Transfer.momentum_bl_d` equals Krang's `p_bl_d` away from the turning points, and at a radial
turning point (the root of the radial potential, found by bisection on a ray with η above the
photon-orbit value) its partials with respect to the spin are finite where Krang's are NaN,
and so are the local frame's redshift, pitch angle and polarization angle.
"""
function test_turning_point_frame()
    met = Krang.Kerr(0.4)
    @testset "momentum at a radial turning point" begin
        rng = Random.MersenneTwister(3)
        for _ in 1:20
            r = 2.5 + 8 * rand(rng); θ = 0.2 + 2.7 * rand(rng); η = 10 + 30 * rand(rng); λ = -5 + 10 * rand(rng)
            Krang.r_potential(met, η, λ, r) > 0 && Krang.θ_potential(met, η, λ, θ) > 0 || continue
            @test Transfer.momentum_bl_d(met, r, θ, η, λ, true, false) == Krang.p_bl_d(met, r, θ, η, λ, true, false)
        end
        # a ray with a radial turning point: η = 30, λ = 0 (the photon orbit needs η ≈ 27 at a = 0); the root outside 3 M
        η = 30.0; λ = 0.0
        f(r) = Krang.r_potential(met, η, λ, r)
        lo, hi = 3.0, 12.0
        @test f(lo) < 0 < f(hi)
        for _ in 1:200
            mid = (lo + hi) / 2
            f(mid) < 0 ? (lo = mid) : (hi = mid)
        end
        r0 = hi
        @test abs(f(r0)) < 1e-9
        D = ForwardDiff.Dual{:a}
        metd = Krang.Kerr(D(0.4, 1.0))
        # a hair inside the turning point the potential is negative: Krang's √max(0, R) is a constant zero whose
        # square-root rule gives NaN partials, safe_sqrt gives zero with zero partials; the values agree
        rin = r0 - 1e-6
        @test f(rin) < 0
        pk = Krang.p_bl_d(metd, D(rin, 0.0), D(1.2, 0.0), D(η, 0.0), D(λ, 0.0), true, false)
        pm = Transfer.momentum_bl_d(metd, D(rin, 0.0), D(1.2, 0.0), D(η, 0.0), D(λ, 0.0), true, false)
        @test !isfinite(ForwardDiff.partials(pk[2], 1))
        @test all(x -> isfinite(ForwardDiff.partials(x, 1)), pm)
        @test all(abs.(ForwardDiff.value.(pm) .- ForwardDiff.value.(pk)) .< 1e-12) && ForwardDiff.value(pm[2]) == 0
        # at the root itself both are finite and equal
        pk0 = Krang.p_bl_d(metd, D(r0, 0.0), D(1.2, 0.0), D(η, 0.0), D(λ, 0.0), true, false)
        pm0 = Transfer.momentum_bl_d(metd, D(r0, 0.0), D(1.2, 0.0), D(η, 0.0), D(λ, 0.0), true, false)
        @test all(abs.(ForwardDiff.value.(pm0) .- ForwardDiff.value.(pk0)) .< 1e-12) && all(x -> isfinite(ForwardDiff.partials(x, 1)), pm0)
        fr = Transfer.local_frame(metd, D(rin, 0.0), D(1.2, 0.0), D(η, 0.0), D(λ, 0.0), true, false, D(0.0, 0.0), D(5.477, 0.0), D(1.0, 0.0),
                                  SVector(D(0.1, 0.0), D(0.5, 0.0), D(0.0, 0.0)), SVector(D(1.0, 0.0), D(0.3, 0.0), D(0.2, 0.0)))
        @test isfinite(ForwardDiff.partials(fr.g, 1)) && isfinite(ForwardDiff.partials(fr.cosθB, 1)) && isfinite(ForwardDiff.partials(fr.χ, 1))
        @info "turning-point frame: root at r = $r0 for η = $η; a hair inside it the momentum values agree with Krang's and the frame's partials are finite ($(ForwardDiff.partials(fr.g, 1)) for the redshift) where Krang's radial momentum has $(ForwardDiff.partials(pk[2], 1))"
    end
end

