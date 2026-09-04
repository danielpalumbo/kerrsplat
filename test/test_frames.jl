# Gate for the per-sample frames: the boost against Krang's fluid-frame Jacobian, and the redshift
# and the screen polarization angle against Krang's `synchrotronPolarization` (the Walker–Penrose
# transport used in Krang's polarized images) on random rays, samples, velocities and fields; the
# pitch angle against an explicit construction; and the same routine inside a kernel on the backend.

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
