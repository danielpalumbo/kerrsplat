# Mixed-mode derivatives with respect to the spacetime parameters (plan §7.1 regime b, §7.6
# Phase 5): ForwardDiff duals for the spin and the inclination propagate through the geodesic
# cache (regenerate!), the transfer coefficients (Bessel functions included), the frames and the
# exact transfer step, to the Stokes image of the polarized splats. Compared with central finite
# differences of the whole pipeline.

using StaticArrays
using LinearAlgebra
using ForwardDiff
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Splats
using KerrSplat.Fit
using Random

function test_spacetime_duals(; res = 8, N = 60, tol = 1e-5)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    L = gravitational_radius(4e6); ν = 230e9
    p = zeros(NPOLARIZEDPARAMS, 1)
    p[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    function totals(a::T, θo::T) where {T}
        cache = GeodesicCache(CPU(), Geodesics.Camera(T.(camera.αs), T.(camera.βs), camera.size), Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        img = polarized_image(cache, T.(p), T(0.0), T(ν), T(L))
        return SVector(sum(getindex.(img, 1)), sum(getindex.(img, 2)), sum(getindex.(img, 3)), sum(getindex.(img, 4)))
    end
    a0 = 0.9; θ0 = deg2rad(60.0)
    @testset "spacetime duals through the polarized image ($(res)² × $N)" begin
        J = ForwardDiff.jacobian(x -> totals(x[1], x[2]), [a0, θ0])
        h = 1e-5
        Ja = (totals(a0 + h, θ0) - totals(a0 - h, θ0)) / 2h
        Jθ = (totals(a0, θ0 + h) - totals(a0, θ0 - h)) / 2h
        err = maximum(abs.(J .- hcat(Ja, Jθ)) ./ max.(abs.(hcat(Ja, Jθ)), 1e-6 * maximum(abs, J)))
        @test err < tol
        @info "spacetime duals: ∂(I, Q, U, V)/∂a = $(round.(J[:, 1], sigdigits = 4)), ∂/∂θo = $(round.(J[:, 2], sigdigits = 4)); max relative error vs finite differences $err"
    end
end

"""
Spin and inclination recovered from a noisy Stokes image of two splats by Levenberg–Marquardt
on the dual Jacobian (the splat parameters held at the truth), starting 0.1 off in spin and 8°
off in inclination.
"""
function test_spacetime_fit(; res = 8, N = 60)
    rng = Random.MersenneTwister(9)
    a_true = 0.9; θ_true = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    L = gravitational_radius(4e6); ν = 230e9
    p = zeros(NPOLARIZEDPARAMS, 2)
    p[:, 1] = [5.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    p[:, 2] = [-3.0, 2.5, 0.2, log(1.0), log(1.0), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(25.0), log(20.0), 0.7, -0.3, 0.1, -0.3, 0.0, 0.0]
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a_true, θ_true; marcher = Fused(64))
    clean = polarized_cube(cache, p, [0.0], [ν], L)
    σ = SVector(0.02, 0.01, 0.01, 0.005) * maximum(norm.(clean))
    data = [clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
    movie = StokesMovie(data, [0.0], [ν], σ)
    @testset "spin and inclination by Levenberg–Marquardt on the dual Jacobian" begin
        x, χ, history = Fit.fit_spacetime([0.8, deg2rad(52.0)], p, movie, camera, L; N, iterations = 8)
        # χ² reaches the noise floor; at 2.25 M per pixel and one frequency the spin is weakly constrained
        # (measured 0.86 for 0.9) while the inclination is recovered to a tenth of a degree
        @test abs(x[1] - a_true) < 0.05
        @test abs(x[2] - θ_true) < deg2rad(1.0)
        @test χ < 1.3 * 4 * length(data)
        @info "spacetime fit: a $(round(x[1], digits = 4)) (true $a_true), θo $(round(rad2deg(x[2]), digits = 2))° (true 60°); χ² $(round(history[1])) → $(round(χ)) for $(4 * length(data)) data points in $(length(history) - 1) iterations"
    end
end

"""
    test_joint_fit(backend; res = 8, N = 40, iterations = 60, label = "CPU backend")

The joint spacetime-and-splat fit. (1) `spacetime_valgrad`'s value equals the stored-sample χ²
(`chi2` on a recurrence cache at the same spacetime: two marchers) and its gradient with respect
to (a, θo, ln L) matches central finite differences of that χ²; `spacetime_jacobian`'s residuals
square to the same χ² and 2 Jᵀr is the same gradient. (2) From a two-splat truth at a = 0.9,
θo = 60°, the fit of (a, θo) started 0.1 off in spin and 8° off in inclination with the splats
perturbed recovers both to `atol` and `θtol` (with the default schedule: no warmup, three inner
Levenberg–Marquardt steps per iteration; see the note of 2026-09-11).
"""
function test_joint_fit(backend; res = 8, N = 40, iterations = 60, tol = 2e-5, θtol = deg2rad(2.0), atol = 0.05, label = "CPU backend")
    rng = Random.MersenneTwister(21)
    a_true = 0.9; θ_true = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    L = gravitational_radius(4e6); ν = 230e9
    p = zeros(NPOLARIZEDPARAMS, 2)
    p[:, 1] = [5.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    p[:, 2] = [-3.0, 2.5, 0.2, log(1.0), log(1.0), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(25.0), log(20.0), 0.7, -0.3, 0.1, -0.3, 0.0, 0.0]
    cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cpu, a_true, θ_true; marcher = Fused(64))
    clean = polarized_cube(cpu, p, [0.0, 10.0], [ν], L)
    σ = SVector(0.02, 0.01, 0.01, 0.005) * maximum(norm.(clean))
    data = [clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
    movie = StokesMovie(data, [0.0, 10.0], [ν], σ)
    cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
    q = p .+ 0.05 .* randn(rng, size(p))
    x3 = [0.8, deg2rad(52.0), log(1.2 * L)]
    @testset "$label joint spacetime-and-splat fit" begin
        regenerate!(cache, x3[1], x3[2]; marcher = Recurrence(64))
        χ_stored = chi2(q, movie, cache, exp(x3[3]))
        v, gx = Fit.spacetime_valgrad(Fit.spacetime_movie_loss(q, movie), x3, camera, backend; N)
        @test abs(v - χ_stored) <= 1e-8 * χ_stored
        function χ_at(y)
            c = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
            regenerate!(c, y[1], y[2]; marcher = Fused(64))
            return chi2(q, movie, c, exp(y[3]))
        end
        h = 1e-5
        gfd = [(χ_at(x3 .+ h .* (1:3 .== i)) - χ_at(x3 .- h .* (1:3 .== i))) / (2h) for i in 1:3]
        e = maximum(abs.(gx .- gfd) ./ max.(abs.(gfd), 1e-6 * maximum(abs, gfd)))
        @test e < tol
        r, J = Fit.spacetime_jacobian((c, Lc) -> Fit.spacetime_movie_residuals(c, Fit._on_backend(c, q), movie, Lc), x3, camera, backend; N)
        @test abs(sum(abs2, r) - χ_stored) <= 1e-8 * χ_stored && size(J) == (length(r), 3)
        @test maximum(abs.(2 .* (J' * r) .- gx)) <= 1e-8 * maximum(abs, gx)
        x0 = x3[1:2]
        qj, xj, hist, acc = Fit.fit_joint!(copy(q), x0, movie, cache, camera; L, iterations, η = 0.03)
        χ0 = hist[1][1]; χ1 = hist[end][1]
        @test length(hist) == iterations && χ1 < 0.5 * χ0 && acc >= 1
        @test abs(xj[2] - θ_true) < θtol && abs(xj[1] - a_true) < atol
        @info "$label joint fit: spacetime gradient vs finite differences $e; χ² $(round(χ0)) → $(round(χ1)) for $(4 * length(data)) values in $iterations iterations, $acc spacetime steps accepted; a $(round(xj[1]; digits = 3)) (from 0.8, true 0.9), θo $(round(rad2deg(xj[2]); digits = 2))° (from 52°, true 60°); trace of θo every 10 iterations: $(round.(rad2deg.(getindex.(hist[1:10:end], 3)); digits = 1))°"
    end
end
