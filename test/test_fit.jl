# Gate for KerrSplat.Fit: a noisy synthetic Stokes movie of two polarized splats is fitted from
# perturbed starting values with staged unfreezing (geometry first, then plasma, then everything)
# and minibatches over frames; χ² must end near the number of data points (the noise floor), the
# positions within a fraction of a pixel and the pattern rates within a few per cent (the density
# is degenerate with temperature and field at one frequency; the Fisher audit of the addendum
# will quantify such degeneracies later).

using StaticArrays
using LinearAlgebra
using Random
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Splats
using KerrSplat.Fit

function test_fit(; res = 10, N = 80, iterations = (40, 40, 60))
    rng = Random.MersenneTwister(3)
    a = 0.9; θo = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    L = gravitational_radius(4e6)
    times = [0.0, 20.0, 40.0]; νs = [230e9]
    p_true = zeros(NPOLARIZEDPARAMS, 2)
    p_true[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 6.0^(-1.5)]
    p_true[:, 2] = [-4.0, 3.0, 0.2, log(1.0), log(1.0), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(7e5), log(25.0), log(20.0), 0.7, -0.3, 0.1, -0.3, 0.0, -0.06]
    clean = polarized_cube(cache, p_true, times, νs, L)
    peak = maximum(norm.(clean))
    σ = SVector(0.02, 0.01, 0.01, 0.005) * peak
    data = [clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
    movie = StokesMovie(data, times, νs, σ)
    p = copy(p_true)
    p[1, :] .+= 0.6; p[2, :] .-= 0.4; p[13, :] .-= 0.3; p[14, :] .+= 0.2; p[16, :] .+= 0.25; p[19, :] .-= 0.15; p[21, :] .*= 1.2
    ndata = 4 * length(data)
    @testset "fit of two polarized splats to a noisy 3-frame Stokes movie ($(res)² × $N)" begin
        χ0 = chi2(p, movie, cache, L)
        h1 = fit!(p, movie, cache, L; free = freeze(p, (:x, :y, :z, :omega)), iterations = iterations[1], η = 0.03)
        h2 = fit!(p, movie, cache, L; free = freeze(p, (:logne, :logTe, :logB, :thB, :phB, :u1, :u2, :u3)), iterations = iterations[2], η = 0.03)
        h3 = fit!(p, movie, cache, L; iterations = iterations[3], η = 0.01, batch = (2, 1), rng = rng)
        χ1 = h3[end]
        # the fit reaches the noise floor (χ² ≈ number of data points); at 1.8 M pixels and one frequency the
        # positions are recovered to a fraction of a pixel and the density only up to the nₑ–B–Θe degeneracy
        # that multi-frequency data break (plan §7.7)
        @test χ1 < 0.2 * χ0
        @test χ1 < 1.3 * ndata
        @test maximum(abs.(p[1:2, :] .- p_true[1:2, :])) < 0.5
        @test maximum(abs.(p[21, :] ./ p_true[21, :] .- 1)) < 0.05
        @info "fit: χ² $χ0 → $(h1[end]) (geometry) → $(h2[end]) (plasma) → $χ1 (all, minibatched) for $ndata data points; position errors $(round.(vec(maximum(abs.(p[1:3, :] .- p_true[1:3, :]), dims = 1)), digits = 3)) M, pattern-rate errors $(round.(vec(p[21, :] ./ p_true[21, :] .- 1), digits = 4)), log-density errors $(round.(vec(abs.(p[13, :] .- p_true[13, :])), digits = 3))"
    end
end
