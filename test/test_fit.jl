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

function test_hygiene(; res = 10, N = 80)
    a = 0.9; θo = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    L = gravitational_radius(4e6); ν = 230e9
    p = zeros(NPOLARIZEDPARAMS, 3)
    p[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    p[:, 2] = p[:, 1]                                       # an identical co-located parcel
    p[:, 3] = [-4.0, 3.0, 0.2, log(1.0), log(1.0), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e-3), log(25.0), log(20.0), 0.7, -0.3, 0.1, -0.3, 0.0, 0.0]   # negligible density
    @testset "partition hygiene: prune, merge, densify" begin
        pp, kept = Fit.prune(p; fraction = 1e-4)
        @test kept == [1, 2]
        pm, groups = Fit.merge(pp)
        @test size(pm, 2) == 1 && groups == [[1, 2]] && pm[13, 1] ≈ log(2e6)
        img2 = polarized_image(cache, pp, 0.0, ν, L); img1 = polarized_image(cache, pm, 0.0, ν, L)
        @test maximum(norm.(img2 .- img1)) < 1e-12 * maximum(norm.(img1))     # coefficients add: two parcels = one with the summed density
        g = zeros(size(pm)); g[1, 1] = 1.0
        pd = Fit.densify(pm, g; threshold = 0.5)
        @test size(pd, 2) == 2
        @test pd[13, :] ≈ fill(pm[13, 1] + 3 * log(1.6) - log(2), 2)         # density × volume conserved
        @test pd[4:6, :] ≈ repeat(pm[4:6, :] .- log(1.6), 1, 2)
        @test norm(pd[1:3, 1] .- pd[1:3, 2]) ≈ exp(pm[4, 1])            # children a full largest-scale apart
        imgd = polarized_image(cache, pd, 0.0, ν, L)
        @test 0.5 < sum(getindex.(imgd, 1)) / sum(getindex.(img1, 1)) < 2
        @info "hygiene: prune kept $kept; merge → density $(exp(pm[13, 1])); densify children at $(round.(pd[1:3, 1], digits = 2)) and $(round.(pd[1:3, 2], digits = 2)); flux ratio after the split $(round(sum(getindex.(imgd, 1)) / sum(getindex.(img1, 1)), digits = 3))"
    end
end

"""
Fisher audit (addendum §5.3): with one frequency the density, temperature and field strength of a
splat are degenerate along one direction (nₑ up, B down); a second frequency raises the smallest
Fisher eigenvalue and lifts the temperature-dominated second one by an order of magnitude.
"""
function test_fisher(; res = 8, N = 60)
    a = 0.9; θo = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    L = gravitational_radius(4e6)
    p = zeros(NPOLARIZEDPARAMS, 1)
    p[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    free = freeze(p, (:logne, :logTe, :logB, :thB, :u2))
    names = [String(POLARIZED_SPLAT_PARAMS[i]) for i in findall(vec(free))]
    @testset "Fisher audit: plasma degeneracies at one and two frequencies" begin
        λ1 = Float64[]; λ2 = Float64[]; weakest = Vector{Float64}[]
        for νs in ([230e9], [230e9, 345e9])
            clean = polarized_cube(cache, p, [0.0], νs, L)
            σ = SVector(0.02, 0.01, 0.01, 0.005) * maximum(norm.(clean))
            movie = StokesMovie(clean, [0.0], νs, σ)
            F, J, idx = Fit.fisher(p, movie, cache, L; free)
            vals, vecs = Fit.audit(F, names; nshow = 2)
            @test all(vals .>= -1e-8 * maximum(vals))          # positive semidefinite
            push!(λ1, vals[1]); push!(λ2, vals[2]); push!(weakest, abs.(vecs[:, 1]))
            @info "Fisher at $(length(νs)) frequenc$(length(νs) == 1 ? "y" : "ies"): σ of the weakest combination $(round(1 / sqrt(vals[1]), digits = 3)), of the second $(round(1 / sqrt(vals[2]), digits = 3)); weakest combination $(join(("$(names[i]) $(round(vecs[i, 1], digits = 2))" for i in sortperm(abs.(vecs[:, 1]); rev = true)[1:3]), ", "))"
        end
        # at one frequency the weakest direction is the nₑ–B–Θe degeneracy (plan §7.7): its weight lies on those rows
        plasma = [findfirst(==(n), names) for n in ("logne", "logTe", "logB")]
        @test sum(weakest[1][plasma] .^ 2) > 0.95
        # a second frequency raises the smallest eigenvalue and lifts the second (temperature-dominated) one by far
        # more than the doubling of the data alone
        @test λ1[2] > λ1[1]
        @test λ2[2] > 4 * λ2[1]
    end
end
