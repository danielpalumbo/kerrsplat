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
using Krang

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
    p_start = copy(p)
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
        # field-level recovery on a voxel grid (plan §7.5 item 7 i): density PSNR and the density-weighted
        # relative errors of temperature and field strength at the first frame
        grid = range(-9.0, 9.0, length = 25)
        m = recovery_metrics(p, p_true, times[1], grid, grid, range(-3.0, 3.0, length = 9))
        m0 = recovery_metrics(p_start, p_true, times[1], grid, grid, range(-3.0, 3.0, length = 9))
        @test m.psnr_density > m0.psnr_density + 3            # RMS density error down by at least √2 (measured: +3.7 dB; the
                                                              # single-frequency nₑ–B–Θe degeneracy of the Fisher audit limits it)
        @test m.temperature < 0.3 && m.field < 0.5
        @info "field recovery on the voxel grid: density PSNR $(round(m0.psnr_density, digits = 1)) → $(round(m.psnr_density, digits = 1)) dB; density-weighted relative errors of Θe $(round(m0.temperature, digits = 3)) → $(round(m.temperature, digits = 3)), of B $(round(m0.field, digits = 3)) → $(round(m.field, digits = 3))"
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

"""
Schedule with hygiene: a single splat is fitted to the movie of two well-separated splats; the
densification pass splits it where the position gradient is large, so the fit ends with two
splats near the true ones and a χ² several times below what one splat reaches (measured:
40585 → 6966, centres 3.0 and 2.8 M from the truth at 2.25 M per pixel after 100 iterations).
Also exercises the frequency curriculum (the 345 GHz channel first, then both) and the annealed
learning rate.
"""
function test_fit_schedule(; res = 8, N = 60)
    rng = Random.MersenneTwister(5)
    a = 0.9; θo = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    L = gravitational_radius(4e6)
    times = [0.0, 25.0]; νs = [230e9, 345e9]
    p_true = zeros(NPOLARIZEDPARAMS, 2)
    p_true[:, 1] = [5.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 5.0^(-1.5)]
    p_true[:, 2] = [-3.5, 2.5, 0.2, log(1.0), log(1.0), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(25.0), log(20.0), 0.7, -0.3, 0.1, -0.3, 0.0, -0.05]
    clean = polarized_cube(cache, p_true, times, νs, L)
    σ = SVector(0.02, 0.01, 0.01, 0.005) * maximum(norm.(clean))
    data = [clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
    movie = StokesMovie(data, times, νs, σ)
    # one splat between the two, elongated toward both
    p = zeros(NPOLARIZEDPARAMS, 1)
    p[:, 1] = [0.75, 1.25, 0.1, log(3.5), log(1.5), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(22.0), log(25.0), 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    stages = [Fit.Stage(free = (:x, :y, :z, :s1, :s2, :s3, :logne), iterations = 40, η = 0.08, η_end = 0.03, freqs = [2], label = "geometry, 345 GHz"),
              Fit.Stage(iterations = 60, η = 0.03, η_end = 0.005, label = "everything, both frequencies")]
    @testset "schedule with hygiene: one splat densifies into two" begin
        χ0 = chi2(p, movie, cache, L)
        q, history, events = Fit.fit!(copy(p), movie, cache, L, stages; hygiene = Fit.Hygiene(every = 20, densify_threshold = 0.0, max_splats = 2), rng = rng)
        χ1 = chi2(q, movie, cache, L)
        @test size(q, 2) == 2
        @test !isempty(events)
        @test χ1 < 0.3 * χ0
        # each recovered splat sits within about a pixel (2.25 M here) of one of the true ones after this
        # short schedule (100 iterations); the noise-floor fit of test_fit covers convergence itself
        d = [minimum(norm(q[1:3, k] .- p_true[1:3, j]) for j in 1:2) for k in 1:size(q, 2)]
        @test all(d .< 3.5)
        @info "schedule: χ² $χ0 → $χ1 with $(size(q, 2)) splats after hygiene events $events; distances of the recovered centres to the nearest true ones $(round.(d, digits = 2)) M"
    end
    @testset "generic loss form matches the movie form" begin
        one = [Fit.Stage(free = (:x, :y, :logne, :logB), iterations = 8, η = 0.05, η_end = 0.02)]
        q1, h1, e1 = Fit.fit!(copy(p), movie, cache, L, one; hygiene = Fit.Hygiene(every = 0))
        q2, h2, e2 = Fit.fit!(copy(p), q -> chi2(q, movie, cache, L), one; hygiene = Fit.Hygiene(every = 0))
        @test length(h1) == length(h2) == 8 && isempty(e1) && isempty(e2)
        @test maximum(abs.(h1 .- h2) ./ abs.(h1)) < 1e-12
        @test maximum(abs.(q1 .- q2)) < 1e-12
    end
end

"""
FITS round trip (the real-data path): a rendered Stokes movie written as ehtim-style FITS files
(one per frame and frequency, Jy/pixel) is read back into a `StokesMovie` with the same
intensities, times in M, frequencies and camera.
"""
function test_fits_io(; res = 8, N = 60)
    a = 0.9; θo = deg2rad(60.0)
    fov = 18.0; Δα = fov / res
    axis = [(i - (res + 1) / 2) * Δα for i in 1:res]
    camera = Geodesics.Camera(vec([axis[i] for i in 1:res, j in 1:res]), vec([axis[j] for i in 1:res, j in 1:res]), (res, res))
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    M_solar = 6.5e9; D_pc = 16.8e6
    L = gravitational_radius(M_solar)
    p = zeros(NPOLARIZEDPARAMS, 1)
    p[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e5), log(20.0), log(10.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    times = [0.0, 30.0]; νs = [230e9, 345e9]
    cube = polarized_cube(cache, p, times, νs, L)
    dir = mktempdir()
    mjd0 = 60000.0
    paths = String[]
    for (l, ν) in enumerate(νs), (k, t) in enumerate(times)
        path = joinpath(dir, "frame_$(k)_$(l).fits")
        write_stokes_fits(path, cube[:, :, k, l], Δα; M_solar, D_pc, freq = ν, mjd = mjd0 + t * Fit.time_unit(M_solar) / 86400)
        push!(paths, path)
    end
    σjy = SVector(1e-3, 5e-4, 5e-4, 2e-4)
    movie, cam2, L2 = read_stokes_movie(reverse(paths); M_solar, D_pc, mjd0, σ = σjy)
    @testset "FITS round trip of a Stokes movie" begin
        @test L2 == L
        @test maximum(norm.(movie.data .- cube)) < 1e-9 * maximum(norm.(cube))
        @test isapprox(movie.times, times; atol = 1e-6)
        @test movie.νs == νs
        @test cam2.αs ≈ camera.αs && cam2.βs ≈ camera.βs
        Ω = (Δα * L / (D_pc * Transfer.PC))^2
        @test movie.σ ≈ σjy .* (Transfer.JY / Ω)
        S, hdr = read_stokes_fits(paths[1])
        @test hdr.bunit == "JY/PIXEL" && hdr.freq == 230e9
        @info "FITS round trip: $(length(paths)) files, pixel $(round(hdr.psize_deg * 3.6e9, digits = 2)) μas, peak $(maximum(norm.(S))) Jy/pixel"
    end
end

"""
Visibilities: the direct transform at the grid frequencies matches the FFT of the image, the
zero-spacing visibility is the flux density, and the χ² of a splat model against visibilities of
its own image is zero while a perturbed model gives a positive value with an Enzyme gradient
matching finite differences.
"""
function test_visibilities(; res = 8, N = 60)
    a = 0.9; θo = deg2rad(60.0)
    fov = 18.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    M_solar = 6.5e9; D_pc = 16.8e6; D = D_pc * Transfer.PC
    L = gravitational_radius(M_solar); ν = 230e9
    p = zeros(NPOLARIZEDPARAMS, 1)
    p[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e5), log(20.0), log(10.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    img = polarized_image(cache, p, 0.0, ν, L)
    psize = Δα * L / D
    @testset "visibilities by direct transform" begin
        # an explicit transform written from the sky coordinates of the pixels (RA offset −α toward the
        # east, declination offset +β), as the definition V(u, v) = ∫ I e^{+2πi(u l + v m)} dΩ (the EHT
        # sign convention; test_uvfits pins it against ehtim)
        I = getindex.(img, 1) .* (psize^2 / Transfer.JY)
        us = Float64[]; vs = Float64[]; ref = ComplexF64[]
        for kv in 0:2, ku in 0:2
            u = ku / (res * psize); v = kv / (res * psize)
            push!(us, u); push!(vs, v)
            acc = 0.0im
            for j in 1:res, i in 1:res
                l = -camera.αs[i + (j - 1) * res] * L / D; m = camera.βs[i + (j - 1) * res] * L / D
                acc += I[i, j] * cis(2π * (u * l + v * m))
            end
            push!(ref, acc)
        end
        V = visibilities(img, Δα, L, D, us, vs)
        @test maximum(abs.(getindex.(V, 1) .- ref)) < 1e-10 * abs(ref[1])
        @test real(V[1][1]) ≈ sum(I) && abs(imag(V[1][1])) < 1e-12 * sum(I)
        # χ² against the model's own visibilities is zero; a perturbed model is not, with a correct gradient
        rng = Random.MersenneTwister(2)
        u = 4e9 .* randn(rng, 20); v = 4e9 .* randn(rng, 20)
        σ = SVector(0.01, 0.005, 0.005, 0.002) * abs(V[1][1])
        data = VisibilityData(u, v, visibilities(img, Δα, L, D, u, v), σ)
        loss(q) = (out = Vector{RadiativeState{Float64}}(undef, npixels(cache)); fill!(out, zero(RadiativeState{Float64}));
                   polarized_image!(out, cache, q, 0.0, ν, L); chi2_visibilities(map(st -> observed_stokes(st, ν), to_screen(cache, out)), Δα, L, D, data))
        @test loss(p) < 1e-18
        q0 = copy(p); q0[1] += 0.5; q0[13] -= 0.2
        χ = loss(q0)
        @test χ > 1
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), q0)[1]
        for i in (1, 13, 16)
            h = 1e-4; f(x) = (q = copy(q0); q[i] = x; loss(q)); x = q0[i]
            fd = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
            @test abs(g[i] - fd) / abs(fd) < 1e-5
        end
        @info "visibilities: zero-spacing $(round(real(V[1][1]), digits = 4)) Jy; χ² of the perturbed model $χ over 20 baselines"
        # closure quantities: gain-independent (unchanged by station gains), zero χ² against the model's own,
        # positive for the perturbed model, with a correct gradient
        tri = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]; quad = [(1, 2, 3, 4), (5, 6, 7, 8)]
        Vd = visibilities(img, Δα, L, D, u, v)
        gains = [cis(0.3k) * (1 + 0.1 * sin(k)) for k in eachindex(u)]
        # a closing triangle has baselines ij, jk, ki: gains cancel in the bispectrum only for such triangles, so
        # emulate them by assigning stations: baseline k joins stations s1[k], s2[k]
        s1 = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 4, 4, 1, 2, 3, 4, 1, 2, 3, 4]; s2 = [2, 3, 1, 2, 3, 1, 2, 3, 1, 3, 2, 3, 4, 4, 4, 2, 3, 4, 1, 3]
        gV = [Vd[k] .* (gains[s1[k]] * conj(gains[s2[k]])) for k in eachindex(u)]
        @test maximum(abs.(rem.(closure_phases(gV, tri) .- closure_phases(Vd, tri), 2π, RoundNearest))) < 1e-12
        cl = ClosureData(u, v, tri, closure_phases(Vd, tri), fill(0.05, 3), quad, log_closure_amplitudes(Vd, quad), fill(0.05, 2))
        lossc(q) = (out = Vector{RadiativeState{Float64}}(undef, npixels(cache)); fill!(out, zero(RadiativeState{Float64}));
                    polarized_image!(out, cache, q, 0.0, ν, L); chi2_closures(map(st -> observed_stokes(st, ν), to_screen(cache, out)), Δα, L, D, cl))
        @test lossc(p) < 1e-18
        χc = lossc(q0)
        @test χc > 0
        gc = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(lossc), q0)[1]
        h = 1e-4; fc(x) = (q = copy(q0); q[1] = x; lossc(q)); x = q0[1]
        fd = (-fc(x + 2h) + 8fc(x + h) - 8fc(x - h) + fc(x - 2h)) / (12h)
        @test abs(gc[1] - fd) / abs(fd) < 1e-5
        @info "closures: χ² of the perturbed model $χc over 3 closure phases and 2 log closure amplitudes"
    end
end

"""
Priors: the penalty of a fixed prior and of hierarchical shrinkage, its Enzyme gradient against
the analytic one, and χ² with priors equal to χ² plus the penalty.
"""
function test_priors(; res = 8, N = 60)
    a = 0.9; θo = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    L = gravitational_radius(4e6)
    p = polarized_test_params()
    clean = polarized_cube(cache, p, [0.0], [230e9], L)
    movie = StokesMovie(clean, [0.0], [230e9], SVector(1.0, 1.0, 1.0, 1.0) * 1e-3 * maximum(norm.(clean)))
    @testset "priors and shrinkage" begin
        pr = Fit.Prior(rows = (:logTe, :logB), μ = (log(25.0), log(20.0)), σ = 0.5)
        sh = Fit.Prior(rows = (:logne,), σ = 0.2, shrink = true)
        pen = Fit.penalty(p, pr)
        @test pen ≈ sum(((p[14, :] .- log(25.0)) ./ 0.5) .^ 2) + sum(((p[15, :] .- log(20.0)) ./ 0.5) .^ 2)
        m = sum(p[13, :]) / 2
        @test Fit.penalty(p, sh) ≈ sum(((p[13, :] .- m) ./ 0.2) .^ 2)
        @test chi2(p, movie, cache, L; priors = [pr, sh]) ≈ chi2(p, movie, cache, L) + pen + Fit.penalty(p, sh)
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(q -> Fit.penalty(q, [pr, sh])), p)[1]
        ga = zeros(size(p))
        ga[14, :] = 2 .* (p[14, :] .- log(25.0)) ./ 0.5^2; ga[15, :] = 2 .* (p[15, :] .- log(20.0)) ./ 0.5^2
        ga[13, :] = 2 .* (p[13, :] .- m) ./ 0.2^2                  # the mean's own dependence cancels: Σ (p − m) = 0
        @test g ≈ ga atol = 1e-10
        @info "priors: fixed penalty $pen, shrinkage penalty $(Fit.penalty(p, sh)); gradient matches the analytic one"
    end
end

"""
Station gains: gained model visibilities against data corrupted with the same gains give zero
χ² (up to the gain priors), the gain priors act, and the Enzyme gradient of the self-calibration
χ² with respect to a gain and to a splat parameter matches finite differences.
"""
function test_gains(; res = 8, N = 60)
    a = 0.9; θo = deg2rad(60.0)
    fov = 18.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    M_solar = 6.5e9; D_pc = 16.8e6; D = D_pc * Transfer.PC
    L = gravitational_radius(M_solar); ν = 230e9
    p = zeros(NPOLARIZEDPARAMS, 1)
    p[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e5), log(20.0), log(10.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    img = polarized_image(cache, p, 0.0, ν, L)
    rng = Random.MersenneTwister(4)
    u = 4e9 .* randn(rng, 12); v = 4e9 .* randn(rng, 12)
    s1 = [1, 1, 1, 2, 2, 3, 1, 2, 3, 4, 4, 2]; s2 = [2, 3, 4, 3, 4, 4, 2, 3, 4, 1, 3, 1]
    gtrue = [0.05 -0.1 0.0 0.08; 0.3 -0.2 0.0 0.5]
    V = visibilities(img, Δα, L, D, u, v)
    data = VisibilityData(u, v, apply_gains(V, gtrue, s1, s2), SVector(0.01, 0.005, 0.005, 0.002) * abs(V[1][1]))
    @testset "station gains" begin
        g0 = zeros(2, 4)
        χtrue = chi2_visibilities(img, Δα, L, D, data, gtrue, s1, s2; σ_logamp = 0.1)
        @test χtrue ≈ sum((gtrue[1, :] ./ 0.1) .^ 2)           # data term zero, only the amplitude prior
        @test chi2_visibilities(img, Δα, L, D, data, g0, s1, s2) > 10 * χtrue
        # gradients through Enzyme: with respect to the gains at a fixed image, and with respect to the splat
        # parameters at fixed gains (a fresh parameter matrix is the active argument in both, as in the fits)
        lossg(g) = chi2_visibilities(img, Δα, L, D, data, g, s1, s2)
        g1 = gtrue .+ [0.02 -0.03 0.01 0.04; 0.05 -0.02 0.03 -0.04]      # a uniform phase offset would be invisible
        gg = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(lossg), g1)[1]
        for i in (1, 2, 5)
            h = 1e-4; f(y) = (g = copy(g1); g[i] = y; lossg(g)); y = g1[i]
            fd = (-f(y + 2h) + 8f(y + h) - 8f(y - h) + f(y - 2h)) / (12h)
            @test abs(gg[i] - fd) / abs(fd) < 1e-6
        end
        lossp(q) = (out = Vector{RadiativeState{Float64}}(undef, npixels(cache)); fill!(out, zero(RadiativeState{Float64}));
                    polarized_image!(out, cache, q, 0.0, ν, L); chi2_visibilities(map(st -> observed_stokes(st, ν), to_screen(cache, out)), Δα, L, D, data, g1, s1, s2))
        q0 = copy(p); q0[1] += 0.3
        gp = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(lossp), q0)[1]
        h = 1e-4; fp(y) = (q = copy(q0); q[1] = y; lossp(q)); y = q0[1]
        fd = (-fp(y + 2h) + 8fp(y + h) - 8fp(y - h) + fp(y - 2h)) / (12h)
        @test abs(gp[1] - fd) / abs(fd) < 1e-5
        @info "gains: χ² at the true gains $χtrue (amplitude prior only), without gains $(chi2_visibilities(img, Δα, L, D, data, g0, s1, s2))"
        # per-scan gains: two scans with different gains as the columns of a matrix indexed by (station, scan)
        scan = [1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2]
        g3 = hcat(gtrue, [-0.04 0.06 0.02 -0.1; -0.3 0.1 0.4 0.0])
        t1 = scan_station.(s1, scan, 4); t2 = scan_station.(s2, scan, 4)
        data3 = VisibilityData(u, v, apply_gains(V, g3, t1, t2), data.σ)
        χ3 = chi2_visibilities(img, Δα, L, D, data3, g3, t1, t2; σ_logamp = 0.1)
        @test χ3 ≈ sum((g3[1, :] ./ 0.1) .^ 2)
        @test chi2_visibilities(img, Δα, L, D, data3, hcat(gtrue, gtrue), t1, t2) > 10 * χ3
        loss3(g) = chi2_visibilities(img, Δα, L, D, data3, g, t1, t2)
        g4 = g3 .+ 0.03 .* reshape(sin.(1:16), 2, 8)
        gg3 = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss3), g4)[1]
        for i in (1, 6, 11, 16)
            h = 1e-4; fs(y) = (g = copy(g4); g[i] = y; loss3(g)); y = g4[i]      # not `f3`: `8f3(…)` is a Float32 literal
            fd = (-fs(y + 2h) + 8fs(y + h) - 8fs(y - h) + fs(y - 2h)) / (12h)
            @test abs(gg3[i] - fd) / abs(fd) < 1e-6
        end
    end
end

"""
    test_pattern_prior()

Motion mode C as a soft constraint: `fluid_pattern_rate` returns the Keplerian angular velocity
1/(r^{3/2} + a) for a splat whose ZAMO velocity is that of a circular equatorial orbit (round
trip through the ZAMO tetrad), the `PatternPrior` penalty vanishes when the pattern rate equals
it and grows as (Δω/σ)², and its Enzyme gradient matches a stencil.
"""
function test_pattern_prior()
    a = 0.9
    met = Krang.Kerr(a)
    p = polarized_test_params()
    @testset "pattern prior (mode C)" begin
        for (i, r0) in enumerate((6.0, 9.0))
            Ω = 1 / (r0^1.5 + a)
            # BL 4-velocity of the circular orbit, normalized, then into the ZAMO frame: ũ = γβ⃗
            g = Krang.metric_dd(met, r0, π / 2)
            ut = 1 / sqrt(-(g[1, 1] + 2 * g[1, 4] * Ω + g[4, 4] * Ω^2))
            u_bl = SVector(ut, 0.0, 0.0, ut * Ω)
            u_zamo = Krang.jac_zamo_u_bl_d(met, r0, π / 2) * u_bl
            @test abs(u_zamo[1] - sqrt(1 + u_zamo[2]^2 + u_zamo[3]^2 + u_zamo[4]^2)) < 1e-12
            p[1, i] = r0 * cos(0.3i); p[2, i] = r0 * sin(0.3i); p[3, i] = 0.0
            p[18:20, i] = u_zamo[2:4]
            @test abs(Fit.fluid_pattern_rate(p, i, met) - Ω) < 1e-12
            p[21, i] = Ω
        end
        pr = Fit.PatternPrior(met, 0.01)
        @test Fit.penalty(p, pr) < 1e-20
        q = copy(p); q[21, 1] += 0.02
        @test abs(Fit.penalty(q, pr) - 4.0) < 1e-8
        q[1, 2] += 0.5; q[19, 2] += 0.1
        f(x) = Fit.penalty(x, pr)
        grad = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(f), q)[1]
        for (i, j) in ((21, 1), (1, 2), (19, 2), (20, 2), (3, 1))
            h = 1e-4; x = q[i, j]
            fd = (-(w = copy(q); w[i, j] = x + 2h; f(w)) + 8(w = copy(q); w[i, j] = x + h; f(w)) - 8(w = copy(q); w[i, j] = x - h; f(w)) + (w = copy(q); w[i, j] = x - 2h; f(w))) / (12h)
            @test abs(grad[i, j] - fd) <= 1e-6 * max(abs(fd), 1e-3)
        end
        @info "pattern prior: Keplerian rates recovered through the ZAMO frame; penalty gradient vs stencil to 1e-6"
    end
end

"""
    test_chi2_gradient(backend; res = 8, N = 40, tol = 1e-9, label = "CPU backend", methods = (:dual, :enzyme))

The movie χ² gradient by the in-kernel sweeps on `backend` (`chi2_gradient!`, stored samples)
against the host Enzyme gradient of `chi2` through the fused march, on a two-frame,
one-frequency movie with a mask; the χ² values agree too.
"""
function test_chi2_gradient(backend; res = 8, N = 40, tol = 1e-9, label = "CPU backend", methods = (:dual, :enzyme))
    a = 0.9; θo = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cpu, a, θo; marcher = Fused(64))
    L = gravitational_radius(4e6); times = [0.0, 20.0]; νs = [230e9]
    p_true = polarized_test_params()
    clean = polarized_cube(cpu, p_true, times, νs, L)
    rng = Random.MersenneTwister(11)
    σ = SVector(0.02, 0.01, 0.01, 0.005) * maximum(norm.(clean))
    data = [clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
    mask = trues(size(data)); mask[1:2, :, 1, 1] .= false                         # a few masked pixels
    movie = StokesMovie(data, times, νs, σ; mask)
    p = p_true .+ 0.05 .* randn(rng, size(p_true))
    χ_host = chi2(p, movie, cpu, L)
    g_host = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(q -> chi2(q, movie, cpu, L)), p)[1]
    cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
    regenerate!(cache, a, θo; marcher = Recurrence(64))
    params = adapt_to(backend, p)
    for method in methods
        @testset "$label movie χ² gradient by the $method sweep, $(res)² × $N" begin
            dparams = adapt_to(backend, zeros(size(p)))
            t0 = time()
            χ = chi2_gradient!(dparams, params, movie, cache, L; method)
            t1 = time() - t0
            g = Array(dparams)
            @test abs(χ - χ_host) <= 1e-10 * χ_host
            e = maximum(abs.(g .- g_host)) / maximum(abs.(g_host))
            @test all(isfinite, g)
            @test e <= tol
            @info "$label chi2_gradient! ($method) vs host Enzyme chi2 gradient: max |Δ|/max = $e; χ² $χ vs $χ_host ($(round(t1; digits = 1)) s including compilation)"
        end
    end
end

"""
    test_image_loss_gradient(backend; res = 8, N = 40, tol = 1e-9, label = "CPU backend", methods = (:dual, :enzyme))

`image_loss_gradient!` (host Enzyme seed on the image, the in-kernel sweeps on the backend)
for the closure χ² of one frame against the host Enzyme gradient of the same closure χ²
through the fused march.
"""
function test_image_loss_gradient(backend; res = 8, N = 40, tol = 1e-9, label = "CPU backend", methods = (:dual, :enzyme))
    a = 0.9; θo = deg2rad(60.0)
    fov = 18.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cpu, a, θo; marcher = Fused(64))
    M_solar = 6.5e9; D = 16.8e6 * Transfer.PC; L = gravitational_radius(M_solar); ν = 230e9
    p = polarized_test_params()
    img = polarized_image(cpu, p, 0.0, ν, L)
    rng = Random.MersenneTwister(5)
    u = 4e9 .* randn(rng, 9); v = 4e9 .* randn(rng, 9)
    tri = [(1, 2, 3), (4, 5, 6), (7, 8, 9)]; quad = [(1, 2, 3, 4), (5, 6, 7, 8)]
    V = visibilities(img, Δα, L, D, u, v)
    data = ClosureData(u, v, tri, closure_phases(V, tri) .+ 0.1, fill(0.05, 3), quad, log_closure_amplitudes(V, quad) .+ 0.05, fill(0.05, 2))
    loss_image(im) = chi2_closures(im, Δα, L, D, data)
    q = p .+ 0.05 .* randn(rng, size(p))
    host(x) = (out = Vector{RadiativeState{Float64}}(undef, npixels(cpu)); fill!(out, zero(RadiativeState{Float64}));
               polarized_image!(out, cpu, x, 0.0, ν, L); loss_image(map(st -> observed_stokes(st, ν), to_screen(cpu, out))))
    value_host = host(q)
    g_host = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(host), q)[1]
    cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
    regenerate!(cache, a, θo; marcher = Recurrence(64))
    params = adapt_to(backend, q)
    for method in methods
        @testset "$label closure χ² gradient by the image seed and the $method sweep" begin
            dparams = adapt_to(backend, zeros(size(q)))
            value = image_loss_gradient!(dparams, loss_image, cache, params, 0.0, ν, L; method)
            g = Array(dparams)
            @test abs(value - value_host) <= 1e-10 * value_host
            e = maximum(abs.(g .- g_host)) / maximum(abs.(g_host))
            @test all(isfinite, g)
            @test e <= tol
            @info "$label image_loss_gradient! ($method, closure χ²) vs host Enzyme: max |Δ|/max = $e"
        end
    end
end
