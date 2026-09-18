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
        # a residual with finite value and non-finite partials has its Jacobian row zeroed, the rest untouched
        rbad, Jbad = Fit.spacetime_jacobian(x3, camera, backend; N) do c, Lc
            rr = Fit.spacetime_movie_residuals(c, Fit._on_backend(c, q), movie, Lc)
            SS = eltype(rr)
            rr[3] = SS(ForwardDiff.value(rr[3]), ForwardDiff.Partials(ntuple(_ -> ForwardDiff.valtype(SS)(NaN), 3)))
            rr
        end
        @test all(isfinite, Jbad) && all(Jbad[3, :] .== 0) && rbad == r && maximum(abs.(Jbad[[1, 2, 4], :] .- J[[1, 2, 4], :])) == 0
        # the stored and the fused dual paths agree, with and without the half-orbit truncation, and stay finite
        for nmax in (-1, 1)
            rs, Js = Fit.spacetime_jacobian((c, Lc) -> Fit.spacetime_movie_residuals(c, Fit._on_backend(c, q), movie, Lc; nmax, slab = 0.5), x3, camera, backend; N, stored = true)
            rf, Jf = Fit.spacetime_jacobian((c, Lc) -> Fit.spacetime_movie_residuals(c, Fit._on_backend(c, q), movie, Lc; nmax, slab = 0.5), x3, camera, backend; N, stored = false)
            @test all(isfinite, Js) && all(isfinite, Jf)
            @test maximum(abs.(rs .- rf)) <= 1e-10 * maximum(abs, rf) && maximum(abs.(Js .- Jf)) <= 1e-8 * maximum(abs, Jf)
        end
        # the pattern prior's residuals depend on the spin through the metric: dual spin vs finite differences
        σp = 0.01
        rp(a) = Fit.prior_residuals(q, Fit.PatternPrior(Geodesics.Krang.Kerr(a), σp))
        da = ForwardDiff.Dual{:a}(0.8, 1.0)
        rpd = Fit.prior_residuals(ForwardDiff.Dual{:a}.(q, 0.0), Fit.PatternPrior(Geodesics.Krang.Kerr(da), σp))
        @test maximum(abs.(ForwardDiff.value.(rpd) .- rp(0.8))) <= 1e-12 * maximum(abs, rp(0.8))
        @test maximum(abs.(ForwardDiff.partials.(rpd, 1) .- (rp(0.8 + 1e-6) .- rp(0.8 - 1e-6)) ./ 2e-6)) <= 1e-6 * maximum(abs, ForwardDiff.partials.(rpd, 1))
        # the Keplerian fluid prior: Schwarzschild at 6 M has the orbital speed 1/2 in the static frame (γv = 1/√3), and the
        # residuals' spin partials match finite differences
        uk = Fit.keplerian_zamo_velocity(Geodesics.Krang.Kerr(0.0), 6.0, π / 2)
        @test abs(norm(uk) - 1 / sqrt(3)) < 1e-12 && count(c -> abs(c) > 1e-12, uk) == 1
        rk(a) = Fit.prior_residuals(q, Fit.KeplerianPrior(Geodesics.Krang.Kerr(a), 0.1))
        rkd = Fit.prior_residuals(ForwardDiff.Dual{:a}.(q, 0.0), Fit.KeplerianPrior(Geodesics.Krang.Kerr(da), 0.1))
        @test maximum(abs.(ForwardDiff.value.(rkd) .- rk(0.8))) <= 1e-12 * maximum(abs, rk(0.8))
        @test maximum(abs.(ForwardDiff.partials.(rkd, 1) .- (rk(0.8 + 1e-6) .- rk(0.8 - 1e-6)) ./ 2e-6)) <= 1e-6 * maximum(abs, ForwardDiff.partials.(rkd, 1))
        qk, xk, hk, acck, _ = Fit.fit_joint!(copy(q), x3[1:2], movie, cache, camera; L, iterations = 3, η = 0.03, keplerian = 0.1)
        @test length(hk) == 3 && all(isfinite, xk) && hk[1][1] > chi2(q, movie, cache, L) * 0.999
        # a short joint fit with the pattern prior runs and its history includes the penalty
        qp, xp, hp, accp, _ = Fit.fit_joint!(copy(q), x3[1:2], movie, cache, camera; L, iterations = 3, η = 0.03, pattern = σp)
        @test length(hp) == 3 && all(isfinite, xp) && hp[1][1] > chi2(q, movie, cache, L) * 0.999
        # the joint fit from an over-complete shell with hygiene: the parcel count falls, χ² falls, the spacetime stays finite
        qs = shell_parcels(24; rin = 2.5, rout = 7.0, height = 0.5, scale = 0.4, spin = 0.8, rng = Random.MersenneTwister(2))
        qs[13, :] .= log(1e6 * 2 / 24)
        χs0 = chi2(qs, movie, cache, L)
        qh, xh, hh, acch, evh = Fit.fit_joint!(copy(qs), x3[1:2], movie, cache, camera; L, iterations = 12, η = 0.05, hygiene = Fit.Hygiene(every = 4, prune_fraction = 0.2, merge_position = 0.3))
        @test size(qh, 2) < 24 && !isempty(evh) && all(isfinite, xh) && hh[end][1] < χs0 && length(hh) == 12
        x0 = x3[1:2]
        qj, xj, hist, acc, _ = Fit.fit_joint!(copy(q), x0, movie, cache, camera; L, iterations, η = 0.03)
        χ0 = hist[1][1]; χ1 = hist[end][1]
        @test length(hist) == iterations && χ1 < 0.5 * χ0 && acc >= 1
        @test abs(xj[2] - θ_true) < θtol && abs(xj[1] - a_true) < atol
        @info "$label joint fit: spacetime gradient vs finite differences $e; χ² $(round(χ0)) → $(round(χ1)) for $(4 * length(data)) values in $iterations iterations, $acc spacetime steps accepted; a $(round(xj[1]; digits = 3)) (from 0.8, true 0.9), θo $(round(rad2deg(xj[2]); digits = 2))° (from 52°, true 60°); trace of θo every 10 iterations: $(round.(rad2deg.(getindex.(hist[1:10:end], 3)); digits = 1))°"
    end
end

"""
The joint fit in the data domain: the spacetime Jacobian of the bands' scan residuals against
finite differences of the summed χ², and a few joint iterations from an offset spacetime.
"""
function test_joint_scans(backend; res = 6, N = 16, tol = 2e-5, label = "CPU backend")
    rng = Random.MersenneTwister(23)
    a_true = 0.9; θ_true = deg2rad(60.0)
    fov = 18.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    M_solar = 6.5e9; D = 16.8e6 * Transfer.PC; L = gravitational_radius(M_solar)
    p = polarized_test_params()
    cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cpu, a_true, θ_true; marcher = Fused(64))
    function scancov(t)
        sts = sort(randperm(rng, 5)[1:4]); s1 = Int[]; s2 = Int[]
        for i in 1:4, j in i+1:4
            push!(s1, sts[i]); push!(s2, sts[j])
        end
        return ScanCoverage(t, 3e9 .* randn(rng, length(s1)), 3e9 .* randn(rng, length(s1)), s1, s2)
    end
    cov = [scancov(0.0), scancov(0.0), scancov(20.0)]
    bands = [BandScans(ν, synthetic_scans(cpu, p, L, Δα, D, ν, cov; noise = 0.02, closures = false, rng), Δα, D) for ν in (230e9, 345e9)]
    q = p .+ 0.05 .* randn(rng, size(p))
    x = [0.85, deg2rad(55.0)]
    @testset "$label joint fit on scans" begin
        function χ_at(y)
            c = GeodesicCache(CPU(), camera, Val(N); store_samples = true)
            regenerate!(c, y[1], y[2]; marcher = Recurrence(64))
            return Fit._joint_chi2(c, q, bands, L)
        end
        χ0 = χ_at(x)                              # (the stored recurrence march, as the joint loop renders; the fused march of
        r, J = Fit.spacetime_jacobian((c, Lc) -> Fit.spacetime_scan_residuals(c, Fit._on_backend(c, q), bands, eltype(c.αs)(L)), x, camera, backend; N)
        @test abs(sum(abs2, r) - χ0) <= 1e-8 * χ0 && size(J) == (length(r), 2)
        h = 1e-5
        gfd = [(χ_at(x .+ h .* (1:2 .== i)) - χ_at(x .- h .* (1:2 .== i))) / (2h) for i in 1:2]
        gx = 2 .* (J' * r)
        @test maximum(abs.(gx .- gfd) ./ max.(abs.(gfd), 1e-6 * maximum(abs, gfd))) < tol
        # every second frame in the spacetime block
        r2, J2 = Fit.spacetime_jacobian((c, Lc) -> Fit.spacetime_scan_residuals(c, Fit._on_backend(c, q), bands, eltype(c.αs)(L); every = 2), x, camera, backend; N)
        @test length(r2) < length(r) && all(isfinite, J2)
        # joint iterations lower the χ² and keep the spacetime near the truth (the few scans of this test constrain it loosely)
        cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
        q1, x1, history, accepted, events = fit_joint!(copy(q), x, bands, cache, camera; L, iterations = 24, η = 0.01, lm_every = 1)
        @test history[end][1] < history[1][1] && accepted > 0
        @test abs(x1[1] - a_true) <= 0.2 && abs(x1[2] - θ_true) <= deg2rad(8.0)
        @info "joint fit on scans ($label): χ² $(round(history[1][1], digits = 1)) → $(round(history[end][1], digits = 1)), spin $(x[1]) → $(round(x1[1], digits = 4)) (truth $a_true), inclination $(round(rad2deg(x1[2]), digits = 2))°, $accepted accepted steps"
    end
end

"""
The matrix-free Gauss–Newton polish on visibility scans: the residuals' χ² against the same
stored rendering, J·v against central differences, the adjoint identity ⟨Jv, w⟩ = ⟨v, Jᵀw⟩, and
two Levenberg–Marquardt steps from a perturbed start that lower the χ².
"""
function test_gauss_newton(backend; res = 6, N = 16, tol = 1e-5, label = "CPU backend")
    rng = Random.MersenneTwister(29)
    a_true = 0.9; θ_true = deg2rad(60.0)
    fov = 18.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    M_solar = 6.5e9; D = 16.8e6 * Transfer.PC; L = gravitational_radius(M_solar)
    p = polarized_test_params()
    cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cpu, a_true, θ_true; marcher = Fused(64))
    function scancov(t)
        sts = sort(randperm(rng, 5)[1:4]); s1 = Int[]; s2 = Int[]
        for i in 1:4, j in i+1:4
            push!(s1, sts[i]); push!(s2, sts[j])
        end
        return ScanCoverage(t, 3e9 .* randn(rng, length(s1)), 3e9 .* randn(rng, length(s1)), s1, s2)
    end
    cov = [scancov(0.0), scancov(0.0), scancov(20.0)]
    bands = [BandScans(ν, synthetic_scans(cpu, p, L, Δα, D, ν, cov; noise = 0.02, closures = false, rng), Δα, D) for ν in (230e9, 345e9)]
    cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
    regenerate!(cache, a_true, θ_true; marcher = Recurrence(64))
    q = p .+ 0.05 .* randn(rng, size(p)); q[12, :] .= p[12, :]
    qd = adapt_to(backend, q)
    @testset "$label Gauss–Newton polish on scans" begin
        sb = Fit.ScanBands(bands, cache)
        m = Fit.residual_length(sb)
        r = adapt_to(backend, zeros(m))
        χ = Fit.residuals!(r, sb, cache, qd, L)
        @test m == sum(8 * length(s.data.u) for b in bands for s in b.tr.scans)
        @test abs(χ - Fit._joint_chi2(cache, qd, bands, L)) <= 1e-9 * χ
        # J·v against central differences of the residual vector
        v = randn(rng, size(p)); v[12, :] .= 0
        Jv = adapt_to(backend, zeros(m)); Fit.jvp!(Jv, sb, cache, qd, adapt_to(backend, v), L)
        rp = adapt_to(backend, zeros(m)); rm = adapt_to(backend, zeros(m)); h = 1e-6
        Fit.residuals!(rp, sb, cache, adapt_to(backend, q .+ h .* v), L); Fit.residuals!(rm, sb, cache, adapt_to(backend, q .- h .* v), L)
        fd = (Array(rp) .- Array(rm)) ./ (2h)
        @test maximum(abs.(Array(Jv) .- fd)) <= tol * maximum(abs.(fd))
        # the adjoint identity
        w = randn(rng, m)
        Jtw = adapt_to(backend, zeros(size(p))); Fit.jtvp!(Jtw, sb, cache, qd, adapt_to(backend, w), L)
        lhs = dot(Array(Jv), w); rhs = dot(v, Array(Jtw))
        @test abs(lhs - rhs) <= 1e-8 * max(abs(lhs), abs(rhs))
        # two Levenberg–Marquardt steps lower the χ²
        infos = []
        q2, history = Fit.polish_timeresolved!(copy(qd), bands, cache, L; iterations = 2, solve_iterations = 8, probes = 4, rng = Random.MersenneTwister(3),
                                               callback = (it, x, v, dmp, info) -> push!(infos, info))
        @test history[end] < history[1] && all(isfinite, Array(q2))
        @test length(infos) == 2 && all(i -> isfinite(i.gain) && i.predicted > 0 && 0 <= i.solve_residual <= 1, infos)
        # the Hutchinson diagonal against the exact one, column by column, on a few columns
        sb = Fit.ScanBands(bands, cache)
        dg = adapt_to(backend, zeros(size(p))); Fit.normal_diagonal!(dg, sb, cache, qd, L; probes = 64, rng = Random.MersenneTwister(5))
        dgh = Array(dg); ratios = Float64[]
        for (i, j) in ((1, 1), (13, 2), (16, 1))
            e = zeros(size(p)); e[i, j] = 1
            Je = adapt_to(backend, zeros(m)); Fit.jvp!(Je, sb, cache, qd, adapt_to(backend, e), L)
            exact = sum(abs2, Array(Je))
            push!(ratios, dgh[i, j] / exact)
            @test exact / 4 <= dgh[i, j] <= 4 * exact                # a 64-probe estimate carries the couplings as noise (2.4× on one column of two overlapping parcels)
        end
        @info "Hutchinson diagonal ($label): estimate / exact on three columns $(round.(ratios, digits = 2))"
        # the explicit Jacobian (chunked duals) against J·v, and the dense Levenberg–Marquardt iterations
        J = zeros(m, length(p)); Fit.jacobian!(J, sb, cache, qd, L; chunk = Val(5))
        Jv2 = J * vec(v)
        @test maximum(abs.(Jv2 .- Array(Jv))) <= 1e-10 * maximum(abs.(Array(Jv)))
        q3, h3, A = Fit.polish_dense!(copy(qd), bands, cache, L; iterations = 2, λ = 1e-2, chunk = Val(5))
        @test h3[end] < h3[1] && all(isfinite, Array(q3)) && size(A) == (length(p), length(p))
        # the LSQR step against the exact damped solve on the explicit Jacobian
        r0 = adapt_to(backend, zeros(m)); Fit.residuals!(r0, sb, cache, qd, L)
        A0 = J' * J; g0 = J' * Float64.(Array(r0)); λ0 = 1e-2
        D0 = max.(diag(A0), 1e-6 * maximum(diag(A0)))
        p_exact = -((A0 + λ0 * Diagonal(D0)) \ g0)
        S0 = adapt_to(backend, reshape(1 ./ sqrt.(D0), size(p)))
        x0 = adapt_to(backend, zeros(size(p)))
        x0, k0, rel0 = Fit.lsqr_step!(x0, sb, cache, qd, r0, L; scale = S0, damp = sqrt(λ0), iterations = 4 * length(p), atol = 1e-12)
        p_lsqr = vec(Array(S0 .* x0))
        e_lsqr = norm(p_lsqr - p_exact) / norm(p_exact)
        @test e_lsqr <= 1e-6
        @info "LSQR step ($label): $(k0) iterations, relative difference from the dense damped solve $e_lsqr, normal residual $rel0"
        @info "dense Levenberg–Marquardt on scans ($label): χ² $(round(h3[1], digits = 1)) → $(round(h3[end], digits = 1)) in two iterations (the LSQR polish: → $(round(history[end], digits = 1)))"
        @info "Gauss–Newton polish on scans ($label): χ² $(round(history[1], digits = 1)) → $(round(history[end], digits = 1)) in two steps (gain ratios $(round.([i.gain for i in infos], digits = 2))); J·v vs FD $(maximum(abs.(Array(Jv) .- fd)) / maximum(abs.(fd))), adjoint identity $(abs(lhs - rhs) / abs(lhs))"
    end
end

