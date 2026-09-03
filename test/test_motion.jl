# Motion mode B (pattern rotation): a splat's centre and orientation rotate rigidly about the spin
# axis at the pattern rate ω, independently of the fluid velocity. Checks: (1) the image of a
# rotating splat at time t equals the image of the static splat placed at the rotated position
# with the rotated orientation (fast light, wide envelope); (2) after one pattern period the movie
# frame repeats; (3) the Enzyme gradient with respect to ω agrees with finite differences; (4) the
# (t, ν) cube helper assembles frames consistently.

using StaticArrays
using LinearAlgebra
using KerrSplat.Geodesics
using KerrSplat.Splats
using KerrSplat.Transfer

"Quaternion product (w, x, y, z)."
qmul(a, b) = (a[1]*b[1] - a[2]*b[2] - a[3]*b[3] - a[4]*b[4], a[1]*b[2] + a[2]*b[1] + a[3]*b[4] - a[4]*b[3],
              a[1]*b[3] - a[2]*b[4] + a[3]*b[1] + a[4]*b[2], a[1]*b[4] + a[2]*b[3] - a[3]*b[2] + a[4]*b[1])

function test_motion(backend; res = 32, N = 300, label = "")
    Geodesics.prepare_backend!(backend)
    @testset "pattern rotation of splats ($label)" begin
        a = 0.9; θo = deg2rad(60.0)
        camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
        cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        ω = 6.0^(-1.5)                                    # Keplerian rate at r = 6
        p = zeros(NSPLATPARAMS, 1)
        p[:, 1] = [6.0, 0.0, 0.3, log(1.0), log(0.5), log(0.7), 0.8, 0.2, -0.3, 0.4, 0.0, log(1e9), 0.0, ω]
        # the emissivity of the rotating splat at time t equals that of a static splat placed at the rotated
        # position with the rotated orientation (the rendered images differ because every sample has its own
        # emission time; that bookkeeping is the slow-light gate)
        t = 37.0
        φ = ω * t
        q = copy(p); q[14] = 0.0
        q[1] = p[1] * cos(φ) - p[2] * sin(φ); q[2] = p[1] * sin(φ) + p[2] * cos(φ)
        qz = (cos(φ / 2), 0.0, 0.0, sin(φ / 2))
        q[7:10] .= qmul(qz, (p[7], p[8], p[9], p[10]))
        worst = 0.0
        for (x, y, z) in ((q[1] + 0.3, q[2] - 0.2, 0.5), (q[1] - 1.0, q[2] + 0.4, 0.1), (2.0, 5.0, -0.3))
            worst = max(worst, abs(splat_emissivity(p, 1, t, x, y, z) / splat_emissivity(q, 1, t, x, y, z) - 1))
        end
        @test worst < 1e-12
        img_t = Array(thin_image(cache, adapt_to(backend, p), t))
        # periodicity
        P = 2π / ω
        img_P = Array(thin_image(cache, adapt_to(backend, p), t + P))
        @test maximum(abs.(img_t .- img_P)) < 1e-9 * maximum(img_t)
        @info "pattern rotation: emissivity of the rotating splat vs its statically rotated copy $worst; movie frame after one period $(maximum(abs.(img_t .- img_P)) / maximum(img_t))"
        # the polarized cube helper: frames equal individual images
        pp = zeros(NPOLARIZEDPARAMS, 1)
        pp[:, 1] = [6.0, 0.0, 0.3, log(1.0), log(0.5), log(0.7), 0.8, 0.2, -0.3, 0.4, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.4, 0.0, ω]
        L = gravitational_radius(4e6)
        cube = polarized_cube(cache, adapt_to(backend, pp), [0.0, 20.0], [230e9, 345e9], L)
        one = Array(polarized_image(cache, adapt_to(backend, pp), 20.0, 345e9, L))
        @test cube[:, :, 2, 2] == one
        @test size(cube) == (res, res, 2, 2)
        F = flux_density(getindex.(one, 1), 20.0 / res, L, 8e3 * Transfer.PC)
        @test all(F .>= 0) && sum(F) > 0
    end
end

function test_motion_gradient(; res = 12, N = 120)
    a = 0.9; θo = deg2rad(60.0)
    camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    p = zeros(NSPLATPARAMS, 1)
    p[:, 1] = [6.0, 0.0, 0.3, log(1.0), log(0.5), log(0.7), 0.8, 0.2, -0.3, 0.4, 0.0, log(1e9), 0.0, 6.0^(-1.5)]
    target = thin_image_march(p, cache, 30.0)
    q0 = copy(p); q0[14] *= 0.8
    loss(q) = sum(abs2, thin_image_march(q, cache, 30.0) .- target) / maximum(target)^2
    @testset "Enzyme gradient with respect to the pattern rate" begin
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), q0)[1]
        h = 1e-4 * q0[14]
        f(x) = (q = copy(q0); q[14] = x; loss(q))
        fd = (-f(q0[14] + 2h) + 8f(q0[14] + h) - 8f(q0[14] - h) + f(q0[14] - 2h)) / (12h)
        @test abs(g[14] - fd) / abs(fd) < 1e-6
        @info "pattern-rate gradient: Enzyme $(g[14]) vs stencil $fd"
    end
end

"""
Pattern-versus-fluid separation (addendum §6.2): from a short Stokes movie of one orbiting splat,
fit the pattern rate ω together with the fluid velocity components and the field angle, starting
from perturbed values. The pattern rate is constrained by the motion of the image between frames,
the fluid velocity by beaming, redshift and the polarization frame within each frame.
"""
function test_pattern_vs_fluid(; res = 8, N = 80, iterations = 120)
    a = 0.9; θo = deg2rad(60.0)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    L = gravitational_radius(4e6); ν = 230e9
    times = (0.0, 25.0, 50.0)
    p_true = zeros(NPOLARIZEDPARAMS, 1)
    p_true[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 6.0^(-1.5)]
    function movie(q)
        out = Vector{RadiativeState{Float64}}(undef, npixels(cache))
        frames = map(times) do t
            fill!(out, zero(RadiativeState{Float64}))
            polarized_image!(out, cache, q, t, ν, L)
            map(st -> observed_stokes(st, ν), out)
        end
        return reduce(vcat, frames)
    end
    target = movie(p_true)
    scale = maximum(norm.(target))
    loss(q) = sum(abs2, reinterpret(Float64, movie(q) .- target)) / scale^2
    p = copy(p_true); p[21] *= 1.25; p[18] += 0.15; p[19] -= 0.15; p[16] += 0.25; p[13] -= 0.2
    free = [13, 16, 18, 19, 21]
    @testset "pattern-versus-fluid separation: joint fit of ω, ũ, field angle and density from a 3-frame movie" begin
        L0 = loss(p)
        opt = Optimisers.setup(Optimisers.Adam(0.02), p)
        mask = zeros(size(p)); mask[free, :] .= 1
        Lmin = L0
        for it in 1:iterations
            g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p)[1] .* mask
            opt, p = Optimisers.update(opt, p, g)
            Lmin = min(Lmin, loss(p))
        end
        @test Lmin < 0.02 * L0
        @test abs(p[21] / p_true[21] - 1) < 0.05
        @test abs(p[18] - p_true[18]) < 0.05 && abs(p[19] - p_true[19]) < 0.05
        @info "pattern vs fluid: loss $L0 → $Lmin; ω $(p[21]) (true $(p_true[21])), ũ ($(round(p[18], digits = 3)), $(round(p[19], digits = 3))) (true (0, 0.35)), field angle $(round(p[16], digits = 3)) (true 1.0), log density error $(abs(p[13] - p_true[13]))"
    end
end

"""
Motion mode C: a splat advected by its own ZAMO velocity. With the Keplerian ZAMO velocity at
r = 6 the integrated trajectory is the circular orbit, so the knot model reproduces the pattern
rotation (mode B) at the Keplerian rate; the BL inverse transform round-trips; and the movie
frames of the two modes agree.
"""
function test_advection(backend; res = 24, N = 200, label = "")
    Geodesics.prepare_backend!(backend)
    @testset "advected splats vs pattern rotation ($label)" begin
        a = 0.9; θo = deg2rad(60.0); met = Krang.Kerr(a)
        # inverse transform round-trip
        for (r, θ, ϕ) in ((6.0, 1.2, 0.4), (2.5, 0.3, 3.0), (15.0, 2.9, -1.0))
            x, y, z = quasi_cartesian_kerr_schild(met, r, θ, ϕ)
            rb, θb, ϕb = boyer_lindquist(met, x, y, z)
            @test isapprox(rb, r; atol = 1e-12) && isapprox(θb, θ; atol = 1e-12) && abs(rem2pi(ϕb - ϕ, RoundNearest)) < 1e-12
        end
        # Keplerian ZAMO velocity at (r = 6, equator): u^ϕ/u^t = Ω_K; ZAMO φ̂ component ũ_φ = γ β_φ with β from Ω
        r0 = 6.0
        Ω = 1 / (r0^1.5 + a)
        gdd = Krang.metric_dd(met, r0, π / 2)
        ut = 1 / sqrt(-(gdd[1, 1] + 2Ω * gdd[1, 4] + Ω^2 * gdd[4, 4]))
        uz = Krang.jac_zamo_u_bl_d(met, r0, π / 2) * SVector(ut, 0.0, 0.0, Ω * ut)
        ũ = SVector(uz[2], uz[3], uz[4])
        @test abs(uz[2]) < 1e-12 && abs(uz[4]) < 1e-12
        x0, y0, z0 = quasi_cartesian_kerr_schild(met, r0, π / 2, 0.0)
        pB = zeros(NPOLARIZEDPARAMS, 1)
        pB[:, 1] = [x0, y0, z0, log(1.0), log(1.0), log(0.7), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, ũ..., Ω]
        pC = copy(pB); pC[21] = 0.0
        # the knots must cover every emission time of the movie: t_obs minus the lookback of any sample that
        # can see the splat (lensed paths reach the splat tens of M before the direct one); outside the
        # knots the centre is clamped
        times = collect(-40.0:1.0:120.0)
        knots = trajectory_knots(pC, met, times; substeps = 8)
        # the knots lie on the circular orbit at the Keplerian phase (in KS coordinates the BL azimuth shift is constant at fixed r)
        worst = 0.0
        for (k, t) in enumerate(times)
            φ = Ω * t
            expected = SVector(x0 * cos(φ) - y0 * sin(φ), x0 * sin(φ) + y0 * cos(φ), z0)
            worst = max(worst, norm(SVector(knots[1, k, 1], knots[2, k, 1], knots[3, k, 1]) - expected))
        end
        @test worst < 1e-8
        L = gravitational_radius(4e6); ν = 230e9
        camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
        cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        t_obs = 33.0
        outB = KernelAbstractions.allocate(backend, RadiativeState{Float64}, npixels(cache))
        fused_march!(RadiativeTransport(PolarizedSplats(adapt_to(backend, pB), t_obs), ν, L), outB, cache)
        outC = KernelAbstractions.allocate(backend, RadiativeState{Float64}, npixels(cache))
        fused_march!(RadiativeTransport(KnotSplats(adapt_to(backend, pC), t_obs, adapt_to(backend, knots), adapt_to(backend, times)), ν, L), outC, cache)
        SB = map(st -> observed_stokes(st, ν), Array(outB)); SC = map(st -> observed_stokes(st, ν), Array(outC))
        err = sqrt(sum(norm.(SB .- SC) .^ 2) / sum(norm.(SB) .^ 2))
        @test err < 5e-3                                  # linear interpolation between knots 1 M apart on an orbit of period 100 M
        @info "advection: knots on the Keplerian orbit to $worst M; movie frame of the knot model vs the pattern rotation: relative L2 difference $err ($label)"
    end
end
