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
