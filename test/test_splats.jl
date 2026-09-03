# Phase 1 of the main plan (§7.6): optically-thin Gaussian splats rendered through the fused
# marcher. Checks: (1) the emissivity and its parameterization; (2) the fused renderer against
# a host loop over Krang's direct samples with the same integrand; (3) CPU and CUDA images
# agree; (4) Enzyme reverse-mode gradients of an image loss with respect to all splat parameters
# against central finite differences (main plan §7.5 gate 6), on the CPU backend.

using Test
using Random
using KernelAbstractions
using Krang
using StaticArrays
using Enzyme
using Optimisers
using KerrSplat.Geodesics
using KerrSplat.Splats

"Two test splats: one near the equatorial plane, one above it, born at different times."
function test_splat_params()
    p = zeros(NSPLATPARAMS, 2)
    p[:, 1] = [6.0, 4.0, 0.5, log(1.5), log(1.2), log(0.8), 1.0, 0.1, -0.2, 0.3, 0.0, log(40.0), log(2.0)]
    p[:, 2] = [-3.0, 2.0, 2.5, log(1.0), log(1.0), log(2.0), 0.9, -0.3, 0.2, 0.1, 10.0, log(15.0), log(1.0)]
    return p
end

"Host reference: the renderer's integrand on Krang's direct samples, one thread per ray."
function thin_image_host(camera::Camera, a, θo, ::Val{N}, params, t_obs) where {N}
    met = Krang.Kerr(a)
    img = zeros(npixels(camera))
    c = ThinRenderer(params, t_obs)
    Threads.@threads for i in 1:npixels(camera)
        pix = Krang.SlowLightIntensityPixel(met, camera.αs[i], camera.βs[i], θo)
        Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
        acc = 0.0
        for k in 1:N
            acc = c(acc, i, k, direct_sample(pix, k * Δτ), Δτ, pix)
        end
        img[i] = acc
    end
    return reshape(img, size(camera))
end

function test_splats(backend; res::Int, N::Int, label::String)
    a, θo = 0.94, deg2rad(60.0)
    t_obs = 20.0
    p = test_splat_params()
    @testset "$label splats $(res)² × $N" begin
        # emissivity: peak at the centre and birth time, principal-axis widths, rotation invariance
        @test splat_emissivity(p, 1, 0.0, 6.0, 4.0, 0.5) ≈ 2.0
        @test splat_emissivity(p, 2, 10.0, -3.0, 2.0, 2.5) ≈ 1.0
        p0 = copy(p); p0[7:10, 1] = [1.0, 0.0, 0.0, 0.0]        # identity rotation
        @test splat_emissivity(p0, 1, 0.0, 6.0 + 1.5, 4.0, 0.5) ≈ 2.0 * exp(-0.5)
        @test splat_emissivity(p0, 1, 0.0, 6.0, 4.0 + 1.2, 0.5) ≈ 2.0 * exp(-0.5)
        @test splat_emissivity(p0, 1, 40.0, 6.0, 4.0, 0.5) ≈ 2.0 * exp(-0.5)
        p1 = copy(p0); p1[7:10, 1] .*= 3.0                         # quaternion scale is irrelevant
        @test splat_emissivity(p1, 1, 0.3, 5.0, 4.4, 0.1) ≈ splat_emissivity(p0, 1, 0.3, 5.0, 4.4, 0.1)

        camera = Camera((-10.0, 10.0), (-10.0, 10.0), res)
        cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        params = KernelAbstractions.allocate(backend, Float64, size(p)...)
        copyto!(params, p)
        img = Array(thin_image(cache, params, t_obs))
        ref = thin_image_host(camera, a, θo, Val(N), p, t_obs)
        e = maximum(abs.(img .- ref)) / maximum(ref)
        @info "$label splats $(res)² × $N: total flux $(sum(img)); fused image vs host loop on Krang's direct samples: max |Δ|/max = $e"
        @test maximum(ref) > 0
        @test e <= 1e-8            # the two marchers differ at 1e-11 in the coordinates; Krang's vortical bug is masked by the splats' positions
        return img
    end
end

# ---- gradients (main plan §7.5 gate 6) ------------------------------------------------------

"""
Host renderer built directly on `march_ray` (no kernel launch), so that Enzyme differentiates
a plain Julia loop: image in sorted order for the parameter matrix `p`.
"""
function thin_image_march(p, cache::GeodesicCache{T,N}, t_obs) where {T,N}
    pcs = cache.consts
    met = Krang.Kerr(cache.spin)
    img = zeros(eltype(p), npixels(cache))
    c = ThinRenderer(p, t_obs)
    r = cache.ranges
    # one type-stable loop per root case (a loop over a heterogeneous tuple would dispatch dynamically, which Enzyme rejects)
    for j in r.case2
        img[j] = march_ray(c, zero(eltype(p)), pcs, j, met, cache.θo, Case2(), Val(N), Val(64))[1]
    end
    for j in r.case3
        img[j] = march_ray(c, zero(eltype(p)), pcs, j, met, cache.θo, Case3(), Val(N), Val(64))[1]
    end
    for j in r.case4
        img[j] = march_ray(c, zero(eltype(p)), pcs, j, met, cache.θo, Case4(), Val(N), Val(64))[1]
    end
    return img
end

function test_splat_gradients(; res::Int, N::Int, tol::Float64)
    a, θo = 0.94, deg2rad(60.0)
    t_obs = 20.0
    camera = Camera((-10.0, 10.0), (-10.0, 10.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    p = test_splat_params()
    target = thin_image_march(p, cache, t_obs)
    p_start = p .+ 0.05 .* randn(MersenneTwister(1), size(p))
    loss(q) = sum(abs2, thin_image_march(q, cache, t_obs) .- target)
    @testset "Enzyme reverse gradient of an image loss, $(res)² × $N, $(length(p)) parameters" begin
        L0 = loss(p_start)
        @test L0 > 0
        t0 = time()
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p_start)[1]
        t_first = time() - t0
        t0 = time()
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p_start)[1]
        t_grad = time() - t0
        t0 = time()
        loss(p_start)
        t_loss = time() - t0
        gfd = similar(p_start)
        h = 1e-5
        for i in eachindex(p_start)
            q = copy(p_start); q[i] += h; Lp = loss(q)
            q[i] -= 2h; Lm = loss(q)
            gfd[i] = (Lp - Lm) / 2h
        end
        err = maximum(abs.(g .- gfd) ./ (abs.(gfd) .+ 1e-8 * maximum(abs.(gfd))))
        @info "Enzyme reverse gradient: loss $L0; max relative error vs central finite differences $err; gradient $(round(t_grad, digits = 3)) s ($(round(t_grad / t_loss, digits = 1))× the forward pass of $(round(t_loss, digits = 3)) s; first call $(round(t_first, digits = 1)) s)"
        @test all(isfinite, g)
        @test err <= tol
    end
end

# ---- a small fit (Phase 1: "fit synthetic hotspot images") ----------------------------------

"""
Recover two splats from a synthetic image with Adam on the Enzyme gradient of the image loss,
starting from perturbed parameters. Checks that the loss falls by orders of magnitude and the
positions and amplitudes come back.
"""
function test_splat_fit(; res::Int, N::Int, iterations::Int)
    a, θo = 0.94, deg2rad(60.0)
    t_obs = 20.0
    camera = Camera((-10.0, 10.0), (-10.0, 10.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    p_true = test_splat_params()
    target = thin_image_march(p_true, cache, t_obs)
    scale = maximum(target)
    loss(q) = sum(abs2, thin_image_march(q, cache, t_obs) .- target) / scale^2
    rng = MersenneTwister(7)
    p = copy(p_true)
    p[1:3, :] .+= 0.3 .* randn(rng, 3, 2)          # positions off by ~0.3 M
    p[13, :] .+= 0.2 .* randn(rng, 2)              # amplitudes off by ~20 %
    L0 = loss(p)
    pos_err0 = maximum(abs.(p[1:3, :] .- p_true[1:3, :]))
    opt = Optimisers.setup(Optimisers.Adam(0.03), p)
    Lmin = L0
    for it in 1:iterations
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p)[1]
        opt, p = Optimisers.update(opt, p, g)
        Lmin = min(Lmin, loss(p))
    end
    L1 = loss(p)
    pos_err = maximum(abs.(p[1:3, :] .- p_true[1:3, :]))
    @testset "Adam fit of two splats, $(res)² × $N, $iterations iterations" begin
        @info "splat fit: loss $L0 → $L1 (best $Lmin); position errors $pos_err0 → $pos_err M, log-amplitude errors $(maximum(abs.(p[13, :] .- p_true[13, :])))"
        # a 12² screen over ±10 M cannot pin a 1.5 M splat much better than a few tenths of M
        @test L1 < 1e-2 * L0
        @test pos_err < 0.5 * pos_err0
        @test pos_err < 0.3
        @test maximum(abs.(p[13, :] .- p_true[13, :])) < 0.1
    end
end
