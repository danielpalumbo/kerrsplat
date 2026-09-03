# Gates for the polarized one-zone splats (Splats.PolarizedSplats through Transfer.RadiativeTransport):
#  1. the fused march on the CPU backend reproduces a host loop over stored samples (plumbing:
#     observer-first ordering, interval lengths, skipped samples, the observation time);
#  2. CUDA reproduces the CPU backend;
#  3. Enzyme reverse-mode gradients of an image loss with respect to all 20 parameters of every
#     splat agree with central finite differences;
#  4. a short Adam fit recovers a perturbed splat from its own I, Q, U, V image.

using StaticArrays
using LinearAlgebra
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Splats
using KerrSplat.Transfer: transfer_step, advance, rotate_to_screen, invariants, cap_polarization

const POL_ν = 230e9
const POL_L = gravitational_radius(4.0e6)

"Two overlapping splats with different fields and velocities (columns of the parameter matrix)."
function polarized_test_params()
    p = zeros(NPOLARIZEDPARAMS, 2)
    p[:, 1] = [4.0, 1.0, 0.0, 0.2, -0.1, -0.3, 1.0, 0.1, 0.0, 0.0, 0.0, log(1e6), log(2e6), log(20.0), log(30.0), 1.2, 0.7, 0.1, 0.45, 0.0]
    p[:, 2] = [3.5, 1.6, 0.2, 0.0, 0.1, -0.2, 0.9, 0.0, 0.2, 0.1, 0.0, log(1e6), log(1e6), log(35.0), log(15.0), 0.6, -1.0, -0.2, 0.3, 0.1]
    return p
end

"Host reference: the same elements and steps, looped over stored samples."
function polarized_image_host(cache_stored::GeodesicCache{T,N}, params, t_obs, ν, L) where {T,N}
    S = host(cache_stored.samples)
    consts = host(cache_stored.consts)
    met = Krang.Kerr(cache_stored.spin); θo = cache_stored.θo
    model = PolarizedSplats(params, fill(T(t_obs), 1))
    npix = npixels(cache_stored)
    out = Vector{SVector{4,T}}(undef, npix)
    for i in 1:npix
        pix = Krang.SlowLightIntensityPixel(met, consts.α[i], consts.β[i], θo)
        Δτ = Krang.total_mino_time(pix) / (N + 1)
        st = zero(RadiativeState{T})
        for k in 1:N
            s = S[i, k]
            (s.ok && s.r > Krang.horizon(met) * (1 + T(1e-3))) || continue
            j4 = zero(SVector{4,T}); α4 = zero(SVector{4,T}); ρ3 = zero(SVector{3,T}); active = false
            for e in 1:Transfer.nelements(model)
                cf, fr = Transfer.element(model, e, pix, s, ν)
                (cf.jI > 0 || cf.αI > 0 || cf.ρQ != 0 || cf.ρV != 0) || continue
                jj, aa, rr = rotate_to_screen(invariants(cap_polarization(cf), ν / fr.g), fr.χ)
                j4 += jj; α4 += aa; ρ3 += rr; active = true
            end
            active || continue
            Σ = s.r^2 + met.spin^2 * cos(s.θ)^2
            O, E = transfer_step(j4, α4, ρ3, L / ν * Σ * Δτ)
            st = advance(st, O, E)
        end
        out[i] = observed_stokes(st, ν)
    end
    return to_screen(cache_stored, out)
end

function polarized_march(q, cache, t_obs)
    out = Vector{RadiativeState{Float64}}(undef, npixels(cache))
    fill!(out, zero(RadiativeState{Float64}))
    polarized_image!(out, cache, q, t_obs, POL_ν, POL_L)
    return map(st -> observed_stokes(st, POL_ν), out)
end

function test_polarized_splats(backend; res = 24, N = 300, label = "")
    Geodesics.prepare_backend!(backend)
    @testset "polarized splats ($label)" begin
        a = 0.9; θo = deg2rad(60.0); t_obs = 0.0
        camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
        p = polarized_test_params()
        cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        img = Array(polarized_image(cache, adapt_to(backend, p), t_obs, POL_ν, POL_L))
        I = getindex.(img, 1)
        @test all(isfinite, I) && maximum(I) > 0
        @test maximum(hypot.(getindex.(img, 2), getindex.(img, 3)) ./ max.(I, 1e-300 * maximum(I))) < 1     # partially polarized
        if backend isa CPU
            cs = GeodesicCache(CPU(), camera, Val(N)); regenerate!(cs, a, θo; marcher = Recurrence(64))
            ref = polarized_image_host(cs, p, t_obs, POL_ν, POL_L)
            err = maximum(norm.(img .- ref)) / maximum(norm.(ref))
            @test err < 1e-10
            @info "polarized splats: fused march vs host loop, $(res)² × $N: max relative difference $err; peak I $(maximum(I)) cgs, LP fraction at the peak $(round(hypot(img[argmax(I)][2], img[argmax(I)][3]) / maximum(I), digits = 3)), V/I $(round(img[argmax(I)][4] / maximum(I), sigdigits = 3))"
        else
            cc = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cc, a, θo; marcher = Fused(64))
            ref = polarized_image(cc, p, t_obs, POL_ν, POL_L)
            err = maximum(norm.(img .- ref)) / maximum(norm.(ref))
            @test err < 1e-11
            @info "polarized splats on $label vs CPU backend: max relative difference $err"
        end
    end
end

function test_polarized_splat_gradients(; res = 10, N = 100, tol = 1e-5)
    a = 0.9; θo = deg2rad(60.0); t_obs = 0.0
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    p = polarized_test_params()
    target = polarized_march(p, cache, t_obs)
    scale = maximum(norm.(target))
    p_start = copy(p); p_start[1, :] .+= 0.4; p_start[13, :] .-= 0.2; p_start[16, :] .+= 0.2; p_start[18, :] .+= 0.1
    loss(q) = sum(abs2, reinterpret(Float64, polarized_march(q, cache, t_obs) .- target)) / scale^2
    @testset "Enzyme reverse gradient of a polarized image loss, $(res)² × $N, $(length(p)) parameters" begin
        L0 = loss(p_start)
        t0 = time()
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p_start)[1]
        t_first = time() - t0
        t0 = time(); g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p_start)[1]; t_grad = time() - t0
        # fourth-order central stencil; the floor covers derivatives at rounding level (t0 and logw of a
        # splat whose temporal envelope is flat over the observation have gradients of 1e-12)
        err = 0.0
        gmax = maximum(abs, g)
        for i in eachindex(p_start)
            h = 1e-3 * max(1.0, abs(p_start[i]))
            f(x) = (q = copy(p_start); q[i] = x; loss(q))
            x = p_start[i]
            fd = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
            err = max(err, abs(g[i] - fd) / max(abs(fd), 1e-6 * gmax))
        end
        @test err < tol
        @info "polarized splats Enzyme gradient: loss $L0; max relative error vs central finite differences $err; gradient $(round(t_grad, digits = 2)) s (first call $(round(t_first, digits = 1)) s)"
    end
end

function test_polarized_splat_fit(; res = 10, N = 100, iterations = 150)
    a = 0.9; θo = deg2rad(60.0); t_obs = 0.0
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    p_true = polarized_test_params()[:, 1:1]
    target = polarized_march(p_true, cache, t_obs)
    scale = maximum(norm.(target))
    loss(q) = sum(abs2, reinterpret(Float64, polarized_march(q, cache, t_obs) .- target)) / scale^2
    p = copy(p_true); p[1] += 0.5; p[2] -= 0.3; p[13] -= 0.3; p[14] += 0.15; p[16] += 0.3; p[18] -= 0.15
    @testset "Adam fit of one polarized splat, $(res)² × $N, $iterations iterations" begin
        L0 = loss(p)
        opt = Optimisers.setup(Optimisers.Adam(0.02), p)
        Lmin = L0
        for it in 1:iterations
            g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p)[1]
            opt, p = Optimisers.update(opt, p, g)
            Lmin = min(Lmin, loss(p))
        end
        @test Lmin < 1e-2 * L0
        @info "polarized splat fit: loss $L0 → $Lmin; position error $(norm(p[1:3] - p_true[1:3])) M, log density error $(abs(p[13] - p_true[13])), field angle error $(abs(p[16] - p_true[16]))"
    end
end
