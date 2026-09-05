# Gates for the half-orbit (sub-image) decomposition: the passage counter in `WindingState` and
# the truncation of rays after their nmax-th passage through the midplane slab.
#
# 1. Identity: with no truncation the winding path reproduces the plain polarized image exactly,
#    on the CPU and on CUDA. (The slab half-thickness must exceed the emitters' vertical extent by
#    several σ: emission beyond the slab after a passage is counted to the next order.)
# 2. Independent reference: the Mino times at which our marcher's rays cross the equatorial
#    plane (detected from consecutive samples and interpolated) agree with Krang's own
#    `Gθ(pix, π/2, isindir, n)`, the Mino time of the n-th image of an equatorial source, for
#    n = 0, 1, 2 (either branch of `isindir`).
# 3. Sub-image geometry: for a thin ring of parcels at radius r₀ in the midplane, the order-n
#    sub-image (the difference of the images truncated at n and n − 1) is confined to the pixels
#    whose n-th crossing radius from Krang's `emission_radius` lies near r₀, and vanishes where
#    it does not; the sub-images are demagnified in turn, and their sum over n ≤ 2 leaves only a
#    small remainder to the untruncated image.
# 4. Enzyme reverse gradients run through the truncated path (vs a stencil).

using Krang

"Consumer recording the Mino times of the first three equatorial crossings of each ray."
struct CrossingRecorder end
@inline function (::CrossingRecorder)(acc::SVector{5,T}, j, k, s::Geodesics.GeodesicSample, Δτ, pix) where {T}
    # acc = (previous cosθ (NaN at start), count, τ₁, τ₂, τ₃)
    s.ok || return acc
    c = cos(s.θ); cp = acc[1]
    isnan(cp) && return SVector(c, acc[2], acc[3], acc[4], acc[5])
    if cp * c < 0
        τ = (k - 1) * Δτ + Δτ * cp / (cp - c)          # linear interpolation between the samples at (k−1)Δτ and kΔτ
        n = Int(acc[2]) + 1
        return SVector(c, T(n), n == 1 ? τ : acc[3], n == 2 ? τ : acc[4], n == 3 ? τ : acc[5])
    end
    return SVector(c, acc[2], acc[3], acc[4], acc[5])
end

function test_winding(backend; res = 24, N = 300, label = "CPU")
    a = 0.94; θo = deg2rad(17.0)
    fov = 16.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    M_solar = 6.5e9; L = gravitational_radius(M_solar); ν = 230e9
    met = Krang.Kerr(a)
    # a thin ring of eight optically thin parcels at r₀ in the midplane, thin in z
    r0 = 6.0; nring = 16
    p = zeros(NPOLARIZEDPARAMS, nring)
    for i in 1:nring
        φ = 2π * (i - 0.5) / nring
        x, y, z = quasi_cartesian_kerr_schild(met, r0, π / 2, φ)
        p[:, i] = [x, y, 0.0, log(0.8), log(0.8), log(0.08), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(2e4), log(30.0), log(10.0), π / 2, π / 2, 0.0, 0.3, 0.0, 0.0]
    end
    pd = adapt_to(backend, p)
    @testset "half-orbit decomposition ($label)" begin
        # 1. identity without truncation
        plain = Array(polarized_image(cache, pd, 0.0, ν, L))
        same = Array(polarized_image(cache, pd, 0.0, ν, L; nmax = 100, slab = 0.5))
        @test maximum(norm.(same .- plain)) <= 1e-13 * maximum(norm.(plain))
        # 2. crossing Mino times vs Krang's n-th image times
        rec = KernelAbstractions.allocate(backend, SVector{5,Float64}, npixels(cache))
        fill!(rec, SVector(NaN, 0.0, 0.0, 0.0, 0.0))
        fused_march!(CrossingRecorder(), rec, cache)
        recs = Array(to_screen(cache, rec))
        worst = 0.0; ncross = 0
        for i in 1:res, j in 1:res
            α = camera.αs[i + (j - 1) * res]; β = camera.βs[i + (j - 1) * res]
            pix = Krang.SlowLightIntensityPixel(met, α, β, θo)
            for n in 0:2
                τ_ours = recs[i, j][3 + n]
                τ_ours > 0 || continue
                best = Inf
                for isindir in (false, true)
                    τk, _, _, _, _, ok = Krang.Gθ(pix, π / 2, isindir, n)
                    ok && isfinite(τk) && (best = min(best, abs(τk - τ_ours)))
                end
                worst = max(worst, best); ncross += 1
            end
        end
        @test ncross > res * res ÷ 2
        @test worst < 2e-3                                        # linear interpolation between samples
        @info "winding ($label): $ncross crossings (n ≤ 2) on $(res)² pixels; worst |Δτ| vs Krang's Gθ $worst"
        # 3. sub-images against Krang's emission radii
        imgs = [Array(polarized_image(cache, pd, 0.0, ν, L; nmax = n, slab = 0.5)) for n in 0:2]     # slab ≥ 6σ_z of the parcels
        sub = [getindex.(imgs[1], 1), getindex.(imgs[2], 1) .- getindex.(imgs[1], 1), getindex.(imgs[3], 1) .- getindex.(imgs[2], 1)]
        full = getindex.(plain, 1)
        @test all(sub[2] .>= -1e-12 * maximum(full)) && all(sub[3] .>= -1e-12 * maximum(full))
        fluxes = sum.(sub)
        @test 0.005 < fluxes[2] / fluxes[1] < 0.5 && 0.005 < fluxes[3] / fluxes[2] < 0.5
        @test (sum(full) - sum(fluxes)) / sum(full) < 0.1
        σring = 0.8
        for n in 0:2
            near = falses(res, res); far = trues(res, res)
            for i in 1:res, j in 1:res
                α = camera.αs[i + (j - 1) * res]; β = camera.βs[i + (j - 1) * res]
                pix = Krang.SlowLightIntensityPixel(met, α, β, θo)
                for isindir in (false, true)
                    rs, _, _, _, ok = Krang.emission_radius(pix, π / 2, isindir, n)
                    ok && rs > 0 || continue
                    abs(rs - r0) < 0.5 * σring && (near[i, j] = true)
                    abs(rs - r0) < 5 * σring && (far[i, j] = false)    # beyond the parcels' Gaussian tails (e^{-12.5})
                end
            end
            peak = maximum(sub[n + 1])
            @test maximum(sub[n + 1][far]; init = 0.0) < 1e-4 * peak
            @test count(near) == 0 || minimum(sub[n + 1][near]) > 0.05 * peak
        end
        @info "winding ($label): sub-image fluxes n = 0, 1, 2: $(round.(fluxes; sigdigits = 4)) (ratios $(round(fluxes[2] / fluxes[1]; sigdigits = 3)), $(round(fluxes[3] / fluxes[2]; sigdigits = 3))); remainder $(round((sum(full) - sum(fluxes)) / sum(full); sigdigits = 2))"
    end
end

function test_winding_gradient(; res = 8, N = 120)
    a = 0.94; θo = deg2rad(17.0)
    fov = 16.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    L = gravitational_radius(6.5e9); ν = 230e9
    p = zeros(NPOLARIZEDPARAMS, 1)
    p[:, 1] = [5.5, 1.0, 0.1, log(1.0), log(1.0), log(0.3), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e5), log(30.0), log(10.0), 1.0, 0.5, 0.0, 0.3, 0.0, 0.0]
    @testset "gradient through the truncated path" begin
        for nmax in (0, 1)
            loss(q) = (out = Vector{WindingState{Float64}}(undef, npixels(cache)); fill!(out, zero(WindingState{Float64}));
                       polarized_image!(out, cache, q, 0.0, ν, L; nmax, slab = 0.3); sum(st -> observed_stokes(st, ν)[1], out))
            g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p)[1]
            for i in (1, 13, 15)
                h = 1e-4; f(x) = (q = copy(p); q[i] = x; loss(q)); x = p[i]
                fd = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
                @test abs(g[i] - fd) / abs(fd) < 1e-5
            end
        end
        # and through the movie χ² with truncation (the accumulator type is chosen in a type-stable branch)
        clean = polarized_cube(cache, p, [0.0], [ν], L; nmax = 0, slab = 0.5)
        movie = StokesMovie(clean .* 1.05, [0.0], [ν], SVector(1.0, 1.0, 1.0, 1.0) * 1e-3 * maximum(norm.(clean)))
        lossχ(q) = chi2(q, movie, cache, L; nmax = 0, slab = 0.5)
        gχ = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(lossχ), p)[1]
        for i in (1, 13)
            h = 1e-4; fχ(x) = (q = copy(p); q[i] = x; lossχ(q)); x = p[i]
            fd = (-fχ(x + 2h) + 8fχ(x + h) - 8fχ(x - h) + fχ(x - 2h)) / (12h)
            @test abs(gχ[i] - fd) / abs(fd) < 1e-6
        end
    end
end
