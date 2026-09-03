# Gate 5 of the plan (slow light): a flaring splat (Gaussian in time) rendered as a movie. The
# flash appears in the direct (n = 0) image and, later, in the lensed (n = 1) image; the arrival
# times of both must equal the flare time plus the regularized lookback time t̃ of the rays that
# cross the splat (Krang's t̃ grows inward along the ray: it is the elapsed time to the observer
# minus the divergent r_obs + 2 ln r_obs), and the n = 1 echo lags the direct image by of order
# the half photon-orbit period 3√3 π M ≈ 16 M (Schwarzschild, face-on; 25 M here for the
# source at r = 6 M). In the limit of a wide temporal envelope the slow-light
# image is independent of the observation time (the fast-light limit).

using StaticArrays
using KerrSplat.Geodesics
using KerrSplat.Splats

function test_slowlight(backend; res = 96, N = 600, label = "")
    Geodesics.prepare_backend!(backend)
    @testset "slow light: flare echoes ($label)" begin
        a = 1e-3; θo = deg2rad(20.0)                      # Schwarzschild limit (Krang needs a > 0), nearly face-on
        fov = 9.0
        camera = Geodesics.Camera((-fov, fov), (-fov, fov), res)
        cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        r0 = 6.0; t_flare = 0.0; w = 1.5
        p = zeros(NSPLATPARAMS, 1)
        p[:, 1] = [r0, 0.0, 0.0, log(0.6), log(0.6), log(0.6), 1.0, 0.0, 0.0, 0.0, t_flare, log(w), 0.0]
        pb = adapt_to(backend, p)
        times = -30.0:0.5:70.0
        movie = [Array(thin_image(cache, pb, T)) for T in times]
        # regions: the direct image of the splat lies near the projected position of r0; the n = 1 image on the
        # far side of the ring (opposite in α, at |b| ≈ 5). Locate both from the movie itself: the pixel of the
        # first peak and the brightest pixel after the direct flash has faded, outside the direct region.
        αs = range(-fov, fov, length = res); βs = range(-fov, fov, length = res)
        tot = [sum(img) for img in movie]
        # direct image: brightest pixel in the brightest frame
        k1 = argmax(tot); img1 = movie[k1]; i1 = argmax(img1)
        lc1 = [img[i1] for img in movie]
        t1 = times[argmax(lc1)]
        # echo: mask the direct image region (radius 2 M around i1) and find the brightest pixel in the later frames
        mask = [hypot(αs[i] - αs[i1[1]], βs[j] - βs[i1[2]]) > 2.0 for i in 1:res, j in 1:res]
        later = findall(t -> t > t1 + 8, times)
        best = (0.0, k1, i1)
        for k in later
            m = movie[k] .* mask
            i2 = argmax(m)
            m[i2] > best[1] && (best = (m[i2], k, i2))
        end
        i2 = best[3]
        lc2 = [img[i2] for img in movie]
        t2 = times[argmax(lc2)]
        # expected arrival times from the geodesics: t_flare − t̃ at the samples nearest the splat centre on each ray
        cs = GeodesicCache(CPU(), Geodesics.Camera([αs[i1[1]], αs[i2[1]]], [βs[i1[2]], βs[i2[2]]]), Val(N))
        regenerate!(cs, a, θo; marcher = Recurrence(64))
        S = host(cs.samples)
        met = Krang.Kerr(a)
        function arrival(i)
            best = (Inf, 0.0)
            for k in 1:N
                s = S[i, k]
                s.ok || continue
                x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
                d = hypot(x - r0, y, z)
                d < best[1] && (best = (d, t_flare + s.t))       # t_obs = t_em + t̃ (t̃ is the regularized lookback)
            end
            return best
        end
        d1, ta1 = arrival(1); d2, ta2 = arrival(2)
        @info "slow light: direct flash peaks at t_obs = $t1 (expected $(round(ta1, digits = 2)) from t̃ at $(round(d1, digits = 2)) M from the centre); echo at pixel ($(round(αs[i2[1]], digits = 2)), $(round(βs[i2[2]], digits = 2))) peaks at $t2 (expected $(round(ta2, digits = 2)), closest approach $(round(d2, digits = 2)) M); delay $(t2 - t1) M vs 3√3π = $(round(3 * sqrt(3) * π, digits = 2)) M"
        @test abs(t1 - ta1) <= 1.0
        @test abs(t2 - ta2) <= 1.5
        @test 10 < t2 - t1 < 30
        # the echo is demagnified in area (its surface brightness is not): compare region-integrated fluxes at the peaks
        region(img, i0) = sum(img[i, j] for i in 1:res, j in 1:res if hypot(αs[i] - αs[i0[1]], βs[j] - βs[i0[2]]) <= 2.0)
        F1 = region(movie[argmax(lc1)], i1); F2 = region(movie[argmax(lc2)], i2)
        @test F2 < 0.5 * F1
        @info "slow light: region fluxes at the peaks, direct $F1 vs echo $F2 (ratio $(round(F2 / F1, digits = 3)))"
        # fast-light limit: a wide envelope makes the image independent of the observation time
        pw = copy(p); pw[12] = log(1e9)
        pwb = adapt_to(backend, pw)
        imA = Array(thin_image(cache, pwb, -100.0)); imB = Array(thin_image(cache, pwb, 300.0))
        @test maximum(abs.(imA .- imB)) < 1e-9 * maximum(imA)
    end
end
