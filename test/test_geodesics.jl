# Gate 1 of the GPU plan (§8): K1 (per-pixel constants) and the stored-mode direct marcher K2
# on a KernelAbstractions backend, against Krang's own CPU evaluation.

using Test
using KernelAbstractions
using Krang
using KerrSplat.Geodesics

const SPACETIMES = (
    (a = 0.94, θo = 60.0),
    (a = 0.5, θo = 30.0),
    (a = 0.94, θo = 1.0),
    (a = 0.94, θo = 89.0),
    (a = 0.998, θo = 45.0),
    (a = 0.7, θo = 120.0),
    (a = 0.2, θo = 75.0),
)

function test_geodesics(backend; res::Int, N::Int, tol::Float64, tol_pos::Float64 = tol, label::String)
    @testset "Camera" begin
        cam = Camera((-10.0, 10.0), (-10.0, 10.0), 5)
        @test size(cam) == (5, 5)
        @test npixels(cam) == 25
        @test cam.αs[1:5] == collect(range(-10.0, 10.0, length = 5))
        @test cam.βs[1:5] == fill(-10.0, 5)
        @test cam.αs[2 + 3 * 5] == -5.0 && cam.βs[2 + 3 * 5] == 5.0   # (i, j) = (2, 4)
        @test_throws DimensionMismatch Camera([1.0, 2.0], [1.0], (2, 1))
    end

    @testset "ConcretePixel" begin
        pixt = typeof(Krang.SlowLightIntensityPixel(Krang.Kerr(0.5), 0.1, 0.2, 1.0))
        @test ConcretePixel{Float64} == pixt
        @test isbitstype(ConcretePixel{Float64})
    end

    camera = Camera((-10.0, 10.0), (-10.0, 10.0), res)
    cache = GeodesicCache(backend, camera, Val(N))
    @test npixels(cache) == res^2
    @test nsamples(cache) == N
    @test isnan(cache.spin)

    seen_cases = (false, false, false)
    for st in SPACETIMES
        a = st.a
        θo = deg2rad(st.θo)
        @testset "$label a=$(st.a) θo=$(st.θo)°" begin
            regenerate!(cache, a, θo)
            @test cache.spin == a && cache.θo == θo

            # the sort is a permutation, grouped by case, with consistent ranges
            perm = cache.perm_host
            @test sort(perm) == 1:npixels(cache)
            @test Array(cache.perm) == perm
            nr = Array(host(cache.consts).numreals)
            r = cache.ranges
            @test length(r.case2) + length(r.case3) + length(r.case4) == npixels(cache)
            @test all(nr[r.case2] .== 4) && all(nr[r.case3] .== 2) && all(nr[r.case4] .== 0)
            @test Array(cache.numreals_screen)[perm] == nr
            @test unsort(cache, host(cache.consts).α) == camera.αs
            @test to_screen(cache, host(cache.consts).β) == reshape(camera.βs, size(camera))
            seen_cases = seen_cases .| (!isempty(r.case2), !isempty(r.case3), !isempty(r.case4))

            refpix = reference_pixels(cache, camera, a, θo)
            pertpix = perturbed_pixels(cache, camera, a, θo)
            errs = compare_constants(host(cache.consts), refpix, pertpix)
            worst = argmax(name -> errs[name].err, keys(errs))
            @info "$label a=$(st.a) θo=$(st.θo)°: cases (4/2/0 real roots) = $(length(r.case2))/$(length(r.case3))/$(length(r.case4)); constants: max scaled error $(errs[worst].err) ($worst; one-ulp sensitivity $(errs[worst].sens)); max excess over $(SENS_FACTOR)×sensitivity $(maximum(v.excess for v in values(errs)))"
            for (name, v) in errs
                @test v.excess <= tol
            end

            se = compare_samples(host(cache.samples), refpix, pertpix, Val(N))
            @info "$label a=$(st.a) θo=$(st.θo)°: samples: $(se.nvalid)/$(npixels(cache) * N) valid; max scaled error $(se.err); one-ulp sensitivity $(se.sens); excess $(se.excess); flag mismatches = $(se.flag_mismatches)"
            @test se.flag_mismatches == 0
            @test se.nvalid > 0
            @test se.excess.t <= tol && se.excess.r <= tol && se.excess.θ <= tol
            @test se.excess.pos <= tol_pos
        end
    end

    # every root case must occur somewhere in the test set, or the case branches go untested
    @test all(seen_cases)

    @testset "$label regenerate! with a new camera" begin
        cam2 = Camera((-6.0, 6.0), (-6.0, 6.0), res)
        g = cache.generation
        regenerate!(cache, 0.94, deg2rad(17.0), cam2)
        @test cache.generation == g + 1
        @test Array(cache.αs) == cam2.αs
        refpix = reference_pixels(cache, cam2, 0.94, deg2rad(17.0))
        pertpix = perturbed_pixels(cache, cam2, 0.94, deg2rad(17.0))
        errs = compare_constants(host(cache.consts), refpix, pertpix)
        @test maximum(v.excess for v in values(errs)) <= tol
        @test_throws DimensionMismatch regenerate!(cache, 0.9, 1.0, Camera((-1.0, 1.0), (-1.0, 1.0), res + 1))
    end
end
