# The large-N dual sweep with and without per-ray parcel lists on the GPU: time and agreement of the tails image and
# the gradient for parcel counts from 32 to 2048 on a 64² screen with 60 stored samples (defaults). Usage:
#     julia -t 8 --project=../.. bench_lists.jl [--res 64] [--samples 60] [--counts 32,128,512,2048] [--dense-max 512] [--tag bench]
# Parcels of two sizes (0.15 and 0.6 M scales) scattered over 3–8 M; the dense path is timed up to --dense-max parcels
# (its per-ray gradient buffer is npix × 21 × N doubles). Output: output/bench_<tag>.txt.
using KerrSplat, KerrSplat.Fit, KerrSplat.Splats, KerrSplat.Geodesics, KerrSplat.Transfer
using KernelAbstractions, CUDA, StaticArrays, Random, Printf
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
res = getopt("--res", 64); N = getopt("--samples", 60); counts = parse.(Int, split(getstr("--counts", "32,128,512,2048"), ",")); densemax = getopt("--dense-max", 512); tag = getstr("--tag", "bench")
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
a, θo = 0.94, deg2rad(60.0); t_obs = 12.0; ν = 230e9; L = gravitational_radius(4e6)
camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
cache = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true)
regenerate!(cache, a, θo; marcher = Recurrence(64))
npix = npixels(cache)
rng = MersenneTwister(11)
function parcels(n)
    p = zeros(NPOLARIZEDPARAMS, n)
    for i in 1:n
        φ = 2π * rand(rng); r0 = 3.0 + 5.0 * rand(rng); sz = i % 3 == 0 ? 0.15 : 0.6
        p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.8 * randn(rng), log(sz), log(sz * 1.3), log(sz * 0.7), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
                   0.0, log(1e9), log(2e4 * 48 / n), log(30.0), log(10.0), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.2 * randn(rng), 0.3 + 0.1 * randn(rng), 0.1 * randn(rng), 0.02 * randn(rng)]
    end
    return p
end
w = CuArray([SVector{4}(randn(rng, 4)) for _ in 1:npix])
lines = String[@sprintf("per-ray lists on the GPU: %d² pixels, %d samples, RTX 2080 SUPER", res, N)]
for n in counts
    params = CuArray(parcels(n))
    tl = @elapsed lists = ray_lists(cache, params, t_obs); CUDA.synchronize()
    tl = @elapsed (lists = ray_lists(cache, params, t_obs); CUDA.synchronize())
    cnt = Array(lists.count); cap = size(lists.ids, 1)
    tails = CUDA.zeros(SVector{4,Float64}, npix, N + 1)
    polarized_tails!(tails, cache, params, t_obs, ν, L; lists); CUDA.synchronize()
    tt = @elapsed (polarized_tails!(tails, cache, params, t_obs, ν, L; lists); CUDA.synchronize())
    img_l = Array(tail_image(tails, ν))
    g_l = CUDA.zeros(Float64, size(params))
    polarized_dual_sweep!(g_l, w, tails, cache, params, t_obs, ν, L; lists); CUDA.synchronize()
    tg = @elapsed (fill!(g_l, 0.0); polarized_dual_sweep!(g_l, w, tails, cache, params, t_obs, ν, L; lists); CUDA.synchronize())
    line = @sprintf("N = %5d  lists %.3f s (capacity %3d, mean length %5.1f)  tails %.3f s  gradient %.3f s", n, tl, cap, sum(cnt) / npix, tt, tg)
    if n <= densemax
        tails_d = CUDA.zeros(SVector{4,Float64}, npix, N + 1)
        polarized_tails!(tails_d, cache, params, t_obs, ν, L); CUDA.synchronize()
        ttd = @elapsed (polarized_tails!(tails_d, cache, params, t_obs, ν, L); CUDA.synchronize())
        img_d = Array(tail_image(tails_d, ν))
        g_d = CUDA.zeros(Float64, size(params))
        polarized_dual_sweep!(g_d, w, tails_d, cache, params, t_obs, ν, L); CUDA.synchronize()
        tgd = @elapsed (fill!(g_d, 0.0); polarized_dual_sweep!(g_d, w, tails_d, cache, params, t_obs, ν, L); CUDA.synchronize())
        ei = maximum(maximum.(abs, img_l .- img_d)) / maximum(x -> maximum(abs, x), img_d)
        eg = maximum(abs.(Array(g_l) .- Array(g_d))) / maximum(abs.(Array(g_d)))
        line *= @sprintf("  | dense: tails %.3f s  gradient %.3f s  (image agrees to %.1e, gradient to %.1e)", ttd, tgd, ei, eg)
    end
    push!(lines, line); println(line)
end
open(joinpath(outdir, "bench_$(tag).txt"), "w") do io
    foreach(l -> println(io, l), lines)
end
