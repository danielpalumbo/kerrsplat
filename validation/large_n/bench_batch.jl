# Frames batched into one launch on the GPU: the tails and the dual-sweep gradient of four frames, one launch per frame
# against one launch for all four, at 64² × 60 with 512 parcels of 0.2 M (and dense for reference). Usage:
#     julia -t 8 --project=../.. bench_batch.jl [--res 64] [--samples 60] [--parcels 512] [--frames 4] [--tag batch]
using KerrSplat, KerrSplat.Fit, KerrSplat.Splats, KerrSplat.Geodesics, KerrSplat.Transfer
using KernelAbstractions, CUDA, StaticArrays, Random, Printf
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
res = getopt("--res", 64); N = getopt("--samples", 60); n = getopt("--parcels", 512); nf = getopt("--frames", 4); tag = getstr("--tag", "batch")
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
a, θo = 0.94, deg2rad(60.0); ν = 230e9; L = gravitational_radius(4e6)
camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
cache = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true); regenerate!(cache, a, θo; marcher = Recurrence(64))
npix = npixels(cache)
rng = MersenneTwister(11)
p = zeros(NPOLARIZEDPARAMS, n)
for i in 1:n
    φ = 2π * rand(rng); r0 = 3.0 + 5.0 * rand(rng)
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.8 * randn(rng), log(0.2), log(0.26), log(0.14), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
               0.0, log(1e9), log(2e4 * 48 / n), log(30.0), log(10.0), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.2 * randn(rng), 0.3 + 0.1 * randn(rng), 0.1 * randn(rng), 0.02 * randn(rng)]
end
params = CuArray(p)
times = collect(range(0.0, 30.0; length = nf))
w = CuArray([SVector{4}(randn(rng, 4)) for _ in 1:npix, _ in 1:nf])
lines = String[@sprintf("frames batched on the GPU: %d² pixels, %d samples, %d parcels of 0.2 M, %d frames, RTX 2080 SUPER", res, N, n, nf)]
for cull in (true, false)
    # per frame
    g1 = CUDA.zeros(Float64, size(p))
    t1 = @elapsed begin
        for (f, t) in enumerate(times)
            l = cull ? ray_lists(cache, params, t) : nothing
            tails = CUDA.zeros(SVector{4,Float64}, npix, N + 1)
            polarized_tails!(tails, cache, params, t, ν, L; lists = l)
            polarized_dual_sweep!(g1, w[:, f], tails, cache, params, t, ν, L; lists = l)
        end
        CUDA.synchronize()
    end
    t1 = @elapsed begin
        fill!(g1, 0.0)
        for (f, t) in enumerate(times)
            l = cull ? ray_lists(cache, params, t) : nothing
            tails = CUDA.zeros(SVector{4,Float64}, npix, N + 1)
            polarized_tails!(tails, cache, params, t, ν, L; lists = l)
            polarized_dual_sweep!(g1, w[:, f], tails, cache, params, t, ν, L; lists = l)
        end
        CUDA.synchronize()
    end
    # batched
    gb = CUDA.zeros(Float64, size(p))
    tb = @elapsed begin
        l = cull ? ray_lists(cache, params, times) : nothing
        tails = CUDA.zeros(SVector{4,Float64}, npix, N + 1, nf)
        polarized_tails!(tails, cache, params, times, ν, L; lists = l)
        polarized_dual_sweep!(gb, w, tails, cache, params, times, ν, L; lists = l)
        CUDA.synchronize()
    end
    tb = @elapsed begin
        fill!(gb, 0.0)
        l = cull ? ray_lists(cache, params, times) : nothing
        tails = CUDA.zeros(SVector{4,Float64}, npix, N + 1, nf)
        polarized_tails!(tails, cache, params, times, ν, L; lists = l)
        polarized_dual_sweep!(gb, w, tails, cache, params, times, ν, L; lists = l)
        CUDA.synchronize()
    end
    e = maximum(abs.(Array(gb) .- Array(g1))) / maximum(abs.(Array(g1)))
    line = @sprintf("lists %-5s  per frame %.3f s  batched %.3f s  (%.2fx; gradients agree to %.1e)", cull, t1, tb, t1 / tb, e)
    push!(lines, line); println(line)
end
open(joinpath(outdir, "bench_$(tag).txt"), "w") do io
    foreach(l -> println(io, l), lines)
end
