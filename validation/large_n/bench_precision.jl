# Float32 against Float64 for the transport at fit size: the tails and the dual-sweep gradient of four batched
# frames over per-ray lists, Float64 geodesics in both cases (`Geodesics.precision` copies the cache), the
# Float32 run timed against the Float64 run and its gradient compared. Usage:
#     julia -t 8 --project=../.. bench_precision.jl [--backend cuda|cpu] [--res 64] [--samples 160] [--parcels 300] [--frames 4] [--tag precision]
using KerrSplat, KerrSplat.Fit, KerrSplat.Splats, KerrSplat.Geodesics, KerrSplat.Transfer
using KernelAbstractions, CUDA, StaticArrays, Random, Printf
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
res = getopt("--res", 64); N = getopt("--samples", 160); n = getopt("--parcels", 300); nf = getopt("--frames", 4); tag = getstr("--tag", "precision")
backend = getstr("--backend", "cuda") == "cuda" ? CUDABackend() : CPU()
device(x) = (y = KernelAbstractions.allocate(backend, eltype(x), size(x)...); copyto!(y, x); y)
sync() = KernelAbstractions.synchronize(backend)
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
a, θo = 0.94, deg2rad(60.0); ν = 230e9; L = gravitational_radius(4e6)
camera = Geodesics.Camera((-8.0, 8.0), (-8.0, 8.0), res)
cache = GeodesicCache(backend, camera, Val(N); store_samples = true); regenerate!(cache, a, θo; marcher = Recurrence(64))
cache32 = Geodesics.precision(cache, Float32)
npix = npixels(cache)
p = Splats.shell_parcels(n; rin = 2.5, rout = 5.0, spin = a, rng = MersenneTwister(11))
times = collect(range(0.0, 30.0; length = nf))
w = [SVector{4}(randn(MersenneTwister(3), 4)) for _ in 1:npix, _ in 1:nf]
lines = String[@sprintf("Float32 against Float64 transport, %s: %d² pixels, %d samples, %d shell parcels, %d frames batched over lists", backend isa CPU ? "CPU" : "RTX 2080 SUPER", res, N, n, nf)]
results = Dict{DataType,Any}()
for (T, c) in ((Float64, cache), (Float32, cache32))
    params = device(T.(p)); wT = device(SVector{4,T}.(w)); tT = T.(times)
    g = KernelAbstractions.zeros(backend, T, size(p))
    run!() = begin
        fill!(g, zero(T))
        l = ray_lists(c, params, tT)
        tails = KernelAbstractions.zeros(backend, SVector{4,T}, npix, N + 1, nf)
        polarized_tails!(tails, c, params, tT, T(ν), T(L); lists = l)
        polarized_dual_sweep!(g, wT, tails, c, params, tT, T(ν), T(L); lists = l)
        sync()
        tails
    end
    run!()                                   # compile
    t = @elapsed tails = run!()
    results[T] = (t = t, g = Float64.(Array(g)), img = map(x -> SVector{4,Float64}(x), Array(tail_image(tails, T(ν)))))
    line = @sprintf("%s: tails + dual sweep %.3f s", T, t)
    push!(lines, line); println(line)
end
r64 = results[Float64]; r32 = results[Float32]
eg = maximum(abs.(r32.g .- r64.g)) / maximum(abs.(r64.g))
ei = maximum(maximum.(abs, r32.img .- r64.img)) / maximum(x -> maximum(abs, x), r64.img)
line = @sprintf("Float32 is %.2fx the Float64 speed; the gradient agrees to %.1e of its largest entry, the image to %.1e of the peak; Float32 gradient finite: %s", r64.t / r32.t, eg, ei, all(isfinite, r32.g))
push!(lines, line); println(line)
open(joinpath(outdir, "bench_$(tag).txt"), "w") do io
    foreach(l -> println(io, l), lines)
end
