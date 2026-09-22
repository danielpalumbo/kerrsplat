# A quick functional check of an installation: the package loads, the card is usable, a polarized image of the six-parcel
# truth renders on the device and on the CPU to the same numbers, and the dual sweep's gradient runs on the device.
#
#     julia -t 8 --project=. cluster/check.jl
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer, KerrSplat.Fit
using KernelAbstractions, CUDA, StaticArrays, Random, LinearAlgebra
println("Julia $(VERSION), $(Threads.nthreads()) threads; CUDA functional: $(CUDA.functional())")
CUDA.functional() || (println("no usable card here: run this from a GPU job"); exit(1))
println("device: $(CUDA.name(CUDA.device())), CUDA runtime $(CUDA.runtime_version()), driver $(CUDA.driver_version())")
a = 0.9; θo = deg2rad(60.0); L = gravitational_radius(6.5e9); res = 24; N = 40; ν = 230e9
camera = Geodesics.Camera((-8.0, 8.0), (-8.0, 8.0), res)
rng = MersenneTwister(1); kepler(r) = 1 / (r^1.5 + a)
p = zeros(NPOLARIZEDPARAMS, 6)
for i in 1:6
    φ = 2π * (i - 1) / 6 + 0.3 * randn(rng); r0 = 2.5 + 2.5 * (i - 1) / 5
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.3 * randn(rng), log(0.7), log(0.7), log(0.5), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
               0.0, log(1e9), log(3e5) + 0.3 * randn(rng), log(30.0) + 0.2 * randn(rng), log(20.0) + 0.2 * randn(rng), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.0, 0.3, 0.05 * randn(rng), kepler(r0)]
end
cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cpu, a, θo; marcher = Fused(64))
gpu = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true); regenerate!(gpu, a, θo; marcher = Recurrence(64))
t0 = time(); img_cpu = polarized_cube(cpu, p, [0.0], [ν], L; nmax = 2, slab = 0.5)[:, :, 1, 1]; t_cpu = time() - t0
pd = CuArray(p)
t0 = time(); img_gpu = polarized_cube(gpu, pd, [0.0], [ν], L; nmax = 2, slab = 0.5)[:, :, 1, 1]; t_gpu = time() - t0
e = maximum(maximum.(abs, img_cpu .- img_gpu)) / maximum(x -> maximum(abs, x), img_cpu)
println("polarized image $(res)² × $N samples: CPU $(round(t_cpu, digits = 2)) s (with compile), device $(round(t_gpu, digits = 2)) s (with compile); relative difference $e")
e < 1e-6 || (println("the device image differs from the CPU's: something is wrong with the installation"); exit(1))
# the dual sweep's gradient of an image χ² on the device
dp = CUDA.zeros(Float64, size(p)); q = CuArray(p .* (1 .+ 0.01 .* randn(rng, size(p))))
σ = 1e-3 * maximum(x -> x[1], img_cpu); movie = StokesMovie(reshape(img_cpu, size(img_cpu)..., 1, 1), [0.0], [ν], SVector(σ, σ, σ, σ))
t0 = time(); χ = Fit.chi2_gradient!(dp, q, movie, gpu, L; nmax = 2, slab = 0.5); t_grad = time() - t0
println("dual-sweep gradient on the device: χ² $(round(χ, digits = 1)), $(round(t_grad, digits = 2)) s with compile, finite: $(all(isfinite, Array(dp)))")
all(isfinite, Array(dp)) || exit(1)
println("installation check passed")
