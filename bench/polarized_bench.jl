# Timings of the polarized pipeline on the GPU: cost per sample-splat of the fused polarized
# transport (thermal coefficients, frames, exact step) against the thin renderer and the bare
# geodesic march, for the cost note of the addendum (§6.3).
#
#     julia -t 8 --project=. bench/polarized_bench.jl
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer, KernelAbstractions, CUDA, StaticArrays
backend = CUDA.functional() ? CUDABackend() : CPU()
Geodesics.prepare_backend!(backend)
res = 128; N = 1000
camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
regenerate!(cache, 0.94, deg2rad(60.0); marcher = Fused(64))
L = gravitational_radius(4e6); ν = 230e9
tosec(f) = (f(); KernelAbstractions.synchronize(backend); t = @elapsed (f(); KernelAbstractions.synchronize(backend)); t)
for nsplat in (1, 4, 16)
    p = zeros(NPOLARIZEDPARAMS, nsplat)
    for i in 1:nsplat
        p[:, i] = [6.0 + 0.5i, 0.5i, 0.1i, log(1.5), log(1.5), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.3, 0.0, 0.0]
    end
    pb = KernelAbstractions.allocate(backend, Float64, size(p)...); copyto!(pb, p)
    out = KernelAbstractions.allocate(backend, RadiativeState{Float64}, npixels(cache))
    t = tosec(() -> polarized_image!(out, cache, pb, 0.0, ν, L))
    pt = zeros(NSPLATPARAMS, nsplat); pt[1:12, :] = p[1:12, :]; pt[13, :] .= 0.0
    ptb = KernelAbstractions.allocate(backend, Float64, size(pt)...); copyto!(ptb, pt)
    outt = KernelAbstractions.allocate(backend, Float64, npixels(cache))
    tt = tosec(() -> thin_image!(outt, cache, ptb, 0.0))
    ns = res^2 * N
    println("nsplat = $nsplat: polarized $(round(t, digits = 3)) s = $(round(1e9 * t / ns, digits = 1)) ns per sample ($(round(1e9 * t / ns / nsplat, digits = 1)) per sample-splat); thin $(round(tt, digits = 3)) s = $(round(1e9 * tt / ns, digits = 1)) ns per sample")
end
