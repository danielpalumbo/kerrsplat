# Timing of the geodesic layer with the package code (compare plan §3, RTX 2080 SUPER):
# K0+K1 (per-pixel constants) and stored-mode K2 with Krang's direct evaluation.
#
#     julia -t 16 --project=. bench/geodesics_bench.jl [res] [N]
#
import Pkg; Pkg.activate(joinpath(@__DIR__, ".."))
using CUDA, KernelAbstractions, Krang, Printf
using KerrSplat.Geodesics
const KA = KernelAbstractions

res = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : 256
N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 200
a, θo = 0.94, deg2rad(60.0)
camera = Camera((-10.0, 10.0), (-10.0, 10.0), res)
println("threads = ", Threads.nthreads(), "; screen ", res, "² × ", N, " samples; a = ", a, ", θo = ", rad2deg(θo), "°")

function timed_stages(cache, a, θo)
    backend = cache.backend
    met = Krang.Kerr(a)
    npix = npixels(cache)
    sync() = KA.synchronize(backend)
    t0 = time()
    Geodesics.root_case_kernel!(backend, 256)(cache.numreals_screen, met, θo, cache.αs, cache.βs; ndrange = npix); sync()
    t_k0 = time() - t0
    t0 = time()
    perm_host, ranges = case_permutation(Array(cache.numreals_screen))
    copyto!(cache.perm, perm_host); cache.perm_host = perm_host; cache.ranges = ranges
    t_sort = time() - t0
    t0 = time()
    Geodesics.pixel_constants_kernel!(backend, 256)(cache.consts, met, θo, cache.αs, cache.βs, cache.perm; ndrange = npix); sync()
    t_k1 = time() - t0
    t0 = time()
    Geodesics.direct_march!(cache.samples, cache.consts, met, θo, cache.nval); sync()
    t_k2 = time() - t0
    Geodesics.recurrence_march!(cache.samples, cache.consts, cache.ranges, met, θo, cache.nval, Val(64)); sync()   # compile
    t0 = time()
    Geodesics.recurrence_march!(cache.samples, cache.consts, cache.ranges, met, θo, cache.nval, Val(64)); sync()
    t_rec = time() - t0
    Geodesics.quadrature_march!(cache.samples, cache.consts, cache.residual_t, cache.residual_ϕ, cache.ranges, met, θo, cache.nval, Val(64)); sync()
    t0 = time()
    Geodesics.quadrature_march!(cache.samples, cache.consts, cache.residual_t, cache.residual_ϕ, cache.ranges, met, θo, cache.nval, Val(64)); sync()
    t_quad = time() - t0
    return (k0 = t_k0, sort = t_sort, k1 = t_k1, k2 = t_k2, rec = t_rec, quad = t_quad)
end

for (name, backend) in (("CUDA", CUDABackend()), ("CPU ($(Threads.nthreads()) threads)", CPU()))
    name == "CUDA" && !CUDA.functional() && continue
    cache = GeodesicCache(backend, camera, Val(N))
    t0 = time(); regenerate!(cache, a, θo); t_first = time() - t0          # includes compilation
    t0 = time(); regenerate!(cache, a, θo); t_warm = time() - t0
    regenerate!(cache, a, θo; marcher = Recurrence(64))
    t0 = time(); regenerate!(cache, a, θo; marcher = Recurrence(64)); t_warm_rec = time() - t0
    s = timed_stages(cache, a, θo)
    npix = npixels(cache)
    @printf("[%s] regenerate!: Direct first %.1f s (compile), warm %.3f s; Recurrence(64) warm %.3f s\n", name, t_first, t_warm, t_warm_rec)
    @printf("[%s]   K0 root cases %.4f s | sort %.4f s | K1 constants %.4f s (%.1f µs/px) | K2 direct march %.3f s (%.1f ns/sample) | K2 recurrence (r, θ; anchor 64) %.4f s (%.2f ns/sample) | K2 recurrence + quadrature (t̃, r, θ, φ) %.4f s (%.2f ns/sample)\n",
            name, s.k0, s.sort, s.k1, s.k1 / npix * 1e6, s.k2, s.k2 / (npix * N) * 1e9, s.rec, s.rec / (npix * N) * 1e9, s.quad, s.quad / (npix * N) * 1e9)
    @printf("[%s]   stored samples: %.2f GB; cases 4/2/0 real roots: %d/%d/%d\n", name, sizeof(cache.samples) / 1e9,
            length(cache.ranges.case2), length(cache.ranges.case3), length(cache.ranges.case4))
    flush(stdout)
    cache = nothing; GC.gc(); name == "CUDA" && CUDA.reclaim()
end
println("BENCH_DONE")
