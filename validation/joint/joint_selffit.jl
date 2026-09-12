# Joint spacetime-and-splat self-fit on the GPU: a six-parcel truth rendered as a polarized slow-light movie with the
# n ≤ 2 sub-images, fitted from a wrong spin and inclination with the parcels perturbed (or fresh), by fit_joint!.
# Usage (from validation/joint):
#     julia -t 8 --project=../.. joint_selffit.jl [--res 40] [--fov 20] [--samples 160] [--nmax 2] [--slab 0.5] [--frames 4]
#                                                  [--frequencies 230] [--iterations 150] [--eta 0.02] [--warmup 0] [--inner 2]
#                                                  [--every 1] [--a0 0.5] [--inc0 45] [--perturb 0.1] [--fresh] [--seed 1] [--tag joint] [--pattern 0]
#                                                  [--rin 4] [--rout 7] [--keplerian 0]
# Truth: spin 0.9, inclination 60°, six parcels on orbits at --rin to --rout M (4–7 by default) with Keplerian pattern
# rates. Output:
# output/joint_<tag>_summary.txt (the spacetime trajectory, χ² trace, recovered parcels) and the parameter files.
using KerrSplat, KerrSplat.Fit, KerrSplat.Splats, KerrSplat.Geodesics, KerrSplat.Transfer
using KernelAbstractions, CUDA, StaticArrays, LinearAlgebra, Random, Printf, DelimitedFiles
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
res = getopt("--res", 40); fov = getopt("--fov", 20.0); N = getopt("--samples", 160); nmax = getopt("--nmax", 2); slab = getopt("--slab", 0.5)
nframes = getopt("--frames", 4); freqs = parse.(Float64, split(getstr("--frequencies", "230"), ",")) .* 1e9
iterations = getopt("--iterations", 150); η = getopt("--eta", 0.02); warmup = getopt("--warmup", 0); inner = getopt("--inner", 2); every = getopt("--every", 1)
a0 = getopt("--a0", 0.5); inc0 = getopt("--inc0", 45.0); perturb = getopt("--perturb", 0.1); fresh = "--fresh" in ARGS; seed = getopt("--seed", 1); tag = getstr("--tag", "joint")
σpattern = getopt("--pattern", 0.0); pattern = σpattern > 0 ? σpattern : nothing            # the Keplerian pattern prior ties the rates to the spin
rin = getopt("--rin", 4.0); rout = getopt("--rout", 7.0)
σkep = getopt("--keplerian", 0.0); keplerian = σkep > 0 ? σkep : nothing                  # Keplerian truth velocities and the fluid prior in the fit
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
a_true = 0.9; θ_true = deg2rad(60.0); L = gravitational_radius(4e6)
rng = MersenneTwister(seed)
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
Δα = fov / res
kepler(r, a) = 1 / (r^1.5 + a)
p = zeros(NPOLARIZEDPARAMS, 6)
for i in 1:6
    φ = 2π * (i - 1) / 6 + 0.3 * randn(rng); r0 = rin + (rout - rin) * (i - 1) / 5
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.3 * randn(rng), log(0.7), log(0.7), log(0.5), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
               0.0, log(1e9), log(3e5) + 0.3 * randn(rng), log(30.0) + 0.2 * randn(rng), log(20.0) + 0.2 * randn(rng), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.0, 0.3, 0.05 * randn(rng), kepler(r0, a_true)]
    if keplerian !== nothing
        met_true = Geodesics.Krang.Kerr(a_true)
        rb, θb, _ = Splats.boyer_lindquist(met_true, p[1, i], p[2, i], p[3, i])
        p[18:20, i] = Fit.keplerian_zamo_velocity(met_true, rb, θb)
    end
end
times = collect(range(0.0, 60.0; length = nframes))
gtruth = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = false); regenerate!(gtruth, a_true, θ_true; marcher = Fused(64))
clean = polarized_cube(gtruth, CuArray(p), times, freqs, L; nmax, slab)
σ = SVector(0.02, 0.01, 0.01, 0.005) * maximum(norm.(clean))
data = [clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
movie = StokesMovie(data, times, freqs, σ)
@info "truth movie" res N nmax slab frames = nframes frequencies_GHz = freqs ./ 1e9 peak = maximum(norm.(clean)) values = 4 * length(data)
q = fresh ? begin
        q0 = zeros(NPOLARIZEDPARAMS, 8)
        for i in 1:8
            φ = 2π * (i - 1) / 8; r0 = (rin + rout) / 2
            q0[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.0, log(1.0), log(1.0), log(0.7), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(2e5), log(30.0), log(20.0), π / 2, 0.0, 0.0, 0.3, 0.0, kepler(r0, a0)]
        end
        q0
    end : p .+ perturb .* randn(rng, size(p))
x0 = [a0, deg2rad(inc0)]
cache = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true)
regenerate!(cache, x0[1], x0[2]; marcher = Recurrence(64))
χ2(qq) = Fit.spacetime_chi2(cache, Fit._on_backend(cache, qq), movie, L; nmax, slab)          # the device χ² (chi2 is the host form)
χstart = χ2(q)
regenerate!(cache, a_true, θ_true; marcher = Recurrence(64))
χtruth_sky = χ2(q); χtruth = χ2(p)
@info "start" chi2 = χstart reduced = χstart / (4 * length(data)) chi2_at_true_spacetime_start_sky = χtruth_sky chi2_truth = χtruth reduced_truth = χtruth / (4 * length(data)) a0 inc0 fresh
t0 = time()
trace = String[]
qj, xj, hist, acc = Fit.fit_joint!(copy(q), x0, movie, cache, camera; L, iterations, η, warmup, inner, every, nmax, slab, pattern, keplerian,
                                   callback = (it, pp, xx, v) -> (it % 10 == 0 && (push!(trace, @sprintf("iteration %3d  chi2 %10.1f  a %.4f  inc %.2f°  %.1f min", it, v, xx[1], rad2deg(xx[2]), (time() - t0) / 60)); @info trace[end])))
regenerate!(cache, xj[1], xj[2]; marcher = Recurrence(64))
χend = χ2(qj)
@info "end" chi2 = χend reduced = χend / (4 * length(data)) a = xj[1] inc = rad2deg(xj[2]) accepted = acc minutes = (time() - t0) / 60
writedlm(joinpath(outdir, "joint_$(tag)_params.csv"), qj, ','); writedlm(joinpath(outdir, "joint_$(tag)_truth.csv"), p, ','); writedlm(joinpath(outdir, "joint_$(tag)_x.csv"), xj, ',')
open(joinpath(outdir, "joint_$(tag)_summary.txt"), "w") do io
    println(io, "joint spacetime-and-splat self-fit: $(res)² pixels, fov $fov M, $N samples, nmax $nmax slab $slab, $nframes frames over $(times[end]) M, frequencies $(freqs ./ 1e9) GHz, $(4 * length(data)) values; parcels at $(rin)–$(rout) M; $iterations iterations, eta $η, warmup $warmup, inner $inner, every $every, pattern prior $(pattern === nothing ? "off" : "σ = $σpattern"), Keplerian fluid prior $(keplerian === nothing ? "off" : "σ = $σkep"); start a $a0 inc $inc0, $(fresh ? "fresh 8 parcels" : "truth perturbed by $perturb"), seed $seed")
    println(io, "truth: a $a_true inc 60.0; chi2 at truth $χtruth (reduced $(χtruth / (4 * length(data)))); chi2 at start $χstart; with the start sky at the true spacetime $χtruth_sky")
    println(io, "end: a $(xj[1]) inc $(rad2deg(xj[2])) chi2 $χend (reduced $(χend / (4 * length(data)))) accepted spacetime steps $acc minutes $((time() - t0) / 60)")
    foreach(l -> println(io, l), trace)
end
