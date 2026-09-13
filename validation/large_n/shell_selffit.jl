# The large-N self-fit: a six-parcel truth rendered as a polarized slow-light movie, fitted from an over-complete shell of
# small parcels (shell_parcels) with the hygiene schedule (prune, merge, densify) at the true spacetime; the recovered
# fields compared with the truth on a voxel grid (recovery_metrics), not the parcels. Usage (from validation/large_n):
#     julia -t 8 --project=../.. shell_selffit.jl [--res 64] [--fov 16] [--samples 160] [--nmax 2] [--slab 0.5] [--frames 4]
#                                                  [--n 300] [--scale 0.25] [--iterations 300] [--eta 0.02] [--every 25]
#                                                  [--prune 0.02] [--densify 0] [--max 600] [--seed 1] [--tag shell] [--frequencies 230]
using KerrSplat, KerrSplat.Fit, KerrSplat.Splats, KerrSplat.Geodesics, KerrSplat.Transfer
using KernelAbstractions, CUDA, StaticArrays, LinearAlgebra, Random, Printf, DelimitedFiles
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
res = getopt("--res", 64); fov = getopt("--fov", 16.0); N = getopt("--samples", 160); nmax = getopt("--nmax", 2); slab = getopt("--slab", 0.5)
nframes = getopt("--frames", 4); n = getopt("--n", 300); scale = getopt("--scale", 0.25); iterations = getopt("--iterations", 300); η = getopt("--eta", 0.02)
every = getopt("--every", 25); prune = getopt("--prune", 0.02); densify = getopt("--densify", 0.0); maxsplats = getopt("--max", 600); seed = getopt("--seed", 1); tag = getstr("--tag", "shell")
freqs = parse.(Float64, split(getstr("--frequencies", "230"), ",")) .* 1e9
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
a = 0.9; θo = deg2rad(60.0); L = gravitational_radius(4e6); ν = freqs[1]
rng = MersenneTwister(seed)
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
kepler(r) = 1 / (r^1.5 + a)
p = zeros(NPOLARIZEDPARAMS, 6)
for i in 1:6
    φ = 2π * (i - 1) / 6 + 0.3 * randn(rng); r0 = 2.5 + 2.5 * (i - 1) / 5
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.3 * randn(rng), log(0.7), log(0.7), log(0.5), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
               0.0, log(1e9), log(3e5) + 0.3 * randn(rng), log(30.0) + 0.2 * randn(rng), log(20.0) + 0.2 * randn(rng), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.0, 0.3, 0.05 * randn(rng), kepler(r0)]
end
times = collect(range(0.0, 60.0; length = nframes))
cache = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true); regenerate!(cache, a, θo; marcher = Recurrence(64))
cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cpu, a, θo; marcher = Fused(64))
clean = polarized_cube(cpu, p, times, freqs, L; nmax, slab)
σ = SVector(0.02, 0.01, 0.01, 0.005) * maximum(norm.(clean))
data = [clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
movie = StokesMovie(data, times, freqs, σ)
ndat = 4 * length(data)
q0 = shell_parcels(n; rin = 2.2, rout = 5.5, height = 0.6, scale, spin = a, rng)
# the shell's emission measure at the truth's: densities scaled to match the first frame's total intensity
F(qq) = sum(x -> x[1], polarized_cube(cpu, qq, [times[1]], [ν], L; nmax, slab))
q0[13, :] .+= log(F(p) / F(q0))
χ0 = chi2(q0, movie, cpu, L; nmax, slab); χt = chi2(p, movie, cpu, L; nmax, slab)
@info "start" parcels = n chi2 = χ0 reduced = χ0 / ndat reduced_truth = χt / ndat values = ndat
xs = range(-6, 6; length = 25); ys = xs; zs = range(-1.5, 1.5; length = 7)
m0 = recovery_metrics(q0, p, 0.0, xs, ys, zs)
@info "field recovery at the start" m0
t0 = time()
trace = String[]
stages = [Fit.Stage(; free = (:x, :y, :z, :s1, :s2, :s3, :logne), iterations = iterations ÷ 3, η, η_end = η / 2, label = "geometry and densities"),
          Fit.Stage(; iterations = iterations - iterations ÷ 3, η, η_end = η / 10, label = "everything")]
hyg = Fit.Hygiene(every = every, prune_fraction = prune, densify_threshold = densify > 0 ? densify : Inf, merge_position = 0.15, max_splats = maxsplats)
q, history, events = Fit.fit!(copy(q0), movie, cache, L, stages; hygiene = hyg, gradient = :dual, nmax, slab,
                              callback = (si, it, x, v) -> (it % 25 == 0 && (push!(trace, @sprintf("stage %d iteration %3d  chi2 %10.1f  reduced %.3f  parcels %4d  %.1f min", si, it, v, v / ndat, size(x, 2), (time() - t0) / 60)); @info trace[end])))
χ1 = chi2(q, movie, cpu, L; nmax, slab)
m1 = recovery_metrics(q, p, 0.0, xs, ys, zs)
@info "end" chi2 = χ1 reduced = χ1 / ndat parcels = size(q, 2) events = length(events) minutes = (time() - t0) / 60 recovery = m1
writedlm(joinpath(outdir, "shell_$(tag)_params.csv"), q, ','); writedlm(joinpath(outdir, "shell_$(tag)_truth.csv"), p, ',')
open(joinpath(outdir, "shell_$(tag)_summary.txt"), "w") do io
    println(io, "large-N self-fit: $(res)² pixels, fov $fov M, $N samples, nmax $nmax slab $slab, $nframes frames, frequencies $(freqs ./ 1e9) GHz, $ndat values; shell of $n parcels of $scale M, $iterations iterations, eta $η, hygiene every $every (prune $prune, densify $densify, max $maxsplats), seed $seed")
    println(io, "chi2 start $χ0 (reduced $(χ0 / ndat)), truth $χt (reduced $(χt / ndat)), end $χ1 (reduced $(χ1 / ndat)); parcels $n → $(size(q, 2)) through $(length(events)) hygiene events $(events)")
    println(io, "field recovery (density PSNR dB, relative density error, density-weighted temperature and field errors): start $m0; end $m1")
    foreach(l -> println(io, l), trace)
end
