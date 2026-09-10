# Self-fit of the half-orbit truth (n ≤ 1) to synthetic VLBI data of its own slow-light movie: the
# EHT 2017 array's coverage (a real uvfits file, scan by scan) lent to the four-parcel movie, with
# the scans' frame times spread over the movie's span, closure quantities (or visibilities) with
# thermal noise, and the time-resolved likelihood (`Fit.chi2_timeresolved`) fitted on the GPU
# through `Fit.timeresolved_gradient!` (one dual sweep per frame). The plan's gate 7 in the data
# domain: the head-to-head with PI-DEF's reconstructions from simulated EHT data.
#
#     julia -t 8 --project=../.. winding_vlbi.jl --data <file.uvfits> [--frames 5] [--span 25] [--noise 0.01] [--mode closures|visibilities]
#                                               [--iterations 1000] [--eta 0.005] [--seed 1] [--tag name] [--polish 0] [--init fitted.csv]
#
# `--polish N` runs N Levenberg–Marquardt iterations on the time-resolved residuals from the Adam
# endpoint (CPU Jacobian by duals) and reports the polished recovery with the Laplace errors of
# the positions and emission rows; `--init file` starts from a saved matrix (`--iterations 0`
# skips Adam).
# `--frames F` groups the scans into F frame times over the first `--span` M of the movie (a
# dual sweep per frame per iteration); `--noise` is the thermal noise as a fraction of the frame's
# total flux density. Writes output/vlbi_tag/: fitted parameters and a summary.
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Optimisers, Krang, CUDA, Adapt

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
const PATH = getstr("--data", "/home/daniel/Dropbox/minimal_closures/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits")
const NFRAMES = getopt("--frames", 5); const SPAN = getopt("--span", 25.0); const NOISE = getopt("--noise", 0.01)
const MODE = getstr("--mode", "closures"); const ITER = getopt("--iterations", 1000); const ETA = getopt("--eta", 0.005); const SEED = getopt("--seed", 1)
const TAG = getstr("--tag", MODE)
const POLISH = getopt("--polish", 0); const INIT = getstr("--init", "")
const a = 0.94; const θo = deg2rad(17.0); const ν = 230e9
const M_solar = 6.5e9; const D = 16.8e6 * Transfer.PC; const L = gravitational_radius(M_solar)
const SLAB = 0.6; const NMAX = 1; const N = 160
const met = Krang.Kerr(a)

# ---- the truth of winding_selffit.jl
function keplerian(r)
    Ω = 1 / (r^1.5 + a); gdd = Krang.metric_dd(met, r, π / 2)
    ut = 1 / sqrt(-(gdd[1, 1] + 2Ω * gdd[1, 4] + Ω^2 * gdd[4, 4]))
    uz = Krang.jac_zamo_u_bl_d(met, r, π / 2) * SVector(ut, 0.0, 0.0, Ω * ut)
    return Ω, SVector(uz[2], uz[3], uz[4])
end
truth = zeros(NPOLARIZEDPARAMS, 4)
for (i, (r, φ)) in enumerate(((5.5, 0.3), (6.5, 2.0), (5.0, 3.6), (7.0, 5.2)))
    x, y, z = quasi_cartesian_kerr_schild(met, r, π / 2, φ)
    Ω, u = keplerian(r)
    truth[:, i] = [x, y, 0.0, log(0.7), log(0.7), log(0.15), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(3e4), log(30.0), log(10.0), π / 2, π / 2, u[1], u[2], u[3], Ω]
end
const FREE = Tuple(r for r in POLARIZED_SPLAT_PARAMS if r ∉ (:t0, :logw))     # every row but the temporal envelope

# ---- screen: the uniform 48² grid over 16 M (the visibilities need a regular grid; the annulus of the movie fits is not needed
# here, the (u, v) coverage sets the resolution)
const fov = 16.0; const res = 48; const Δα = fov / res
camera, binning = binned_grid((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
regenerate!(cache, a, θo; marcher = Fused(64))

# ---- coverage: the real array's scans, their frame times spread over the movie's span in NFRAMES groups
obs = average_scans(read_uvfits(PATH))                     # one row per baseline and scan
scans = scan_index(obs); nscans = maximum(scans)
frame_of_scan = [round(Int, (k - 1) / max(nscans - 1, 1) * (NFRAMES - 1)) for k in 1:nscans]
times = [SPAN * f / max(NFRAMES - 1, 1) for f in frame_of_scan]
cov = coverage(obs, times)
@info "coverage" file = basename(PATH) scans = length(cov) baselines = sum(length(c.u) for c in cov) frames = NFRAMES span_M = SPAN mode = MODE noise = NOISE
rng = MersenneTwister(SEED)
tr = synthetic_scans(cache, truth, L, Δα, D, ν, cov; noise = NOISE, closures = MODE == "closures", rng, nmax = NMAX, slab = SLAB, binning)
ndat = ndata(tr)
@info "synthetic data" data_values = ndat frames = frame_times(tr)

# ---- the GPU likelihood
gcache = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true)
regenerate!(gcache, a, θo; marcher = Recurrence(64))
function value_and_gradient(q)
    dp = CUDA.zeros(Float64, size(q))
    χ = timeresolved_gradient!(dp, CuArray(q), tr, gcache, L, Δα, D, ν; nmax = NMAX, slab = SLAB, binning)
    return χ, Array(dp)
end
χ_truth = value_and_gradient(truth)[1]
@info "χ² at the truth" chi2 = χ_truth reduced = χ_truth / ndat

# ---- the perturbed start of winding_selffit.jl
rng0 = MersenneTwister(SEED + 1000)
p0 = copy(truth)
for i in 1:4
    p0[1, i] += 0.3 * randn(rng0); p0[2, i] += 0.3 * randn(rng0); p0[3, i] += 0.05 * randn(rng0)
    p0[4:6, i] .+= 0.15 .* randn(rng0, 3)
    p0[13, i] += 0.2 * randn(rng0); p0[14, i] += 0.1 * randn(rng0); p0[15, i] += 0.15 * randn(rng0)
    p0[19, i] += 0.05 * randn(rng0); p0[21, i] *= 1 + 0.02 * randn(rng0)
end
isempty(INIT) || (p0 = Matrix{Float64}(readdlm(INIT, ',')); @info "starting from $INIT")
χ0 = value_and_gradient(p0)[1]
@info "start" chi2 = χ0 reduced = χ0 / ndat
t0 = time()
stages = [Fit.Stage(free = FREE, iterations = ITER, η = ETA, η_end = ETA / 10)]
q, history, _ = Fit.fit!(copy(p0), x -> chi2_timeresolved(x, tr, cache, L, Δα, D, ν; nmax = NMAX, slab = SLAB, binning), stages;
                         hygiene = Fit.Hygiene(every = 0), gradient = value_and_gradient,
                         callback = (si, it, x, v) -> (it % 50 == 0 && @info "iteration $it" chi2 = v reduced = v / ndat minutes = (time() - t0) / 60))
χ1 = value_and_gradient(q)[1]
laplace = nothing
if POLISH > 0
    tP = time()
    freemask = freeze(truth, FREE); freerows = [i for i in 1:NPOLARIZEDPARAMS if freemask[i, 1]]; nper = length(freerows)
    q, hist, cov, idx = polish!(copy(q), x -> timeresolved_residuals(x, tr, cache, L, Δα, D, ν; nmax = NMAX, slab = SLAB, binning); free = freemask, iterations = POLISH, chunk = 12)
    χ1 = hist[end]
    σp = sqrt.(max.(diag(cov), 0.0))
    laplace = [σp[(i - 1) * nper + findfirst(==(r), freerows)] for r in (1, 2, 13, 14, 15), i in 1:4]
    @info "Levenberg–Marquardt polish ($POLISH iterations)" chi2 = hist[1] => hist[end] reduced = χ1 / ndat history = round.(hist ./ ndat; digits = 4) minutes = (time() - tP) / 60
    @info "Laplace errors per parcel" x_M = round.(laplace[1, :]; sigdigits = 3) y_M = round.(laplace[2, :]; sigdigits = 3) ln_ne = round.(laplace[3, :]; sigdigits = 3) ln_Te = round.(laplace[4, :]; sigdigits = 3) ln_B = round.(laplace[5, :]; sigdigits = 3)
end
pos_err = [hypot(q[1, i] - truth[1, i], q[2, i] - truth[2, i], q[3, i] - truth[3, i]) for i in 1:4]
pos0 = [hypot(p0[1, i] - truth[1, i], p0[2, i] - truth[2, i], p0[3, i] - truth[3, i]) for i in 1:4]
rel(row) = [abs(exp(q[row, i] - truth[row, i]) - 1) for i in 1:4]
ωerr = [abs(q[21, i] / truth[21, i] - 1) for i in 1:4]
@info "recovery from synthetic VLBI data ($MODE)" chi2 = χ1 reduced = χ1 / ndat truth_reduced = χ_truth / ndat position_M = round.(pos_err; sigdigits = 2) position_start_M = round.(pos0; sigdigits = 2) ne = round.(rel(13); sigdigits = 2) Te = round.(rel(14); sigdigits = 2) B = round.(rel(15); sigdigits = 2) pattern_rate = round.(ωerr; sigdigits = 2) minutes = (time() - t0) / 60
outdir = joinpath(@__DIR__, "output", "vlbi_$TAG"); mkpath(outdir)
writedlm(joinpath(outdir, "truth.csv"), truth, ','); writedlm(joinpath(outdir, "start.csv"), p0, ','); writedlm(joinpath(outdir, "fitted.csv"), q, ',')
open(joinpath(outdir, "summary.txt"), "w") do io
    println(io, "synthetic VLBI self-fit ($MODE): $(length(cov)) scans of $(basename(PATH)) in $NFRAMES frames over $SPAN M, noise $NOISE of the flux, $ndat data values, $ITER iterations, eta $ETA")
    println(io, "chi2 truth $χ_truth start $χ0 end $χ1 (reduced $(χ1 / ndat), truth $(χ_truth / ndat))")
    println(io, "position errors (M): start $pos0 end $pos_err")
    println(io, "relative errors: ne $(rel(13)) Te $(rel(14)) B $(rel(15)) pattern rate $ωerr")
    laplace === nothing || println(io, "after $POLISH LM iterations, Laplace errors per parcel: x $(laplace[1, :]) y $(laplace[2, :]) ln ne $(laplace[3, :]) ln Te $(laplace[4, :]) ln B $(laplace[5, :])")
end
