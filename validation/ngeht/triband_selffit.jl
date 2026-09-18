# The triband ngEHT self-fit: a six-parcel slow-light truth at M87's mass and distance observed with the campaign of
# make_campaign.py (the reference array's (u, v) sampling and thermal noise per band and day), fitted from an over-complete
# shell of small parcels with the hygiene schedule at the true spacetime, the three bands' time-resolved likelihoods summed.
# Frames are shared by the scans of every --frame-hours of the campaign (M87's GM/c³ is 8.9 h, so four hours of scans see
# 0.45 M of motion, a few degrees of orbit at the shell's radii). Usage (from validation/ngeht):
#     julia -t 8 --project=../.. triband_selffit.jl [--days 5] [--start 2026-04-01] [--bands 86,230,345] [--res 64] [--fov 16]
#         [--samples 160] [--nmax 2] [--slab 0.5] [--frame-hours 4] [--shell 300] [--scale 0.25] [--iterations 300] [--eta 0.02]
#         [--every 25] [--prune 0.02] [--max 600] [--flux 0.6] [--closures 0] [--precision Float32] [--backend cuda] [--batch 8]
#         [--seed 1] [--tag triband] [--free-spacetime 0] [--a0 0.7] [--inc0 50] [--lm-every 3] [--inner 3] [--pattern 0.005] [--keplerian 0.05]
#         [--resume output/<tag>_params.csv] [--single-stage 0] [--eta-end 0] [--polish 0] [--solve 40] [--lambda 0.01] [--probes 8] [--probe-every 4] [--dense 0] [--chunk 8] [--reuse 1]
# --dense N runs N Levenberg–Marquardt iterations on the explicit Jacobian (Fit.polish_dense!, --chunk columns per pass).
# --polish N runs N Levenberg–Marquardt steps of the matrix-free Gauss–Newton polish (Fit.polish_timeresolved!, --solve
# conjugate-gradient iterations each, --lambda the initial damping) after the Adam stages (with --iterations 0, only the
# polish, e.g. on a --resume'd state); the spacetime stays where it is.
# --resume resumes from the parameters a previous run wrote (the same --seed regenerates the same data; Adam's moments
# start afresh); --single-stage 1 runs one stage with everything free (the way to continue a fit), --eta-end its final
# step (0: --eta/10).
# --flux is the truth's total flux density at 230 GHz in Jy (the densities are scaled to it, so ngehtsim's thermal noise
# applies as it is); --closures 1 fits closure phases and log closure amplitudes instead of the visibilities. With
# --free-spacetime 1 the spin and inclination are fitted jointly (Fit.fit_joint! on the bands, Levenberg–Marquardt on
# the spacetime block from every --lm-every-th frame, in Float64 whatever --precision says) from --a0 and --inc0, with
# the pattern and Keplerian priors tied to the current spin.
using KerrSplat, KerrSplat.Fit, KerrSplat.Splats, KerrSplat.Geodesics, KerrSplat.Transfer
using KernelAbstractions, CUDA, StaticArrays, LinearAlgebra, Random, Printf, DelimitedFiles, Dates
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
days = getopt("--days", 5); start = Date(getstr("--start", "2026-04-01")); bands = parse.(Float64, split(getstr("--bands", "86,230,345"), ","))
res = getopt("--res", 64); fov = getopt("--fov", 16.0); N = getopt("--samples", 160); nmax = getopt("--nmax", 2); slab = getopt("--slab", 0.5)
frame_hours = getopt("--frame-hours", 4.0); n = getopt("--shell", 300); scale = getopt("--scale", 0.25); iterations = getopt("--iterations", 300)
η = getopt("--eta", 0.02); every = getopt("--every", 25); prune = getopt("--prune", 0.02); maxsplats = getopt("--max", 600); flux_target = getopt("--flux", 0.6)
closures = getopt("--closures", 0) == 1; T = getstr("--precision", "Float32") == "Float32" ? Float32 : Float64
backend = getstr("--backend", "cuda") == "cuda" ? CUDABackend() : CPU(); batch = getopt("--batch", 8); seed = getopt("--seed", 1); tag = getstr("--tag", "triband")
free_spacetime = getopt("--free-spacetime", 0) == 1; a0 = getopt("--a0", 0.7); inc0 = getopt("--inc0", 50.0); lm_every = getopt("--lm-every", 3); inner = getopt("--inner", 3)
pattern_σ = getopt("--pattern", 0.005); keplerian_σ = getopt("--keplerian", 0.05)
start_file = getstr("--resume", ""); single_stage = getopt("--single-stage", 0) == 1; η_end = getopt("--eta-end", 0.0); η_end = η_end > 0 ? η_end : η / 10
npolish = getopt("--polish", 0); nsolve = getopt("--solve", 40); λ0 = getopt("--lambda", 0.01); nprobes = getopt("--probes", 8); probe_every = getopt("--probe-every", 4); ndense = getopt("--dense", 0); chunk = getopt("--chunk", 8); reuse = getopt("--reuse", 1)
free_spacetime && T !== Float64 && (@warn "the joint fit runs in Float64 (the geodesics are regenerated at every iteration)"; global T = Float64)
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
device(x) = (y = KernelAbstractions.allocate(backend, eltype(x), size(x)...); copyto!(y, x); y)
_mean(x) = sum(x) / length(x); _median(x) = (s = sort(x); s[(length(s) + 1) ÷ 2])

# ---- M87 and the screen
const M_solar = 6.5e9; const D = 16.8e6 * Transfer.PC; const L = gravitational_radius(M_solar)
t_M = L / Transfer.CL / 3600                                  # hours per M
a = 0.9; θo = deg2rad(60.0); Δα = fov / res
rng = MersenneTwister(seed)
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cpu, a, θo; marcher = Fused(64))

# ---- the coverage: every day's uvfits per band, the scans of each --frame-hours block sharing a frame
function band_coverage(f)
    cov = ScanCoverage{Float64}[]
    for day in 0:days - 1
        date = start + Day(day)
        path = joinpath(outdir, "ngeht_M87_$(date)_$(round(Int, f))GHz.uvfits")
        isfile(path) || error("$path is missing: run make_campaign.py --days $days --start $start first")
        obs = read_uvfits(path)
        scans = scan_index(obs)
        hours = [24day + _mean(obs.time[scans .== k]) for k in 1:maximum(scans)]              # each scan's mean time, campaign hours
        frames = (floor.(hours ./ frame_hours) .+ 0.5) .* frame_hours ./ t_M                  # the block's centre in M
        append!(cov, coverage(obs, frames))
    end
    return cov
end
covs = Dict(f => band_coverage(f) for f in bands)
for f in bands
    c = covs[f]
    @info "coverage at $(f) GHz" scans = length(c) baselines = sum(length(x.u) for x in c) frames = length(unique(x.time for x in c)) span_M = round(maximum(x.time for x in c) - minimum(x.time for x in c), digits = 2) median_sigma_mJy = round(1e3 * _median(vcat((getindex.(x.σ, 1) for x in c)...)), digits = 2)
end
frame_span = maximum(maximum(x.time for x in covs[f]) for f in bands)

# ---- the truth: shell_selffit's six parcels, their densities scaled to --flux Jy at 230 GHz on the first frame
kepler(r) = 1 / (r^1.5 + a)
p = zeros(NPOLARIZEDPARAMS, 6)
for i in 1:6
    φ = 2π * (i - 1) / 6 + 0.3 * randn(rng); r0 = 2.5 + 2.5 * (i - 1) / 5
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.3 * randn(rng), log(0.7), log(0.7), log(0.5), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
               0.0, log(1e9), log(3e5) + 0.3 * randn(rng), log(30.0) + 0.2 * randn(rng), log(20.0) + 0.2 * randn(rng), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.0, 0.3, 0.05 * randn(rng), kepler(r0)]
end
t0f = minimum(x.time for x in covs[bands[argmin(abs.(bands .- 230))]])
function total_flux(q, ν)                                   # Jy, from the zero-spacing visibility of the frame
    img = Fit.pixel_stokes(to_screen(cpu, (out = Vector{Splats.accumulator_type(Float64, nmax)}(undef, npixels(cpu)); fill!(out, zero(eltype(out))); polarized_image!(out, cpu, q, t0f, ν, L; nmax, slab); out)), ν, nothing)
    return real(visibilities(img, Δα, L, D, [0.0], [0.0])[1][1])
end
for _ in 1:3                                                 # absorption makes the scaling slightly nonlinear
    p[13, :] .+= log(flux_target / total_flux(p, 230e9))
end
@info "truth" flux_Jy = Dict(f => round(total_flux(p, f * 1e9), digits = 3) for f in bands) hours_per_M = round(t_M, digits = 2) frame_span_M = round(frame_span, digits = 2)

# ---- the synthetic data per band with the campaign's own noise
trs = Dict(f => synthetic_scans(cpu, p, L, Δα, D, f * 1e9, covs[f]; noise = nothing, closures, rng, nmax, slab) for f in bands)
ndat = Dict(f => ndata(trs[f]) for f in bands)
@info "synthetic data" values = ndat total = sum(values(ndat)) closures

# ---- the shell start, its densities scaled to the truth's 230 GHz flux
q0 = shell_parcels(n; rin = 2.2, rout = 5.5, height = 0.6, scale, spin = a, rng)
q0[13, :] .+= log(total_flux(p, 230e9) / total_flux(q0, 230e9))
if !isempty(start_file)                                     # resume from a previous run's end state (the data above are the same)
    q0 = Matrix{Float64}(readdlm(start_file, ','))
    @info "resuming" from = start_file parcels = size(q0, 2)
end

# ---- the likelihood on the backend in the chosen precision: the three bands summed
gcache = GeodesicCache(backend, camera, Val(N); store_samples = true); regenerate!(gcache, a, θo; marcher = Recurrence(64))
gcacheT = T === Float64 ? gcache : Geodesics.precision(gcache, T)
trsT = Dict(f => Geodesics.precision(trs[f], T) for f in bands)
function valgrad(q)
    dp = KernelAbstractions.zeros(backend, T, size(q))
    qd = device(T.(q))
    χ = zero(T)
    for f in bands
        χ += timeresolved_gradient!(dp, qd, trsT[f], gcacheT, T(L), T(Δα), T(D), T(f * 1e9); nmax, slab, batch_frames = batch)
    end
    return Float64(χ), Float64.(Array(dp))
end
band_chi2(q) = Dict(f => chi2_timeresolved(q, trs[f], cpu, L, Δα, D, f * 1e9; nmax, slab) for f in bands)
χt = band_chi2(p); χ0 = band_chi2(q0)
@info "χ² per band" truth = Dict(f => round(χt[f] / ndat[f], digits = 3) for f in bands) start = Dict(f => round(χ0[f] / ndat[f], digits = 3) for f in bands)
xs = range(-6, 6; length = 25); ys = xs; zs = range(-1.5, 1.5; length = 7)
m0 = recovery_metrics(q0, p, t0f, xs, ys, zs)

# ---- the fit
t_start = time(); trace = String[]
stages = single_stage ? [Fit.Stage(; iterations, η, η_end, label = "everything")] :
         [Fit.Stage(; free = (:x, :y, :z, :s1, :s2, :s3, :logne), iterations = iterations ÷ 3, η, η_end = η / 2, label = "geometry and densities"),
          Fit.Stage(; iterations = iterations - iterations ÷ 3, η, η_end, label = "everything")]
hyg = Fit.Hygiene(every = every, prune_fraction = prune, merge_position = 0.15, max_splats = maxsplats)
ntot = sum(values(ndat))
last_state = Ref(copy(q0))
cb = (si, it, x, v) -> begin
    last_state[] = copy(x)
    it % 20 == 0 && (push!(trace, @sprintf("stage %d iteration %3d  chi2 %10.1f  reduced %.3f  parcels %4d  %.1f min", si, it, v, v / ntot, size(x, 2), (time() - t_start) / 60)); @info trace[end])   # not every 25: the loop skips the callback on a hygiene iteration
end
x_end = [a, θo]; accepted = 0
q, history, events = try
    if free_spacetime
        bandsT = [BandScans(T(f * 1e9), trsT[f], T(Δα), T(D)) for f in bands]
        xj = fit_joint!(copy(q0), [a0, deg2rad(inc0)], bandsT, gcacheT, camera; L = T(L), iterations, η, η_end = η / 10, inner, nmax, slab,
                        pattern = pattern_σ, keplerian = keplerian_σ, hygiene = hyg, lm_every,
                        callback = (it, xq, xs, v) -> (last_state[] = copy(xq); it % 20 == 0 && (push!(trace, @sprintf("iteration %3d  chi2 %10.1f  reduced %.3f  a %.4f  inc %.2f°  parcels %4d  %.1f min", it, v, v / ntot, xs[1], rad2deg(xs[2]), size(xq, 2), (time() - t_start) / 60)); @info trace[end])))
        global x_end = xj[2]; global accepted = xj[4]
        (xj[1], [h[1] for h in xj[3]], xj[5])
    elseif iterations > 0
        Fit.fit!(copy(q0), x -> valgrad(x)[1], stages; hygiene = hyg, gradient = valgrad, callback = cb)
    else
        (copy(q0), Float64[], Tuple{Int,Int,Int,Int}[])
    end
catch err
    writedlm(joinpath(outdir, "$(tag)_failed_params.csv"), last_state[], ',')
    @error "the fit threw; the parameters of its last iteration are in output/$(tag)_failed_params.csv" exception = (err, catch_backtrace())
    rethrow()
end
polish_history = Float64[]
if npolish > 0                                               # the Gauss–Newton polish on the backend, in T, at the held spacetime
    bandsT = [BandScans(T(f * 1e9), trsT[f], T(Δα), T(D)) for f in bands]
    qdev = device(T.(q))
    qdev, ph = polish_timeresolved!(qdev, bandsT, gcacheT, T(L); iterations = npolish, solve_iterations = nsolve, λ = λ0, probes = nprobes, probe_every, nmax, slab, batch_frames = batch,
                                    callback = (it, x, v, dmp, info) -> (push!(trace, @sprintf("polish step %2d  chi2 %10.1f  reduced %.4f  damping %.1e  gain %.2f  predicted %.1f  solve %d its residual %.2e  %.1f min", it, v, v / ntot, dmp, info.gain, info.predicted, info.solve_iterations, info.solve_residual, (time() - t_start) / 60)); @info trace[end]))
    global q = Float64.(Array(qdev)); global polish_history = Float64.(ph)
    global history = vcat(history, polish_history)
end
dense_history = Float64[]
if ndense > 0                                                # Levenberg–Marquardt on the explicit Jacobian (chunked duals), at the held spacetime
    bandsT = [BandScans(T(f * 1e9), trsT[f], T(Δα), T(D)) for f in bands]
    qdev = device(T.(q))
    qdev, dh, _ = polish_dense!(qdev, bandsT, gcacheT, T(L); iterations = ndense, λ = λ0, chunk = Val(chunk), reuse, nmax, slab, batch_frames = batch,
                                callback = (it, x, v, dmp, info) -> (push!(trace, @sprintf("dense step %2d  chi2 %10.1f  reduced %.4f  damping %.1e  gain %.2f  predicted %.1f  tries %d  max step %.2e  reused %d  jacobian %.1f min  %.1f min", it, v, v / ntot, dmp, info.gain, info.predicted, info.tries, info.max_step, info.reused, info.jacobian_seconds / 60, (time() - t_start) / 60)); @info trace[end]))
    global q = Float64.(Array(qdev)); global dense_history = Float64.(dh)
    global history = vcat(history, dense_history)
end
minutes = (time() - t_start) / 60
χ1 = band_chi2(q)
m1 = recovery_metrics(q, p, t0f, xs, ys, zs)
free_spacetime && (cpu_end = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cpu_end, x_end[1], x_end[2]; marcher = Fused(64)); global χ1 = Dict(f => chi2_timeresolved(q, trs[f], cpu_end, L, Δα, D, f * 1e9; nmax, slab) for f in bands))
@info "end" reduced = Dict(f => round(χ1[f] / ndat[f], digits = 3) for f in bands) total = round(sum(values(χ1)) / ntot, digits = 4) parcels = size(q, 2) minutes = round(minutes, digits = 1) spacetime = free_spacetime ? (a = round(x_end[1], digits = 4), inc = round(rad2deg(x_end[2]), digits = 2), accepted) : "held at the truth"
@info "field recovery" start = m0 fin = m1

writedlm(joinpath(outdir, "$(tag)_params.csv"), q, ',')
writedlm(joinpath(outdir, "$(tag)_history.csv"), history, ',')       # the total χ² at every iteration
writedlm(joinpath(outdir, "$(tag)_truth.csv"), p, ',')
open(joinpath(outdir, "$(tag)_summary.txt"), "w") do io
    isempty(start_file) || println(io, "resumed from $start_file ($(single_stage ? "one stage" : "two stages"), eta $η → $η_end)")
    npolish > 0 && println(io, "Gauss–Newton polish: $npolish steps of at most $nsolve LSQR iterations ($nprobes Hutchinson probes for the scaling) from damping $λ0: chi2 $(round.(polish_history, digits = 1))")
    ndense > 0 && println(io, "dense Levenberg–Marquardt: $ndense steps on the explicit Jacobian ($chunk columns per pass, up to $reuse steps per Jacobian) from damping $λ0: chi2 $(round.(dense_history, digits = 1))")
    println(io, "triband ngEHT self-fit ($T on $(backend isa CPU ? "CPU" : "CUDA")): M87 (M $(M_solar) M☉, D 16.8 Mpc), $days days from $start, bands $(bands) GHz, $(res)² pixels of $(round(Δα, digits = 3)) M, $N samples, nmax $nmax slab $slab, frames per $frame_hours h ($(round(frame_span, digits = 1)) M of campaign), $(closures ? "closures" : "visibilities"), values per band $ndat; shell of $n parcels of $scale M, $iterations iterations, eta $η, hygiene every $every (prune $prune, max $maxsplats), seed $seed, truth flux $flux_target Jy at 230 GHz")
    free_spacetime && println(io, "spacetime: start a $a0 inc $inc0; end a $(x_end[1]) inc $(rad2deg(x_end[2])) (truth a $a inc $(rad2deg(θo))); accepted spacetime steps $accepted; lm_every $lm_every inner $inner pattern $pattern_σ keplerian $keplerian_σ")
    println(io, "chi2/N per band: truth $(Dict(f => round(χt[f] / ndat[f], digits = 4) for f in bands)), start $(Dict(f => round(χ0[f] / ndat[f], digits = 3) for f in bands)), end $(Dict(f => round(χ1[f] / ndat[f], digits = 4) for f in bands)); total end $(round(sum(values(χ1)) / ntot, digits = 4)) (truth $(round(sum(values(χt)) / ntot, digits = 4))); parcels $n → $(size(q, 2)) through $(length(events)) hygiene events $events; $(round(minutes, digits = 1)) minutes")
    println(io, "field recovery (density PSNR dB, relative density error, density-weighted temperature and field errors): start $m0; end $m1")
    foreach(l -> println(io, l), trace)
end
