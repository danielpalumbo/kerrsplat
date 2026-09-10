# The first real time-resolved fit: Sgr A* (EHT 2017, April 11, low band; the HOPS file with the
# ALMA feed rotation applied and the leakage calibrated, so the instrument is the per-scan R and L
# gains alone) through the time-resolved likelihood with the instrument model. Every scan sees the
# slow-light frame at its own time (`scan_times`, GM/c³ = 20.5 s: the 4.9-hour track spans 865 M),
# or the scans are grouped into `--frames` frames. The sky: `--nsplat` thermal parcels on a ring
# of radius 5 M (the 52 μas ring is 10 M across for 4.15 × 10⁶ M⊙ at 8.15 kpc) with free
# velocities, fields and pattern rates (starting at zero: a static model, which the data may then
# move), fitted jointly with the gains by `Fit.selfcal!` on the GPU with a total-flux prior per
# frame; then per-scan χ² and the model's light curve.
#
#     julia -t 8 --project=../.. sgra_jones.jl [--data <file.uvfits>] [--res 32] [--samples 60] [--nsplat 8] [--iterations 600] [--eta 0.005] [--eta-gain 0.02]
#                                          [--spin 0.94] [--inc 150] [--flux 2.4] [--sigma-flux 0.05] [--fnoise 0.02] [--uvmin 0.1] [--frames 0] [--seed 1]
#                                          [--init params.csv] [--polish 0] [--tag jones] [--stage closures|selfcal] [--staged] [--static] [--eta-omega 0.05]
#
# The protocol of the real-data pipelines: `--stage closures` fits the sky to the closure phases and log closure amplitudes
# of every scan (gain-free, `closure_scans`) with the flux prior, from the ring; `--stage selfcal` (the default) fits the
# products through the per-scan gains jointly with the sky, from `--init` (the closure stage's parameters). A first attempt
# that self-calibrated straight from the ring stalled at χ²/N 222 with the gains absorbing the sky's wrong flux scale.
# In both stages the parcels' densities are first scaled so that the model's flux matches the prior. `--staged` runs the
# closure stage as the M87 fits did: the geometry and densities first (pattern rates at zero: a static sky), then the
# plasma rows, then everything with the pattern rates free, `--iterations` each. `--static` freezes the pattern rates at
# zero in every stage (the staged run showed the static geometry stage reaching a closure χ²/N of 5 and the pattern rates,
# freed at the common Adam step of 0.005 rad/M per iteration, wrecking it within a hundred iterations); when they are free,
# `--eta-omega` is the factor of the common step they move with (0.05: 2.5e-4 rad/M per iteration at the default step).
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Optimisers, Krang, CUDA, Adapt, Printf, Statistics

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
path = getstr("--data", "/home/daniel/Dropbox/b2vis/realdata/obsdata/sgra_2017/hops_3601_SGRA_LO_netcal_LMTcal_10s_ALMArot_dcal.uvfits")
res = getopt("--res", 32); N = getopt("--samples", 60); nsplat = getopt("--nsplat", 8); iterations = getopt("--iterations", 600)
η = getopt("--eta", 0.005); ηgain = getopt("--eta-gain", 0.02); a = getopt("--spin", 0.94); inc = getopt("--inc", 150.0)
fluxprior = getopt("--flux", 2.4); σflux = getopt("--sigma-flux", 0.05); fnoise = getopt("--fnoise", 0.02); uvmin = getopt("--uvmin", 0.1)
nframes = getopt("--frames", 0); seed = getopt("--seed", 1); init = getstr("--init", ""); npolish = getopt("--polish", 0); tag = getstr("--tag", "jones")
stage = getstr("--stage", "selfcal"); stage in ("closures", "selfcal") || error("--stage must be closures or selfcal"); staged = "--staged" in ARGS; static = "--static" in ARGS
ηω = getopt("--eta-omega", 0.05); steps = (; omega = ηω)
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)

M_solar = 4.15e6; D = 8.15e3 * Transfer.PC; L = gravitational_radius(M_solar)
obs = average_scans(read_uvfits(path)); ν = obs.freq
if uvmin > 0
    keep = hypot.(obs.u, obs.v) .>= uvmin * 1e9
    obs = Fit.Observation{Float64}(obs.time[keep], obs.tint[keep], obs.s1[keep], obs.s2[keep], obs.stations, obs.u[keep], obs.v[keep], obs.vis[keep], obs.σ[keep],
                                   obs.freq, obs.bandwidth, obs.ra, obs.dec, obs.mjd, obs.source, obs.coh[keep], obs.σ_coh[keep])
end
for r in 1:length(obs)                                   # Comrade's fractional noise floor
    obs.σ_coh[r] = SVector{4}(sqrt(obs.σ_coh[r][p]^2 + (fnoise * abs(obs.coh[r][p]))^2) for p in 1:4)
    obs.σ[r] = SVector{4}(sqrt(obs.σ[r][p]^2 + (fnoise * abs(obs.vis[r][p]))^2) for p in 1:4)
end
inst = InstrumentModel(obs; leakage = false, reference = SingleReference("AA"))       # d-terms calibrated in this file; the feed rotation commutes with diagonal gains
gm, dm = free_mask(inst, obs)
tscan = scan_times(obs, M_solar)
times = nframes > 0 ? (edges = range(minimum(tscan), maximum(tscan), length = nframes + 1); [edges[min(searchsortedlast(edges, t), nframes)] + step(edges) / 2 for t in tscan]) : tscan
tr = stage == "closures" ? closure_scans(obs, times) : observed_scans(obs, times)
ndat = ndata(tr)
@info "data" file = basename(path) rows = length(obs) scans = inst.nseg frames = length(frame_times(tr)) span_M = round(maximum(tscan); digits = 0) stations = obs.stations products = ndat free_gains = count(gm) fractional_noise = fnoise

fov = 20.0; Δα = fov / res
camera, binning = binned_grid((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
θo = deg2rad(inc)
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cache, a, θo; marcher = Fused(64))
gcache = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true); regenerate!(gcache, a, θo; marcher = Recurrence(64))
psize = Δα * L / D
@info "screen" res pixel_μas = psize * 206264.806247e6 M_μas = L / D * 206264.806247e6

rng = MersenneTwister(seed)
p = zeros(NPOLARIZEDPARAMS, nsplat)
for i in 1:nsplat
    φ = 2π * (i - 1) / nsplat + 0.05 * randn(rng); r0 = 5.0 + 0.2 * randn(rng)
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.1 * randn(rng), log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0,
               0.0, log(1e9), log(1e6), log(20.0), log(30.0), π / 2, π / 2, 0.0, 0.35, 0.0, 0.0]
end
isempty(init) || (p = Matrix{Float64}(readdlm(init, ',')); nsplat = size(p, 2); @info "initialized from $init" nsplat)
free = freeze(p, static ? (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :logne, :logTe, :logB, :thB, :phB, :u1, :u2, :u3) :
                          (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :logne, :logTe, :logB, :thB, :phB, :u1, :u2, :u3, :omega))
image_prior = img -> [(total_flux(img, Δα, L, D) - fluxprior) / σflux]
# the densities scaled so that the first frame's flux matches the prior (the parcels' emission is linear in nₑ where thin)
F0 = total_flux(bin(binning, polarized_cube(cache, p, [times[1]], [ν], L))[:, :, 1, 1], Δα, L, D)
if isempty(init)
    p[13, :] .+= log(fluxprior / F0)
    F1 = total_flux(bin(binning, polarized_cube(cache, p, [times[1]], [ν], L))[:, :, 1, 1], Δα, L, D)
    @info "densities scaled to the flux prior" flux_before = F0 flux_after = F1
end
gains, dterms = zero_instrument(inst)
instrument = stage == "closures" ? nothing : (inst, gains, dterms)
χ0 = chi2_timeresolved(p, tr, cache, L, Δα, D, ν; binning, instrument, image_prior)
@info "start ($stage)" chi2 = χ0 reduced = χ0 / ndat splats = nsplat
t0 = time()
if stage == "closures"
    function valgrad(x)
        dp = CUDA.zeros(Float64, size(x))
        χ = timeresolved_gradient!(dp, CuArray(x), tr, gcache, L, Δα, D, ν; binning, image_prior)
        return χ, Array(dp)
    end
    allrows = static ? (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :logne, :logTe, :logB, :thB, :phB, :u1, :u2, :u3) :
                       (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :logne, :logTe, :logB, :thB, :phB, :u1, :u2, :u3, :omega)
    stages = staged ? [Fit.Stage(; free = (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :logne), iterations, η = 2η, η_end = η / 2, label = "geometry, static"),
                       Fit.Stage(; free = (:logne, :logTe, :logB, :thB, :phB, :u1, :u2, :u3), iterations, η, η_end = η / 5, label = "plasma"),
                       Fit.Stage(; free = allrows, iterations, η, η_end = η / 10, steps, label = "everything, pattern rates free")] :
                      [Fit.Stage(; free = allrows, iterations, η, η_end = η / 10, steps)]
    q, history, _ = Fit.fit!(copy(p), x -> chi2_timeresolved(x, tr, cache, L, Δα, D, ν; binning, image_prior), stages; hygiene = Fit.Hygiene(every = 0), gradient = valgrad,
                             callback = (si, it, x, v) -> (it % 25 == 0 && @info "stage $si iteration $it" chi2 = v reduced = v / ndat minutes = (time() - t0) / 60))
else
    q, gains, dterms, history = selfcal!(copy(p), gains, dterms, tr, gcache, L, Δα, D, ν; inst, masks = (gm, dm), free, iterations, η, η_inst = ηgain, steps, binning, image_prior,
                                         callback = (it, x, g, d, v) -> (it % 25 == 0 && @info "iteration $it" chi2 = v reduced = v / ndat minutes = (time() - t0) / 60))
    instrument = (inst, gains, dterms)
end
χ1 = chi2_timeresolved(q, tr, cache, L, Δα, D, ν; binning, instrument, image_prior)
@info "after Adam ($stage)" chi2 = χ1 reduced = χ1 / ndat minutes = (time() - t0) / 60
if npolish > 0 && stage == "selfcal"
    tP = time()
    x0 = pack(q, free, gains, gm, dterms, dm)
    resid(x) = (t = unpack(q, free, gains, gm, dterms, dm, x); timeresolved_residuals(t[1], tr, cache, L, Δα, D, ν; binning, instrument = (inst, t[2], t[3]), image_prior))
    x, hist, covj = levenberg_marquardt!(x0, resid; iterations = npolish, chunk = 12)
    q, gains, dterms = unpack!(copy(q), free, copy(gains), gm, copy(dterms), dm, x)
    χ1 = hist[end]
    @info "joint Levenberg–Marquardt polish ($npolish iterations)" chi2 = hist[1] => hist[end] reduced = χ1 / ndat minutes = (time() - tP) / 60
end
# per scan: the products' χ² through the fitted instrument, the closure χ² of the sky alone at the scan's frame, the model's flux
scans_of_rows = scan_index(obs)
tri = scan_triangles(obs); quad = scan_quadrangles(obs)
phases = closure_phases(obs.vis, tri); σ_phase = [sqrt(sum((obs.σ[abs(k)][1] / abs(obs.vis[abs(k)][1]))^2 for k in t)) for t in tri]
logamps = log_closure_amplitudes(obs.vis, quad); σ_logamp = [sqrt(sum((obs.σ[abs(k)][1] / abs(obs.vis[abs(k)][1]))^2 for k in qd)) for qd in quad]
keep_t = σ_phase .< 1; keep_q = σ_logamp .< 1
frames = Dict(t => bin(binning, polarized_cube(cache, q, [t], [ν], L))[:, :, 1, 1] for t in frame_times(tr))
lines = String[]; χtot_c = 0.0; ntot_c = 0
for (k, s) in enumerate(tr.scans)
    rows = findall(==(k), scans_of_rows); img = frames[s.time]
    χs = stage == "closures" ? scan_loss(img, Δα, L, D, s) : scan_loss(img, Δα, L, D, s, (inst, gains, dterms)); ns = ndata(s)
    kt = [i for i in eachindex(tri) if keep_t[i] && scans_of_rows[abs(tri[i][1])] == k]; kq = [i for i in eachindex(quad) if keep_q[i] && scans_of_rows[abs(quad[i][1])] == k]
    cdata = ClosureData(obs.u, obs.v, tri[kt], phases[kt], σ_phase[kt], quad[kq], logamps[kq], σ_logamp[kq])
    χc = chi2_closures(img, Δα, L, D, cdata); nc = length(kt) + length(kq)
    global χtot_c += χc; global ntot_c += nc
    F = total_flux(img, Δα, L, D)
    sts = join(obs.stations[sort(unique(vcat(obs.s1[rows], obs.s2[rows])))], " ")
    push!(lines, @sprintf("scan %2d  UT %6.3f h  t = %6.1f M  %3d %s χ²/N %6.2f  %3d closures χ²/N %6.2f  flux %.3f Jy  stations %s", k, mean(obs.time[rows]), s.time, ns, stage == "closures" ? "closures" : "products", χs / max(ns, 1), nc, nc > 0 ? χc / nc : NaN, F, sts))
end
@info "per scan\n" * join(lines, "\n")
@info "sky alone" closure_chi2 = χtot_c closures = ntot_c reduced = χtot_c / max(ntot_c, 1) pattern_rates = round.(q[21, :]; sigdigits = 3) flux_range = extrema(total_flux(frames[t], Δα, L, D) for t in frame_times(tr))
glines = [@sprintf("%-3s  scans %2d  |gR| %.3f ± %.3f  phase rms %.2f rad  L/R amp %.3f", name, count(g -> gm[1, gain_column(inst, s, g)], 1:inst.nseg), exp(mean(gains[1, [gain_column(inst, s, g) for g in 1:inst.nseg if gm[1, gain_column(inst, s, g)]]])),
           std(exp.(gains[1, [gain_column(inst, s, g) for g in 1:inst.nseg if gm[1, gain_column(inst, s, g)]]])), sqrt(mean(abs2, gains[2, [gain_column(inst, s, g) for g in 1:inst.nseg if gm[1, gain_column(inst, s, g)]]])),
           exp(mean(gains[3, [gain_column(inst, s, g) for g in 1:inst.nseg if gm[1, gain_column(inst, s, g)]]]))) for (s, name) in enumerate(obs.stations) if any(gm[1, gain_column(inst, s, g)] for g in 1:inst.nseg)]
@info "gains per station\n" * join(glines, "\n")
writedlm(joinpath(outdir, "sgra_$(tag)_params.csv"), q, ','); writedlm(joinpath(outdir, "sgra_$(tag)_gains.csv"), gains, ',')
open(joinpath(outdir, "sgra_$(tag)_summary.txt"), "w") do io
    println(io, "Sgr A* 2017 April 11 low band through the time-resolved likelihood, stage $stage$(staged ? " (staged)" : "")$(static ? " (static)" : ""): $(length(obs)) rows, $(inst.nseg) scans in $(length(frame_times(tr))) frames over $(round(maximum(tscan); digits = 0)) M, $ndat data values, $(count(gm)) gains, $nsplat parcels, flux prior $fluxprior ± $σflux Jy, noise floor $fnoise, uvmin $uvmin, $iterations iterations, spin $a inclination $inc, init '$init'")
    println(io, "chi2 start $χ0 (reduced $(χ0 / ndat)) end $χ1 (reduced $(χ1 / ndat)); closure chi2 of the sky per frame $χtot_c over $ntot_c (reduced $(χtot_c / max(ntot_c, 1)))")
    println(io, "pattern rates: $(q[21, :])")
    foreach(l -> println(io, l), lines); foreach(l -> println(io, l), glines)
end
