# The M87 2017 full-polarization data (all four correlation products, HOPS, netcal) fitted
# through the instrument model with Comrade's structure: per-scan R and L gains with a reference
# station, d-terms per station over the track, the feed rotation of the EHT array from the antenna
# table (the products are feed-rotation corrected, so J = R† G D R), Comrade's priors and a 2%
# fractional noise floor; the sky the six-parcel model of the earlier self-calibrated fit
# (validation/m87_fit/output/m87_selfcal6_params.csv), fitted jointly with the instrument by
# `Fit.selfcal!` and polished by the joint Levenberg–Marquardt with the Laplace covariance.
# M87 does not change over a night, so every scan sees the frame at t = 0 (one dual sweep per
# iteration). Reports the χ² of the products, the closure χ² of the sky alone (gain-independent),
# and the d-terms per station with their Laplace errors, to be read against the published
# 2017 D-terms (EHT Collaboration 2021, Paper VII).
#
#     julia -t 8 --project=../.. m87_jones.jl [--data <file.uvfits>] [--init params.csv] [--res 32] [--samples 60] [--iterations 300] [--eta 0.005] [--eta-gain 0.02] [--polish 6] [--fnoise 0.02] [--tag jones]
#                                          [--no-leakage] [--sigma-lg 0.2] [--sigma-lgrat 0.1] [--raw] [--fix-sky] [--uvmin 0.1]
# `--uvmin` (Gλ) drops the shorter baselines, as the earlier M87 runs did: the intra-site baselines see the jet's extended flux.
# `--no-leakage` drops the d-terms, `--sigma-lg` sets the log-amplitude prior width (Comrade's 0.2; the LMT keeps 1.0 unless it is
# tighter), `--raw` uses the raw-data Jones chain G D R instead of R† G D R, `--fix-sky` fits the instrument alone on the starting sky.
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Optimisers, Krang, CUDA, Adapt, Printf, Statistics

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
path = getstr("--data", "/home/daniel/Downloads/M87_Uncertainty/hops_3601_M87+netcal.uvfits")
init = getstr("--init", joinpath(@__DIR__, "output", "m87_selfcal6_params.csv"))
res = getopt("--res", 32); N = getopt("--samples", 60); iterations = getopt("--iterations", 300)
η = getopt("--eta", 0.005); ηgain = getopt("--eta-gain", 0.02); npolish = getopt("--polish", 6); fnoise = getopt("--fnoise", 0.02)
tag = getstr("--tag", "jones"); a = getopt("--spin", 0.94); inc = getopt("--inc", 163.0)
leakage = !("--no-leakage" in ARGS); σ_lg = getopt("--sigma-lg", 0.2); σ_lgrat = getopt("--sigma-lgrat", 0.1); corrected = !("--raw" in ARGS); fixsky = "--fix-sky" in ARGS
uvmin = getopt("--uvmin", 0.1)
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)

M_solar = 6.5e9; D = 16.8e6 * Transfer.PC; L = gravitational_radius(M_solar)
obs = average_scans(read_uvfits(path)); ν = obs.freq
if uvmin > 0
    keep = hypot.(obs.u, obs.v) .>= uvmin * 1e9
    obs = Fit.Observation{Float64}(obs.time[keep], obs.tint[keep], obs.s1[keep], obs.s2[keep], obs.stations, obs.u[keep], obs.v[keep], obs.vis[keep], obs.σ[keep],
                                   obs.freq, obs.bandwidth, obs.ra, obs.dec, obs.mjd, obs.source, obs.coh[keep], obs.σ_coh[keep])
    @info "baselines shorter than $uvmin Gλ dropped" remaining = length(obs)
end
# Comrade's fractional noise floor on the products: σ → √(σ² + (f |V|)²)
for r in 1:length(obs)
    obs.σ_coh[r] = SVector{4}(sqrt(obs.σ_coh[r][p]^2 + (fnoise * abs(obs.coh[r][p]))^2) for p in 1:4)
    obs.σ[r] = SVector{4}(sqrt(obs.σ[r][p]^2 + (fnoise * abs(obs.vis[r][p]))^2) for p in 1:4)
end
xyz = antenna_positions(path)
mounts = Dict(s => EHT_MOUNTS[s] for s in obs.stations)
φ1, φ2 = feed_angles(obs, xyz, mounts)
inst = InstrumentModel(obs; feedangles = (φ1, φ2), reference = SingleReference("AA"), leakage, corrected, σ_lg, σ_lg_station = Dict("LM" => max(1.0, σ_lg)), σ_lgrat)
gm, dm = free_mask(inst, obs)
tr = observed_scans(obs, zeros(inst.nseg))                                # one frame: M87 is static over the night
ndat = ndata(tr)
@info "data" file = basename(path) rows = length(obs) scans = inst.nseg stations = obs.stations products = ndat free_gains = count(gm) dterm_parts = count(dm) fractional_noise = fnoise feed_angle_range = extrema(vcat(φ1, φ2))

fov = 20.0; Δα = fov / res
camera, binning = binned_grid((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
θo = deg2rad(inc)
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cache, a, θo; marcher = Fused(64))
gcache = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true); regenerate!(gcache, a, θo; marcher = Recurrence(64))
p = Matrix{Float64}(readdlm(init, ','))
free = fixsky ? falses(size(p)) : freeze(p, (:x, :y, :z, :s1, :s2, :s3, :q1, :q2, :q3, :q4, :logne, :logTe, :logB, :thB, :phB, :u1, :u2, :u3))
gains, dterms = zero_instrument(inst)
# the closure quantities of the data (Stokes I, gain-independent) for the sky alone, before and after
tri = scan_triangles(obs); quad = scan_quadrangles(obs)
phases = closure_phases(obs.vis, tri); σ_phase = [sqrt(sum((obs.σ[abs(k)][1] / abs(obs.vis[abs(k)][1]))^2 for k in t)) for t in tri]
logamps = log_closure_amplitudes(obs.vis, quad); σ_logamp = [sqrt(sum((obs.σ[abs(k)][1] / abs(obs.vis[abs(k)][1]))^2 for k in qd)) for qd in quad]
keep_t = σ_phase .< 1; keep_q = σ_logamp .< 1
cdata = ClosureData(obs.u, obs.v, tri[keep_t], phases[keep_t], σ_phase[keep_t], quad[keep_q], logamps[keep_q], σ_logamp[keep_q])
nclos = count(keep_t) + count(keep_q)
closure_chi2(x) = chi2_closures(bin(binning, polarized_cube(cache, x, [0.0], [ν], L))[:, :, 1, 1], Δα, L, D, cdata)
χc0 = closure_chi2(p)
@info "starting sky alone" closure_chi2 = χc0 reduced = χc0 / nclos closure_quantities = nclos leakage = leakage corrected = corrected σ_lg = σ_lg fix_sky = fixsky
χ0 = chi2_timeresolved(p, tr, cache, L, Δα, D, ν; binning, instrument = (inst, gains, dterms))
@info "start (unit instrument)" chi2 = χ0 reduced = χ0 / ndat splats = size(p, 2)
t0 = time()
q, gains, dterms, history = selfcal!(copy(p), gains, dterms, tr, gcache, L, Δα, D, ν; inst, masks = (gm, dm), free, iterations, η, η_inst = ηgain, binning,
                                     callback = (it, x, g, d, v) -> (it % 25 == 0 && @info "iteration $it" chi2 = v reduced = v / ndat minutes = (time() - t0) / 60))
χ1 = chi2_timeresolved(q, tr, cache, L, Δα, D, ν; binning, instrument = (inst, gains, dterms))
@info "after Adam" chi2 = χ1 reduced = χ1 / ndat minutes = (time() - t0) / 60
laplace_d = nothing
if npolish > 0
    tP = time()
    x0 = pack(q, free, gains, gm, dterms, dm)
    resid(x) = (t = unpack(q, free, gains, gm, dterms, dm, x); timeresolved_residuals(t[1], tr, cache, L, Δα, D, ν; binning, instrument = (inst, t[2], t[3])))
    x, hist, covj = levenberg_marquardt!(x0, resid; iterations = npolish, chunk = 12)
    q, gains, dterms = unpack!(copy(q), free, copy(gains), gm, copy(dterms), dm, x)
    χ1 = hist[end]
    σx = sqrt.(max.(diag(covj), 0.0))
    nsky = count(free); ng = count(gm)
    laplace_d = fill(NaN, 4, nstations(inst)); laplace_d[dm] .= σx[nsky+ng+1:end]           # the free d-term parts back in station columns
    @info "joint Levenberg–Marquardt polish ($npolish iterations)" chi2 = hist[1] => hist[end] reduced = χ1 / ndat minutes = (time() - tP) / 60
end
# the sky alone against the closure quantities (gain-independent), as the earlier runs reported it
img = bin(binning, polarized_cube(cache, q, [0.0], [ν], L))[:, :, 1, 1]
χc = chi2_closures(img, Δα, L, D, cdata)
flux = real(visibilities(img, Δα, L, D, [0.0], [0.0])[1][1])
@info "sky alone" closure_chi2 = χc reduced = χc / nclos flux_Jy = flux
# the instrument: gain amplitudes and phases per station, the d-terms with their Laplace errors
lines = String[]
for (s, name) in enumerate(obs.stations)
    cols = [gain_column(inst, s, g) for g in 1:inst.nseg if gm[1, gain_column(inst, s, g)]]
    lgR = gains[1, cols]; gpR = gains[2, cols]; lgrat = gains[3, cols]; gprat = gains[4, cols]
    dR = complex(dterms[1, s], dterms[2, s]); dL = complex(dterms[3, s], dterms[4, s])
    σd = laplace_d === nothing ? fill(NaN, 4) : laplace_d[:, s]
    push!(lines, @sprintf("%-3s  scans %2d  |gR| %.3f ± %.3f  phase rms %.2f rad  L/R amp %.3f  L−R phase %.3f rad  D_R = %+.4f%+.4fi (±%.4f, ±%.4f)  D_L = %+.4f%+.4fi (±%.4f, ±%.4f)",
                  name, length(cols), exp(sum(lgR) / max(length(cols), 1)), std(exp.(lgR)), sqrt(sum(abs2, gpR) / max(length(cols), 1)), exp(sum(lgrat) / max(length(cols), 1)), sum(gprat) / max(length(cols), 1),
                  real(dR), imag(dR), σd[1], σd[2], real(dL), imag(dL), σd[3], σd[4]))
end
@info "instrument per station\n" * join(lines, "\n")
writedlm(joinpath(outdir, "m87_$(tag)_params.csv"), q, ','); writedlm(joinpath(outdir, "m87_$(tag)_gains.csv"), gains, ','); writedlm(joinpath(outdir, "m87_$(tag)_dterms.csv"), dterms, ',')
open(joinpath(outdir, "m87_$(tag)_summary.txt"), "w") do io
    println(io, "M87 2017 full polarization through the instrument model: $(length(obs)) rows, $(inst.nseg) scans, $ndat product values, $(count(gm)) gains, $(count(dm)) d-term parts, fractional noise $fnoise")
    println(io, "options: leakage $leakage, corrected $corrected, sigma_lg $σ_lg, fix_sky $fixsky")
    println(io, "chi2 start $χ0 (reduced $(χ0 / ndat)) end $χ1 (reduced $(χ1 / ndat)); closure chi2 of the sky $χc0 → $χc over $nclos (reduced $(χc0 / nclos) → $(χc / nclos)); flux $flux Jy")
    foreach(l -> println(io, l), lines)
end
