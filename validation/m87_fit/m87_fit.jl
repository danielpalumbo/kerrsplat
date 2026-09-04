# Fit thermal splats in Kerr to the public EHT 2017 M87 data (closure phases and log closure
# amplitudes, Stokes I) through Fit.read_uvfits: the first real-data run of the pipeline.
#
#   julia -t 8 --project=../.. m87_fit.jl --data <file.uvfits> [--res 32] [--samples 60] [--iterations 300]
#                                          [--nsplat 6] [--spin 0.94] [--inc 163] [--flux 0.6] [--sigma-flux 0.01] [--seed 1]
#                                          [--mode closures|selfcal] [--scan-average] [--sigma-gain 0.1] [--tag name]
#
# Mode `closures` (default) fits the Stokes I closure phases and log closure amplitudes with a
# flux prior. Mode `selfcal` fits the complex visibilities of all four Stokes parameters with one
# complex gain per station and scan as nuisance parameters (the same gain for every Stokes
# parameter: R/L gain ratios and leakage are not modelled), with a Gaussian prior of `sigma-gain`
# on the log-amplitudes and free phases; the flux is then set by the data. `--scan-average`
# averages the data coherently over scans first (ehtim's rules).
#
# Geometry: M = 6.5e9 M⊙, D = 16.8 Mpc; the observer at inclination `inc` from the spin axis
# (163° puts the jet toward us with the ring's southern side approaching). The splats start on a
# ring of radius 4.5 M in the equatorial plane with a mildly sub-Keplerian toroidal velocity and a
# toroidal field; every parameter but the temporal envelope and the pattern rate is free. The
# loss is the closure χ² plus a Gaussian prior on the total flux (compact flux `flux` ± `sigma-flux`
# Jy; closures do not constrain the flux, so the prior must be tight to matter against ~10⁴ closure
# quantities).
# Outputs in output/: the fitted image (ehtim-style FITS), the parameters and a summary.

using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Enzyme, Optimisers, Statistics

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
res = getopt("--res", 32); N = getopt("--samples", 60); iterations = getopt("--iterations", 300)
nsplat = getopt("--nsplat", 6); a = getopt("--spin", 0.94); inc = getopt("--inc", 163.0)
flux = getopt("--flux", 0.6); σflux = getopt("--sigma-flux", 0.01); seed = getopt("--seed", 1); η = getopt("--eta", 0.02)
mode = getstr("--mode", "closures"); σgain = getopt("--sigma-gain", 0.1); tag = getstr("--tag", mode)
path = getstr("--data", "/home/daniel/Dropbox/minimal_closures/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits")
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)

M_solar = 6.5e9; D_pc = 16.8e6; D = D_pc * Transfer.PC; L = gravitational_radius(M_solar)
obs = read_uvfits(path); ν = obs.freq
"--scan-average" in ARGS && (obs = average_scans(obs))
scans = scan_index(obs); nscans = maximum(scans)
@info "data" rows = length(obs) stations = obs.stations mjd = obs.mjd freq = ν scans = nscans mode = mode polarized = count(r -> isfinite(obs.σ[r][2]), 1:length(obs))
tri = scan_triangles(obs); quad = scan_quadrangles(obs)
absidx(k) = abs(k)
phases = closure_phases(obs.vis, tri)
σ_phase = [sqrt(sum((obs.σ[absidx(k)][1] / abs(obs.vis[absidx(k)][1]))^2 for k in t)) for t in tri]
logamps = log_closure_amplitudes(obs.vis, quad)
σ_logamp = [sqrt(sum((obs.σ[absidx(k)][1] / abs(obs.vis[absidx(k)][1]))^2 for k in q)) for q in quad]
# drop closure quantities dominated by noise (|V|/σ < 3 on any leg)
keep_t = [σ < 1.0 for σ in σ_phase]; keep_q = [σ < 1.0 for σ in σ_logamp]
data = ClosureData(obs.u, obs.v, tri[keep_t], phases[keep_t], σ_phase[keep_t], quad[keep_q], logamps[keep_q], σ_logamp[keep_q])
nclosure = count(keep_t) + count(keep_q)
@info "closures" triangles = count(keep_t) quadrangles = count(keep_q) scans = length(unique(round.(obs.time; digits = 5)))

fov = 20.0; Δα = fov / res
camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
regenerate!(cache, a, deg2rad(inc); marcher = Fused(64))
psize = Δα * L / D
@info "screen" res = res fov_M = fov pixel_μas = psize * 206264.806247 * 1e6

rng = MersenneTwister(seed)
p = zeros(NPOLARIZEDPARAMS, nsplat)
for i in 1:nsplat
    φ = 2π * (i - 1) / nsplat + 0.05 * randn(rng)
    r0 = 4.5 + 0.2 * randn(rng)
    # field and velocity components follow the axes (r̂, φ̂, −θ̂): B = Bmag (sin thB cos phB, sin thB sin phB, cos thB),
    # so thB = phB = π/2 is the toroidal field and the toroidal ZAMO velocity is the second component
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.1 * randn(rng), log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0,
               0.0, log(1e9), log(3e5), log(30.0), log(10.0), π / 2, π / 2, 0.0, 0.35, 0.0, 0.0]
end
free = trues(size(p)); free[11, :] .= false; free[12, :] .= false; free[21, :] .= false

function image_of(q)
    out = Vector{RadiativeState{Float64}}(undef, npixels(cache)); fill!(out, zero(RadiativeState{Float64}))
    polarized_image!(out, cache, q, 0.0, ν, L)
    return map(st -> observed_stokes(st, ν), to_screen(cache, out))
end
vdata = VisibilityData(obs)
ndata_vis = 2 * sum(count(isfinite, obs.σ[r]) for r in 1:length(obs))            # real and imaginary parts of the present Stokes visibilities
function loss(q)
    img = image_of(q)
    F = sum(getindex.(img, 1)) * psize^2 / Transfer.JY
    return chi2_closures(img, Δα, L, D, data) + ((F - flux) / σflux)^2
end
nst = length(obs.stations)
t1 = scan_station.(obs.s1, scans, nst); t2 = scan_station.(obs.s2, scans, nst)      # per-scan gains as matrix columns
selfcal_loss(q, g) = chi2_visibilities(image_of(q), Δα, L, D, vdata, g, t1, t2; σ_logamp = σgain)
total_flux(q) = sum(getindex.(image_of(q), 1)) * psize^2 / Transfer.JY

gains = zeros(2, nst * nscans)
if mode == "closures"
    @info "start" loss = loss(p) flux_Jy = total_flux(p)
else
    @info "start" loss = selfcal_loss(p, gains) reduced = selfcal_loss(p, gains) / ndata_vis flux_Jy = total_flux(p)
end
opt = Optimisers.setup(Optimisers.Adam(η), p)
optg = Optimisers.setup(Optimisers.Adam(0.05), gains)
t0 = time()
for it in 1:iterations
    if mode == "closures"
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p)[1]
    else
        g, gg = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(selfcal_loss), p, gains)
        global optg, gains = Optimisers.update!(optg, gains, gg)
    end
    g[.!free] .= 0
    global opt, p = Optimisers.update!(opt, p, g)
    if it % 10 == 0 || it == 1
        χ = mode == "closures" ? loss(p) : selfcal_loss(p, gains)
        @info "iteration $it" loss = χ reduced = χ / (mode == "closures" ? nclosure : ndata_vis) flux_Jy = total_flux(p) elapsed_min = (time() - t0) / 60
    end
end
img = image_of(p)
χ = chi2_closures(img, Δα, L, D, data)
if mode == "selfcal"
    χv = selfcal_loss(p, gains)
    @info "final (self-calibration)" chi2 = χv reduced = χv / ndata_vis closure_chi2 = χ closure_reduced = χ / nclosure flux_Jy = total_flux(p) minutes = (time() - t0) / 60 gain_amplitude_rms = sqrt(sum(abs2, gains[1, :]) / size(gains, 2))
else
    @info "final" closure_chi2 = χ reduced = χ / nclosure flux_Jy = total_flux(p) minutes = (time() - t0) / 60
end
write_stokes_fits(joinpath(outdir, "m87_$(tag).fits"), img, Δα; M_solar, D_pc, freq = ν, mjd = obs.mjd, source = "M87")
writedlm(joinpath(outdir, "m87_$(tag)_params.csv"), p, ',')
mode == "selfcal" && writedlm(joinpath(outdir, "m87_$(tag)_gains.csv"), gains, ',')
open(joinpath(outdir, "m87_$(tag)_summary.txt"), "w") do io
    println(io, "mode $mode; closure χ² $χ over $nclosure closure quantities (reduced $(χ / nclosure)); total flux $(total_flux(p)) Jy")
    mode == "selfcal" && println(io, "self-calibration χ² $(selfcal_loss(p, gains)) over $ndata_vis data values (reduced $(selfcal_loss(p, gains) / ndata_vis)); net polarization Q, U, V / I of the model: $(sum(getindex.(img, 2)) / sum(getindex.(img, 1))), $(sum(getindex.(img, 3)) / sum(getindex.(img, 1))), $(sum(getindex.(img, 4)) / sum(getindex.(img, 1)))")
    println(io, "spin $a inclination $inc res $res samples $N iterations $iterations nsplat $nsplat")
    for i in 1:nsplat
        println(io, "splat $i: r = $(hypot(p[1, i], p[2, i])) M, z = $(p[3, i]), ne = $(exp(p[13, i])), Θe = $(exp(p[14, i])), B = $(exp(p[15, i])) G, u = $(p[18:20, i])")
    end
end
