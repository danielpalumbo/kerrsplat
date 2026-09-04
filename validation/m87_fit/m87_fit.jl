# Fit thermal splats in Kerr to the public EHT 2017 M87 data (closure phases and log closure
# amplitudes, Stokes I) through Fit.read_uvfits: the first real-data run of the pipeline.
#
#   julia -t 8 --project=../.. m87_fit.jl --data <file.uvfits> [--res 32] [--samples 60] [--iterations 300]
#                                          [--nsplat 6] [--spin 0.94] [--inc 163] [--flux 0.6] [--seed 1]
#
# Geometry: M = 6.5e9 M⊙, D = 16.8 Mpc; the observer at inclination `inc` from the spin axis
# (163° puts the jet toward us with the ring's southern side approaching). The splats start on a
# ring of radius 4.5 M in the equatorial plane with a mildly sub-Keplerian toroidal velocity and a
# toroidal field; every parameter but the temporal envelope and the pattern rate is free. The
# loss is the closure χ² plus a Gaussian prior on the total flux (compact flux `flux` ± 10%).
# Outputs in output/: the fitted image (ehtim-style FITS), the parameters and a summary.

using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Enzyme, Optimisers, Statistics

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
res = getopt("--res", 32); N = getopt("--samples", 60); iterations = getopt("--iterations", 300)
nsplat = getopt("--nsplat", 6); a = getopt("--spin", 0.94); inc = getopt("--inc", 163.0)
flux = getopt("--flux", 0.6); seed = getopt("--seed", 1); η = getopt("--eta", 0.02)
path = getstr("--data", "/home/daniel/Dropbox/minimal_closures/SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits")
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)

M_solar = 6.5e9; D_pc = 16.8e6; D = D_pc * Transfer.PC; L = gravitational_radius(M_solar)
obs = read_uvfits(path); ν = obs.freq
@info "data" rows = length(obs) stations = obs.stations mjd = obs.mjd freq = ν
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
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.1 * randn(rng), log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0,
               0.0, log(1e9), log(3e5), log(30.0), log(10.0), 0.0, 0.0, 0.0, 0.0, 0.35, 0.0]   # B = φ̂ (thB = 0); u = (0, 0, u_φ)
end
free = trues(size(p)); free[11, :] .= false; free[12, :] .= false; free[21, :] .= false

function image_of(q)
    out = Vector{RadiativeState{Float64}}(undef, npixels(cache)); fill!(out, zero(RadiativeState{Float64}))
    polarized_image!(out, cache, q, 0.0, ν, L)
    return map(st -> observed_stokes(st, ν), to_screen(cache, out))
end
function loss(q)
    img = image_of(q)
    F = sum(getindex.(img, 1)) * psize^2 / Transfer.JY
    return chi2_closures(img, Δα, L, D, data) + ((F - flux) / (0.1 * flux))^2
end
total_flux(q) = sum(getindex.(image_of(q), 1)) * psize^2 / Transfer.JY

@info "start" loss = loss(p) flux_Jy = total_flux(p)
opt = Optimisers.setup(Optimisers.Adam(η), p)
t0 = time()
for it in 1:iterations
    g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p)[1]
    g[.!free] .= 0
    global opt, p = Optimisers.update!(opt, p, g)
    if it % 10 == 0 || it == 1
        χ = loss(p)
        @info "iteration $it" loss = χ reduced = χ / nclosure flux_Jy = total_flux(p) elapsed_min = (time() - t0) / 60
    end
end
img = image_of(p)
χ = chi2_closures(img, Δα, L, D, data)
@info "final" closure_chi2 = χ reduced = χ / nclosure flux_Jy = total_flux(p) minutes = (time() - t0) / 60
write_stokes_fits(joinpath(outdir, "m87_fit.fits"), img, Δα; M_solar, D_pc, freq = ν, mjd = obs.mjd, source = "M87")
writedlm(joinpath(outdir, "m87_params.csv"), p, ',')
open(joinpath(outdir, "m87_summary.txt"), "w") do io
    println(io, "closure χ² $χ over $nclosure closure quantities (reduced $(χ / nclosure)); total flux $(total_flux(p)) Jy")
    println(io, "spin $a inclination $inc res $res samples $N iterations $iterations nsplat $nsplat")
    for i in 1:nsplat
        println(io, "splat $i: r = $(hypot(p[1, i], p[2, i])) M, z = $(p[3, i]), ne = $(exp(p[13, i])), Θe = $(exp(p[14, i])), B = $(exp(p[15, i])) G, u = $(p[18:20, i])")
    end
end
