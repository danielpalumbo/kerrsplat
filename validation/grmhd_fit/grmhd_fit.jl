# Cross-model recovery against a GRMHD snapshot (plan §7.5 item 7 ii): fit polarized splats to an
# ipole full-Stokes image of a KHARMA simulation (converted with ipole_h5_to_fits.py; the default
# is the Sgr A* MAD a = 0.9375, R_high = 10 snapshot seen from 130°, 200 μas across at 8.13 kpc for
# 4.14e6 M⊙, 400² pixels block-averaged to RES²), starting from a ring of splats with densification.
#
#     julia -t 8 --project=../.. grmhd_fit.jl --image <file.fits> [--res 40] [--samples 200] [--iterations 300]
#                                             [--spin 0.9375] [--inc 130] [--msolar 4.14e6] [--dpc 8127]
#
# Writes output/ (untracked): fitted parameters, model and data images (CSV), a summary.
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Enzyme, Optimisers, Krang, Statistics

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
const RES = getopt("--res", 40); const N = getopt("--samples", 200); const ITER = getopt("--iterations", 300)
const a = getopt("--spin", 0.9375); const θo = deg2rad(getopt("--inc", 130.0))
const M_solar = getopt("--msolar", 4.14e6); const D_pc = getopt("--dpc", 8127.0)
const path = getstr("--image", joinpath(@__DIR__, "kharma_a094_i130.fits"))
const D = D_pc * Transfer.PC; const L = gravitational_radius(M_solar)

# ---- data: the ipole image block-averaged to RES² --------------------------------------------------
img_jy, hdr = read_stokes_fits(path)
nx = size(img_jy, 1); b = nx ÷ RES; nx % RES == 0 || error("RES must divide $nx")
ν = hdr.freq
psize_rad = hdr.psize_deg * π / 180; fov = psize_rad * nx * D / L                     # in M
Ω_block = (b * psize_rad)^2
data = Array{SVector{4,Float64}}(undef, RES, RES, 1, 1)
for i in 1:RES, j in 1:RES
    data[i, j, 1, 1] = sum(img_jy[(i-1)*b+1:i*b, (j-1)*b+1:j*b]) * Transfer.JY / Ω_block   # cgs intensity of the block
end
peak = maximum(norm.(data[:, :, 1, 1]))
σ = SVector(0.02, 0.01, 0.01, 0.005) * peak
movie = StokesMovie(data, [0.0], [ν], σ)
xs = [(((i - 1) * b + (b - 1) / 2 + 0.49) / nx - 0.5) * fov for i in 1:RES]         # ipole's pixel centres (its 0.49-pixel x offset)
ys = [(((j - 1) * b + (b - 1) / 2 + 0.5) / nx - 0.5) * fov for j in 1:RES]
camera = Geodesics.Camera(vec([xs[i] for i in 1:RES, j in 1:RES]), vec([ys[j] for i in 1:RES, j in 1:RES]), (RES, RES))
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
regenerate!(cache, a, θo; marcher = Fused(64))
met = Krang.Kerr(a)
flux(im) = sum(getindex.(im, 1)) * Ω_block / Transfer.JY
println("data: $(nx)² ipole image → $(RES)² blocks of $(round(b * psize_rad * 206264.806247e6; digits = 2)) μas; fov $(round(fov; digits = 2)) M; flux $(round(flux(data[:, :, 1, 1]); digits = 3)) Jy; a = $a, θo = $(rad2deg(θo))°, M = $M_solar M⊙, D = $D_pc pc")

# ---- initial splats: a ring at r = 4 M with Keplerian velocity and toroidal field -----------------
nring = 6
p = zeros(NPOLARIZEDPARAMS, nring)
for k in 1:nring
    ϕ = 2π * (k - 0.5) / nring; r = 4.0
    x, y, z = quasi_cartesian_kerr_schild(met, r, π / 2, ϕ)
    Ω = 1 / (r^1.5 + a); gdd = Krang.metric_dd(met, r, π / 2)
    ut = 1 / sqrt(-(gdd[1, 1] + 2Ω * gdd[1, 4] + Ω^2 * gdd[4, 4]))
    uz = Krang.jac_zamo_u_bl_d(met, r, π / 2) * SVector(ut, 0.0, 0.0, Ω * ut)
    p[:, k] = [x, y, z, log(1.5), log(1.5), log(0.6), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(3e6), log(30.0), log(30.0), 0.0, 0.0, uz[2], uz[3], uz[4], 0.0]
end
χ0 = chi2(p, movie, cache, L)
println("start: $nring splats, χ² = $χ0 for $(4 * length(data)) data points, model flux $(round(flux(polarized_image(cache, p, 0.0, ν, L)); digits = 3)) Jy")
rng = MersenneTwister(1)
stages = [Fit.Stage(free = (:logne, :logTe, :logB, :thB, :phB), iterations = ITER ÷ 5, η = 0.05, η_end = 0.02, label = "plasma"),
          Fit.Stage(iterations = ITER - ITER ÷ 5, η = 0.03, η_end = 0.003, label = "everything")]
t0 = time()
q, history, events = Fit.fit!(p, movie, cache, L, stages; hygiene = Fit.Hygiene(every = 40, densify_threshold = 0.0, max_splats = 16, prune_fraction = 1e-3),
                              rng = rng, callback = (si, it, q, χ) -> (it % 20 == 0 && println("stage $si iteration $it: χ² = $χ, $(size(q, 2)) splats, $(round(time() - t0)) s")))
model = polarized_image(cache, q, 0.0, ν, L)
χ1 = chi2(q, movie, cache, L)
println("end: $(size(q, 2)) splats, χ² = $χ1 (χ²/N = $(χ1 / (4 * length(data)))); hygiene events $events; $(round(time() - t0)) s")
for (k, s) in enumerate(("I", "Q", "U", "V"))
    d = getindex.(data[:, :, 1, 1], k); m = getindex.(model, k)
    println("Stokes $s: relative L2 error $(round(norm(m .- d) / norm(getindex.(data[:, :, 1, 1], 1)); sigdigits = 3)) (of the I norm); data total $(round(sum(d) * Ω_block / Transfer.JY; digits = 4)) Jy, model $(round(sum(m) * Ω_block / Transfer.JY; digits = 4)) Jy")
end
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
writedlm(joinpath(outdir, "fitted_params.csv"), q, ',')
for (k, s) in enumerate(("I", "Q", "U", "V"))
    writedlm(joinpath(outdir, "model_$(s).csv"), getindex.(model, k), ',')
    writedlm(joinpath(outdir, "data_$(s).csv"), getindex.(data[:, :, 1, 1], k), ',')
end
println("wrote ", outdir)
