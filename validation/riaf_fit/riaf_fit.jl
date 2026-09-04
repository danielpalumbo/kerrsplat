# Cross-model recovery (plan §7.5 item 7 ii, with ipole's RIAF standing in for a GRMHD movie):
# fit polarized splats to ipole's full-Stokes RIAF image at 230 GHz (validation/ipole_riaf,
# block-averaged to a coarse grid) starting from a ring of splats, with densification, and compare
# the recovered fields with the RIAF model's analytic density, temperature and field strength.
#
#     julia -t 4 --project=. validation/riaf_fit/riaf_fit.jl [--res 16] [--samples 200] [--iterations 300] [--freqs 1|2]
#
# Writes validation/riaf_fit/output/ (untracked): the fitted parameters (CSV), the model image (CSV)
# and a summary; the numbers go into docs/notes.
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Enzyme, Optimisers, Krang, Statistics
include(joinpath(@__DIR__, "..", "..", "test", "riaf_model.jl"))

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
const RES = getopt("--res", 16)
const N = getopt("--samples", 200)
const ITER = getopt("--iterations", 300)
const NFREQ = getopt("--freqs", 1)                          # 1: 230 GHz; 2: 230 and 345 GHz
const a = 0.9375; const θo = deg2rad(85.0); const ν = 230e9
const νs = NFREQ == 1 ? [230e9] : [230e9, 345e9]
const tags = NFREQ == 1 ? ["riaf_fine"] : ["riaf_fine", "riaf_345"]
const M_solar = 4.3e6; const D = 8.3e3 * Transfer.PC; const L = gravitational_radius(M_solar)
const fov = 200e-6 / 206264.806247 * D / L                     # 39.10 M, ipole's field of view

# ---- data: ipole's 100² image block-averaged to RES² (RES must divide 100) -----------------------
dir = joinpath(@__DIR__, "..", "ipole_riaf")
b = 100 ÷ RES
data = Array{SVector{4,Float64}}(undef, RES, RES, 1, length(νs))
for (l, tag) in enumerate(tags)
    ref = [readdlm(joinpath(dir, "$(tag)_$(s).csv"), ',') for s in ("I", "Q", "U", "V")]
    data[:, :, 1, l] = [SVector{4}(mean(ref[k][(i-1)*b+1:i*b, (j-1)*b+1:j*b]) for k in 1:4) for i in 1:RES, j in 1:RES]
end
peak = maximum(norm.(data[:, :, 1, 1]))
σ = SVector(0.02, 0.01, 0.01, 0.005) * peak
movie = StokesMovie(data, [0.0], νs, σ)
# camera: the block centres of ipole's pixel grid (its x offset of 0.49 pixel included)
xs = [(((i - 1) * b + (b - 1) / 2 + 0.49) / 100 - 0.5) * fov for i in 1:RES]
ys = [(((j - 1) * b + (b - 1) / 2 + 0.5) / 100 - 0.5) * fov for j in 1:RES]
camera = Geodesics.Camera(vec([xs[i] for i in 1:RES, j in 1:RES]), vec([ys[j] for i in 1:RES, j in 1:RES]), (RES, RES))
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
regenerate!(cache, a, θo; marcher = Fused(64))
met = Krang.Kerr(a)

# ---- initial splats: a ring at r = 4 M with Keplerian velocity and toroidal field ---------------
nring = 6
p = zeros(NPOLARIZEDPARAMS, nring)
for k in 1:nring
    ϕ = 2π * (k - 0.5) / nring; r = 4.0
    x, y, z = quasi_cartesian_kerr_schild(met, r, π / 2, ϕ)
    Ω = 1 / (r^1.5 + a); gdd = Krang.metric_dd(met, r, π / 2)
    ut = 1 / sqrt(-(gdd[1, 1] + 2Ω * gdd[1, 4] + Ω^2 * gdd[4, 4]))
    uz = Krang.jac_zamo_u_bl_d(met, r, π / 2) * SVector(ut, 0.0, 0.0, Ω * ut)
    p[:, k] = [x, y, z, log(1.5), log(1.5), log(0.6), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(3e6), log(30.0), log(30.0), π / 2, π / 2, uz[2], uz[3], uz[4], 0.0]
end
χ0 = chi2(p, movie, cache, L)
println("start: $nring splats at $(length(νs)) frequenc$(length(νs) == 1 ? "y" : "ies"), χ² = $χ0 for $(4 * length(data)) data points")
rng = MersenneTwister(1)
stages = [Fit.Stage(free = (:logne, :logTe, :logB, :thB, :phB), iterations = ITER ÷ 5, η = 0.05, η_end = 0.02, label = "plasma"),
          Fit.Stage(iterations = ITER - ITER ÷ 5, η = 0.03, η_end = 0.003, label = "everything")]
t0 = time()
q, history, events = Fit.fit!(p, movie, cache, L, stages; hygiene = Fit.Hygiene(every = 40, densify_threshold = 0.0, max_splats = 12, prune_fraction = 1e-3),
                              rng = rng, callback = (si, it, q, χ) -> (it % 20 == 0 && println("stage $si iteration $it: χ² = $χ, $(size(q, 2)) splats, $(round(time() - t0)) s")))
χ1 = chi2(q, movie, cache, L)
println("end: $(size(q, 2)) splats, χ² = $χ1 (χ²/N = $(χ1 / (4 * length(data)))); hygiene events $events; $(round(time() - t0)) s")

# ---- field comparison with the analytic RIAF on a voxel grid ------------------------------------
m = riaf_example(a)
function riaf_fields(x, y, z)
    r, θ, ϕ = boyer_lindquist(met, x, y, z)
    (r > Krang.horizon(met) + 0.1 && r < m.rout) || return (0.0, 0.0, 0.0)
    zc = r * cos(θ); rc = r * sin(θ)
    n = m.nth0 * exp(-zc * zc / 2 / rc / rc / m.disk_h / m.disk_h) * r^m.pow_nth * m.Ne_unit
    Θe = m.Te0 * r^m.pow_T * m.Te_unit * Transfer.KBOL / (Transfer.ME * Transfer.CL^2)
    return n, Θe, sqrt(8π * m.ε * n * Transfer.MP * Transfer.CL^2 / 6 / r)
end
grid = range(-10.0, 10.0, length = 41); zgrid = range(-2.0, 2.0, length = 9)
nf, Θf, Bf = field_grid(q, 0.0, grid, grid, zgrid)
nt = [riaf_fields(x, y, z)[1] for x in grid, y in grid, z in zgrid]
Θt = [riaf_fields(x, y, z)[2] for x in grid, y in grid, z in zgrid]
Bt = [riaf_fields(x, y, z)[3] for x in grid, y in grid, z in zgrid]
# compare where the RIAF has density and the fit put some (the emitting region), weighting by the RIAF density
mask = (nt .> 0.01 * maximum(nt)) .& (nf .> 0.01 * maximum(nf))
w = nt .* mask ./ sum(nt .* mask)
println("fields vs the analytic RIAF, RIAF-density-weighted over the voxels where both have density (", count(mask), " of ", length(mask), "): density ratio fitted/true $(sum((w .* nf ./ nt)[mask])), Θe relative error $(sum((w .* abs.(Θf .- Θt) ./ Θt)[mask])), B relative error $(sum((w .* abs.(Bf .- Bt) ./ Bt)[mask]))")
println("fraction of the RIAF's density (r < 10 M, |z| < 2 M) covered by the splats' 1% contours: ", sum(nt .* (nf .> 0.01 * maximum(nf))) / sum(nt))
outdir = joinpath(@__DIR__, "output", NFREQ == 1 ? "one_frequency" : "two_frequencies")
mkpath(outdir)
writedlm(joinpath(outdir, "fitted_params.csv"), q, ',')
for (l, νl) in enumerate(νs)
    img = polarized_image(cache, q, 0.0, νl, L)
    for (k, s) in enumerate(("I", "Q", "U", "V"))
        writedlm(joinpath(outdir, "model_$(s)_$(round(Int, νl / 1e9)).csv"), getindex.(img, k), ',')
        writedlm(joinpath(outdir, "data_$(s)_$(round(Int, νl / 1e9)).csv"), getindex.(data[:, :, 1, l], k), ',')
    end
end
println("wrote ", outdir)
