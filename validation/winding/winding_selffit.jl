# Self-fits of the splat model to movies truncated by their half-orbit content (Johnson et al.
# 2020): the same four-parcel truth rendered with rays truncated after the direct passage
# (n = 0), after the first lensed passage (n ≤ 1) and after the second (n ≤ 2), each fitted by
# the model with the same truncation from a perturbed start, on screens whose resolution grows
# with the order: a uniform 48² grid over 16 M plus, for n ≥ 1, an annulus of fine pixels around
# the screen radii where Krang's n-th emission radius falls in the source region. The Fisher
# information of the spin and inclination at the truth (joint with the free parcel parameters,
# by ForwardDiff duals through the whole pipeline) quantifies what each order adds.
#
#     julia -t 8 --project=../.. winding_selffit.jl [--case 0|1|2|3] [--iterations 300] [--eta 0.005] [--seed 1] [--frames 2] [--backend cpu|cuda] [--subsamples 1] [--fisher fd|duals] [--fisher-only] [--frequencies 230] [--tag name] [--polish 0] [--init fitted.csv]
#
# `--polish N` runs N Levenberg–Marquardt iterations (`Fit.polish!`, CPU Jacobian by duals) from
# the Adam endpoint and reports the polished recovery with its Laplace errors; `--init file`
# starts from a saved parameter matrix (with `--iterations 0` the Adam stage is skipped).
# `--frequencies 86,230,345` fits the movie at several frequencies (GHz) at once, each band with its
# own noise (1%, 0.5%, 0.5%, 0.2% of that band's peak in I, Q, U, V): the multifrequency test of
# whether bands on both sides of the parcels' synchrotron turnover (near 180 GHz for the truth;
# τ_z ≈ 3 at 86 GHz, 0.3 at 230 GHz, 0.12 at 345 GHz) pin down nₑ, Θe and B. `--tag` names the
# output directory `output/case_n_tag`.
# The Fisher audit at the truth takes the parcel columns of the Jacobian by ForwardDiff duals through
# the transfer at fixed geodesics (`Fit.fisher`) and, with `--fisher fd` (the default), the spin and
# inclination columns by central finite differences of the residuals (step 1e-4 in a, 1e-4 rad in θo),
# tile by tile; `--fisher duals` differentiates the whole pipeline, geodesics included, by duals (the
# 2026-09-05 method, whose spin and inclination columns turned out to depend on the screen sampling:
# Krang's closed forms differentiate noisily near polar observers). `--fisher-only` skips the fit.
# `--subsamples K` integrates every pixel over K × K points (the annulus cells over 2K × K in ρ, ψ)
# through `Geodesics.Binning`: with K = 1 the screens are point-sampled at the pixel centres.
# With `--backend cuda` the fits take their χ² and gradient from the truncated dual sweep on the
# GPU (`Fit.chi2_gradient!(...; nmax, slab)` over stored samples, one cache for the whole screen);
# the CPU path tiles the screen and differentiates with Enzyme on the host. The Fisher audit runs on
# the CPU by ForwardDiff duals in both cases.
# Writes output/case_n/ (untracked): data and model images, fitted parameters, a summary.
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Enzyme, Optimisers, Krang, ForwardDiff, Statistics, CUDA, Adapt

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
const CASE = getopt("--case", 0); const ITER = getopt("--iterations", 300); const SEED = getopt("--seed", 1); const NFRAMES = getopt("--frames", 2)
const BACKEND = (i = findfirst(==("--backend"), ARGS); i === nothing ? "cpu" : lowercase(ARGS[i+1]))
BACKEND in ("cpu", "cuda") || error("--backend must be cpu or cuda")
const SUB = getopt("--subsamples", 1)
const FISHER = (i = findfirst(==("--fisher"), ARGS); i === nothing ? "fd" : lowercase(ARGS[i+1]))
FISHER in ("fd", "duals") || error("--fisher must be fd or duals")
const FISHER_ONLY = "--fisher-only" in ARGS
const FDSTEP = 1e-4
const ETA = getopt("--eta", 0.005)                           # Adam step in parameter units: the pattern rates are ~0.05 rad/M
const a = 0.94; const θo = deg2rad(17.0)
const νs = [parse(Float64, v) * 1e9 for v in split((i = findfirst(==("--frequencies"), ARGS); i === nothing ? "230" : ARGS[i+1]), ",")]
const ν = νs[1]                                              # the band of the printed images and fluxes
const TAG = (i = findfirst(==("--tag"), ARGS); i === nothing ? "" : "_" * ARGS[i+1])
const POLISH = getopt("--polish", 0)
const INIT = (i = findfirst(==("--init"), ARGS); i === nothing ? "" : ARGS[i+1])
const M_solar = 6.5e9; const L = gravitational_radius(M_solar)
const SLAB = 0.6                                              # ≥ 4σ_z of the parcels
const met = Krang.Kerr(a)
const times = collect(range(0.0, 25.0 * (NFRAMES - 1), length = NFRAMES))

# ---- truth: four parcels in a thin midplane layer on Keplerian orbits with a rigid pattern rotation
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
free = trues(size(truth)); free[11, :] .= false; free[12, :] .= false
freerows = [i for i in 1:NPOLARIZEDPARAMS if free[i, 1]]

# ---- screens: a uniform grid plus an annulus where the n-th images of the source region fall
function annulus_bounds(n; rsrc = (4.0, 8.0))
    ρs = Float64[]
    for ψ in range(0, 2π, length = 25)[1:end-1], ρ in range(3.0, 9.0, length = 601)
        pix = Krang.SlowLightIntensityPixel(met, ρ * cos(ψ), ρ * sin(ψ), θo)
        for isindir in (false, true)
            rs, _, _, _, ok = Krang.emission_radius(pix, π / 2, isindir, n)
            ok && rsrc[1] <= rs <= rsrc[2] && push!(ρs, ρ)
        end
    end
    return minimum(ρs) - 0.1, maximum(ρs) + 0.1
end
function screen(n)
    fov = 16.0; res = 48
    parts = [binned_grid((-fov / 2, fov / 2), (-fov / 2, fov / 2), res; subsamples = SUB)]
    if n >= 1
        lo, hi = annulus_bounds(n)
        Δρ = n == 1 ? 0.03 : 0.006; nψ = n == 1 ? 96 : 128
        nρ = min(ceil(Int, (hi - lo) / Δρ), 140)
        push!(parts, binned_polar(range(lo, hi, length = nρ + 1), nψ; subsamples = (2SUB, SUB)))
        @info "case n ≤ $n: annulus ρ ∈ [$(round(lo; digits = 2)), $(round(hi; digits = 2))] M, $nρ × $nψ fine pixels (Δρ = $(round((hi - lo) / nρ; digits = 4)) M), $(2SUB) × $SUB points per pixel"
    end
    camera, binning = concatenate(parts...)
    return camera, binning, res * res
end

const TILE = 400                                              # pixels per tile: Enzyme's reverse pass over the CPU kernel needs ~1 GB per 8e4 pixel-samples

"""
Joint Fisher matrix of (spin, inclination, free parcel parameters) at the truth, summed over the
tiles: with `FISHER == "fd"` the spin and inclination columns of each tile's Jacobian are central
finite differences of `spacetime_residuals` (fresh geodesics at a ± h, θo ± h) and the parcel
columns come from `Fit.fisher` (duals through the transfer at fixed geodesics); with
`FISHER == "duals"` the whole Jacobian is ForwardDiff through `spacetime_residuals`.
"""
function fisher_at_truth(tls, movies, N, n, x0)
    tF = time()
    F = zeros(length(x0), length(x0))
    for (t, (cam, b, r)) in enumerate(tls)
        if FISHER == "duals"
            function residuals(x)
                params = similar(x, size(truth)); params .= truth
                params[free] .= x[3:end]
                return spacetime_residuals(x[1:2], params, movies[t], cam, L; N, nmax = n, slab = SLAB, binning = b)
            end
            J = ForwardDiff.jacobian(residuals, x0, ForwardDiff.JacobianConfig(residuals, x0, ForwardDiff.Chunk{12}()))
        else
            res(y) = spacetime_residuals(y, truth, movies[t], cam, L; N, nmax = n, slab = SLAB, binning = b)
            Ja = (res([a + FDSTEP, θo]) .- res([a - FDSTEP, θo])) ./ (2FDSTEP)
            Jθ = (res([a, θo + FDSTEP]) .- res([a, θo - FDSTEP])) ./ (2FDSTEP)
            c = GeodesicCache(CPU(), cam, Val(N); store_samples = false); regenerate!(c, a, θo; marcher = Fused(64))
            _, Jp, _ = fisher(truth, movies[t], c, L; free, chunk = 12, nmax = n, slab = SLAB, binning = b)
            J = hcat(Ja, Jθ, Jp)
        end
        F .+= J' * J
    end
    @info "Fisher Jacobians (n ≤ $n, $FISHER)" minutes = (time() - tF) / 60
    return F
end

"Marginal Fisher errors (joint with everything else) of the rows `rows` of every parcel, from the inverse Fisher matrix."
function marginal_errors(Finv, rows)
    nper = length(freerows)
    return [sqrt(Finv[2 + (i - 1) * nper + findfirst(==(r), freerows), 2 + (i - 1) * nper + findfirst(==(r), freerows)]) for r in rows, i in 1:size(truth, 2)]
end

function report_fisher(F, n, npix, nuni)
    Finv = inv(F + 1e-12 * I)
    σa_joint = sqrt(Finv[1, 1]); σθ_joint = sqrt(Finv[2, 2])
    σa_alone = 1 / sqrt(F[1, 1]); σθ_alone = 1 / sqrt(F[2, 2])
    @info "Fisher at the truth (n ≤ $n, $FISHER, $SUB × $SUB sub-samples, $(length(νs)) bands)" σ_spin_joint = σa_joint σ_inclination_deg_joint = rad2deg(σθ_joint) σ_spin_alone = σa_alone σ_inclination_deg_alone = rad2deg(σθ_alone)
    me = marginal_errors(Finv, (13, 14, 15))
    @info "marginal Fisher errors of the emission rows per parcel (joint)" ln_ne = round.(me[1, :]; sigdigits = 3) ln_Te = round.(me[2, :]; sigdigits = 3) ln_B = round.(me[3, :]; sigdigits = 3)
    return σa_joint, σθ_joint, σa_alone, σθ_alone, me
end

"Slice of a movie cube for the pixels `rng` of a screen stored as (npix, 1, nt, nν)."
tile_movie(movie, rng) = StokesMovie(movie.data[rng, :, :, :], movie.times, movie.νs, movie.σ)

function run_case(n)
    N = (80, 160, 240)[n + 1]
    camera, binning, nuni = screen(n)
    npix = Geodesics.npixels(binning)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    @info "case n ≤ $n" pixels = npix points = npixels(camera) samples = N
    clean = bin(binning, polarized_cube(cache, truth, times, νs, L; nmax = n, slab = SLAB))
    peaks = [maximum(norm.(clean[:, :, :, l])) for l in eachindex(νs)]              # each band's own noise level
    σ = [SVector(0.01, 0.005, 0.005, 0.002) * peaks[idx[4]] for idx in CartesianIndices(clean)]
    rng = MersenneTwister(SEED)
    data = [clean[idx] + σ[idx] .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
    movie = StokesMovie(data, times, νs, σ)
    @info "bands" GHz = νs ./ 1e9 peak_I = round.(peaks; sigdigits = 3)
    fluxes = [sum(getindex.(bin(binning, polarized_cube(cache, truth, times[1:1], [ν], L; nmax = m, slab = SLAB))[1:nuni, 1, 1, 1], 1)) for m in 0:n]
    @info "sub-image fluxes on the uniform grid (arbitrary units)" cumulative = fluxes
    # tiles: a cache and a movie slice per tile, so that the CPU reverse pass runs tile by tile (the Fisher audit uses
    # them too); a tile takes whole pixels, with its own binning over its points
    ntiles = cld(npix, TILE)
    pixbounds = round.(Int, range(0, npix, length = ntiles + 1))
    tls = Tuple{Geodesics.Camera,Binning,UnitRange{Int}}[]
    for t in 1:ntiles
        r = pixbounds[t]+1:pixbounds[t+1]
        isempty(r) && continue
        pts = findall(m -> binning.pixel[m] in r, eachindex(binning.pixel))
        push!(tls, (Geodesics.Camera(camera.αs[pts], camera.βs[pts]), Binning(binning.pixel[pts] .- (first(r) - 1), (length(r), 1)), r))
    end
    movies = [tile_movie(movie, r) for (_, _, r) in tls]
    value_and_gradient = if BACKEND == "cuda"
        gcache = GeodesicCache(CUDABackend(), camera, Val(N); store_samples = true)
        regenerate!(gcache, a, θo; marcher = Recurrence(64))
        @info "GPU gradient: the truncated dual sweep over stored samples" pixels = npixels(camera) samples = N
        q -> begin
            dp = CUDA.zeros(Float64, size(q))
            χ = chi2_gradient!(dp, CuArray(q), movie, gcache, L; nmax = n, slab = SLAB, binning)
            (χ, Array(dp))
        end
    else
        caches = [(c = GeodesicCache(CPU(), cam, Val(N); store_samples = false); regenerate!(c, a, θo; marcher = Fused(64)); c) for (cam, _, _) in tls]
        @info "tiles" count = length(tls) pixels_per_tile = TILE
        q -> begin
            χ = 0.0; g = zero(q)
            for t in eachindex(tls)
                b = tls[t][2]
                χ += chi2(q, movies[t], caches[t], L; nmax = n, slab = SLAB, binning = b)
                g .+= Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(x -> chi2(x, movies[t], caches[t], L; nmax = n, slab = SLAB, binning = b)), q)[1]
            end
            (χ, g)
        end
    end
    loss(q) = value_and_gradient(q)[1]
    nfree = count(free)
    x0 = vcat([a, θo], truth[free])
    if FISHER_ONLY
        F = fisher_at_truth(tls, movies, N, n, x0)
        report_fisher(F, n, npix, nuni)
        return nothing
    end
    # perturbed start, from its own stream so that every band set and screen starts from the same point
    rng0 = MersenneTwister(SEED + 1000)
    p0 = copy(truth)
    for i in 1:4
        p0[1, i] += 0.3 * randn(rng0); p0[2, i] += 0.3 * randn(rng0); p0[3, i] += 0.05 * randn(rng0)
        p0[4:6, i] .+= 0.15 .* randn(rng0, 3)
        p0[13, i] += 0.2 * randn(rng0); p0[14, i] += 0.1 * randn(rng0); p0[15, i] += 0.15 * randn(rng0)
        p0[19, i] += 0.05 * randn(rng0); p0[21, i] *= 1 + 0.02 * randn(rng0)
    end
    if !isempty(INIT)
        p0 = Matrix{Float64}(readdlm(INIT, ','))
        @info "starting from $INIT"
    end
    χ0 = loss(p0)
    ndata = 4 * length(data)
    @info "multifrequency" bands = length(νs) data_values = ndata
    @info "start" chi2 = χ0 reduced = χ0 / ndata
    t0 = time()
    q = copy(p0); mask = Float64.(free)
    opt = Optimisers.setup(Optimisers.Adam(ETA), q)
    best = (χ0, copy(q))
    for it in 1:ITER
        η = ETA / 10 + (ETA - ETA / 10) * (1 + cos(π * (it - 1) / max(ITER - 1, 1))) / 2
        Optimisers.adjust!(opt, η)
        χ, g = value_and_gradient(q)                       # χ² at the current point, before the update
        χ < best[1] && (best = (χ, copy(q)))
        opt, q = Optimisers.update!(opt, q, g .* mask)
        if it % 20 == 0 || it == ITER
            @info "iteration $it" chi2 = χ reduced = χ / ndata minutes = (time() - t0) / 60
        end
    end
    χend = loss(q); χend < best[1] && (best = (χend, copy(q)))
    χ1, q = best
    laplace = nothing
    if POLISH > 0
        tP = time()
        q, hist, cov, idx = polish!(copy(q), movie, cache, L; free, iterations = POLISH, chunk = 12, nmax = n, slab = SLAB, binning)
        χ1 = hist[end]
        σp = sqrt.(diag(cov))
        nper = length(freerows)
        laplace = [σp[(i - 1) * nper + findfirst(==(r), freerows)] for r in (13, 14, 15), i in 1:size(truth, 2)]
        @info "Levenberg–Marquardt polish ($POLISH iterations)" chi2 = hist[1] => hist[end] reduced = χ1 / ndata history = round.(hist ./ ndata; digits = 4) minutes = (time() - tP) / 60
        @info "Laplace errors of the emission rows per parcel" ln_ne = round.(laplace[1, :]; sigdigits = 3) ln_Te = round.(laplace[2, :]; sigdigits = 3) ln_B = round.(laplace[3, :]; sigdigits = 3)
    end
    # recovery metrics
    pos_err = [hypot(q[1, i] - truth[1, i], q[2, i] - truth[2, i], q[3, i] - truth[3, i]) for i in 1:4]
    pos0 = [hypot(p0[1, i] - truth[1, i], p0[2, i] - truth[2, i], p0[3, i] - truth[3, i]) for i in 1:4]
    rel(row) = [abs(exp(q[row, i] - truth[row, i]) - 1) for i in 1:4]
    ωerr = [abs(q[21, i] / truth[21, i] - 1) for i in 1:4]
    uerr = [abs(q[19, i] - truth[19, i]) for i in 1:4]
    @info "recovery (n ≤ $n)" chi2 = χ1 reduced = χ1 / ndata position_M = round.(pos_err; sigdigits = 2) position_start_M = round.(pos0; sigdigits = 2) ne = round.(rel(13); sigdigits = 2) Te = round.(rel(14); sigdigits = 2) B = round.(rel(15); sigdigits = 2) pattern_rate = round.(ωerr; sigdigits = 2) u_phi = round.(uerr; sigdigits = 2) minutes = (time() - t0) / 60
    # joint Fisher information at the truth (spin, inclination, free parcel parameters), tile by tile
    F = fisher_at_truth(tls, movies, N, n, x0)
    σa_joint, σθ_joint, σa_alone, σθ_alone, me = report_fisher(F, n, npix, nuni)
    outdir = joinpath(@__DIR__, "output", "case_$n$TAG"); mkpath(outdir)
    writedlm(joinpath(outdir, "truth.csv"), truth, ','); writedlm(joinpath(outdir, "start.csv"), p0, ','); writedlm(joinpath(outdir, "fitted.csv"), q, ',')
    uni = reshape(clean[1:nuni, 1, 1, 1], 48, 48)
    for (k, s) in enumerate(("I", "Q", "U", "V"))
        writedlm(joinpath(outdir, "data_$(s).csv"), getindex.(uni, k), ',')
    end
    for (l, νl) in enumerate(νs)
        writedlm(joinpath(outdir, "data_I_$(round(Int, νl / 1e9))GHz.csv"), getindex.(reshape(clean[1:nuni, 1, 1, l], 48, 48), 1), ',')
    end
    model = bin(binning, polarized_cube(cache, q, times[1:1], [ν], L; nmax = n, slab = SLAB))
    writedlm(joinpath(outdir, "model_I.csv"), getindex.(reshape(model[1:nuni, 1, 1, 1], 48, 48), 1), ',')
    writedlm(joinpath(outdir, "camera.csv"), hcat(camera.αs, camera.βs, binning.pixel), ',')
    open(joinpath(outdir, "summary.txt"), "w") do io
        println(io, "case n ≤ $n: $npix pixels ($nuni uniform) from $(npixels(camera)) points ($SUB × $SUB per uniform pixel), $N samples, $NFRAMES frames, bands $(νs ./ 1e9) GHz, $ITER iterations, eta $ETA")
        println(io, "chi2 start $χ0 end $χ1 (reduced $(χ1 / ndata)) over $ndata data values")
        println(io, "position errors (M): start $pos0 end $pos_err")
        println(io, "relative errors: ne $(rel(13)) Te $(rel(14)) B $(rel(15)) pattern rate $ωerr; u_phi absolute $uerr")
        println(io, "Fisher at the truth ($FISHER): σ(a) joint $σa_joint alone $σa_alone; σ(θo) joint $(rad2deg(σθ_joint))° alone $(rad2deg(σθ_alone))°")
        println(io, "marginal Fisher errors per parcel: ln ne $(me[1, :]); ln Te $(me[2, :]); ln B $(me[3, :])")
        laplace === nothing || println(io, "after $POLISH LM iterations, Laplace errors per parcel: ln ne $(laplace[1, :]); ln Te $(laplace[2, :]); ln B $(laplace[3, :])")
        println(io, "cumulative sub-image fluxes on the uniform grid: $fluxes")
    end
    return nothing
end

for n in (CASE == 0 ? (0, 1, 2) : (CASE - 1,))
    run_case(n)
end
