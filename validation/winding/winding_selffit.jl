# Self-fits of the splat model to movies truncated by their half-orbit content (Johnson et al.
# 2020): the same four-parcel truth rendered with rays truncated after the direct passage
# (n = 0), after the first lensed passage (n ≤ 1) and after the second (n ≤ 2), each fitted by
# the model with the same truncation from a perturbed start, on screens whose resolution grows
# with the order: a uniform 48² grid over 16 M plus, for n ≥ 1, an annulus of fine pixels around
# the screen radii where Krang's n-th emission radius falls in the source region. The Fisher
# information of the spin and inclination at the truth (joint with the free parcel parameters,
# by ForwardDiff duals through the whole pipeline) quantifies what each order adds.
#
#     julia -t 8 --project=../.. winding_selffit.jl [--case 0|1|2|3] [--iterations 120] [--seed 1] [--frames 2]
#
# Writes output/case_n/ (untracked): data and model images, fitted parameters, a summary.
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Transfer, KerrSplat.Splats, KerrSplat.Fit
using KernelAbstractions, StaticArrays, LinearAlgebra, Random, DelimitedFiles, Enzyme, Optimisers, Krang, ForwardDiff, Statistics

getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
const CASE = getopt("--case", 0); const ITER = getopt("--iterations", 120); const SEED = getopt("--seed", 1); const NFRAMES = getopt("--frames", 2)
const a = 0.94; const θo = deg2rad(17.0); const ν = 230e9
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
    fov = 16.0; res = 48; Δ = fov / res
    αs = [(i - (res + 1) / 2) * Δ for i in 1:res, j in 1:res]; βs = [(j - (res + 1) / 2) * Δ for i in 1:res, j in 1:res]
    α = vec(αs); β = vec(βs)
    if n >= 1
        lo, hi = annulus_bounds(n)
        Δρ = n == 1 ? 0.03 : 0.006; nψ = n == 1 ? 96 : 128
        nρ = min(ceil(Int, (hi - lo) / Δρ), 140)
        for ρ in range(lo, hi, length = nρ), ψ in range(0, 2π, length = nψ + 1)[1:end-1]
            push!(α, ρ * cos(ψ)); push!(β, ρ * sin(ψ))
        end
        @info "case n ≤ $n: annulus ρ ∈ [$(round(lo; digits = 2)), $(round(hi; digits = 2))] M, $nρ × $nψ fine pixels (Δρ = $(round((hi - lo) / (nρ - 1); digits = 4)) M)"
    end
    return Geodesics.Camera(α, β), res * res
end

function run_case(n)
    N = (80, 160, 240)[n + 1]
    camera, nuni = screen(n)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    @info "case n ≤ $n" pixels = npixels(camera) samples = N
    clean = polarized_cube(cache, truth, times, [ν], L; nmax = n, slab = SLAB)
    peak = maximum(norm.(clean))
    σ = SVector(0.01, 0.005, 0.005, 0.002) * peak
    rng = MersenneTwister(SEED)
    data = [clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)]
    movie = StokesMovie(data, times, [ν], σ)
    # the sub-image content of the data (fluxes of the orders present, on the uniform part of the screen)
    fluxes = [sum(getindex.(polarized_cube(cache, truth, times[1:1], [ν], L; nmax = m, slab = SLAB)[1:nuni, 1, 1, 1], 1)) for m in 0:n]
    @info "sub-image fluxes on the uniform grid (arbitrary units)" cumulative = fluxes
    # perturbed start
    p0 = copy(truth)
    for i in 1:4
        p0[1, i] += 0.3 * randn(rng); p0[2, i] += 0.3 * randn(rng); p0[3, i] += 0.05 * randn(rng)
        p0[4:6, i] .+= 0.15 .* randn(rng, 3)
        p0[13, i] += 0.2 * randn(rng); p0[14, i] += 0.1 * randn(rng); p0[15, i] += 0.15 * randn(rng)
        p0[19, i] += 0.05 * randn(rng); p0[21, i] *= 1 + 0.1 * randn(rng)
    end
    χ0 = chi2(p0, movie, cache, L; nmax = n, slab = SLAB)
    ndata = 4 * length(data)
    @info "start" chi2 = χ0 reduced = χ0 / ndata
    t0 = time()
    stages = [Fit.Stage(free = Tuple(POLARIZED_SPLAT_PARAMS[freerows]), iterations = ITER, η = 0.03, η_end = 0.003)]
    q, history, _ = Fit.fit!(copy(p0), movie, cache, L, stages; hygiene = Fit.Hygiene(every = 0), nmax = n, slab = SLAB,
                             callback = (si, it, q, χ) -> (it % 20 == 0 && @info "iteration $it" chi2 = χ reduced = χ / ndata minutes = (time() - t0) / 60))
    χ1 = chi2(q, movie, cache, L; nmax = n, slab = SLAB)
    # recovery metrics
    pos_err = [hypot(q[1, i] - truth[1, i], q[2, i] - truth[2, i], q[3, i] - truth[3, i]) for i in 1:4]
    pos0 = [hypot(p0[1, i] - truth[1, i], p0[2, i] - truth[2, i], p0[3, i] - truth[3, i]) for i in 1:4]
    rel(row) = [abs(exp(q[row, i] - truth[row, i]) - 1) for i in 1:4]
    ωerr = [abs(q[21, i] / truth[21, i] - 1) for i in 1:4]
    uerr = [abs(q[19, i] - truth[19, i]) for i in 1:4]
    @info "recovery (n ≤ $n)" chi2 = χ1 reduced = χ1 / ndata position_M = round.(pos_err; sigdigits = 2) position_start_M = round.(pos0; sigdigits = 2) ne = round.(rel(13); sigdigits = 2) Te = round.(rel(14); sigdigits = 2) B = round.(rel(15); sigdigits = 2) pattern_rate = round.(ωerr; sigdigits = 2) u_phi = round.(uerr; sigdigits = 2) minutes = (time() - t0) / 60
    # joint Fisher information at the truth: spin, inclination and the free parcel parameters
    nfree = count(free)
    function residuals(x)
        params = similar(x, size(truth)); params .= truth
        params[free] .= x[3:end]
        return spacetime_residuals(x[1:2], params, movie, camera, L; N, nmax = n, slab = SLAB)
    end
    x0 = vcat([a, θo], truth[free])
    tF = time()
    J = ForwardDiff.jacobian(residuals, x0, ForwardDiff.JacobianConfig(residuals, x0, ForwardDiff.Chunk{12}()))
    F = J' * J
    Finv = inv(F + 1e-12 * I)
    σa_joint = sqrt(Finv[1, 1]); σθ_joint = sqrt(Finv[2, 2])
    σa_alone = 1 / sqrt(F[1, 1]); σθ_alone = 1 / sqrt(F[2, 2])
    @info "Fisher at the truth (n ≤ $n)" σ_spin_joint = σa_joint σ_inclination_deg_joint = rad2deg(σθ_joint) σ_spin_alone = σa_alone σ_inclination_deg_alone = rad2deg(σθ_alone) minutes = (time() - tF) / 60
    outdir = joinpath(@__DIR__, "output", "case_$n"); mkpath(outdir)
    writedlm(joinpath(outdir, "truth.csv"), truth, ','); writedlm(joinpath(outdir, "start.csv"), p0, ','); writedlm(joinpath(outdir, "fitted.csv"), q, ',')
    uni = reshape(clean[1:nuni, 1, 1, 1], 48, 48)
    for (k, s) in enumerate(("I", "Q", "U", "V"))
        writedlm(joinpath(outdir, "data_$(s).csv"), getindex.(uni, k), ',')
    end
    model = polarized_cube(cache, q, times[1:1], [ν], L; nmax = n, slab = SLAB)
    writedlm(joinpath(outdir, "model_I.csv"), getindex.(reshape(model[1:nuni, 1, 1, 1], 48, 48), 1), ',')
    writedlm(joinpath(outdir, "camera.csv"), hcat(camera.αs, camera.βs), ',')
    open(joinpath(outdir, "summary.txt"), "w") do io
        println(io, "case n ≤ $n: $(npixels(camera)) pixels ($nuni uniform), $N samples, $NFRAMES frames, $ITER iterations")
        println(io, "chi2 start $χ0 end $χ1 (reduced $(χ1 / ndata)) over $ndata data values")
        println(io, "position errors (M): start $pos0 end $pos_err")
        println(io, "relative errors: ne $(rel(13)) Te $(rel(14)) B $(rel(15)) pattern rate $ωerr; u_phi absolute $uerr")
        println(io, "Fisher at the truth: σ(a) joint $σa_joint alone $σa_alone; σ(θo) joint $(rad2deg(σθ_joint))° alone $(rad2deg(σθ_alone))°")
        println(io, "cumulative sub-image fluxes on the uniform grid: $fluxes")
    end
    return nothing
end

for n in (CASE == 0 ? (0, 1, 2) : (CASE - 1,))
    run_case(n)
end
