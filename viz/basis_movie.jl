# The model basis as a movie: the same slow-light rendering of one parcel, of the six-parcel truth of the large-N
# self-fits, and of the three hundred small parcels of the over-complete shell start, at 86, 230 and 345 GHz side by
# side, with polarization ticks in eht-imaging's style (a tick per cell of a coarse grid where Stokes I exceeds
# `pcut` of the peak, its length proportional to the polarized intensity √(Q² + U²), its colour the fractional
# polarization, the EVPA measured from north toward east). Runs on the CPU so that it can share the machine with
# a fit on the card.
#
#     nice -n 10 julia -t 6 --project=viz viz/basis_movie.jl [--res 96] [--samples 160] [--frames 60] [--span 60] [--nvec 20] [--pcut 0.05] [--fps 12]
#
# Output (not tracked): viz/output/basis_movie.mp4 and viz/output/basis_frame_1..6.png.
using CairoMakie
using CUDA
using KernelAbstractions
using StaticArrays
using LinearAlgebra
using Random
using Statistics: quantile
using Printf
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
res = getopt("--res", 96); N = getopt("--samples", 160); nframes = getopt("--frames", 60); span = getopt("--span", 60.0)
nvec = getopt("--nvec", 24); pcut = getopt("--pcut", 0.03); fps = getopt("--fps", 12)
outdir = joinpath(@__DIR__, "output"); mkpath(outdir)

# ---- the spacetime, the screen and the three sets of parcels (the large-N self-fits' setup)
a = 0.9; θo = deg2rad(60.0); fov = 16.0; L = gravitational_radius(4e6); nmax = 2; slab = 0.5
freqs = [86e9, 230e9, 345e9]
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cache, a, θo; marcher = Fused(64))
rng = MersenneTwister(1)
kepler(r) = 1 / (r^1.5 + a)
truth = zeros(NPOLARIZEDPARAMS, 6)
for i in 1:6
    φ = 2π * (i - 1) / 6 + 0.3 * randn(rng); r0 = 2.5 + 2.5 * (i - 1) / 5
    truth[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.3 * randn(rng), log(0.7), log(0.7), log(0.5), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
                   0.0, log(1e9), log(3e5) + 0.3 * randn(rng), log(30.0) + 0.2 * randn(rng), log(20.0) + 0.2 * randn(rng), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.0, 0.3, 0.05 * randn(rng), kepler(r0)]
end
one = truth[:, 2:2]
shell = shell_parcels(300; rin = 2.2, rout = 5.5, height = 0.6, scale = 0.25, spin = a, rng)
flux(p) = sum(x -> x[1], polarized_cube(cache, p, [0.0], [230e9], L; nmax, slab))
shell[13, :] .+= log(flux(truth) / flux(shell))
rows = [("one parcel", one), ("six parcels (the truth of the self-fits)", truth), ("three hundred parcels (the over-complete basis)", shell)]
times = collect(range(0.0, span; length = nframes))
@info "rendering" rows = length(rows) bands = length(freqs) frames = nframes res N threads = Threads.nthreads()
t0 = time()
cubes = [polarized_cube(cache, p, times, freqs, L; nmax, slab) for (_, p) in rows]      # res × res × frames × bands
@info "rendered" minutes = round((time() - t0) / 60, digits = 1)

# ---- eht-imaging's polarization ticks: cells of a coarse grid, length ∝ √(Q² + U²), colour m = P/I, EVPA from north toward east
function ticks(img; nvec, pcut, maxlen = 1.6)                 # the lengths normalized to the frame's longest tick, as eht-imaging does
    n = size(img, 1); cell = max(n ÷ nvec, 1)
    imax = maximum(x -> x[1], img)
    cells = Tuple{Float64,Float64,Float64,Float64,Float64}[]     # (ci, cj, P, m, χ) of the cells above the cut
    for j0 in 1:cell:n-cell+1, i0 in 1:cell:n-cell+1
        I = 0.0; Q = 0.0; U = 0.0
        for j in j0:j0+cell-1, i in i0:i0+cell-1
            I += img[i, j][1]; Q += img[i, j][2]; U += img[i, j][3]
        end
        I /= cell^2; Q /= cell^2; U /= cell^2
        I > pcut * imax || continue
        P = hypot(Q, U)
        push!(cells, (i0 + (cell - 1) / 2, j0 + (cell - 1) / 2, P, P / I, atan(U, Q) / 2))
    end
    pmax = max(maximum(c -> c[3], cells; init = 0.0), eps())
    segs = Point2f[]; cols = Float32[]
    for (ci, cj, P, m, χ) in cells
        len = maxlen * cell * P / pmax
        di = -sin(χ) * len / 2; dj = cos(χ) * len / 2               # x runs to the west, so east is −x; north is +y
        push!(segs, Point2f(ci - di, cj - dj)); push!(segs, Point2f(ci + di, cj + dj)); push!(cols, m); push!(cols, m)
    end
    return segs, cols
end
stokesI(img) = [x[1] for x in img]
# the brightness scale of each panel: the 99.8th percentile of its pixels over the frames (a few lensed pixels are far brighter)
imax = [quantile(vec([x[1] for x in cubes[r][:, :, :, c]]), 0.998) for r in eachindex(rows), c in eachindex(freqs)]

# ---- the figure
fig = Figure(size = (1500, 1500), fontsize = 18)
Label(fig[0, 1:3], "the same slow-light rendering of one, six and three hundred parcels at three bands", fontsize = 22, tellwidth = false)
img_obs = Matrix{Observable}(undef, length(rows), length(freqs)); seg_obs = similar(img_obs); col_obs = similar(img_obs)
for (r, (label, _)) in enumerate(rows), (c, ν) in enumerate(freqs)
    ax = Axis(fig[r, c], title = r == 1 ? "$(round(Int, ν / 1e9)) GHz" : "", aspect = DataAspect(), xticksvisible = false, yticksvisible = false,
              xticklabelsvisible = false, yticklabelsvisible = false, ylabel = c == 1 ? label : "", ylabelsize = 16)
    img_obs[r, c] = Observable(stokesI(cubes[r][:, :, 1, c]))
    heatmap!(ax, img_obs[r, c], colormap = :afmhot, colorrange = (0, imax[r, c]))
    s, m = ticks(cubes[r][:, :, 1, c]; nvec, pcut)
    seg_obs[r, c] = Observable(s); col_obs[r, c] = Observable(m)
    linesegments!(ax, seg_obs[r, c], color = col_obs[r, c], colormap = :rainbow, colorrange = (0, 0.7), linewidth = 3)
end
Colorbar(fig[1:3, 4], colormap = :rainbow, limits = (0, 0.7), label = "fractional polarization of the tick", width = 14)
clock = Observable(@sprintf("t = %.1f M", times[1]))
Label(fig[4, 1:3], clock, fontsize = 20, tellwidth = false)
function set_frame!(k)
    for r in eachindex(rows), c in eachindex(freqs)
        img = cubes[r][:, :, k, c]
        img_obs[r, c][] = stokesI(img)
        s, m = ticks(img; nvec, pcut)
        seg_obs[r, c][] = s; col_obs[r, c][] = m
    end
    clock[] = @sprintf("t = %.1f M   (M = 4×10⁶ M☉, a = 0.9, inclination 60°, 0.25 M pixels, n ≤ 2)", times[k])
end
moviepath = joinpath(outdir, "basis_movie.mp4")
CairoMakie.record(fig, moviepath, 1:nframes; framerate = fps) do k
    set_frame!(k)
end
for (n, k) in enumerate(round.(Int, range(1, nframes; length = 6)))
    set_frame!(k); save(joinpath(outdir, "basis_frame_$(n).png"), fig)
end
@info "written" moviepath frames = nframes minutes = round((time() - t0) / 60, digits = 1)
