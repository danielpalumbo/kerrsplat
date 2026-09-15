# The triband ngEHT self-fit as a movie: for every frame of the campaign the truth and the fitted model at the three
# bands (Stokes I with the polarization ticks), the campaign's (u, v) coverage filling in scan by scan, and the midplane
# density and temperature of the truth and the fit; the χ² per band against the iteration as a still.
#
#     julia -t 8 --project=viz viz/triband_movie.jl [--tag triband] [--days 5] [--start 2026-04-01] [--bands 86,230,345]
#         [--res 96] [--fov 16] [--samples 160] [--nmax 2] [--slab 0.5] [--frame-hours 4] [--cpu]
#
# Reads validation/ngeht/output/<tag>_params.csv and <tag>_truth.csv (the fit's end and its truth) and the campaign's
# uvfits files; the screen and truncation should match the fit's. Output (not tracked): viz/output/<tag>_movie.mp4,
# viz/output/<tag>_frame_<k>.png for six frames, viz/output/<tag>_chi2.png.
using CairoMakie
using CUDA
using KernelAbstractions
using StaticArrays
using LinearAlgebra
using DelimitedFiles
using Printf
using Dates
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer, KerrSplat.Fit
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
tag = getstr("--tag", "triband"); days = getopt("--days", 5); start = Date(getstr("--start", "2026-04-01"))
bands = parse.(Float64, split(getstr("--bands", "86,230,345"), ","))
res = getopt("--res", 96); fov = getopt("--fov", 16.0); N = getopt("--samples", 160); nmax = getopt("--nmax", 2); slab = getopt("--slab", 0.5)
frame_hours = getopt("--frame-hours", 4.0)
backend = "--cpu" in ARGS ? CPU() : CUDABackend()
datadir = joinpath(@__DIR__, "..", "validation", "ngeht", "output"); outdir = joinpath(@__DIR__, "output"); mkpath(outdir)

# ---- the fit's inputs and outputs
q = readdlm(joinpath(datadir, "$(tag)_params.csv"), ','); p = readdlm(joinpath(datadir, "$(tag)_truth.csv"), ',')
const M_solar = 6.5e9; const D = 16.8e6 * Transfer.PC; const L = gravitational_radius(M_solar)
t_M = L / Transfer.CL / 3600
a = 0.9; θo = deg2rad(60.0); Δα = fov / res
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
cache = GeodesicCache(backend, camera, Val(N); store_samples = true); regenerate!(cache, a, θo; marcher = Recurrence(64))
# a fit with the spacetime free is rendered at its own spin and inclination (the summary's "spacetime: ... end a X inc Y" line)
fitted = let m = match(r"spacetime: .*end a ([-\d.eE+]+) inc ([-\d.eE+]+)", read(joinpath(datadir, "$(tag)_summary.txt"), String))
    m === nothing ? nothing : (parse(Float64, m[1]), deg2rad(parse(Float64, m[2])))
end
cache_fit = fitted === nothing ? cache : (c = GeodesicCache(backend, camera, Val(N); store_samples = true); regenerate!(c, fitted[1], fitted[2]; marcher = Recurrence(64)); c)
fitted === nothing || @info "the fit is rendered at its own spacetime" a = fitted[1] inc_deg = rad2deg(fitted[2])

# ---- the campaign: every scan's (u, v) points, band and campaign hour
uvpoints = Dict(f => (u = Float64[], v = Float64[], hour = Float64[]) for f in bands)
for day in 0:days - 1, f in bands
    path = joinpath(datadir, "ngeht_M87_$(start + Day(day))_$(round(Int, f))GHz.uvfits")
    isfile(path) || continue
    obs = read_uvfits(path)
    append!(uvpoints[f].u, obs.u); append!(uvpoints[f].v, obs.v); append!(uvpoints[f].hour, 24day .+ obs.time)
end
hours_all = vcat((uvpoints[f].hour for f in bands)...)
blocks = sort(unique(floor.(hours_all ./ frame_hours)))
frame_times = (blocks .+ 0.5) .* frame_hours ./ t_M                    # M, as the fit's frames
frame_hours_centre = (blocks .+ 0.5) .* frame_hours
@info "campaign" frames = length(frame_times) span_M = round(frame_times[end] - frame_times[1], digits = 2) uv_points = length(hours_all)

# ---- rendering
device(x) = (y = KernelAbstractions.allocate(backend, eltype(x), size(x)...); copyto!(y, x); y)
qdev = device(q); pdev = device(p)                                      # the parameters on the backend (a host matrix is no kernel argument)
function render(params, t, ν)
    c = params === q ? cache_fit : cache
    out = Vector{Splats.accumulator_type(Float64, nmax)}(undef, npixels(c)); fill!(out, zero(eltype(out)))
    dev = KernelAbstractions.allocate(backend, eltype(out), length(out)); fill!(dev, zero(eltype(out)))
    polarized_image!(dev, c, params === q ? qdev : pdev, t, ν, L; nmax, slab)
    img = Fit.pixel_stokes(to_screen(c, Array(dev)), ν, nothing)
    return img                                                           # res × res of SVector{4} in cgs
end
stokesI(img) = [x[1] for x in img]
function ticks(img; every = 8, scale = 1.0)                             # EVPA ticks on a coarse grid as line segments, length ∝ polarized fraction
    segs = Point2f[]
    n = size(img, 1)
    for j in every÷2:every:n, i in every÷2:every:n
        I, Q, U, V = img[i, j]
        I > 0 || continue
        m = hypot(Q, U) / I; χ = atan(U, Q) / 2
        dx = scale * every * m * cos(χ) / 2; dy = scale * every * m * sin(χ) / 2
        push!(segs, Point2f(i - dx, j - dy)); push!(segs, Point2f(i + dx, j + dy))
    end
    return segs
end
xs = range(-6, 6; length = 61); zs = [0.0]
function slices(params, t)                                              # the temperature only where there is plasma
    ne, Θ, B = field_grid(params, t, xs, xs, zs)
    d = ne[:, :, 1]; floor_ne = 1e-3 * maximum(d)
    return log10.(max.(d, floor_ne)), [d[i, j] > floor_ne ? Θ[i, j, 1] : NaN for i in axes(d, 1), j in axes(d, 2)]
end
peak = Dict(f => maximum(stokesI(render(p, frame_times[1], f * 1e9))) for f in bands)

# ---- the figure
fig = Figure(size = (1800, 1150), fontsize = 16)
Label(fig[0, 1:6], "triband ngEHT self-fit of a six-parcel slow-light truth at M87 from an over-complete shell", fontsize = 22, tellwidth = false)
img_axes = Dict{Tuple{Float64,Symbol},Axis}(); img_obs = Dict{Tuple{Float64,Symbol},Observable}()
tick_obs = Dict{Tuple{Float64,Symbol},Observable}()
for (col, f) in enumerate(bands), (row, which) in enumerate((:truth, :fit))
    ax = Axis(fig[row, col], title = "$(which == :truth ? "truth" : "fit") at $(round(Int, f)) GHz", aspect = DataAspect(), xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false)
    img_axes[(f, which)] = ax
    img_obs[(f, which)] = Observable(stokesI(render(which == :truth ? p : q, frame_times[1], f * 1e9)))
    heatmap!(ax, img_obs[(f, which)], colormap = :afmhot, colorrange = (0, peak[f]))
    tick_obs[(f, which)] = Observable(ticks(render(which == :truth ? p : q, frame_times[1], f * 1e9)))
    linesegments!(ax, tick_obs[(f, which)], color = :cyan, linewidth = 1.5)
end
ax_uv = Axis(fig[1:2, 4:5], title = "(u, v) coverage to date", xlabel = "u (Gλ)", ylabel = "v (Gλ)", aspect = DataAspect())
uv_obs = Dict(f => Observable(Point2f[]) for f in bands)
colors = Dict(86.0 => :dodgerblue, 230.0 => :orange, 345.0 => :crimson)
for f in bands
    scatter!(ax_uv, uv_obs[f], color = get(colors, f, :black), markersize = 3, label = "$(round(Int, f)) GHz")
end
axislegend(ax_uv, position = :rt); limits!(ax_uv, -13, 13, -13, 13)
ax_time = Axis(fig[1:2, 6], title = "campaign clock")
hidedecorations!(ax_time); hidespines!(ax_time)
clock = Observable("")
text!(ax_time, 0.05, 0.6, text = clock, fontsize = 20, space = :relative)
sl = Dict{Tuple{Symbol,Symbol},Observable}()
for (col, which) in enumerate((:truth, :fit)), (row, field) in enumerate((:density, :temperature))
    ax = Axis(fig[row + 2, col], title = "$(which) midplane $(field == :density ? "log₁₀ nₑ" : "Θe")", aspect = DataAspect(), xlabel = "x (M)", ylabel = "y (M)")
    d, T = slices(which == :truth ? p : q, frame_times[1])
    sl[(which, field)] = Observable(field == :density ? d : T)
    heatmap!(ax, xs, xs, sl[(which, field)], colormap = field == :density ? :viridis : :plasma)
end
# the total χ² at every iteration (the fit's history file), as a still on the right
histpath = joinpath(datadir, "$(tag)_history.csv")
history = isfile(histpath) ? vec(readdlm(histpath, ',')) : Float64[]
ax_chi = Axis(fig[3:4, 3:6], title = "the three bands' χ² against the iteration", xlabel = "iteration", ylabel = "χ²", yscale = log10)
isempty(history) || lines!(ax_chi, 1:length(history), max.(history, 1e-300), color = :black, linewidth = 2)

function set_frame!(k)
    t = frame_times[k]; hour = frame_hours_centre[k]
    for f in bands, which in (:truth, :fit)
        img = render(which == :truth ? p : q, t, f * 1e9)
        img_obs[(f, which)][] = stokesI(img); tick_obs[(f, which)][] = ticks(img)
    end
    for f in bands
        sel = uvpoints[f].hour .<= hour + frame_hours / 2
        uv_obs[f][] = [Point2f(u / 1e9, v / 1e9) for (u, v) in zip(uvpoints[f].u[sel], uvpoints[f].v[sel])]
    end
    for which in (:truth, :fit)
        d, T = slices(which == :truth ? p : q, t)
        sl[(which, :density)][] = d; sl[(which, :temperature)][] = T
    end
    clock[] = @sprintf("day %d, hour %02d\n%.1f M since the start", floor(Int, hour / 24) + 1, round(Int, mod(hour, 24)), t - frame_times[1])
end
moviepath = joinpath(outdir, "$(tag)_movie.mp4")
CairoMakie.record(fig, moviepath, eachindex(frame_times); framerate = 6) do k
    set_frame!(k)
end
for (n, k) in enumerate(round.(Int, range(1, length(frame_times); length = 6)))
    set_frame!(k); save(joinpath(outdir, "$(tag)_frame_$(n).png"), fig)
end
@info "written" moviepath frames = length(frame_times)
