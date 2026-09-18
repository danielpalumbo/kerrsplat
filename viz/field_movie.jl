# The self-fit's fields in three dimensions against the truth's, frame by frame over the campaign: the 230 GHz image with
# polarization ticks, the density as nested isosurfaces at absolute levels, the temperature and the field strength painted
# on each model's own density surface, and the field direction on the midplane; the truth in the upper row, the fit in the
# lower. CPU only (the fields on a voxel grid, marching cubes, the images through the stored geodesics), so that it shares
# the machine with a fit on the card.
#
#     nice -n 10 julia -t 6 --project=viz viz/field_movie.jl [--tag triband_pcg4] [--label "fit: 70 parcels, chi2/N 1.11"]
#         [--frames 60] [--span 13.3] [--res 64] [--fov 16] [--samples 160] [--nmax 2] [--slab 0.5] [--voxel 0.2] [--fps 12]
#
# Reads validation/ngeht/output/<tag>_params.csv and <tag>_truth.csv (M87's mass; a = 0.9, inclination 60°, the triband
# self-fit's spacetime). Output (not tracked): viz/output/<tag>_fields.mp4 and <tag>_fields_1..6.png.
using CairoMakie
const GeometryBasics = CairoMakie.Makie.GeometryBasics      # Makie's own, not a direct dependency of the viz environment
using Meshing
using StaticArrays
using KernelAbstractions: CPU
using LinearAlgebra
using DelimitedFiles
using Statistics: quantile
using Printf
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
tag = getstr("--tag", "triband_pcg4"); label = getstr("--label", "fit")
nframes = getopt("--frames", 60); span = getopt("--span", 13.3); res = getopt("--res", 64); fov = getopt("--fov", 16.0)
N = getopt("--samples", 160); nmax = getopt("--nmax", 2); slab = getopt("--slab", 0.5); voxel = getopt("--voxel", 0.2); fps = getopt("--fps", 12)
datadir = joinpath(@__DIR__, "..", "validation", "ngeht", "output"); outdir = joinpath(@__DIR__, "output"); mkpath(outdir)
q = readdlm(joinpath(datadir, "$(tag)_params.csv"), ','); p = readdlm(joinpath(datadir, "$(tag)_truth.csv"), ',')
const a = 0.9; const θo = deg2rad(60.0); const L = gravitational_radius(6.5e9); hours_per_M = L / Transfer.CL / 3600
times = collect(range(0.0, span; length = nframes))
models = [("truth: six parcels", p), (label, q)]
@info "field movie" tag parcels = size.(getindex.(models, 2), 2) frames = nframes span voxel threads = Threads.nthreads()

# ---- the fields on a voxel grid: density, density-weighted temperature and field strength, and the field's unit vector
# (the parcels' (B_r, B_θ, B_φ) in the ZAMO tetrad turned into Cartesian components at the voxel)
xs = collect(-6.5:voxel:6.5); ys = xs; zs = collect(-1.6:voxel:1.6)
function field_grids(p, t)
    n = size(p, 2)
    A = [exp(p[13, i]) for i in 1:n]; Θi = [exp(p[14, i]) for i in 1:n]; Bi = [exp(p[15, i]) for i in 1:n]
    b = [SVector(sin(p[16, i]) * cos(p[17, i]), sin(p[16, i]) * sin(p[17, i]), cos(p[16, i])) for i in 1:n]
    ne = zeros(length(xs), length(ys), length(zs)); Θ = zeros(size(ne)); B = zeros(size(ne)); Bv = fill(SVector(0.0, 0.0, 0.0), size(ne))
    Threads.@threads for k in eachindex(zs)
        z = zs[k]
        for (j, y) in enumerate(ys), (i, x) in enumerate(xs)
            s = 0.0; sΘ = 0.0; sB = 0.0; sv = SVector(0.0, 0.0, 0.0)
            for m in 1:n
                w = A[m] * Splats.splat_weight(p, m, t, x, y, z)
                w > 1e-30 || continue
                s += w; sΘ += w * Θi[m]; sB += w * Bi[m]; sv += w * Bi[m] * b[m]
            end
            s > 0 || continue
            ne[i, j, k] = s; Θ[i, j, k] = sΘ / s; B[i, j, k] = sB / s
            r = max(hypot(x, y, z), 1e-6); ct = z / r; st = sqrt(max(1 - ct^2, 0.0)); φ = atan(y, x)
            rhat = SVector(st * cos(φ), st * sin(φ), ct); that = SVector(ct * cos(φ), ct * sin(φ), -st); phat = SVector(-sin(φ), cos(φ), 0.0)
            v = sv[1] * rhat + sv[2] * that + sv[3] * phat
            Bv[i, j, k] = v / max(norm(v), 1e-30)
        end
    end
    return ne, Θ, B, Bv
end
# trilinear interpolation of a grid at a point
function interp3(vol, x, y, z)
    fx = clamp((x - xs[1]) / voxel + 1, 1, length(xs) - 1e-9); fy = clamp((y - ys[1]) / voxel + 1, 1, length(ys) - 1e-9); fz = clamp((z - zs[1]) / voxel + 1, 1, length(zs) - 1e-9)
    i = floor(Int, fx); j = floor(Int, fy); k = floor(Int, fz); u = fx - i; v = fy - j; w = fz - k
    return (1 - u) * (1 - v) * (1 - w) * vol[i, j, k] + u * (1 - v) * (1 - w) * vol[i+1, j, k] + (1 - u) * v * (1 - w) * vol[i, j+1, k] + u * v * (1 - w) * vol[i+1, j+1, k] +
           (1 - u) * (1 - v) * w * vol[i, j, k+1] + u * (1 - v) * w * vol[i+1, j, k+1] + (1 - u) * v * w * vol[i, j+1, k+1] + u * v * w * vol[i+1, j+1, k+1]
end
function surface_mesh(vol, level)
    vts, fcs = isosurface(vol, MarchingCubes(iso = level), xs, ys, zs)
    isempty(fcs) && return nothing
    pts = [Point3f(v[1], v[2], v[3]) for v in vts]
    return GeometryBasics.Mesh(pts, [GeometryBasics.TriangleFace{Int}(f[1], f[2], f[3]) for f in fcs]), pts
end
n_ref = maximum(field_grids(p, 0.0)[1])                       # the truth's peak density at the start, the absolute levels' reference

# ---- the 230 GHz images (Stokes I with eht-imaging-style ticks) through the stored geodesics on the CPU
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cache, a, θo; marcher = Fused(64))
t0 = time()
cubes = [polarized_cube(cache, m[2], times, [230e9], L; nmax, slab) for m in models]
@info "images rendered" minutes = round((time() - t0) / 60, digits = 1)
function ticks(img; nvec = 20, pcut = 0.03, maxlen = 1.6)
    n = size(img, 1); cell = max(n ÷ nvec, 1); imax = maximum(x -> x[1], img)
    cells = Tuple{Float64,Float64,Float64,Float64,Float64}[]
    for j0 in 1:cell:n-cell+1, i0 in 1:cell:n-cell+1
        I = 0.0; Q = 0.0; U = 0.0
        for j in j0:j0+cell-1, i in i0:i0+cell-1
            I += img[i, j][1]; Q += img[i, j][2]; U += img[i, j][3]
        end
        I /= cell^2; Q /= cell^2; U /= cell^2
        I > pcut * imax || continue
        P = hypot(Q, U); push!(cells, (i0 + (cell - 1) / 2, j0 + (cell - 1) / 2, P, P / I, atan(U, Q) / 2))
    end
    pmax = max(maximum(c -> c[3], cells; init = 0.0), eps()); segs = Point2f[]; cols = Float32[]
    for (ci, cj, P, m, χ) in cells
        len = maxlen * cell * P / pmax; di = -sin(χ) * len / 2; dj = cos(χ) * len / 2
        push!(segs, Point2f(ci - di, cj - dj)); push!(segs, Point2f(ci + di, cj + dj)); push!(cols, m); push!(cols, m)
    end
    return segs, cols
end
imax = quantile(vec([x[1] for x in cubes[1]]), 0.998)      # one brightness scale for both, the truth's

# ---- the figure: two rows (truth, fit), five columns
Θrange = (18.0, 42.0); Brange = (8.0, 34.0); levels = [(0.5, (:orangered, 0.95)), (0.2, (:orange, 0.45)), (0.08, (:gold, 0.18))]
fig = Figure(size = (2600, 900), fontsize = 17)
titles = ["230 GHz: Stokes I, EVPA ticks", @sprintf("density: isosurfaces at 50, 20, 8%% of the truth's peak (%.2g cm⁻³)", n_ref),
          "temperature Θe on the model's 20%-of-peak density surface", "field strength |B| on the same surface", "field direction on the midplane (colour |B|)"]
img_obs = Vector{Observable}(undef, 2); seg_obs = similar(img_obs); col_obs = similar(img_obs); ax3 = Matrix{Axis3}(undef, 2, 4)
for (r, (name, _)) in enumerate(models)
    Label(fig[r, 0], name, rotation = π / 2, fontsize = 20, tellheight = false)
    ax = Axis(fig[r, 1], title = r == 1 ? titles[1] : "", aspect = DataAspect(), xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false)
    img_obs[r] = Observable([x[1] for x in cubes[r][:, :, 1, 1]])
    heatmap!(ax, img_obs[r], colormap = :afmhot, colorrange = (0, imax))
    s, m = ticks(cubes[r][:, :, 1, 1]); seg_obs[r] = Observable(s); col_obs[r] = Observable(m)
    linesegments!(ax, seg_obs[r], color = col_obs[r], colormap = :rainbow, colorrange = (0, 0.7), linewidth = 2.5)
    for c in 1:4
        ax3[r, c] = Axis3(fig[r, c + 1], title = r == 1 ? titles[c + 1] : "", aspect = :data, limits = ((-6.5, 6.5), (-6.5, 6.5), (-1.6, 1.6)), viewmode = :fitzoom,
                          azimuth = deg2rad(-55), elevation = deg2rad(28), perspectiveness = 0.25, xlabel = "x [M]", ylabel = "y [M]", zlabel = "z [M]",
                          xlabelsize = 12, ylabelsize = 12, zlabelsize = 12, xticklabelsize = 10, yticklabelsize = 10, zticklabelsize = 10, titlesize = 15)
    end
end
Colorbar(fig[1:2, 6], colormap = :plasma, limits = Θrange, label = "Θe on the surface", width = 14)
Colorbar(fig[1:2, 7], colormap = :viridis, limits = Brange, label = "|B| [G]", width = 14)
Colorbar(fig[1:2, 8], colormap = :rainbow, limits = (0, 0.7), label = "fractional polarization of the tick", width = 14)
colsize!(fig.layout, 1, Relative(0.13))
clock = Observable(""); Label(fig[0, 1:5], clock, fontsize = 21, tellwidth = false)
Label(fig[3, 1:5], "the parcels move at their pattern rates; the fit's spacetime is the truth's (a = 0.9, inclination 60°); M87's mass and distance; 0.25 M pixels, n ≤ 2; the black sphere is the horizon, the dashed circle r = 6 M; midplane arrows where the density exceeds 20% of the model's peak, outlines at 5 and 20%", fontsize = 14, tellwidth = false)
const r_hor = 1 + sqrt(1 - a^2)
circle = [Point3f(6cos(φ), 6sin(φ), 0) for φ in range(0, 2π; length = 200)]
draw_arrows! = isdefined(Makie, :arrows3d!) ? (ax, pts, dirs, vals) -> arrows3d!(ax, pts, dirs; color = vals, colormap = :viridis, colorrange = Brange, lengthscale = 0.7, shaftradius = 0.07, tipradius = 0.16, tiplength = 0.28, align = :center) :
                                              (ax, pts, dirs, vals) -> arrows!(ax, pts, dirs; color = vals, colormap = :viridis, colorrange = Brange, lengthscale = 0.7, linewidth = 0.08, arrowsize = Vec3f(0.2, 0.2, 0.3), align = :center)
function decorate!(ax)
    mesh!(ax, Sphere(Point3f(0), Float32(r_hor)), color = :black)
    lines!(ax, circle, color = (:gray40, 0.7), linestyle = :dash, linewidth = 1)
end
function set_frame!(k)
    t = times[k]
    for (r, (_, pm)) in enumerate(models)
        img = cubes[r][:, :, k, 1]; img_obs[r][] = [x[1] for x in img]
        s, m = ticks(img); seg_obs[r][] = s; col_obs[r][] = m
        ne, Θ, B, Bv = field_grids(pm, t)
        own = 0.2 * maximum(ne)
        for c in 1:4
            empty!(ax3[r, c]); decorate!(ax3[r, c])
        end
        for (frac, col) in reverse(levels)                          # outer, most transparent surfaces first
            sm = surface_mesh(ne, frac * n_ref); sm === nothing && continue
            mesh!(ax3[r, 1], sm[1], color = col, transparency = true)
        end
        sm = surface_mesh(ne, own)
        if sm !== nothing
            msh, pts = sm
            mesh!(ax3[r, 2], msh, color = [interp3(Θ, v...) for v in pts], colormap = :plasma, colorrange = Θrange)
            mesh!(ax3[r, 3], msh, color = [interp3(B, v...) for v in pts], colormap = :viridis, colorrange = Brange)
        end
        kmid = argmin(abs.(zs)); step = max(round(Int, 0.6 / voxel), 1)
        for (lev, col) in ((0.05, (:gray55, 0.8)), (own / maximum(ne), (:gray25, 0.9)))          # the material on the midplane: outlines at 5% and 20% of the peak
            for line in Makie.Contours.lines(Makie.Contours.contour(xs, ys, ne[:, :, kmid], lev * maximum(ne)))
                cx, cy = Makie.Contours.coordinates(line)
                lines!(ax3[r, 4], [Point3f(x, y, 0) for (x, y) in zip(cx, cy)], color = col, linewidth = 1.2)
            end
        end
        pts = Point3f[]; dirs = Vec3f[]; vals = Float32[]
        for j in 1:step:length(ys), i in 1:step:length(xs)
            ne[i, j, kmid] > own || continue
            v = Bv[i, j, kmid]; push!(pts, Point3f(xs[i], ys[j], 0.05)); push!(dirs, Vec3f(v...)); push!(vals, B[i, j, kmid])
        end
        isempty(pts) || draw_arrows!(ax3[r, 4], pts, dirs, vals)
    end
    clock[] = @sprintf("t = %.1f M  (%.0f h of the five-day campaign)", t, t * hours_per_M)
end
moviepath = joinpath(outdir, "$(tag)_fields.mp4")
t0 = time()
CairoMakie.record(fig, moviepath, 1:nframes; framerate = fps) do k
    set_frame!(k)
    k % 10 == 0 && @info "frame $k" minutes = round((time() - t0) / 60, digits = 1)
end
for (n, k) in enumerate(round.(Int, range(1, nframes; length = 6)))
    set_frame!(k); save(joinpath(outdir, "$(tag)_fields_$(n).png"), fig)
end
@info "written" moviepath minutes = round((time() - t0) / 60, digits = 1)
