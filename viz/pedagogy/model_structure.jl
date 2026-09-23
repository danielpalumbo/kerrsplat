# The structure of the model in one figure: the six-parcel truth of the self-fits in three dimensions around the horizon
# with a dozen geodesics from the observer's screen threading it (the stored samples of the cache), the observer's line of
# sight and the spin axis; beside it, what the observer sees at 230 GHz with the polarization ticks, and the same parcels'
# contribution at 86 and 345 GHz. A pedagogical summary figure (docs/pedagogy/README.md); regenerate when the model changes.
#
#     julia -t 6 --project=viz viz/pedagogy/model_structure.jl
using CairoMakie, StaticArrays, LinearAlgebra, Random, Statistics, Printf, Meshing
using KernelAbstractions: CPU
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
const GeometryBasics = CairoMakie.Makie.GeometryBasics
outdir = joinpath(@__DIR__, "..", "..", "docs", "pedagogy"); mkpath(outdir)
a = 0.9; θo = deg2rad(60.0); L = gravitational_radius(6.5e9); res = 96; N = 160; fov = 16.0; nmax = 2; slab = 0.5
rng = MersenneTwister(1); kepler(r) = 1 / (r^1.5 + a)
p = zeros(NPOLARIZEDPARAMS, 6)
for i in 1:6
    φ = 2π * (i - 1) / 6 + 0.3 * randn(rng); r0 = 2.5 + 2.5 * (i - 1) / 5
    p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.3 * randn(rng), log(0.7), log(0.7), log(0.5), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
               0.0, log(1e9), log(3e5) + 0.3 * randn(rng), log(30.0) + 0.2 * randn(rng), log(20.0) + 0.2 * randn(rng), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.0, 0.3, 0.05 * randn(rng), kepler(r0)]
end
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
cache = GeodesicCache(CPU(), camera, Val(N); store_samples = true); regenerate!(cache, a, θo; marcher = Recurrence(64))
freqs = [86e9, 230e9, 345e9]
cube = polarized_cube(cache, p, [0.0], freqs, L; nmax, slab)          # res × res × 1 × 3
# ---- the fields on a grid for the parcels' surfaces
voxel = 0.15; xs = collect(-6.5:voxel:6.5); ys = xs; zs = collect(-2.0:voxel:2.0)
ne = zeros(length(xs), length(ys), length(zs)); Θ = zeros(size(ne))
for (k, z) in enumerate(zs), (j, y) in enumerate(ys), (i, x) in enumerate(xs)
    s = 0.0; sΘ = 0.0
    for m in 1:6
        w = exp(p[13, m]) * Splats.splat_weight(p, m, 0.0, x, y, z)
        s += w; sΘ += w * exp(p[14, m])
    end
    ne[i, j, k] = s; Θ[i, j, k] = s > 0 ? sΘ / s : 0.0
end
vts, fcs = isosurface(ne, MarchingCubes(iso = 0.2 * maximum(ne)), xs, ys, zs)
pts = [Point3f(v[1], v[2], v[3]) for v in vts]
msh = GeometryBasics.Mesh(pts, [GeometryBasics.TriangleFace{Int}(f[1], f[2], f[3]) for f in fcs])
function interp3(vol, x, y, z)
    fx = clamp((x - xs[1]) / voxel + 1, 1, length(xs) - 1e-9); fy = clamp((y - ys[1]) / voxel + 1, 1, length(ys) - 1e-9); fz = clamp((z - zs[1]) / voxel + 1, 1, length(zs) - 1e-9)
    i = floor(Int, fx); j = floor(Int, fy); k = floor(Int, fz); u = fx - i; v = fy - j; w = fz - k
    return (1 - u) * (1 - v) * (1 - w) * vol[i, j, k] + u * (1 - v) * (1 - w) * vol[i+1, j, k] + (1 - u) * v * (1 - w) * vol[i, j+1, k] + u * v * (1 - w) * vol[i+1, j+1, k] +
           (1 - u) * (1 - v) * w * vol[i, j, k+1] + u * (1 - v) * w * vol[i+1, j, k+1] + (1 - u) * v * w * vol[i, j+1, k+1] + u * v * w * vol[i+1, j+1, k+1]
end
# ---- a dozen geodesics: the rays of the brightest 230 GHz pixels, from the stored samples (BL → Cartesian), inside r < 11 M
img = cube[:, :, 1, 2]; I = [x[1] for x in img]
order = sortperm(vec(I); rev = true)
picked = Int[]
for idx in order
    i, jj = Tuple(CartesianIndices(I)[idx])
    any(q -> hypot(i - Tuple(CartesianIndices(I)[q])[1], jj - Tuple(CartesianIndices(I)[q])[2]) < 11, picked) && continue
    push!(picked, idx); length(picked) >= 12 && break
end
S = cache.samples; perm = cache.perm_host; nα = res
rays = Vector{Vector{Point3f}}()
for idx in picked
    i, jj = Tuple(CartesianIndices(I)[idx]); screen = (jj - 1) * nα + i
    j = findfirst(==(screen), perm); j === nothing && continue
    path = Point3f[]
    for k in 1:N
        f = S.flags[j, k]; (f & 0x01) != 0 || continue
        r = S.r[j, k]; θ = S.θ[j, k]; φ = S.ϕ[j, k]
        r < 11 || continue
        push!(path, Point3f(r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ)))
    end
    length(path) > 5 && push!(rays, path)
end
# the observer's direction: the mean unit vector of the rays' outermost samples
far = [normalize(Vec3f(ray[1])) for ray in rays]; obs = normalize(sum(far))
# ---- the figure
fig = Figure(size = (2000, 1050), fontsize = 15)
ax3 = Axis3(fig[1, 1], aspect = :data, limits = ((-7.5, 7.5), (-7.5, 7.5), (-4, 8)), viewmode = :fitzoom, azimuth = deg2rad(-40), elevation = deg2rad(22), perspectiveness = 0.3,
            xlabel = "x [M]", ylabel = "y [M]", zlabel = "z [M]", title = "the model: fluid parcels on pattern orbits in Kerr spacetime, seen along rays from the observer's screen")
r_hor = 1 + sqrt(1 - a^2)
mesh!(ax3, Sphere(Point3f(0), Float32(r_hor)), color = :black)
lines!(ax3, [Point3f(6cos(φ), 6sin(φ), 0) for φ in range(0, 2π; length = 200)], color = (:gray40, 0.6), linestyle = :dash)
lines!(ax3, [Point3f(0, 0, -4), Point3f(0, 0, 7)], color = :gray20, linewidth = 2); text!(ax3, Point3f(0, 0, 7.3), text = "spin axis, a = 0.9", fontsize = 12)
mesh!(ax3, msh, color = [interp3(Θ, v...) for v in pts], colormap = :plasma, colorrange = (24, 42), transparency = false)
for ray in rays
    lines!(ax3, ray, color = (:dodgerblue, 0.8), linewidth = 1.6)
end
arrows!(ax3, [Point3f(obs * 7.5)], [Vec3f(obs * 3)], color = :firebrick, linewidth = 0.05, arrowsize = Vec3f(0.18, 0.18, 0.3))
text!(ax3, Point3f(obs * 11), text = "to the observer\n(inclination 60°)", fontsize = 12, color = :firebrick)
Colorbar(fig[1, 2], colormap = :plasma, limits = (24, 42), label = "Θe on the parcels' 20%-of-peak density surface", width = 12)
# the images
gl = fig[1, 3] = GridLayout()
function ticks(img; nvec = 20, pcut = 0.03, maxlen = 1.6)
    n = size(img, 1); cell = max(n ÷ nvec, 1); imax = maximum(x -> x[1], img)
    cells = Tuple{Float64,Float64,Float64,Float64,Float64}[]
    for j0 in 1:cell:n-cell+1, i0 in 1:cell:n-cell+1
        Iv = 0.0; Q = 0.0; U = 0.0
        for j in j0:j0+cell-1, i in i0:i0+cell-1
            Iv += img[i, j][1]; Q += img[i, j][2]; U += img[i, j][3]
        end
        Iv /= cell^2; Q /= cell^2; U /= cell^2
        Iv > pcut * imax || continue
        P = hypot(Q, U); push!(cells, (i0 + (cell - 1) / 2, j0 + (cell - 1) / 2, P, P / Iv, atan(U, Q) / 2))
    end
    pmax = max(maximum(c -> c[3], cells; init = 0.0), eps()); segs = Point2f[]; cols = Float32[]
    for (ci, cj, P, m, χ) in cells
        len = maxlen * cell * P / pmax; di = -sin(χ) * len / 2; dj = cos(χ) * len / 2
        push!(segs, Point2f(ci - di, cj - dj)); push!(segs, Point2f(ci + di, cj + dj)); push!(cols, m); push!(cols, m)
    end
    return segs, cols
end
for (c, ν) in enumerate(freqs)
    im = cube[:, :, 1, c]
    ax = Axis(gl[c, 1], title = @sprintf("%d GHz: Stokes I and EVPA ticks", round(Int, ν / 1e9)), aspect = DataAspect(), xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false)
    heatmap!(ax, [x[1] for x in im], colormap = :afmhot, colorrange = (0, quantile(vec([x[1] for x in im]), 0.998)))
    s, m = ticks(im); linesegments!(ax, s, color = m, colormap = :rainbow, colorrange = (0, 0.7), linewidth = 2)
    c == 2 && scatter!(ax, [Point2f(Tuple(CartesianIndices(I)[idx])...) for idx in picked], color = :dodgerblue, markersize = 9, strokecolor = :white, strokewidth = 1)
end
Label(fig[2, 1:3], "Six parcels at 2.5–5 M (Gaussian ellipsoids of 0.7 × 0.7 × 0.5 M, thermal electrons at Θe ≈ 30 and B ≈ 20 G, azimuthal velocity 0.3c) rotate about the spin axis at their Keplerian rates; " *
      "each ray is a Kerr geodesic marched from the screen (blue: the twelve brightest pixels' rays, marked on the 230 GHz image), and the full-Stokes transport along it, with slow light, gives the image the observer sees at each band.",
      fontsize = 13, tellwidth = false, word_wrap = true, justification = :left)
colsize!(fig.layout, 1, Relative(0.55))
save(joinpath(outdir, "model_structure.png"), fig, px_per_unit = 2)
println("written docs/pedagogy/model_structure.png")
