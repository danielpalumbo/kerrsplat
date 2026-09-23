# The model at the scale of a RIAF: a puffy disk and a jet built from a couple of thousand parcels on a lattice that
# follows the semi-analytic RIAF of Broderick & Loeb (2006, 2009) and Pu & Broderick (2018) — thermal density
# n = n₀ r^{-1.1} exp(−z²/2(h r)²), temperature Θe = Θ₀ r^{-0.84}, a field in equipartition with the ions at β = 10, a
# Keplerian disk — with a parabolic jet sheath above and below it (Broderick & Loeb 2009's M87 jet in spirit: density
# falling as z^{-2}, an outflow accelerating along the streamline, a toroidal field). Left: the structure in three
# dimensions with geodesics from the screen. Right: the images at 86, 230 and 345 GHz for an optically thin disk
# (n₀ a hundredth of Sgr A*'s) and for Broderick & Loeb's Sgr A* density, where the disk is optically thick at the
# lower bands and hides the shadow. A pedagogical summary figure (docs/pedagogy/README.md).
#
#     julia -t 8 --project=viz viz/pedagogy/model_structure_riaf.jl [--cpu]
using CairoMakie, StaticArrays, LinearAlgebra, Random, Statistics, Printf, Meshing
using KernelAbstractions, CUDA
using KerrSplat, KerrSplat.Geodesics, KerrSplat.Splats, KerrSplat.Transfer
const GeometryBasics = CairoMakie.Makie.GeometryBasics
outdir = joinpath(@__DIR__, "..", "..", "docs", "pedagogy"); mkpath(outdir)
backend = "--cpu" in ARGS ? CPU() : CUDABackend()
a = 0.9375; θo = deg2rad(60.0); M_solar = 4.3e6; L = gravitational_radius(M_solar)
res = 128; N = 200; fov = 50.0; nmax = 2; slab = 0.5
freqs = [86e9, 230e9, 345e9]
Θ_unit = Transfer.KBOL / (Transfer.ME * Transfer.CL^2)

"""
    riaf_parcels(n0; h = 0.7, rin = 2.0, rout = 20.0, ratio = 1.3, Θ0 = 1.7e11 K, β = 10, jet = true, zmax = 30.0)

Parcels on a lattice through the RIAF: shells of radii growing by `ratio`, azimuths and heights spaced by the shell's
radial spacing, every parcel a Gaussian of scale Δr/1.5 so that neighbours overlap at their 1.5σ; the peak density of
each is the profile's value there divided by the lattice's overlap factor, so the mixture reproduces the profile. The
jet: rings of parcels along two parabolic streamlines r_cyl = r_fp √(z / z_fp) from z_fp = 3 M to `zmax`, with the
density falling as (z/z_fp)^{-2} from n0/30, the outflow's ZAMO velocity along the streamline growing from 0.2 to 1.5
in γβ, a hotter Θe = 50 and a toroidal field.
"""
function riaf_parcels(n0; h = 0.7, rin = 2.0, rout = 20.0, ratio = 1.3, T0 = 1.7e11, β = 10.0, jet = true, zmax = 30.0, rng = MersenneTwister(3))
    cols = Vector{Vector{Float64}}()
    overlap = (sqrt(2π) / 1.5)^3                                  # a lattice of spacing d with σ = d/1.5 sums to this times the peak
    r = rin
    while r < rout
        Δr = r * (ratio - 1); σ = Δr / 1.5
        nφ = max(round(Int, 2π * r / Δr), 6)
        zlev = -2h * r:Δr:2h * r
        for φ in range(0, 2π; length = nφ + 1)[1:end-1] .+ 2π * rand(rng) / nφ, z in zlev
            n = n0 * r^-1.1 * exp(-z^2 / (2 * (h * r)^2)) / overlap
            n > 1e-4 * n0 || continue
            Θe = T0 * Θ_unit * r^-0.84
            B = sqrt(8π * n * overlap * Transfer.MP * Transfer.CL^2 / (β * 12 * r))
            ω = 1 / (r^1.5 + a)
            q = normalize(SVector(1.0, 0.05 * randn(rng), 0.05 * randn(rng), 0.05 * randn(rng)))
            push!(cols, [r * cos(φ), r * sin(φ), z, log(σ), log(σ), log(σ), q[1], q[2], q[3], q[4], 0.0, log(1e9),
                         log(n), log(Θe), log(B), π / 2 + 0.2 * randn(rng), π / 2 + 0.4 * randn(rng), -0.05, 0.3 / sqrt(r / 3), 0.0, ω])
        end
        r *= ratio
    end
    if jet
        for sign in (1, -1)
            z = 3.0
            while z < zmax
                rc = 3.0 * sqrt(z / 3.0); Δ = 0.3 * z; σ = Δ / 1.5
                nφ = max(round(Int, 2π * rc / Δ), 4)
                n = n0 / 30 * (z / 3)^-2 / overlap
                γβ = 0.2 + 1.3 * (z - 3) / (zmax - 3)
                ψ = atan(rc / (2z))                                   # the streamline's angle from the axis (parabola)
                for φ in range(0, 2π; length = nφ + 1)[1:end-1]
                    push!(cols, [rc * cos(φ), rc * sin(φ), sign * z, log(σ), log(σ), log(1.4σ), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9),
                                 log(n), log(50.0), log(sqrt(8π * n * overlap * Transfer.MP * Transfer.CL^2 / (β * 12 * hypot(rc, z)))), π / 2, π / 2 + 0.15 * randn(rng),
                                 γβ * sin(ψ), 0.1, sign * γβ * cos(ψ), 0.02])
                end
                z += Δ
            end
        end
    end
    return reduce(hcat, cols)
end
n_sgra = 3e7                                                          # Broderick & Loeb's Sgr A* thermal density at r = 1 M
p_thick = riaf_parcels(n_sgra); p_thin = riaf_parcels(n_sgra / 10)
ndisk = count(i -> abs(p_thick[3, i]) < 2 * 0.7 * hypot(p_thick[1, i], p_thick[2, i]) + 0.1 && hypot(p_thick[1, i], p_thick[2, i]) < 20.5, 1:size(p_thick, 2))
@info "parcels" total = size(p_thick, 2) disk = ndisk jet = size(p_thick, 2) - ndisk
# ---- the images on the backend
camera = Geodesics.Camera((-fov / 2, fov / 2), (-fov / 2, fov / 2), res)
cache = GeodesicCache(backend, camera, Val(N); store_samples = false); regenerate!(cache, a, θo; marcher = Fused(64))
dev(x) = (y = KernelAbstractions.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)
t0 = time()
cube_thin = polarized_cube(cache, dev(p_thin), [0.0], freqs, L; nmax, slab)
cube_thick = polarized_cube(cache, dev(p_thick), [0.0], freqs, L; nmax, slab)
@info "rendered" seconds = round(time() - t0, digits = 1)
flux(cube, c) = sum(x -> x[1], cube[:, :, 1, c]) * Transfer.pixel_solid_angle(fov / res, L, 8.3e3 * Transfer.PC) / Transfer.JY
pk = p_thick
# ---- geodesics from the stored samples of a CPU cache at a coarser screen
ccpu = GeodesicCache(CPU(), camera, Val(N); store_samples = true); regenerate!(ccpu, a, θo; marcher = Recurrence(64))
I230 = [x[1] for x in cube_thin[:, :, 1, 2]]
order = sortperm(vec(I230); rev = true); picked = Int[]
for idx in order
    i, jj = Tuple(CartesianIndices(I230)[idx])
    any(q -> hypot(i - Tuple(CartesianIndices(I230)[q])[1], jj - Tuple(CartesianIndices(I230)[q])[2]) < 14, picked) && continue
    push!(picked, idx); length(picked) >= 14 && break
end
S = ccpu.samples; perm = ccpu.perm_host
rays = Vector{Vector{Point3f}}()
for idx in picked
    i, jj = Tuple(CartesianIndices(I230)[idx]); screen = (jj - 1) * res + i
    j = findfirst(==(screen), perm); j === nothing && continue
    path = Point3f[]
    for k in 1:N
        (S.flags[j, k] & 0x01) != 0 || continue
        r = S.r[j, k]; θ = S.θ[j, k]; φ = S.ϕ[j, k]; r < 34 || continue
        push!(path, Point3f(r * sin(θ) * cos(φ), r * sin(θ) * sin(φ), r * cos(θ)))
    end
    length(path) > 5 && push!(rays, path)
end
obs = normalize(sum(normalize(Vec3f(ray[1])) for ray in rays))
# ---- the figure
fig = Figure(size = (2200, 1250), fontsize = 15)
ax3 = Axis3(fig[1, 1], aspect = :data, limits = ((-25, 25), (-25, 25), (-30, 34)), viewmode = :fitzoom, azimuth = deg2rad(-90), elevation = deg2rad(12), perspectiveness = 0.3,
            xlabel = "x [M]", ylabel = "y [M]", zlabel = "z [M]", title = @sprintf("a RIAF disk (h/r = 0.7, n ∝ r⁻¹·¹, Θe ∝ r⁻⁰·⁸⁴, β = 10) and a parabolic jet: %d parcels, the near half cut away", size(p_thick, 2)))
mesh!(ax3, Sphere(Point3f(0), Float32(1 + sqrt(1 - a^2))), color = :black)
lines!(ax3, [Point3f(0, 0, -30), Point3f(0, 0, 33)], color = :gray20, linewidth = 2)
# every parcel as a sphere of its scale, coloured by its peak density: the lattice through the disk and the jet sheath
# a cut-away: the half y < 0 (toward the camera) is left out, so the meridional section shows the disk's thickness and the jet cones inside
keep = [i for i in 1:size(pk, 2) if pk[2, i] >= 0]
centres = [Point3f(pk[1, i], pk[2, i], pk[3, i]) for i in keep]
logn = [pk[13, i] / log(10) for i in keep]
meshscatter!(ax3, centres, markersize = [0.8 * exp(pk[4, i]) for i in keep], color = logn, colormap = :inferno, colorrange = (2, 6.5), transparency = true, alpha = 0.7)
for ray in rays
    lines!(ax3, ray, color = (:dodgerblue, 0.95), linewidth = 1.6)
end
arrows!(ax3, [Point3f(obs * 26)], [Vec3f(obs * 6)], color = :firebrick, linewidth = 0.12, arrowsize = Vec3f(0.5, 0.5, 0.9))
text!(ax3, Point3f(obs * 34), text = "to the observer (60°)", fontsize = 12, color = :firebrick)
Colorbar(fig[2, 1], colormap = :inferno, limits = (2, 6.5), label = "log₁₀ of the parcel's peak electron density [cm⁻³] (Sgr A*'s normalization); blue: geodesics of the fourteen brightest thin-case pixels at 230 GHz", vertical = false, width = Relative(0.7), height = 12)
gl = fig[1, 2] = GridLayout()
function ticks(img; nvec = 24, pcut = 0.03, maxlen = 1.6)
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
for (r, (cube, name)) in enumerate(((cube_thin, "ten times thinner:\nn₀ = 3×10⁶ cm⁻³"), (cube_thick, "Sgr A*'s density:\nn₀ = 3×10⁷ cm⁻³")))
    Label(gl[r, 0], name, rotation = π / 2, fontsize = 14, tellheight = false)
    for (c, ν) in enumerate(freqs)
        crop = (res ÷ 2 - round(Int, 0.3 * res)):(res ÷ 2 + round(Int, 0.3 * res))            # the central 30 M of the 50 M field
        im = cube[crop, crop, 1, c]; Iv = [x[1] for x in im]
        ax = Axis(gl[r, c], title = @sprintf("%d GHz, %.2f Jy", round(Int, ν / 1e9), flux(cube, c)), aspect = DataAspect(), titlesize = 13,
                  xticksvisible = false, yticksvisible = false, xticklabelsvisible = false, yticklabelsvisible = false)
        heatmap!(ax, Iv, colormap = :afmhot, colorrange = (0, quantile(vec(Iv), 0.999)))
        s, m = ticks(im; nvec = 20); linesegments!(ax, s, color = m, colormap = :rainbow, colorrange = (0, 0.7), linewidth = 1.5)
        if r == 1 && c == 2
            inside = [idx for idx in picked if all(in(crop), Tuple(CartesianIndices(I230)[idx]))]
            scatter!(ax, [Point2f((Tuple(CartesianIndices(I230)[idx]) .- (first(crop) - 1))...) for idx in inside], color = :dodgerblue, markersize = 7, strokecolor = :white, strokewidth = 1)
            limits!(ax, 0.5, length(crop) + 0.5, 0.5, length(crop) + 0.5)
        end
    end
end
Label(fig[3, 1:2], "Broderick & Loeb (2006, 2009) and Pu & Broderick (2018): n = n₀ r⁻¹·¹ exp(−z²/2(hr)²), T = 1.7×10¹¹ K r⁻⁰·⁸⁴, B²/8π = n mₚc²/(12 β r), a Keplerian disk; here on a lattice of Gaussian parcels whose scale grows with radius, " *
      "with a jet sheath along parabolic streamlines (density ∝ z⁻², γβ from 0.2 to 1.5, Θe = 50, a toroidal field). M = 4.3×10⁶ M☉, a = 0.9375, rendered 50 M across at 0.39 M per pixel, the central 30 M shown; the tick colour is the fractional polarization. " *
      "At Sgr A*'s density the disk is optically thick at 86 GHz and the shadow is hidden by the foreground flow; it opens up as the frequency rises, and at a tenth of the density the thin ring and shadow show at every band.",
      fontsize = 12.5, tellwidth = false, word_wrap = true, justification = :left)
colsize!(fig.layout, 1, Relative(0.42))
save(joinpath(outdir, "model_structure_riaf.png"), fig, px_per_unit = 2)
println("written docs/pedagogy/model_structure_riaf.png")
