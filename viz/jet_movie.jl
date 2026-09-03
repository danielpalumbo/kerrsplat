# Pairs of plasma blobs spiral into a Kerr black hole on the midplane, split inside the ergosphere,
# and one member of each pair whips up the spin axis on an accelerating out-spiral, beside the
# ray-traced slow-light movie. Evocative of Penrose-process / Blandford–Znajek energy extraction,
# not a physical model of either.
#
#     julia -t 8 --project=viz viz/jet_movie.jl [--frames 300] [--res 320] [--samples 400] [--cpu]
#
# Left panel: a 3D view from the observer's side (camera azimuth and elevation match the
# inclination; the y axis is mirrored so that left and right agree with the image), with the
# horizon, the ergosphere, the spin axis, the blobs as spheres and their trails, at coordinate
# time T. Right panel: the optically-thin image at the observation time t_obs = T + t̃_ref,
# where t̃_ref is the marcher's regularized lookback time at r = 6 M on the central ray (slow
# light, as in spiral_movie.jl).
#
# Trajectories are analytic and piecewise in coordinate time, in quasi-Cartesian Kerr–Schild
# coordinates. The centre of a pair inspirals as r(t) = r₀ − v (t − t_birth) with the Keplerian
# phase rate Ω = r^(−3/2); its two members orbit the centre at a tightening separation. When
# the centre reaches R_SPLIT (inside the ergosphere, whose equatorial radius is 2 M) the pair
# splits: one member plunges through the horizon on a faster inspiral and fades, the other is
# launched along the spin axis (alternately north and south) on a cone, z ∝ v_z τ + ½ a_z τ²,
# winding rapidly at first and more slowly further out, brightening as it goes and fading as it
# leaves the field of view. The image is rendered by the fused geodesic marcher with the same
# optically-thin integrand as KerrSplat.Splats.ThinRenderer, every blob evaluated at the
# emission time of each sample.
#
# Output (not tracked): viz/output/jet_movie.mp4 and viz/output/jet_frame_1..6.png.

using Adapt
using CairoMakie
using CUDA
using KernelAbstractions
using Krang
using StaticArrays
using KerrSplat.Geodesics
using KerrSplat.Splats
const KA = KernelAbstractions

# ---- options ---------------------------------------------------------------------------------
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
const NFRAMES = getopt("--frames", 300)
const RES = getopt("--res", 320)
const NSAMPLES = getopt("--samples", 400)
const USE_CPU = "--cpu" in ARGS
const SPIN = 0.94
const INCLINATION = deg2rad(60.0)
const FOV = 12.0                      # half-width of the screen and of the 3D box, in M
const T_END = 400.0                   # coordinate time covered by the movie, in M
const FRAMERATE = 24

# ---- trajectory constants ---------------------------------------------------------------------
const R_SPLIT = 1.9                   # BL radius at which a pair splits (ergosphere: r = 2 on the equator)
const V_PLUNGE = 0.03                 # infall speed of the captured member after the split
const V_Z = 0.03                      # initial vertical speed of the escaping member
const A_Z = 0.0006                    # its vertical acceleration
const K_RHO = 0.25                    # cone: cylindrical radius = R_SPLIT + K_RHO |z|
const W_JET = 4.0                     # azimuthal winding of the jet member, ϕ = ϕ_split + W_JET ln(1 + τ/10)
const Z_FADE = 10.5                   # the jet member fades between |z| = Z_FADE and Z_FADE + 2

smoothstep(u::T) where {T} = u <= 0 ? zero(T) : (u >= 1 ? one(T) : u * u * (3 - 2u))

struct BlobPair{T}
    r0::T          # birth radius of the pair's centre
    ϕ0::T          # birth azimuth
    v::T           # infall speed of the centre
    t_birth::T
    d::T           # separation of the two members at birth
    ψ0::T          # orientation of the pair at birth
    ω::T           # rotation rate of the pair about its centre
    σ::T           # width of each member (isotropic, in M)
    A::T           # emissivity amplitude
    jet::T         # +1: the escaping member goes north, −1: south
end

split_time(p::BlobPair) = p.t_birth + (p.r0 - R_SPLIT) / p.v

"Centre (x, y, z) and amplitude factor of member m (1 plunges, 2 escapes) of a pair at coordinate time t."
@inline function member_state(p::BlobPair{T}, m::Int, t, rh) where {T}
    τ = t - p.t_birth
    τ <= 0 && return (zero(T), zero(T), zero(T), zero(T))
    ramp = smoothstep(τ / 10)
    τsplit = (p.r0 - R_SPLIT) / p.v
    sgn = m == 1 ? one(T) : -one(T)
    if τ < τsplit
        r = p.r0 - p.v * τ
        ϕ = p.ϕ0 + 2 / p.v * (inv(sqrt(r)) - inv(sqrt(p.r0)))
        dh = p.d / 2 * (T(0.3) + T(0.7) * (r - R_SPLIT) / (p.r0 - R_SPLIT))
        ψ = p.ψ0 + p.ω * τ
        return (r * cos(ϕ) + sgn * dh * cos(ψ), r * sin(ϕ) + sgn * dh * sin(ψ), zero(T), ramp)
    end
    τs = τ - τsplit
    ϕs = p.ϕ0 + 2 / p.v * (inv(sqrt(R_SPLIT)) - inv(sqrt(p.r0)))
    ψs = p.ψ0 + p.ω * τsplit
    dh = p.d / 2 * T(0.3)
    if m == 1
        r = R_SPLIT - V_PLUNGE * τs
        rlo = rh * T(1.1)
        fade = smoothstep((r - rlo) / T(0.4))
        r = max(r, rlo)
        ϕ = ϕs + 2 / V_PLUNGE * (inv(sqrt(r)) - inv(sqrt(R_SPLIT)))
        return (r * cos(ϕ) + dh * cos(ψs), r * sin(ϕ) + dh * sin(ψs), zero(T), fade)
    else
        z = p.jet * (V_Z * τs + A_Z / 2 * τs^2)
        ρ = R_SPLIT + K_RHO * abs(z)
        ϕ = ϕs + W_JET * log1p(τs / 10)
        fade = 1 - smoothstep((abs(z) - Z_FADE) / 2)
        amp = fade * (1 + T(0.6) * min(τs / 20, one(T)))
        return (ρ * cos(ϕ) - dh * cos(ψs), ρ * sin(ϕ) - dh * sin(ψs), z, amp)
    end
end

"Fused-march consumer: the optically-thin integrand with every blob evaluated at each sample's emission time."
struct JetRenderer{S,V}
    pairs::S       # NTuple of BlobPair (bits)
    t_obs::V       # one-element device array
end
Adapt.@adapt_structure JetRenderer

@inline function (c::JetRenderer)(acc, j, k, s::GeodesicSample{T}, Δτ, pix) where {T}
    met = Krang.metric(pix)
    rh = Krang.horizon(met)
    (s.ok && (rh * (1 + T(1e-3)) < s.r < T(1e3))) || return acc
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    t = @inbounds(c.t_obs[1]) - s.t
    jtot = zero(T)
    for p in c.pairs
        for m in 1:2
            xc, yc, zc, amp = member_state(p, m, t, rh)
            amp > 0 || continue
            d2 = ((x - xc)^2 + (y - yc)^2 + (z - zc)^2) / (2 * p.σ^2)
            jtot += p.A * amp * exp(-d2)
        end
    end
    jtot > 0 || return acc
    pbl = Krang.p_bl_d(met, s.r, s.θ, Krang.η(pix), Krang.λ(pix), s.νr, s.νθ)
    pzamo = Krang.jac_zamo_u_bl_d(met, s.r, s.θ) * (Krang.metric_uu(met, s.r, s.θ) * pbl)
    g = inv(pzamo[1])
    Σ = s.r * s.r + met.spin^2 * cos(s.θ)^2
    return acc + g * g * jtot * Σ * Δτ
end

const PAIRS = (
    BlobPair(10.0, 0.0, 0.050, 0.0, 1.6, 0.0, 0.12, 0.65, 1.0, 1.0),
    BlobPair(8.0, 2.5, 0.040, 50.0, 1.4, 1.0, 0.15, 0.55, 1.3, -1.0),
    BlobPair(11.0, 4.5, 0.060, 100.0, 1.8, 2.0, 0.10, 0.75, 0.8, 1.0),
    BlobPair(7.0, 1.2, 0.035, 150.0, 1.2, 0.5, 0.18, 0.5, 1.5, -1.0),
)

# ---- geodesics --------------------------------------------------------------------------------
backend = USE_CPU || !CUDA.functional() ? CPU() : CUDABackend()
println("backend: ", nameof(typeof(backend)), "; ", RES, "² pixels × ", NSAMPLES, " samples, ", NFRAMES, " frames")
camera = Geodesics.Camera((-FOV, FOV), (-FOV, FOV), RES)
cache = GeodesicCache(backend, camera, Val(NSAMPLES); store_samples = false)
t0 = time(); regenerate!(cache, SPIN, INCLINATION; marcher = Fused(64)); println("regenerate!: ", round(time() - t0, digits = 2), " s")
const MET = Krang.Kerr(SPIN)
rh = Krang.horizon(MET)

let c1 = GeodesicCache(CPU(), Geodesics.Camera([0.01], [0.01]), Val(NSAMPLES))
    regenerate!(c1, SPIN, INCLINATION; marcher = Recurrence(64))
    S = host(c1.samples)
    k = argmin(abs.([S[1, k].r for k in 1:NSAMPLES] .- 6.0))
    global const TLOOKBACK_REF = S[1, k].t
end
println("t̃ at r ≈ 6 M on the central ray: ", round(TLOOKBACK_REF, digits = 3), " M")

tvec = KA.allocate(backend, Float64, 1)
out = KA.allocate(backend, Float64, npixels(cache))
function frame_image(T)
    fill!(tvec, T + TLOOKBACK_REF)
    fused_march!(JetRenderer(PAIRS, tvec), out, cache)
    return Array(to_screen(cache, out))
end

# ---- frames -----------------------------------------------------------------------------------
times = range(0.0, T_END, length = NFRAMES)
t0 = time()
images = [frame_image(T) for T in times]
println("rendered ", NFRAMES, " frames: ", round(time() - t0, digits = 2), " s (", round((time() - t0) / NFRAMES * 1e3, digits = 1), " ms per frame)")
vmax = 0.9 * maximum(maximum.(images))

# ---- figure -----------------------------------------------------------------------------------
# 3D scene coordinates: y mirrored so that left and right agree with the image; the camera sits on
# the +x side (observer's side), 30° above the plane, i.e. at the 60° inclination of the image.
scene3(x, y, z) = Point3f(x, -y, z)
white(α) = (:white, α)
colors = Makie.wong_colors()[1:length(PAIRS)]
fig = Figure(size = (1500, 760), fontsize = 20, backgroundcolor = :black)
ax1 = Axis3(fig[1, 1]; aspect = :data, limits = (-FOV, FOV, -FOV, FOV, -FOV, FOV), azimuth = 0.0, elevation = π / 2 - INCLINATION,
            perspectiveness = 0.4, xlabel = "x  [M]", ylabel = "−y  [M]", zlabel = "z  [M]",
            title = "blob pairs: inspiral, split in the ergosphere, plunge and jet",
            backgroundcolor = :black, xypanelcolor = :black, xzpanelcolor = :black, yzpanelcolor = :black,
            xgridcolor = white(0.35), ygridcolor = white(0.35), zgridcolor = white(0.35),
            xspinecolor_1 = white(0.6), xspinecolor_2 = white(0.6), xspinecolor_3 = white(0.6),
            yspinecolor_1 = white(0.6), yspinecolor_2 = white(0.6), yspinecolor_3 = white(0.6),
            zspinecolor_1 = white(0.6), zspinecolor_2 = white(0.6), zspinecolor_3 = white(0.6),
            xticklabelcolor = :white, yticklabelcolor = :white, zticklabelcolor = :white,
            xlabelcolor = :white, ylabelcolor = :white, zlabelcolor = :white, titlecolor = :white)
ax2 = Axis(fig[1, 2]; aspect = DataAspect(), limits = (-FOV, FOV, -FOV, FOV), xlabel = "α  [M]", ylabel = "β  [M]",
           title = "ray-traced image, a = $(SPIN), θo = $(round(Int, rad2deg(INCLINATION)))°",
           backgroundcolor = :black, xticklabelcolor = :white, yticklabelcolor = :white, xlabelcolor = :white, ylabelcolor = :white,
           titlecolor = :white, xtickcolor = :white, ytickcolor = :white, bottomspinecolor = :white, topspinecolor = :white,
           leftspinecolor = :white, rightspinecolor = :white, xgridvisible = false, ygridvisible = false)

# horizon (a sphere of radius r₊ in these coordinates), ergosphere wireframe, spin axis
mesh!(ax1, Sphere(Point3f(0), Float32(rh)); color = :gray35)
ergo(θ) = 1 + sqrt(1 - SPIN^2 * cos(θ)^2)
for ϕ in range(0, 2π, length = 13)[1:end-1]
    lines!(ax1, [scene3(ergo(θ) * sin(θ) * cos(ϕ), ergo(θ) * sin(θ) * sin(ϕ), ergo(θ) * cos(θ)) for θ in range(0, π, length = 61)]; color = (:mediumpurple, 0.45), linewidth = 1)
end
for θ in range(π / 6, 5π / 6, length = 5)
    lines!(ax1, [scene3(ergo(θ) * sin(θ) * cos(ϕ), ergo(θ) * sin(θ) * sin(ϕ), ergo(θ) * cos(θ)) for ϕ in range(0, 2π, length = 91)]; color = (:mediumpurple, 0.45), linewidth = 1)
end
lines!(ax1, [Point3f(0, 0, -FOV), Point3f(0, 0, FOV)]; color = white(0.35), linestyle = :dash, linewidth = 1)
text!(ax1, Point3f(0, 0, FOV); text = "spin axis", color = white(0.6), fontsize = 14, align = (:left, :bottom))
text!(ax1, scene3(0.0, -2.6, -0.8); text = "ergosphere", color = (:mediumpurple, 0.9), fontsize = 14, align = (:left, :top))

# blobs, trails and split flashes
positions = [Observable(Point3f[]) for _ in PAIRS, _ in 1:2]
trails = [Observable(Point3f[]) for _ in PAIRS, _ in 1:2]
flashes = Observable(Point3f[]); flashsizes = Observable(Float32[])
for (i, p) in enumerate(PAIRS), m in 1:2
    lines!(ax1, trails[i, m]; color = (colors[i], 0.6), linewidth = m == 1 ? 1.5 : 2.5)
    meshscatter!(ax1, positions[i, m]; color = m == 1 ? colors[i] : Makie.RGBAf(0.55 .+ 0.45 .* (colors[i].r, colors[i].g, colors[i].b)..., 1.0),
                 markersize = Float32(1.2 * p.σ))
end
scatter!(ax1, flashes; color = white(0.7), markersize = flashsizes, marker = :circle, strokewidth = 0)
timelabel = Observable("T = 0 M")
Label(fig[1, 1, TopLeft()], timelabel; color = :white, fontsize = 18, halign = :left, padding = (60, 0, -40, 0), tellwidth = false)

img = Observable(images[1])
αs = range(-FOV, FOV, length = RES); βs = range(-FOV, FOV, length = RES)
heatmap!(ax2, αs, βs, img; colormap = :afmhot, colorrange = (0, vmax), colorscale = sqrt)
Colorbar(fig[1, 3]; colormap = :afmhot, limits = (0, vmax), scale = sqrt, label = "intensity (arbitrary units, √ stretch)",
         labelcolor = :white, ticklabelcolor = :white, tickcolor = :white)
text!(ax2, -FOV + 0.5, FOV - 0.5; text = "slow light: observed at t_obs = T + ($(round(TLOOKBACK_REF, digits = 1))) M", align = (:left, :top), color = :white, fontsize = 16)

function update_frame!(iframe)
    T = times[iframe]
    fl = Point3f[]; fs = Float32[]
    for (i, p) in enumerate(PAIRS)
        for m in 1:2
            xc, yc, zc, amp = member_state(p, m, T, rh)
            if amp > 0 && abs(zc) < FOV - 1.0        # hide a blob once it leaves the 3D box
                positions[i, m][] = [scene3(xc, yc, zc)]
                trail = Point3f[]
                for Tp in range(max(p.t_birth, T - 80.0), T, length = 120)
                    xp, yp, zp, ap = member_state(p, m, Tp, rh)
                    ap > 0 && push!(trail, scene3(xp, yp, zp))
                end
                trails[i, m][] = trail
            else
                positions[i, m][] = Point3f[]
                trails[i, m][] = Point3f[]
            end
        end
        τs = T - split_time(p)
        if 0 <= τs <= 12
            xs, ys, _, _ = member_state(p, 2, split_time(p), rh)
            push!(fl, scene3(xs, ys, 0.0)); push!(fs, Float32(50 * (1 - τs / 12)))
        end
    end
    flashes[] = fl; flashsizes[] = fs
    timelabel[] = "T = $(round(Int, T)) M"
    img[] = images[iframe]
    return nothing
end

mkpath(joinpath(@__DIR__, "output"))
moviepath = joinpath(@__DIR__, "output", "jet_movie.mp4")
t0 = time()
CairoMakie.record(fig, moviepath, 1:NFRAMES; framerate = FRAMERATE) do iframe
    update_frame!(iframe)
end
println("wrote ", moviepath, " (", round(time() - t0, digits = 1), " s)")

for (n, iframe) in enumerate(round.(Int, range(1, NFRAMES, length = 6)))
    update_frame!(iframe)
    save(joinpath(@__DIR__, "output", "jet_frame_$(n).png"), fig)
end
println("wrote six frames to ", joinpath(@__DIR__, "output"))
