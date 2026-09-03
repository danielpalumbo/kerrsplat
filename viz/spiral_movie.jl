# Splats spiraling into a Kerr black hole, side by side with the ray-traced movie.
#
#     julia -t 8 --project=viz viz/spiral_movie.jl [--frames 240] [--res 160] [--samples 400] [--cpu]
#
# Left panel: the equatorial plane seen from the observer's side (a top view mirrored so that left
# and right agree with the image; the observer is toward the bottom), with the horizon, the splats
# as their 1σ circles and their trails, at coordinate time T. Right panel: the optically-thin
# image at the observation time t_obs = T + t̃_ref, where t̃_ref is the marcher's regularized
# lookback time at r = 6 M on the central ray, so that emission from r ≈ 6 M is seen as it was at
# time T and emission from deeper in is seen from slightly earlier (slow light).
#
# Each splat follows an analytic inspiral in quasi-Cartesian Kerr–Schild coordinates,
# r(t) = r₀ − v (t − t_birth) with the Keplerian phase rate Ω = r^(−3/2), which integrates to
# φ(t) = φ₀ + (2/v)(r(t)^(−1/2) − r₀^(−1/2)); it is born smoothly at t_birth and fades as it
# reaches the horizon. The image is rendered by the fused geodesic marcher on the GPU (or the
# CPU backend with --cpu) with the same optically-thin integrand as KerrSplat.Splats.ThinRenderer,
# but with the splat centre evaluated at the emission time of every sample.
#
# Output (not tracked): viz/output/spiral_movie.mp4 and viz/output/spiral_frames.png.

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
const NFRAMES = getopt("--frames", 240)
const RES = getopt("--res", 160)
const NSAMPLES = getopt("--samples", 400)
const USE_CPU = "--cpu" in ARGS
const SPIN = 0.94
const INCLINATION = deg2rad(60.0)
const FOV = 12.0                      # half-width of the screen in M
const T_END = 320.0                   # coordinate time covered by the movie, in M
const FRAMERATE = 24

# ---- the inspiraling splats -------------------------------------------------------------------
struct SpiralSplat{T}
    r0::T          # birth radius
    ϕ0::T          # birth azimuth (Kerr–Schild)
    v::T           # infall speed dr/dt
    t_birth::T
    σ::T           # width (isotropic, in M)
    A::T           # emissivity amplitude
    z::T           # height above the equatorial plane
end

"Centre (x, y, z), radius, and amplitude factor (birth ramp × horizon fade) of a splat at coordinate time t."
@inline function splat_state(s::SpiralSplat{T}, t, rh) where {T}
    τ = t - s.t_birth
    r = s.r0 - s.v * τ
    ramp = τ <= 0 ? zero(T) : (τ >= 10 ? one(T) : (τ / 10)^2 * (3 - 2τ / 10))
    rfade_lo = rh * T(1.1)
    fade = r <= rfade_lo ? zero(T) : (r >= rfade_lo + T(0.4) ? one(T) : ((r - rfade_lo) / T(0.4))^2 * (3 - 2(r - rfade_lo) / T(0.4)))
    r = max(r, rfade_lo)
    ϕ = s.ϕ0 + 2 / s.v * (inv(sqrt(r)) - inv(sqrt(s.r0)))
    return r * cos(ϕ), r * sin(ϕ), s.z, r, ramp * fade
end

"Fused-march consumer: the optically-thin integrand with the splats evaluated at each sample's emission time."
struct SpiralRenderer{S,V}
    splats::S      # NTuple of SpiralSplat (bits)
    t_obs::V       # one-element device array
end
Adapt.@adapt_structure SpiralRenderer

@inline function (c::SpiralRenderer)(acc, j, k, s::GeodesicSample{T}, Δτ, pix) where {T}
    met = Krang.metric(pix)
    rh = Krang.horizon(met)
    (s.ok && (rh * (1 + T(1e-3)) < s.r < T(1e3))) || return acc
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    t = @inbounds(c.t_obs[1]) - s.t
    jtot = zero(T)
    for sp in c.splats
        xc, yc, zc, _, amp = splat_state(sp, t, rh)
        amp > 0 || continue
        d2 = ((x - xc)^2 + (y - yc)^2 + (z - zc)^2) / (2 * sp.σ^2)
        jtot += sp.A * amp * exp(-d2)
    end
    jtot > 0 || return acc
    pbl = Krang.p_bl_d(met, s.r, s.θ, Krang.η(pix), Krang.λ(pix), s.νr, s.νθ)
    pzamo = Krang.jac_zamo_u_bl_d(met, s.r, s.θ) * (Krang.metric_uu(met, s.r, s.θ) * pbl)
    g = inv(pzamo[1])
    Σ = s.r * s.r + met.spin^2 * cos(s.θ)^2
    return acc + g * g * jtot * Σ * Δτ
end

const SPLATS = (
    SpiralSplat(11.0, 0.0, 0.035, 0.0, 0.9, 1.0, 0.0),
    SpiralSplat(8.5, 2.1, 0.030, 40.0, 0.7, 1.4, 0.3),
    SpiralSplat(12.5, 4.0, 0.040, 90.0, 1.1, 0.8, -0.2),
    SpiralSplat(7.0, 5.5, 0.025, 150.0, 0.6, 1.8, 0.0),
)

# ---- geodesics --------------------------------------------------------------------------------
backend = USE_CPU || !CUDA.functional() ? CPU() : CUDABackend()
println("backend: ", nameof(typeof(backend)), "; ", RES, "² pixels × ", NSAMPLES, " samples, ", NFRAMES, " frames")
camera = Geodesics.Camera((-FOV, FOV), (-FOV, FOV), RES)
cache = GeodesicCache(backend, camera, Val(NSAMPLES); store_samples = false)
t0 = time(); regenerate!(cache, SPIN, INCLINATION; marcher = Fused(64)); println("regenerate!: ", round(time() - t0, digits = 2), " s")
rh = Krang.horizon(Krang.Kerr(SPIN))

# lookback reference: t̃ at r ≈ 6 M on the central ray (stored samples of a one-pixel cache)
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
    fused_march!(SpiralRenderer(SPLATS, tvec), out, cache)
    return Array(to_screen(cache, out))
end

# ---- frames -----------------------------------------------------------------------------------
times = range(0.0, T_END, length = NFRAMES)
t0 = time()
images = [frame_image(T) for T in times]
println("rendered ", NFRAMES, " frames: ", round(time() - t0, digits = 2), " s (", round((time() - t0) / NFRAMES * 1e3, digits = 1), " ms per frame)")
vmax = 0.9 * maximum(maximum.(images))

# ---- figure -----------------------------------------------------------------------------------
# schematic coordinates: mirrored top view, observer toward the bottom (screen x = −y_KS, screen y = −x_KS)
schematic(x, y) = Point2f(-y, -x)
colors = Makie.wong_colors()[1:length(SPLATS)]
fig = Figure(size = (1500, 760), fontsize = 20, backgroundcolor = :white)
ax1 = Axis(fig[1, 1]; aspect = DataAspect(), limits = (-FOV, FOV, -FOV, FOV), xlabel = "−y  [M]", ylabel = "−x  [M]",
           title = "plasma splats in the equatorial plane (observer toward the bottom)")
ax2 = Axis(fig[1, 2]; aspect = DataAspect(), limits = (-FOV, FOV, -FOV, FOV), xlabel = "α  [M]", ylabel = "β  [M]",
           title = "ray-traced image, a = $(SPIN), θo = $(round(Int, rad2deg(INCLINATION)))°")
# horizon and photon-orbit-ish guide in the schematic
poly!(ax1, Circle(Point2f(0, 0), Float32(rh)); color = :black)
lines!(ax1, [Point2f(0, -FOV + 2.4), Point2f(0, -FOV + 1.0)]; color = :gray, linewidth = 2)
scatter!(ax1, [Point2f(0, -FOV + 0.9)]; marker = :dtriangle, color = :gray, markersize = 16)
text!(ax1, 0.4, -FOV + 1.6; text = "to observer", align = (:left, :center), color = :gray, fontsize = 16)
positions = [Observable(Point2f[]) for _ in SPLATS]
trails = [Observable(Point2f[]) for _ in SPLATS]
sizes = [Observable(Float32[]) for _ in SPLATS]
for (i, s) in enumerate(SPLATS)
    lines!(ax1, trails[i]; color = (colors[i], 0.5), linewidth = 2)
    scatter!(ax1, positions[i]; color = (colors[i], 0.85), markersize = sizes[i], markerspace = :data, strokecolor = :black, strokewidth = 1)
end
timelabel = Observable("T = 0 M")
text!(ax1, -FOV + 0.5, FOV - 0.5; text = timelabel, align = (:left, :top), fontsize = 18)
img = Observable(images[1])
αs = range(-FOV, FOV, length = RES); βs = range(-FOV, FOV, length = RES)
heatmap!(ax2, αs, βs, img; colormap = :afmhot, colorrange = (0, vmax), colorscale = sqrt)
Colorbar(fig[1, 3]; colormap = :afmhot, limits = (0, vmax), scale = sqrt, label = "intensity (arbitrary units, √ stretch)")
text!(ax2, -FOV + 0.5, FOV - 0.5; text = "slow light: observed at t_obs = T + ($(round(TLOOKBACK_REF, digits = 1))) M", align = (:left, :top), color = :white, fontsize = 16)

function update_frame!(iframe)
    T = times[iframe]
    for (i, s) in enumerate(SPLATS)
        xc, yc, _, r, amp = splat_state(s, T, rh)
        if amp > 0
            positions[i][] = [schematic(xc, yc)]
            sizes[i][] = [Float32(2 * s.σ)]
            trail = Point2f[]
            for Tp in range(max(s.t_birth, T - 60.0), T, length = 60)
                xp, yp, _, _, ap = splat_state(s, Tp, rh)
                ap > 0 && push!(trail, schematic(xp, yp))
            end
            trails[i][] = trail
        else
            positions[i][] = Point2f[]
            sizes[i][] = Float32[]
            trails[i][] = Point2f[]
        end
    end
    timelabel[] = "T = $(round(Int, T)) M"
    img[] = images[iframe]
    return nothing
end

mkpath(joinpath(@__DIR__, "output"))
moviepath = joinpath(@__DIR__, "output", "spiral_movie.mp4")
t0 = time()
CairoMakie.record(fig, moviepath, 1:NFRAMES; framerate = FRAMERATE) do iframe
    update_frame!(iframe)
end
println("wrote ", moviepath, " (", round(time() - t0, digits = 1), " s)")

# a contact sheet of six frames for a quick look
sheet = Figure(size = (1500, 2 * 760 * 3 ÷ 2), fontsize = 14)
for (n, iframe) in enumerate(round.(Int, range(1, NFRAMES, length = 6)))
    update_frame!(iframe)
    save(joinpath(@__DIR__, "output", "frame_$(n).png"), fig)
end
println("wrote six frames to ", joinpath(@__DIR__, "output"))
