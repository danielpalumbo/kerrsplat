import Pkg
Pkg.activate(@__DIR__)

using Krang, Enzyme, StaticArrays, FiniteDifferences
using Printf

# ---------------------------------------------------------------------------
# Toy forward model: one optically-thin Gaussian "splat" of emissivity placed
# in the Kerr spacetime, volume-rendered along analytic Krang geodesics by
# marching in Mino time (slow-light pixels, so emission time ts is available).
# Splat params p = [x0, y0, z0, log_sigma, log_amplitude]
# ---------------------------------------------------------------------------

function ray_intensity(p::AbstractVector{T}, pix, N::Int) where {T}
    met = Krang.metric(pix)
    rh = Krang.horizon(met)
    tauf = Krang.total_mino_time(pix)
    dtau = tauf / (N + 1)
    sigma = exp(p[4])
    amp = exp(p[5])
    acc = zero(T)
    for i = 1:N
        tau = dtau * i
        ts, rs,ths, phs, nur, nuth, ok = Krang.emission_coordinates(pix, tau)
        if ok && (rh * (1 + 1e-3) < rs < 1e3)
            x, y, z = Krang.boyer_lindquist_to_quasi_cartesian_kerr_schild_fast_light(
                met, rs, ths, phs)
            d2 = ((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2) / (2 * sigma^2)
            # crude redshift-like weight exercising the momentum utilities:
            pbl = Krang.p_bl_d(met, rs, ths, Krang.η(pix), Krang.λ(pix), nur, nuth)
            pzamo = Krang.jac_zamo_u_bl_d(met, rs, ths) *
                    (Krang.metric_uu(met, rs, ths) * pbl)
            g = inv(pzamo[1])
            acc += amp * exp(-d2) * g^3 * dtau
        end
    end
    return acc
end

function image_intensity(p::AbstractVector{T}, pixels, N::Int) where {T}
    s = zero(T)
    for pix in pixels
        s += ray_intensity(p, pix, N)
    end
    return s
end

met = Krang.Kerr(0.94)
thetao = 60 * pi / 180
# one scattering ray, one plunging ray, one nearly-critical ray
pixels = (
    Krang.SlowLightIntensityPixel(met, -6.0, 4.0, thetao),
    Krang.SlowLightIntensityPixel(met, -2.0, 1.5, thetao),
    Krang.SlowLightIntensityPixel(met, -5.0, 1.0, thetao),
)

p0 = [6.0, 4.0, 0.5, log(1.5), log(2.0)]
N = 200

f(p) = image_intensity(p, pixels, N)

println("== primal ==")
t0 = time()
I0 = f(p0)
@printf("I = %.12e   (%.3f s first call incl. compile)\n", I0, time() - t0)
t0 = time(); I0 = f(p0)
@printf("I = %.12e   (%.6f s warm)\n", I0, time() - t0)

println("\n== Enzyme reverse-mode gradient w.r.t. 5 splat params ==")
t0 = time()
gE = Enzyme.gradient(Reverse, f, p0)[1]
@printf("compile+run: %.3f s\n", time() - t0)
t0 = time()
gE = Enzyme.gradient(Reverse, f, p0)[1]
tg = time() - t0
println("grad  = ", gE)
@printf("warm gradient time: %.6f s  (%.1fx primal)\n", tg, tg / (time() - t0 + eps()))

println("\n== check vs central finite differences ==")
fdm = central_fdm(5, 1)
gFD = FiniteDifferences.grad(fdm, f, p0)[1]
println("gradFD= ", gFD)
relerr = maximum(abs.(gE .- gFD) ./ (abs.(gFD) .+ 1e-30))
@printf("max rel err = %.3e\n", relerr)

println("\n== Enzyme reverse-mode w.r.t. spin (through pixel construction) ==")
function spin_intensity(a::T, p, N) where {T}
    m = Krang.Kerr(a)
    th = T(60 * pi / 180)
    pxs = (
        Krang.SlowLightIntensityPixel(m, T(-6.0), T(4.0), th),
        Krang.SlowLightIntensityPixel(m, T(-2.0), T(1.5), th),
        Krang.SlowLightIntensityPixel(m, T(-5.0), T(1.0), th),
    )
    s = zero(T)
    for pix in pxs
        s += ray_intensity(p, pix, N)
    end
    return s
end

ga = try
    t0 = time()
    da = Enzyme.autodiff(Enzyme.Reverse, spin_intensity, Active,
                         Active(0.94), Const(p0), Const(N))[1][1]
    @printf("dI/da (Enzyme reverse) = %.10e   (%.3f s incl. compile)\n", da, time() - t0)
    da
catch err
    println("Enzyme reverse through pixel construction FAILED:")
    showerror(stdout, err)
    println()
    nothing
end

t0 = time()
daFD = fdm(a -> spin_intensity(a, p0, N), 0.94)
@printf("dI/da (finite diff)    = %.10e   (%.3f s)\n", daFD, time() - t0)
if ga !== nothing
    @printf("rel err = %.3e\n", abs(ga - daFD) / abs(daFD))
end

println("\nDONE")
