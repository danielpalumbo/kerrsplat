import Pkg
Pkg.activate(@__DIR__)

using Krang, Enzyme, StaticArrays, FiniteDifferences
using Printf

# ---------------------------------------------------------------------------
# Smoke test 2: "one-zone splat" — a single Gaussian splat carrying its OWN
# fluid velocity (speed + direction in the ZAMO frame) and its OWN magnetic
# field orientation, rendered as polarized (I,Q,U) emission along Krang
# slow-light geodesics via Krang.synchrotronPolarization (Walker-Penrose
# transport + ZAMO->fluid boosts). Reverse-mode Enzyme gradient w.r.t. ALL
# 10 per-splat parameters, checked against finite differences.
#
# p = [x0, y0, z0, log_sigma, log_amp, atanh_beta, theta_z, phi_z, b_theta, b_phi]
#      splat position/shape/amp     |  fluid velocity (ZAMO)  |  B orientation
# ---------------------------------------------------------------------------

function ray_stokes(p::AbstractVector{T}, pix, N::Int) where {T}
    met = Krang.metric(pix)
    rh = Krang.horizon(met)
    thetao = Krang.inclination(pix)
    alpha, beta = Krang.screen_coordinate(pix)
    tauf = Krang.total_mino_time(pix)
    dtau = tauf / (N + 1)

    sigma = exp(p[4])
    amp = exp(p[5])
    betav = tanh(p[6])                       # fluid speed in (0,1)
    bfield = SVector(sin(p[9]) * cos(p[10]), sin(p[9]) * sin(p[10]), cos(p[9]))
    betafluid = SVector(betav, p[7], p[8])   # Krang convention (speed, incl, az)
    specidx = one(T)                         # fixed spectral index

    Iacc = zero(T); Qacc = zero(T); Uacc = zero(T)
    for i = 1:N
        tau = dtau * i
        ts, rs, ths, phs, nur, nuth, ok = Krang.emission_coordinates(pix, tau)
        if ok && (rh * (1 + 1e-3) < rs < 1e3)
            x, y, z = Krang.boyer_lindquist_to_quasi_cartesian_kerr_schild_fast_light(
                met, rs, ths, phs)
            w = amp * exp(-((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2) / (2 * sigma^2))
            ea, eb, redshift, lp = Krang.synchrotronPolarization(
                met, alpha, beta, rs, ths, thetao, bfield, betafluid, nur, nuth)
            qt = -(ea^2 - eb^2) + eps(T)
            ut = -2 * ea * eb + eps(T)
            mag = hypot(qt, ut)
            ii = mag^(1 + specidx) * min(lp, T(1e2)) * w * redshift^(3 + specidx)
            Iacc += ii * dtau
            Qacc += ii * qt / mag * dtau
            Uacc += ii * ut / mag * dtau
        end
    end
    return Iacc, Qacc, Uacc
end

function loss(p::AbstractVector{T}, pixels, N::Int) where {T}
    s = zero(T)
    for pix in pixels
        I, Q, U = ray_stokes(p, pix, N)
        s += I + T(0.3) * Q - T(0.2) * U   # arbitrary functional of Stokes map
    end
    return s
end

met = Krang.Kerr(0.94)
thetao = 60 * pi / 180
pixels = (
    Krang.SlowLightIntensityPixel(met, -6.0, 4.0, thetao),
    Krang.SlowLightIntensityPixel(met, -2.0, 1.5, thetao),
    Krang.SlowLightIntensityPixel(met, -5.0, 1.0, thetao),
)

p0 = [6.0, 4.0, 0.5, log(1.5), log(2.0),
      atanh(0.5), 1.0, -0.7, 1.1, 0.4]
N = 200

f(p) = loss(p, pixels, N)

println("== primal (polarized one-zone splat) ==")
t0 = time(); v = f(p0)
@printf("loss = %.12e  (%.2f s incl. compile)\n", v, time() - t0)
t0 = time(); v = f(p0)
@printf("loss = %.12e  (%.4f s warm)\n", v, time() - t0)

println("\n== Enzyme reverse gradient w.r.t. all 10 one-zone params ==")
t0 = time()
g = Enzyme.gradient(Reverse, f, p0)[1]
@printf("compile+run: %.2f s\n", time() - t0)
t0 = time()
g = Enzyme.gradient(Reverse, f, p0)[1]
@printf("warm: %.4f s\n", time() - t0)
for (i, name) in enumerate(["x0","y0","z0","logσ","logA","atanhβ","θz","φz","bθ","bφ"])
    @printf("  d/d%-7s = % .10e\n", name, g[i])
end

println("\n== finite-difference check ==")
gFD = FiniteDifferences.grad(central_fdm(5, 1), f, p0)[1]
relerr = maximum(abs.(g .- gFD) ./ (abs.(gFD) .+ 1e-30))
@printf("max rel err (Enzyme vs FD) = %.3e\n", relerr)

println("\nDONE")
