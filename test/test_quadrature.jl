# Gate 3 of the GPU plan (§8): t̃(τ) and φ(τ) from the anchored quadrature of the Mino-time
# rates, on random pixels, bands around the critical curve and columns grazing the polar axis,
# on a KernelAbstractions backend. Two references:
#
# 1. Krang's closed forms at every sample (independent method: elliptic integrals), on every
#    non-vortical pixel. Criterion |Δ| ≤ tol + SENS_FACTOR × (one-ulp input sensitivity of
#    Krang's value), as in gate 1, because Krang's φ is ill-conditioned on axis-grazing rays.
#    Vortical rays (η < 0) are excluded from this comparison: Krang's φ and t̃ are inconsistent
#    with its own θ(τ) after the polar turning point of those rays with β < 0
#    (docs/notes/upstream_issues.md), so the quadrature anchored to them inherits the error.
# 2. A BigFloat 32-point Gauss–Legendre integration of the rates on the BigFloat closed-form
#    coordinates (test/highprec_reference.jl), for a subset: the pixels where the two Float64
#    paths disagree most, a few random ones, and every vortical one (there with re-anchoring
#    switched off, so that the quadrature itself is judged). Compared on samples inside 50 M as
#    increments from the first such sample: outside, Krang's I0_inf (accurate to ~1e-12) shifts
#    the Float64 trajectory in τ and dt/dτ ≈ r² turns that into 1e-8 absolute differences that
#    say nothing about either method. Samples inside r₊(1 + 0.15) are excluded as well: in the
#    final plunge of a near-extremal black hole the horizon pole's residual is not smooth on the
#    sample spacing (a = 0.999: 2e-7 in t̃ at r − r₊ = 0.01 M and 1e-8 at r = 1.1 r₊, where Krang itself is at 2e-8 and 1e-9),
#    and every consumer discards those samples anyway (plan §10).
#
# Anchor residual statistics are reported.

using Test
using Random
using Statistics
using KernelAbstractions
using Krang
using KerrSplat.Geodesics

"Random pixels, near-critical bands and axis-grazing columns for gate 3."
function gate3_camera(a, θo, rng; nrandom = 200, nψ = 6, δs = (1e-2, 1e-3, 1e-4), ncol = 24)
    cam = gate2_camera(a, θo, rng; nrandom = nrandom, nψ = nψ, δs = δs)
    αs = copy(cam.αs)
    βs = copy(cam.βs)
    met = Krang.Kerr(a)
    for β in range(-10.0, 10.0, length = ncol), α in (0.05, -0.05, 0.2)   # small |λ| = |α sin θo|
        isfinite(Krang.SlowLightIntensityPixel(met, α, β, θo).total_mino_time) || continue
        push!(αs, α)
        push!(βs, β)
    end
    # a few central pixels, some of them vortical
    for (α, β) in ((-0.21, -0.21), (0.21, -0.21), (0.3, 0.2), (-0.4, 0.1))
        push!(αs, α)
        push!(βs, β)
    end
    return Camera(αs, βs)
end

const GATE3_SPACETIMES = ((0.2, 45.0), (0.5, 17.0), (0.94, 1.0), (0.94, 60.0), (0.94, 89.0), (0.999, 30.0), (0.7, 120.0))
const GATE3_RMAX = 50.0
const GATE3_HORIZON_MARGIN = 0.15     # samples with r < r₊ (1 + margin) are not compared

"""
Per-pixel comparison of the increments of t̃ and φ (from the first sample inside GATE3_RMAX)
with the BigFloat rate reference, for the quadrature samples `S` and Krang's direct samples
`D`: returns (t̃ quadrature, φ quadrature, t̃ Krang, φ Krang) maximum absolute errors.
"""
function highprec_increment_errors(S::GeodesicSamples, D::GeodesicSamples, j, ray, λb, τs, rmin)
    N = length(τs)
    Δt, Δϕ = highprec_increments(ray, λb, τs)
    k1 = findfirst(k -> D[j, k].ok && D[j, k].r <= GATE3_RMAX, 1:N)
    k1 === nothing && return 0.0, 0.0, 0.0, 0.0
    e = zeros(4)
    for k in k1+1:N
        (D[j, k].ok && rmin <= D[j, k].r <= GATE3_RMAX) || continue
        e[1] = max(e[1], abs((S[j, k].t - S[j, k1].t) - (Δt[k] - Δt[k1])))
        e[2] = max(e[2], abs((S[j, k].ϕ - S[j, k1].ϕ) - (Δϕ[k] - Δϕ[k1])))
        e[3] = max(e[3], abs((D[j, k].t - D[j, k1].t) - (Δt[k] - Δt[k1])))
        e[4] = max(e[4], abs((D[j, k].ϕ - D[j, k1].ϕ) - (Δϕ[k] - Δϕ[k1])))
    end
    return e[1], e[2], e[3], e[4]
end

"""
    test_quadrature(backend; N, M, tol_ϕ, tol_t, tol_hp_ϕ, tol_hp_t, label)

Tolerances are absolute (φ in radians, t̃ in M). Against Krang: `tol_ϕ`, `tol_t` on top of
SENS_FACTOR × sensitivity (Krang's closed forms carry isolated glitches of order 1e-7 in t̃,
docs/notes/). Against the BigFloat rate reference: `tol_hp_ϕ`, `tol_hp_t`, or twice Krang's own
error against it where the shared Float64 inputs limit both (near-critical rays).
"""
function test_quadrature(backend; N::Int, M::Int, tol_ϕ::Float64, tol_t::Float64, tol_hp_ϕ::Float64, tol_hp_t::Float64, label::String)
    rng = MersenneTwister(3)
    for (a, θdeg) in GATE3_SPACETIMES
        θo = deg2rad(θdeg)
        camera = gate3_camera(a, θo, rng)
        cache = GeodesicCache(backend, camera, Val(N))
        met = Krang.Kerr(a)
        @testset "$label a=$a θo=$(θdeg)° ($(npixels(camera)) px)" begin
            regenerate!(cache, a, θo; marcher = Direct())
            D = host(cache.samples)
            @test all(iszero, Array(cache.residual_t))
            regenerate!(cache, a, θo; marcher = Recurrence(M))
            S = host(cache.samples)
            pcs = host(cache.consts)
            rt = Array(cache.residual_t)
            rϕ = Array(cache.residual_ϕ)
            pertpix = perturbed_pixels(cache, camera, a, θo)
            vortical = pcs.η .< 0
            eϕ = zeros(npixels(cache))
            et = zeros(npixels(cache))
            xϕ = zeros(npixels(cache))     # excess over SENS_FACTOR × sensitivity
            xt = zeros(npixels(cache))
            nflag = 0
            nvalid = 0
            for j in 1:npixels(cache)
                perts = map(p -> ref_samples(p, Val(N)), pertpix[j])
                for k in 1:N
                    d = D[j, k]
                    s = S[j, k]
                    (d.ok, d.νr) == (s.ok, s.νr) || (nflag += 1)
                    d.ok || continue
                    nvalid += 1
                    @test isfinite(s.t) && isfinite(s.ϕ)
                    e1 = abs(s.ϕ - d.ϕ)
                    e2 = abs(s.t - d.t)
                    sϕ = maximum(abs(p[k].ϕ - d.ϕ) for p in perts if p[k].ok; init = 0.0)
                    st = maximum(abs(p[k].t - d.t) for p in perts if p[k].ok; init = 0.0)
                    eϕ[j] = max(eϕ[j], e1)
                    et[j] = max(et[j], e2)
                    xϕ[j] = max(xϕ[j], e1 - SENS_FACTOR * sϕ)
                    xt[j] = max(xt[j], e2 - SENS_FACTOR * st)
                end
            end
            nv = .!vortical
            jϕ = argmax(xϕ .* nv)
            jt = argmax(xt .* nv)
            @info "$label a=$a θo=$(θdeg)°: $(npixels(cache)) px ($(count(vortical)) vortical), $nvalid valid samples. vs Krang (non-vortical): max |Δφ| $(maximum(eϕ[nv])) (α=$(pcs.α[jϕ]) β=$(pcs.β[jϕ]) 1-k=$(1 - pcs.k_r[jϕ])), max |Δt̃| $(maximum(et[nv])) (α=$(pcs.α[jt]) β=$(pcs.β[jt])); excess over $(SENS_FACTOR)×sensitivity φ $(maximum(xϕ[nv])) t̃ $(maximum(xt[nv])); anchor residuals (non-vortical) t̃ median $(median(rt[nv])) max $(maximum(rt[nv])), φ median $(median(rϕ[nv])) max $(maximum(rϕ[nv])); vortical residual max t̃ $(maximum(rt[vortical]; init = 0.0)) φ $(maximum(rϕ[vortical]; init = 0.0)); (ok, νr) mismatches $nflag"
            @test nvalid > 0
            @test nflag == 0
            @test maximum(xϕ[nv]) <= tol_ϕ
            @test maximum(xt[nv]) <= tol_t

            # BigFloat rate reference on a subset; vortical rays without re-anchoring
            sub = unique(vcat(sortperm(xϕ .* nv; rev = true)[1:2], sortperm(xt .* nv; rev = true)[1:2],
                              rand(rng, 1:npixels(cache), 2), findall(vortical)))
            regenerate!(cache, a, θo; marcher = Recurrence(10^6))
            S0 = host(cache.samples)
            worst = (0.0, 0.0, 0.0, 0.0, 0)
            nsub = 0
            for j in sub
                i = cache.perm_host[j]
                pix = Krang.SlowLightIntensityPixel(met, camera.αs[i], camera.βs[i], θo)
                pm = PolarMarcher(pix)
                acos(min(pm.sqrt_up, 1.0)) < 0.02 && continue        # axis-grazing: Gauss–Legendre does not converge
                near_critical(pcs, j) && continue
                nsub += 1
                ray = HighPrecisionRay(a, camera.αs[i], camera.βs[i], θo, pix)
                λb = -big(camera.αs[i]) * sin(big(θo))
                Δτ = mino_step(pcs.τ_total[j], Val(N))
                τs = [k * Δτ for k in 1:N]
                rmin = Krang.horizon(met) * (1 + GATE3_HORIZON_MARGIN)
                eq_t, eq_ϕ, ek_t, ek_ϕ = highprec_increment_errors(vortical[j] ? S0 : S, D, j, ray, λb, τs, rmin)
                max(eq_t, eq_ϕ) > max(worst[1], worst[2]) && (worst = (eq_t, eq_ϕ, ek_t, ek_ϕ, j))
                if vortical[j]
                    @test eq_t <= tol_hp_t
                    @test eq_ϕ <= tol_hp_ϕ
                else
                    @test eq_t <= max(tol_hp_t, 2 * ek_t + 1e-12)
                    @test eq_ϕ <= max(tol_hp_ϕ, 2 * ek_ϕ + 1e-12)
                end
            end
            @info "$label a=$a θo=$(θdeg)°: vs BigFloat rate reference on $nsub pixels inside $(GATE3_RMAX) M: worst quadrature t̃ $(worst[1]) φ $(worst[2]) (Krang there: t̃ $(worst[3]) φ $(worst[4]); slot $(worst[5]): α=$(pcs.α[max(worst[5],1)]) β=$(pcs.β[max(worst[5],1)]) 1-k=$(1 - pcs.k_r[max(worst[5],1)]) vortical=$(worst[5] > 0 && vortical[worst[5]]))"
        end
    end
end
