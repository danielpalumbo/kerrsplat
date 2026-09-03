# Gate 2 of the GPU plan (§8): the addition-theorem recurrence for r(τ), θ(τ) against an
# independent high-precision evaluation of the same closed forms (test/highprec_reference.jl)
# and against Krang's direct Float64 evaluation, at every sample, for random pixels of every
# root case and a band of pixels on both sides of the critical curve.
#
# Criterion. Against the BigFloat reference, at every pixel the recurrence must reach 1e-10 in
# r and θ or else be within twice the error of Krang's direct evaluation (+1e-12): the inputs
# limit both paths near the critical curve (a near-double root r₃ ≈ r₄ of the radial quartic
# is determined only to ε/r₄₃ by Float64 root finding; docs/notes/) and, rarely, elsewhere
# (a = 0.999). The maxima of both paths on well-conditioned pixels (1 − k_r ≥ 1e-2) are
# reported. The validity flag
# and ν_r must match Krang's exactly; ν_θ must match the sign of −dθ/dτ obtained by finite
# differences of Krang's own θ(τ), because Krang's flag is wrong after the polar turning point
# of vortical rays (docs/notes/upstream_issues.md, Krang item 5); Krang's disagreements with
# that sign are counted and reported.

using Test
using Random
using KernelAbstractions
using Krang
using JacobiElliptic
using KerrSplat.Geodesics

function test_jacobi()
    @testset "Jacobi addition theorem" begin
        for μ in (0.1, 0.5, 0.9, 0.99, 0.9999), δ in (1e-3, 3e-2)
            K = JacobiElliptic.K(μ)
            Δ = jacobi_step_constants(δ, μ)
            # 64 unanchored steps (one re-anchoring interval) from starts spread over a period
            worst64 = 0.0
            for u0 in range(0.0, 4K, length = 25)
                x = jacobi_state(u0, μ)
                for i in 1:64
                    x = jacobi_step(x, Δ, μ)
                    y = jacobi_state(u0 + i * δ, μ)
                    worst64 = max(worst64, abs(x.sn - y.sn), abs(x.cn - y.cn), abs(x.dn - y.dn))
                end
            end
            @test worst64 < 1e-12
            # drift over a full period without anchoring (documents why we re-anchor)
            x = jacobi_state(0.37, μ)
            n = round(Int, 4K / δ)
            for i in 1:n
                x = jacobi_step(x, Δ, μ)
            end
            y = jacobi_state(0.37 + n * δ, μ)
            @test max(abs(x.sn - y.sn), abs(x.cn - y.cn), abs(x.dn - y.dn)) < 1e-9 * max(1, n / 1000)
            # negative-parameter transform reproduces JacobiElliptic's sn/cn/dn at k < 0
            k = -3μ
            μk, s = Geodesics.negative_parameter_transform(k)
            for u in (0.3, 1.7, 4.1)
                z = jacobi_state(s * u, μk)
                @test isapprox(z.sn / (s * z.dn), JacobiElliptic.sn(u, k); atol = 1e-13)
                @test isapprox(z.cn / z.dn, JacobiElliptic.cn(u, k); atol = 1e-13)
                @test isapprox(1 / z.dn, JacobiElliptic.dn(u, k); atol = 1e-13)
            end
            # step constants are odd in sn, even in cn, dn
            Δm = jacobi_step_constants(-δ, μ)
            @test Δm.sn == -Δ.sn && Δm.cn == Δ.cn && Δm.dn == Δ.dn
        end
    end
end

numreals_at(met, θo, ρ, ψ) =
    sum(Krang._isreal2, Krang.get_radial_roots(met, Krang.η(met, ρ * cos(ψ), ρ * sin(ψ), θo), Krang.λ(met, ρ * cos(ψ), θo)))

"Critical screen radius along direction ψ (four real roots outside, plunging inside)."
function critical_radius(met, θo, ψ)
    lo, hi = 1.0, 12.0
    numreals_at(met, θo, hi, ψ) == 4 || return NaN
    for _ in 1:60
        mid = (lo + hi) / 2
        numreals_at(met, θo, mid, ψ) == 4 ? (hi = mid) : (lo = mid)
    end
    return hi
end

"Random screen pixels plus a band on both sides of the critical curve."
function gate2_camera(a, θo, rng; nrandom = 200, nψ = 8, δs = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6))
    met = Krang.Kerr(a)
    αs = Float64[]
    βs = Float64[]
    while length(αs) < nrandom
        α, β = 12 * (2rand(rng) - 1), 12 * (2rand(rng) - 1)
        isfinite(Krang.SlowLightIntensityPixel(met, α, β, θo).total_mino_time) || continue
        push!(αs, α)
        push!(βs, β)
    end
    for ψ in range(0, 2π, length = nψ + 1)[1:nψ] .+ 0.1rand(rng)
        ρc = critical_radius(met, θo, ψ)
        isfinite(ρc) || continue
        for δ in δs, side in (1 + δ, 1 - δ)
            α, β = ρc * side * cos(ψ), ρc * side * sin(ψ)
            isfinite(Krang.SlowLightIntensityPixel(met, α, β, θo).total_mino_time) || continue
            push!(αs, α)
            push!(βs, β)
        end
    end
    return Camera(αs, βs)
end

const GATE2_SPACETIMES = ((0.2, 45.0), (0.5, 17.0), (0.94, 1.0), (0.94, 60.0), (0.94, 89.0), (0.999, 30.0), (0.7, 120.0))

"""
    gate2_references() -> Dict((a, θdeg) => (camera, rays))

The cameras and the BigFloat ray constants (`HighPrecisionRay`) of the gate-2 pixel sets,
shared by all backends. The reference coordinates are evaluated per backend on the backend's
own Mino-time grid (`highprec_samples`), because τ_total of a near-critical ray differs by
~1e-10 between GPU and CPU and θ(τ) oscillates quickly there.
"""
function gate2_references()
    rng = MersenneTwister(2026)
    refs = Dict{Tuple{Float64,Float64},Any}()
    for (a, θdeg) in GATE2_SPACETIMES
        θo = deg2rad(θdeg)
        camera = gate2_camera(a, θo, rng)
        met = Krang.Kerr(a)
        rays = [HighPrecisionRay(a, camera.αs[i], camera.βs[i], θo,
                                 Krang.SlowLightIntensityPixel(met, camera.αs[i], camera.βs[i], θo))
                for i in 1:npixels(camera)]
        refs[(a, θdeg)] = (camera, rays)
    end
    return refs
end

"BigFloat r, θ at every sample of every ray, on the Mino-time grid the cache marched (sorted order)."
function highprec_samples(cache::GeodesicCache, rays, ::Val{N}) where {N}
    τ_total = host(cache.consts).τ_total
    R = zeros(npixels(cache), N)
    Θ = zeros(npixels(cache), N)
    for j in 1:npixels(cache)
        ray = rays[cache.perm_host[j]]
        Δτ = mino_step(τ_total[j], Val(N))
        for k in 1:N
            r, θ = highprec_coordinates(ray, k * Δτ)
            R[j, k] = Float64(r)
            Θ[j, k] = Float64(θ)
        end
    end
    return R, Θ
end

function test_recurrence(backend, refs; N::Int, M::Int, label::String)
    seen = (false, false, false)
    for (a, θdeg) in GATE2_SPACETIMES
        θo = deg2rad(θdeg)
        camera, rays = refs[(a, θdeg)]
        cache = GeodesicCache(backend, camera, Val(N))
        @testset "$label a=$a θo=$(θdeg)° ($(npixels(camera)) px)" begin
            regenerate!(cache, a, θo; marcher = Direct())
            D = host(cache.samples)
            regenerate!(cache, a, θo; marcher = Recurrence(M))
            S = host(cache.samples)
            Rref, Θref = highprec_samples(cache, rays, Val(N))
            @test cache.marcher == Recurrence(M)
            r = cache.ranges
            seen = seen .| (!isempty(r.case2), !isempty(r.case3), !isempty(r.case4))
            omk = 1 .- host(cache.consts).k_r
            # per pixel: max over samples of the error of each path against the reference
            erec_r = zeros(npixels(cache)); edir_r = zeros(npixels(cache))
            erec_θ = zeros(npixels(cache)); edir_θ = zeros(npixels(cache))
            nflag = 0
            nνθ_rec = 0
            nνθ_krang = 0
            nvalid = 0
            met = Krang.Kerr(a)
            for j in 1:npixels(cache)
                i = cache.perm_host[j]
                pix = Krang.SlowLightIntensityPixel(met, camera.αs[i], camera.βs[i], θo)
                Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
                for k in 1:N
                    d = D[j, k]
                    s = S[j, k]
                    (d.ok, d.νr) == (s.ok, s.νr) || (nflag += 1)
                    d.ok || continue
                    nvalid += 1
                    # ν_θ against the finite-difference sign of the direct θ(τ), away from turning points
                    δ = 1e-7 * Δτ * N
                    dθ = (direct_sample(pix, k * Δτ + δ).θ - direct_sample(pix, k * Δτ - δ).θ) / (2δ)
                    if abs(dθ) > 1e-6
                        (dθ < 0) == s.νθ || (nνθ_rec += 1)
                        (dθ < 0) == d.νθ || (nνθ_krang += 1)
                    end
                    @test isnan(s.t) && isnan(s.ϕ)
                    rr, θr = Rref[j, k], Θref[j, k]
                    erec_r[j] = max(erec_r[j], abs(s.r - rr) / max(rr, 1.0))
                    edir_r[j] = max(edir_r[j], abs(d.r - rr) / max(rr, 1.0))
                    erec_θ[j] = max(erec_θ[j], abs(s.θ - θr))
                    edir_θ[j] = max(edir_θ[j], abs(d.θ - θr))
                end
            end
            good = omk .>= 1e-2
            # per pixel: the recurrence meets 1e-10, or is no worse than twice the direct path
            excess_r = maximum(min.(erec_r .- 1e-10, erec_r .- 2 .* edir_r .- 1e-12))
            excess_θ = maximum(min.(erec_θ .- 1e-10, erec_θ .- 2 .* edir_θ .- 1e-12))
            @info "$label a=$a θo=$(θdeg)°: $(npixels(cache)) px (cases $(length(r.case2))/$(length(r.case3))/$(length(r.case4)); $(count(good)) with 1-k ≥ 1e-2), $nvalid valid samples. vs BigFloat, well-conditioned pixels: recurrence r $(maximum(erec_r[good])) θ $(maximum(erec_θ[good])), direct r $(maximum(edir_r[good])) θ $(maximum(edir_θ[good])); all pixels: recurrence r $(maximum(erec_r)) direct r $(maximum(edir_r)), recurrence θ $(maximum(erec_θ)) direct θ $(maximum(edir_θ)); excess over criterion r $excess_r θ $excess_θ; (ok, νr) mismatches $nflag; νθ vs finite differences: recurrence $nνθ_rec wrong, Krang $nνθ_krang wrong"
            @test nvalid > 0
            @test nflag == 0
            @test nνθ_rec == 0
            @test excess_r <= 0
            @test excess_θ <= 0
            @test maximum(erec_θ[good]) <= 1e-10
        end
    end
    @test all(seen)
end
