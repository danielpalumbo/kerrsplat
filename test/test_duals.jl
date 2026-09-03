# Gate 4 of the GPU plan (§8): ForwardDiff duals for (a, θo) through K1 and both K2 marchers.
#
# Reference 1: the direct marcher's own duals (exact derivatives of Krang's closed forms).
# The recurrence/quadrature duals must agree with them wherever the values agree, i.e. on
# non-vortical rays (Krang's closed forms are wrong on vortical rays after the polar turning
# point, so their derivatives are too).
# Reference 2: central finite differences of the Float64 direct marcher, step 1e-5, whose
# noise floor is ~1e-8 on ordinary rays (plan §3) and worse on axis-grazing ones.

using Test
using KernelAbstractions
using Krang
using ForwardDiff
using KerrSplat.Geodesics

const Dual2 = ForwardDiff.Dual{Nothing,Float64,2}
dual2(v, d1, d2) = Dual2(v, ForwardDiff.Partials((d1, d2)))

const GATE4_PIXELS = ((5.0, 3.0), (-2.0, 1.5), (7.0, 1.0), (0.159, 10.0), (2.0, 10.0), (3.1374, -4.892), (-5.0, 1.0), (-6.0, 4.0), (1.5, -7.0))

function dual_samples(backend, marcher, a, θo, cam, ::Val{N}) where {N}
    cache = GeodesicCache(backend, cam, Val(N))
    regenerate!(cache, a, θo; marcher = marcher)
    return host(cache.samples), cache.perm_host, host(cache.consts)
end

function test_duals(backend; N::Int, tol_rec::Float64, tol_fd::Float64, label::String)
    a, θo = 0.94, deg2rad(60.0)
    αs = [p[1] for p in GATE4_PIXELS]
    βs = [p[2] for p in GATE4_PIXELS]
    cam64 = Camera(αs, βs)
    camD = Camera(dual2.(αs, 0.0, 0.0), dual2.(βs, 0.0, 0.0))
    aD, θD = dual2(a, 1.0, 0.0), dual2(θo, 0.0, 1.0)
    @testset "$label duals for (a, θo)" begin
        Sd, perm, pcs = dual_samples(backend, Direct(), aD, θD, camD, Val(N))
        Sr, perm2, _ = dual_samples(backend, Recurrence(64), aD, θD, camD, Val(N))
        @test perm == perm2
        @test eltype(pcs.τ_total) == Dual2
        h = 1e-5
        fd = Dict(s => dual_samples(backend, Direct(), a + s[1] * h, θo + s[2] * h, cam64, Val(N))[1]
                  for s in ((1, 0), (-1, 0), (0, 1), (0, -1)))
        getters = ((:t, s -> s.t), (:r, s -> s.r), (:θ, s -> s.θ), (:ϕ, s -> s.ϕ))
        met = Krang.Kerr(a)
        nnan = 0
        for j in 1:length(αs)
            i = perm[j]
            pix = Krang.SlowLightIntensityPixel(met, αs[i], βs[i], θo)
            omk = 1 - ForwardDiff.value(pcs.k_r[j])
            θmin = acos(min(PolarMarcher(pix).sqrt_up, 1.0))
            well = omk >= 1e-2 && θmin > 0.1        # finite differences are only meaningful here
            worst_rec = Dict(f => 0.0 for (f, _) in getters)
            worst_fd = Dict(f => 0.0 for (f, _) in getters)
            for k in 1:N
                d = Sd[j, k]
                r_ = Sr[j, k]
                (d.ok && r_.ok) || continue
                for (f, get) in getters
                    vd = get(d)
                    vr = get(r_)
                    pd = (ForwardDiff.partials(vd, 1), ForwardDiff.partials(vd, 2))
                    pr = (ForwardDiff.partials(vr, 1), ForwardDiff.partials(vr, 2))
                    all(isfinite, pr) || (nnan += 1)
                    scale = max(abs(pd[1]), abs(pd[2]), 1.0)
                    worst_rec[f] = max(worst_rec[f], max(abs(pd[1] - pr[1]), abs(pd[2] - pr[2])) / scale)
                    all(s -> fd[s][j, k].ok, keys(fd)) || continue
                    fa = (get(fd[(1, 0)][j, k]) - get(fd[(-1, 0)][j, k])) / 2h
                    fθ = (get(fd[(0, 1)][j, k]) - get(fd[(0, -1)][j, k])) / 2h
                    worst_fd[f] = max(worst_fd[f], max(abs(pd[1] - fa), abs(pd[2] - fθ)) / scale)
                end
            end
            @info "$label duals α=$(αs[i]) β=$(βs[i]) (1-k=$omk, θmin=$(round(θmin, digits = 3)), $(well ? "well-conditioned" : "ill-conditioned")): recurrence vs direct duals $(worst_rec); direct duals vs finite differences (h = $h) $(worst_fd)"
            for (f, _) in getters
                @test worst_rec[f] <= (well ? tol_rec : 100 * tol_rec)
                well && @test worst_fd[f] <= tol_fd
            end
        end
        @test nnan == 0
    end
end
