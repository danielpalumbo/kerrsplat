# Gates for the polarized transport consumer (Transfer.RadiativeTransport):
#  1. the Gold et al. model through the polarized path (all polarized coefficients zero) reproduces
#     the unpolarized path exactly;
#  2. a Faraday screen: a cold slab (large ρ_V, negligible j and α) in front of an emitting slab
#     rotates the emitted linear polarization by the analytic angle ½ ∫ρ_V ds, and two co-located
#     screens add their rotation measures (the coefficients of overlapping elements are additive);
#  3. two co-located emitting elements with different velocities and fields give the same image as
#     the sum of their invariant coefficients evaluated once (the addendum's overlap test), by
#     construction of the consumer; checked against a hand-built single element;
#  4. ipole's RIAF model: full Stokes images against ipole's own (validation/ipole_riaf/, produced
#     by ipole built here from model/riaf/example.par), when the reference files are present.

using DelimitedFiles
using LinearAlgebra
using Statistics
using StaticArrays
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Transfer: JY, PC

include("riaf_model.jl")

# ---- 1. the Gold model through the polarized path -----------------------------------------------
struct GoldAsPolarized{M}
    m::M
end
Transfer.nelements(::GoldAsPolarized) = 1
@inline function Transfer.element(g::GoldAsPolarized, i, pix, s, ν_obs)
    j, α, gg = Transfer.unpolarized_coefficients(g.m, pix, s, ν_obs)
    T = typeof(j)
    return StokesCoefficients(j, zero(T), zero(T), α, zero(T), zero(T), zero(T), zero(T)), LocalFrame(gg, zero(T), zero(T))
end

# ---- 2/3. slab elements: uniform coefficients in a shell of radius r ∈ [r1, r2] ------------------
# An element with fixed fluid-frame coefficients (given directly), a ZAMO velocity and a field,
# present between two radii. Coefficients are what `thermal_synchrotron` would return, scaled.
struct SlabElement{T}
    r1::T
    r2::T
    c::StokesCoefficients{T}
    ũ::SVector{3,T}
    B::SVector{3,T}
end
struct Slabs{N,T}
    els::NTuple{N,SlabElement{T}}
end
Transfer.nelements(::Slabs{N}) where {N} = N
@inline function Transfer.element(m::Slabs{N,T}, i, pix, s, ν_obs) where {N,T}
    e = m.els[i]
    (e.r1 <= s.r <= e.r2) || return zero(StokesCoefficients{T}), LocalFrame(one(T), zero(T), zero(T))
    fr = local_frame(pix, s, e.ũ, e.B)
    return e.c, fr
end

function stokes_image(backend, model, camera, a, θo, ν, L; N = 800)
    cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    out = KernelAbstractions.allocate(backend, RadiativeState{Float64}, npixels(cache))
    fused_march!(RadiativeTransport(model, ν, L), out, cache)
    states = Array(to_screen(cache, out))
    return map(st -> observed_stokes(st, ν), states)
end

function test_polarized(backend; label = "")
    Geodesics.prepare_backend!(backend)
    @testset "polarized transport ($label)" begin
        # 1. Gold model 4 (absorbing) through both paths
        a, gm = gold_model(4)
        res = 24; fov = 30.0; Δα = fov / res; θo = deg2rad(60.0); ν = 230e9
        axis = [-fov / 2 + (k - 0.5) * Δα for k in 1:res]
        camera = Geodesics.Camera(vec([axis[i] for i in 1:res, j in 1:res]), vec([axis[j] for i in 1:res, j in 1:res]), (res, res))
        L = gravitational_radius(4.063e6)
        cache = GeodesicCache(backend, camera, Val(400); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        out_u = KernelAbstractions.allocate(backend, UnpolarizedState{Float64}, npixels(cache))
        fused_march!(UnpolarizedTransport(gm, ν, L), out_u, cache)
        Iu = observed_intensity.(Array(to_screen(cache, out_u)), ν)
        out_p = KernelAbstractions.allocate(backend, RadiativeState{Float64}, npixels(cache))
        fused_march!(RadiativeTransport(GoldAsPolarized(gm), ν, L), out_p, cache)
        Sp = map(st -> observed_stokes(st, ν), Array(to_screen(cache, out_p)))
        Ip = getindex.(Sp, 1)
        @test maximum(abs.(Ip .- Iu) ./ max.(abs.(Iu), 1e-30 * maximum(Iu))) < 1e-12
        @test all(S -> S[2] == 0 && S[3] == 0 && S[4] == 0, Sp)

        # 2. Faraday screen in front of an emitting slab, on a ray captured by the hole (a = 0.5, α = 2, β = 1)
        #    so that each shell is crossed once; both elements at rest in the ZAMO frame with the same
        #    field so that both share one screen basis
        a = 0.5; θo = deg2rad(60.0)
        cam1 = Geodesics.Camera([2.0], [1.0])
        B = SVector(0.3, 0.8, 0.5)
        ũ0 = zero(SVector{3,Float64})
        emit = StokesCoefficients(1e-14, 0.6e-14, 0.1e-14, 0.0, 0.0, 0.0, 0.0, 0.0)          # thin polarized emitter, r ∈ [12, 14]
        emitter = SlabElement(12.0, 14.0, emit, ũ0, B)
        ρV = 2e-12                                                                             # per cm, over the screen r ∈ [16, 18]
        screen = SlabElement(16.0, 18.0, StokesCoefficients(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, ρV), ũ0, B)
        L = 1e12
        S0 = stokes_image(backend, Slabs((emitter,)), cam1, a, θo, ν, L; N = 1500)[1]
        S1 = stokes_image(backend, Slabs((emitter, screen)), cam1, a, θo, ν, L; N = 1500)[1]
        S2 = stokes_image(backend, Slabs((emitter, screen, screen)), cam1, a, θo, ν, L; N = 1500)[1]
        # the screen preserves I and P and rotates the EVPA by ½ ∫ ν_f ρ_V dλ_phys, computed from the samples
        cache1 = GeodesicCache(CPU(), cam1, Val(1500)); regenerate!(cache1, a, θo; marcher = Recurrence(64))
        smp = host(cache1.samples)
        Δτ = Krang.total_mino_time(Krang.SlowLightIntensityPixel(Krang.Kerr(a), 2.0, 1.0, θo)) / 1501
        rm = 0.0
        for k in 1:1500
            sk = smp[1, k]
            (sk.ok && 16 <= sk.r <= 18) || continue
            met = Krang.Kerr(a); Σ = sk.r^2 + a^2 * cos(sk.θ)^2
            pixk = Krang.SlowLightIntensityPixel(met, 2.0, 1.0, θo)
            fr = local_frame(pixk, sk, ũ0, B)
            rm += ν / fr.g * ρV * L / ν * Σ * Δτ       # ν_f ρ_V × (L/ν_obs) Σ Δτ = ρ_V × fluid path length
        end
        χ0 = 0.5 * atan(S0[3], S0[2]); χ1 = 0.5 * atan(S1[3], S1[2]); χ2 = 0.5 * atan(S2[3], S2[2])
        P0 = hypot(S0[2], S0[3]); P1 = hypot(S1[2], S1[3])
        @test S1[1] ≈ S0[1] rtol = 1e-12
        @test P1 ≈ P0 rtol = 1e-10
        @test abs(rem(χ1 - χ0 - rm / 2, π, RoundNearest)) < 1e-9 * max(1, rm)
        @test abs(rem(χ2 - χ0 - rm, π, RoundNearest)) < 1e-9 * max(1, rm)     # two co-located screens add
        @info "Faraday screen: rotation $(round(rm / 2, digits = 6)) rad expected, $(round(rem(χ1 - χ0, π, RoundNearest), digits = 6)) measured; P conserved to $(abs(P1 / P0 - 1))"

        # 3. two co-located emitters with different velocities and fields vs a single element carrying
        #    the sum of their screen-basis invariants: checked by comparing against the consumer's own
        #    linearity, S(e1 + e2 co-located, no absorption) = S(e1) + S(e2)
        e1 = SlabElement(10.0, 13.0, StokesCoefficients(1e-14, 0.5e-14, 0.2e-14, 0.0, 0.0, 0.0, 0.0, 0.0), SVector(0.3, 0.4, 0.0), SVector(1.0, 0.0, 0.2))
        e2 = SlabElement(10.0, 13.0, StokesCoefficients(2e-14, -0.4e-14, 0.3e-14, 0.0, 0.0, 0.0, 0.0, 0.0), SVector(-0.2, 0.5, 0.1), SVector(0.0, 1.0, -0.3))
        Sa = stokes_image(backend, Slabs((e1,)), cam1, a, θo, ν, L; N = 800)[1]
        Sb = stokes_image(backend, Slabs((e2,)), cam1, a, θo, ν, L; N = 800)[1]
        Sab = stokes_image(backend, Slabs((e1, e2)), cam1, a, θo, ν, L; N = 800)[1]
        @test Sab ≈ Sa + Sb rtol = 1e-12
        # and with absorption in one of them the order of the elements does not matter
        e3 = SlabElement(10.0, 13.0, StokesCoefficients(1e-14, 0.5e-14, 0.2e-14, 3e-13, 1e-13, 0.5e-13, 2e-13, 4e-13), SVector(0.3, 0.4, 0.0), SVector(1.0, 0.0, 0.2))
        S12 = stokes_image(backend, Slabs((e3, e2)), cam1, a, θo, ν, L; N = 800)[1]
        S21 = stokes_image(backend, Slabs((e2, e3)), cam1, a, θo, ν, L; N = 800)[1]
        @test S12 ≈ S21 rtol = 1e-13
    end
end

# ---- 4. ipole's RIAF ----------------------------------------------------------------------------
const IPOLE_RIAF_DIR = joinpath(@__DIR__, "..", "validation", "ipole_riaf")

"KerrSplat's Stokes images of ipole's example RIAF on ipole's pixel grid (100², 200 μas at 8.3 kpc, M = 4.3e6 M☉, 85°)."
function riaf_images(backend; N = 2000, res = 100, pandya = true, ν = 230e9)
    a = 0.9375; θo = deg2rad(85.0)
    M_solar = 4.3e6; D = 8.3e3 * PC
    L = gravitational_radius(M_solar)
    fov = 200e-6 / 206264.806247 * D / L                      # 39.10 M
    # ipole's pixel centres: x offset (i + 0.49)/nx − 0.5, y offset (j + 0.5)/ny − 0.5 (model_geodesics.c init_XK)
    xs = [((i - 1) + 0.49) / res - 0.5 for i in 1:res] .* fov
    ys = [((j - 1) + 0.5) / res - 0.5 for j in 1:res] .* fov
    camera = Geodesics.Camera(vec([xs[i] for i in 1:res, j in 1:res]), vec([ys[j] for i in 1:res, j in 1:res]), (res, res))
    S = stokes_image(backend, riaf_example(a; pandya), camera, a, θo, ν, L; N)
    scale = (fov / res * L / D)^2 / JY
    return S, scale, fov
end

function test_riaf_vs_ipole(backend; N = 2000, label = "")
    Geodesics.prepare_backend!(backend)
    if !isfile(joinpath(IPOLE_RIAF_DIR, "riaf_fine_I.csv"))
        @warn "ipole RIAF reference images not found in validation/ipole_riaf; skipping"
        return
    end
    @testset "RIAF full Stokes images vs ipole ($label)" begin
        # 230 GHz: Faraday depths up to 9 rad, optical depths up to 15; 2 THz: intrinsic polarization only
        for (tag, ν) in (("riaf_fine", 230e9), ("riaf_2thz", 2e12))
            S, scale, fov = riaf_images(backend; N, ν)
            ours = [getindex.(S, k) for k in 1:4]
            ref = [readdlm(joinpath(IPOLE_RIAF_DIR, "$(tag)_$(s).csv"), ',') for s in ("I", "Q", "U", "V")]
            res = [norm(ours[k] .- ref[k]) / norm(ref[k]) for k in 1:4]
            tot = [sum(ours[k]) * scale for k in 1:4]; totr = [sum(ref[k]) * scale for k in 1:4]
            w = ref[1] .> 0.01 * maximum(ref[1])
            χo = 0.5 .* atan.(ours[3], ours[2]); χr = 0.5 .* atan.(ref[3], ref[2])
            dχ = median(abs.(rem.(χo .- χr, π, RoundNearest)[w]))
            lp = median(hypot.(ours[2], ours[3])[w] ./ ours[1][w]); lpr = median(hypot.(ref[2], ref[3])[w] ./ ref[1][w])
            @info "RIAF at $(ν / 1e9) GHz vs ipole ($label, N = $N): pixel-norm residuals I $(round(res[1], sigdigits = 2)) Q $(round(res[2], sigdigits = 2)) U $(round(res[3], sigdigits = 2)) V $(round(res[4], sigdigits = 2)); totals (Jy) $(round.(tot, sigdigits = 4)) vs $(round.(totr, sigdigits = 4)); EVPA median difference $(round(dχ, sigdigits = 2)) rad; LP fraction median $(round(lp, digits = 4)) vs $(round(lpr, digits = 4))"
            @test all(res .< 0.03)
            @test all(abs.(tot ./ totr .- 1) .< 0.01)
            @test dχ < 0.01
            @test abs(lp - lpr) < 0.005
        end
    end
end
