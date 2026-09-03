# Gate 5 of the GPU plan (§8): an end-to-end image through a test splat from the stored samples
# of both marchers, on a KernelAbstractions backend, against a host loop over Krang's direct
# evaluation. The splat is the one of the feasibility probes: a Gaussian emissivity in
# quasi-Cartesian coordinates weighted by the cube of the ZAMO redshift factor obtained from
# the analytic momentum (which exercises ν_r, ν_θ), integrated in Mino time along each ray.
#
# Rays whose anchor residual exceeds RESIDUAL_FLAG are excluded from the comparison with the
# host and must be vortical: there Krang's closed forms (the host reference) are wrong after
# the polar turning point (docs/notes/upstream_issues.md), which shows up as 7e-6 of the peak
# in this image, and the cache's residuals are what flags them.

using Test
using KernelAbstractions
using Krang
using StaticArrays
using KerrSplat.Geodesics

const SPLAT = (x = 6.0, y = 4.0, z = 0.5, σ = 1.5, A = 2.0)
const RESIDUAL_FLAG = 1e-6

"Splat emissivity × g³ at one sample (zero for invalid samples and inside r_h(1 + 1e-3))."
@inline function splat_weight(met::Krang.Kerr, η, λ, s::GeodesicSample{T}, splat) where {T}
    rh = Krang.horizon(met)
    (s.ok && (rh * (1 + T(1e-3)) < s.r < T(1e3))) || return zero(T)
    x, y, z = quasi_cartesian_kerr_schild(met, s.r, s.θ, s.ϕ)
    d2 = ((x - splat.x)^2 + (y - splat.y)^2 + (z - splat.z)^2) / (2 * splat.σ^2)
    pbl = Krang.p_bl_d(met, s.r, s.θ, η, λ, s.νr, s.νθ)
    pzamo = Krang.jac_zamo_u_bl_d(met, s.r, s.θ) * (Krang.metric_uu(met, s.r, s.θ) * pbl)
    g = inv(pzamo[1])
    return splat.A * exp(-d2) * g^3
end

@kernel function splat_image_kernel!(out, S, pc, met::Krang.Kerr, splat, ::Val{N}) where {N}
    j = @index(Global, Linear)
    @inbounds begin
        η = pc.η[j]
        λ = pc.λ[j]
        Δτ = mino_step(pc.τ_total[j], Val(N))
        acc = zero(eltype(out))
        for k in 1:N
            acc += splat_weight(met, η, λ, S[j, k], splat) * Δτ
        end
        out[j] = acc
    end
end

"Image (screen order) from the stored samples of `cache`."
function splat_image(cache::GeodesicCache{T,N}, splat) where {T,N}
    out = KernelAbstractions.allocate(cache.backend, T, npixels(cache))
    splat_image_kernel!(cache.backend, 128)(out, cache.samples, cache.consts, Krang.Kerr(cache.spin), splat, Val(N);
                                            ndrange = npixels(cache))
    KernelAbstractions.synchronize(cache.backend)
    return to_screen(cache, out)
end

"The same image from Krang's direct evaluation on the host."
function splat_image_host(camera::Camera, a, θo, ::Val{N}, splat) where {N}
    met = Krang.Kerr(a)
    img = zeros(npixels(camera))
    Threads.@threads for i in 1:npixels(camera)
        pix = Krang.SlowLightIntensityPixel(met, camera.αs[i], camera.βs[i], θo)
        Δτ = mino_step(Krang.total_mino_time(pix), Val(N))
        acc = 0.0
        for k in 1:N
            acc += splat_weight(met, pix.η, pix.λ, direct_sample(pix, k * Δτ), splat) * Δτ
        end
        img[i] = acc
    end
    return reshape(img, size(camera))
end

function test_image(backend; res::Int, N::Int, tol::Float64, label::String)
    a, θo = 0.94, deg2rad(60.0)
    # the GPU-safe transform equals Krang's on the host
    met = Krang.Kerr(a)
    for (r, θ, ϕ) in ((5.0, 1.0, 0.3), (1.5, 2.5, -4.0), (30.0, 0.1, 7.0))
        @test all(isapprox.(quasi_cartesian_kerr_schild(met, r, θ, ϕ),
                            Krang.boyer_lindquist_to_quasi_cartesian_kerr_schild_fast_light(met, r, θ, ϕ); rtol = 1e-14))
    end
    camera = Camera((-10.0, 10.0), (-10.0, 10.0), res)
    cache = GeodesicCache(backend, camera, Val(N))
    ref = splat_image_host(camera, a, θo, Val(N), SPLAT)
    scale = maximum(ref)
    @testset "$label splat image $(res)² × $N" begin
        regenerate!(cache, a, θo; marcher = Direct())
        img_d = Array(splat_image(cache, SPLAT))
        regenerate!(cache, a, θo; marcher = Recurrence(64))
        img_r = Array(splat_image(cache, SPLAT))
        flagged = to_screen(cache, max.(Array(cache.residual_t), Array(cache.residual_ϕ))) .> RESIDUAL_FLAG
        η = to_screen(cache, Array(cache.consts.η))
        e_d = maximum(abs.(img_d .- ref)) / scale
        e_r = maximum(abs.(img_r .- ref)[.!flagged]) / scale
        e_r_all = maximum(abs.(img_r .- ref)) / scale
        @info "$label splat image $(res)² × $N: total flux $(sum(ref)); max |image − host| / max: direct marcher $e_d, recurrence marcher $e_r on the $(count(.!flagged)) unflagged pixels ($e_r_all with the $(count(flagged)) residual-flagged, all vortical: $(all(η[flagged] .< 0)))"
        @test scale > 0
        @test e_d <= tol
        @test e_r <= tol
        @test all(η[flagged] .< 0)
        @test count(flagged) <= 0.01 * length(flagged)
    end
end
