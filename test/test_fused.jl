# Plan step 6: fused mode. The consumer runs inside the marcher and nothing is stored per
# sample; the result must equal what the same consumer computes from the stored samples.
# Also the tile utility for screens whose samples do not fit.

using Test
using KernelAbstractions
using Krang
using KerrSplat.Geodesics

"The splat integrand of test_image.jl as a fused consumer."
struct SplatConsumer{S}
    splat::S
end
@inline (c::SplatConsumer)(acc, j, k, s::GeodesicSample, Δτ, pix) =
    acc + splat_weight(Krang.metric(pix), Krang.η(pix), Krang.λ(pix), s, c.splat) * Δτ

function test_fused(backend; res::Int, N::Int, label::String)
    a, θo = 0.94, deg2rad(60.0)
    camera = Camera((-10.0, 10.0), (-10.0, 10.0), res)
    @testset "$label fused mode $(res)² × $N" begin
        stored = GeodesicCache(backend, camera, Val(N))
        regenerate!(stored, a, θo; marcher = Recurrence(64))
        img_stored = Array(splat_image(stored, SPLAT))
        res_stored = Array(stored.residual_ϕ)

        fused = GeodesicCache(backend, camera, Val(N); store_samples = false)
        @test !has_samples(fused)
        @test_throws ArgumentError regenerate!(fused, a, θo; marcher = Recurrence(64))
        regenerate!(fused, a, θo; marcher = Fused(64))
        @test fused.marcher == Fused(64)
        out = KernelAbstractions.allocate(backend, Float64, npixels(fused))
        fused_march!(SplatConsumer(SPLAT), out, fused)
        img_fused = Array(to_screen(fused, out))
        e = maximum(abs.(img_fused .- img_stored)) / maximum(img_stored)
        # (the two kernels are compiled separately; on CUDA their floating-point contraction differs at 1e-13)
        @info "$label fused mode $(res)² × $N: fused vs stored image max |Δ|/max = $e; max residual difference $(maximum(abs.(Array(fused.residual_ϕ) .- res_stored)))"
        @test e <= 1e-12
        @test all(isapprox.(Array(fused.residual_ϕ), res_stored; atol = 1e-10, rtol = 1e-6))

        # tiles: the same image assembled from four tiles of a quarter of the pixels each
        img_tiles = zeros(npixels(camera))
        for (tile, idx) in tiles(camera, 4)
            c = GeodesicCache(backend, tile, Val(N); store_samples = false)
            regenerate!(c, a, θo; marcher = Fused(64))
            o = KernelAbstractions.allocate(backend, Float64, npixels(c))
            fused_march!(SplatConsumer(SPLAT), o, c)
            img_tiles[idx] = Array(unsort(c, o))
        end
        @test reshape(img_tiles, size(camera)) == img_fused
        @test sum(length(idx) for (_, idx) in tiles(camera, 4)) == npixels(camera)
    end
end
