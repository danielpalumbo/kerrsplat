# Mixed-mode derivatives with respect to the spacetime parameters (plan §7.1 regime b, §7.6
# Phase 5): ForwardDiff duals for the spin and the inclination propagate through the geodesic
# cache (regenerate!), the transfer coefficients (Bessel functions included), the frames and the
# exact transfer step, to the Stokes image of the polarized splats. Compared with central finite
# differences of the whole pipeline.

using StaticArrays
using LinearAlgebra
using ForwardDiff
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Splats

function test_spacetime_duals(; res = 8, N = 60, tol = 1e-5)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    L = gravitational_radius(4e6); ν = 230e9
    p = zeros(NPOLARIZEDPARAMS, 1)
    p[:, 1] = [6.0, 0.0, 0.0, log(1.2), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e6), log(20.0), log(30.0), 1.0, 0.5, 0.0, 0.35, 0.0, 0.0]
    function totals(a::T, θo::T) where {T}
        cache = GeodesicCache(CPU(), Geodesics.Camera(T.(camera.αs), T.(camera.βs), camera.size), Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        img = polarized_image(cache, T.(p), T(0.0), T(ν), T(L))
        return SVector(sum(getindex.(img, 1)), sum(getindex.(img, 2)), sum(getindex.(img, 3)), sum(getindex.(img, 4)))
    end
    a0 = 0.9; θ0 = deg2rad(60.0)
    @testset "spacetime duals through the polarized image ($(res)² × $N)" begin
        J = ForwardDiff.jacobian(x -> totals(x[1], x[2]), [a0, θ0])
        h = 1e-5
        Ja = (totals(a0 + h, θ0) - totals(a0 - h, θ0)) / 2h
        Jθ = (totals(a0, θ0 + h) - totals(a0, θ0 - h)) / 2h
        err = maximum(abs.(J .- hcat(Ja, Jθ)) ./ max.(abs.(hcat(Ja, Jθ)), 1e-6 * maximum(abs, J)))
        @test err < tol
        @info "spacetime duals: ∂(I, Q, U, V)/∂a = $(round.(J[:, 1], sigdigits = 4)), ∂/∂θo = $(round.(J[:, 2], sigdigits = 4)); max relative error vs finite differences $err"
    end
end
