# The CPU-only subset of the gates that GitHub Actions can run in minutes: no CUDA, no BigFloat
# geodesic references, no Enzyme compilations. The full suite is test/runtests.jl.
using Test
using KerrSplat
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Splats
using KerrSplat.Fit
using Enzyme
using LinearAlgebra
using KernelAbstractions
using CUDA
using Random
using StaticArrays

include("reference.jl")
include("highprec_reference.jl")
include("test_geodesics.jl")
include("test_coefficients.jl")
include("test_transfer_step.jl")
include("test_frames.jl")
include("test_gold2020.jl")
include("test_polarized.jl")
include("riaf_model.jl")
include("test_uvfits.jl")
include("test_scattering.jl")
include("test_instrument.jl")
include("test_polarized_splats.jl")
include("test_winding.jl")
include("test_fit.jl")
include("test_spacetime.jl")

@testset "KerrSplat CI subset" begin
    test_geodesics(CPU(); res = 24, N = 16, tol = 1e-12, label = "CPU")
    test_coefficients(CPU(); label = "CPU")
    test_powerlaw_rotativities()
    test_kappa_coefficients()
    test_transfer_step(CPU(); label = "CPU")
    test_frames(CPU(); N = 200, label = "CPU")
    test_turning_point_frame()
    test_gold2020(CPU(); res = 24, N = 300, tol = 0.03, label = "CPU coarse")
    test_polarized(CPU(); label = "CPU")
    test_uvfits()
    test_scattering()
    test_scan_average()
    test_instrument()
    test_feed_rotation()
    test_crosshand_rotation()
    test_dterm_recovery()
    test_reflection()
    test_winding(CPU(); res = 16, N = 200, label = "CPU")
    test_step_adjoint()
    test_polarized_kernel_gradient(CPU(); K = 4, label = "CPU")
    test_polarized_gradient(CPU(); res = 6, N = 16, tol = 1e-10, label = "CPU backend")
    test_ray_lists(CPU(); res = 6, N = 16, tol = 1e-12, label = "CPU backend")
    test_joint_fit(CPU(); res = 6, N = 16, iterations = 30, tol = 5e-5, θtol = deg2rad(4.0), atol = 0.2, label = "CPU backend")
    test_polarized_gradient_winding(CPU(); res = 6, N = 40, tol = 1e-10, label = "CPU backend")
    test_binning(CPU(); res = 4, N = 24, tol = 1e-9, label = "CPU backend")
    test_polish(; res = 6, N = 24)
    test_timeresolved(CPU(); res = 4, N = 12, tol = 1e-9, label = "CPU backend")
    test_selfcal(CPU(); res = 4, N = 12, tol = 1e-9, label = "CPU backend")
end
