using Test
using KerrSplat
using KerrSplat.Geodesics
using KernelAbstractions
using CUDA
using Enzyme
using Optimisers
using Random

include("reference.jl")
include("highprec_reference.jl")
include("test_geodesics.jl")
include("test_recurrence.jl")
include("test_quadrature.jl")
include("test_duals.jl")
include("test_image.jl")
include("test_fused.jl")
include("test_splats.jl")
include("test_coefficients.jl")
include("test_transfer_step.jl")
include("test_frames.jl")
include("test_gold2020.jl")
include("test_polarized.jl")
include("test_polarized_splats.jl")
include("test_slowlight.jl")
include("test_motion.jl")
include("test_fit.jl")
include("test_uvfits.jl")
include("test_instrument.jl")
include("test_winding.jl")
include("test_spacetime.jl")

const GATE2_N = 130          # two re-anchoring intervals of the default Recurrence(64)
@info "computing the BigFloat gate-2 references"
gate2_refs = gate2_references()
@info "references ready; starting the CPU tests"

@testset "KerrSplat" begin
    test_jacobi()
    @testset "Geodesics on the CPU backend" begin
        test_geodesics(CPU(); res = 40, N = 24, tol = 1e-12, label = "CPU")
        test_recurrence(CPU(), gate2_refs; N = GATE2_N, M = 64, label = "CPU")
        # vs Krang: Krang's closed forms are noisy at ~1e-7 (isolated glitches, near-polar observers);
        # the sharp check is the BigFloat rate reference (tol_hp_*), see test_quadrature.jl.
        test_quadrature(CPU(); N = 1000, M = 64, tol_ϕ = 1e-7, tol_t = 5e-7, tol_hp_ϕ = 2e-9, tol_hp_t = 2e-8, label = "CPU")
        test_duals(CPU(); N = 400, tol_rec = 1e-8, tol_fd = 1e-5, label = "CPU")
        test_image(CPU(); res = 48, N = 400, tol = 1e-9, label = "CPU")
        test_fused(CPU(); res = 32, N = 200, label = "CPU")
        test_splats(CPU(); res = 32, N = 300, label = "CPU")
        test_splat_gradients(; res = 12, N = 120, tol = 1e-6)
        test_splat_fit(; res = 12, N = 120, iterations = 300)
        test_kernel_gradient(CPU(); res = 12, N = 120, tol = 1e-12, label = "CPU backend")
        test_stored_gradient(CPU(); res = 12, N = 120, tol = 1e-11, label = "CPU backend")
        test_coefficients(CPU(); label = "CPU")
        test_powerlaw_rotativities()
        test_kappa_coefficients()
        test_transfer_step(CPU(); label = "CPU")
        test_step_adjoint()
        test_frames(CPU(); label = "CPU")
        test_gold2020(CPU(); res = 48, N = 500, tol = 0.03, label = "CPU coarse")
        test_polarized(CPU(); label = "CPU")
        test_composite(CPU(); res = 12, N = 200, label = "CPU")
        test_polarized_splats(CPU(); res = 24, N = 300, label = "CPU")
        test_populations(CPU(); res = 12, N = 150, label = "CPU")
        test_reflection()
        test_polarized_kernel_gradient(CPU(); K = 8, label = "CPU")
        test_polarized_gradient(CPU(); res = 8, N = 40, tol = 1e-10, label = "CPU backend")
        test_winding(CPU(); res = 24, N = 300, label = "CPU")
        test_winding_gradient()
        test_polarized_gradient_winding(CPU(); res = 8, N = 120, tol = 1e-10, label = "CPU backend")
        test_polarized_splat_gradients(; res = 10, N = 100, tol = 1e-5)
        test_polarized_splat_fit(; res = 10, N = 100, iterations = 150)
        test_slowlight(CPU(); res = 48, N = 300, label = "CPU")
        test_motion(CPU(); res = 24, N = 200, label = "CPU")
        test_motion_gradient(; res = 12, N = 120)
        test_advection(CPU(); res = 24, N = 200, label = "CPU")
        test_pattern_vs_fluid(; res = 8, N = 80, iterations = 120)
        test_fit(; res = 10, N = 80, iterations = (40, 40, 60))
        test_hygiene(; res = 10, N = 80)
        test_fit_schedule(; res = 8, N = 60)
        test_fits_io(; res = 8, N = 60)
        test_visibilities(; res = 8, N = 60)
        test_uvfits()
        test_scan_average()
        test_instrument()
        test_feed_rotation()
        test_priors(; res = 8, N = 60)
        test_pattern_prior()
        test_gains(; res = 8, N = 60)
        test_chi2_gradient(CPU(); res = 8, N = 40, tol = 1e-9, label = "CPU backend")
        test_image_loss_gradient(CPU(); res = 8, N = 40, tol = 1e-9, label = "CPU backend")
        test_binning(CPU(); res = 6, N = 40, tol = 1e-9, label = "CPU backend")
        test_spacetime_duals(; res = 8, N = 60)
        test_spacetime_fit(; res = 8, N = 60)
        test_fisher(; res = 8, N = 60)
        test_polish(; res = 8, N = 40)
        test_timeresolved(CPU(); res = 6, N = 16, tol = 1e-9, label = "CPU backend")
        test_selfcal(CPU(); res = 6, N = 16, tol = 1e-9, label = "CPU backend")
    end
    if CUDA.functional()
        @info "CPU tests done; starting the CUDA tests"
        @testset "Geodesics on CUDA" begin
            # Measured GPU/CPU agreement: r, θ to 1e-12 and t̃ to 1e-10 everywhere. The looser
            # tolerances cover rounding differences amplified by Krang's internal cancellation in
            # the polar constants of axis-grazing rays (every ray, for the θo = 1° observer): up
            # to 1.3e-9 in the constants and 6.5e-9 in the position through the azimuth. See the
            # header of test/reference.jl and docs/notes/2026-09-03_direct_azimuth_conditioning.md.
            test_geodesics(CUDABackend(); res = 64, N = 32, tol = 2e-9, tol_pos = 1e-8, label = "CUDA")
            test_recurrence(CUDABackend(), gate2_refs; N = GATE2_N, M = 64, label = "CUDA")
            test_quadrature(CUDABackend(); N = 1000, M = 64, tol_ϕ = 1e-7, tol_t = 5e-7, tol_hp_ϕ = 2e-9, tol_hp_t = 2e-8, label = "CUDA")
            test_duals(CUDABackend(); N = 400, tol_rec = 1e-8, tol_fd = 1e-5, label = "CUDA")
            test_image(CUDABackend(); res = 128, N = 1000, tol = 1e-9, label = "CUDA")
            test_fused(CUDABackend(); res = 128, N = 1000, label = "CUDA")
            test_splats(CUDABackend(); res = 64, N = 1000, label = "CUDA")
            test_coefficients(CUDABackend(); label = "CUDA")
            test_transfer_step(CUDABackend(); label = "CUDA")
            test_frames(CUDABackend(); label = "CUDA")
            test_gold2020(CUDABackend(); res = 128, N = 2000, label = "CUDA")
            test_polarized(CUDABackend(); label = "CUDA")
            test_composite(CUDABackend(); res = 24, N = 400, label = "CUDA")
            test_riaf_vs_ipole(CUDABackend(); N = 2000, label = "CUDA")
            test_polarized_splats(CUDABackend(); res = 48, N = 600, label = "CUDA")
            test_populations(CUDABackend(); res = 24, N = 300, label = "CUDA")
            test_winding(CUDABackend(); res = 32, N = 400, label = "CUDA")
            test_polarized_gradient_winding(CUDABackend(); res = 24, N = 300, tol = 1e-9, label = "CUDA")
            test_stored_gradient(CUDABackend(); res = 32, N = 300, tol = 1e-9, label = "CUDA")
            test_polarized_kernel_gradient(CUDABackend(); K = 8, label = "CUDA")
            test_polarized_gradient(CUDABackend(); res = 32, N = 300, tol = 1e-9, label = "CUDA")
            test_chi2_gradient(CUDABackend(); res = 8, N = 40, tol = 1e-9, label = "CUDA")
            test_image_loss_gradient(CUDABackend(); res = 8, N = 40, tol = 1e-9, label = "CUDA")
            test_binning(CUDABackend(); res = 8, N = 100, tol = 1e-9, label = "CUDA")
            test_timeresolved(CUDABackend(); res = 8, N = 60, tol = 1e-9, label = "CUDA")
            test_selfcal(CUDABackend(); res = 8, N = 60, tol = 1e-9, label = "CUDA")
            test_slowlight(CUDABackend(); res = 96, N = 600, label = "CUDA")
            test_motion(CUDABackend(); res = 48, N = 400, label = "CUDA")
            test_advection(CUDABackend(); res = 48, N = 400, label = "CUDA")
            @test CUDA.limit(CUDA.LIMIT_STACK_SIZE) >= Geodesics.cuda_stack_bytes(Float64)
        end
    else
        @warn "CUDA is not functional on this machine; GPU tests skipped"
    end
end
