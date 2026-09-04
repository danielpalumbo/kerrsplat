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

const GATE2_N = 130          # two re-anchoring intervals of the default Recurrence(64)
gate2_refs = gate2_references()

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
        test_coefficients(CPU(); label = "CPU")
        test_transfer_step(CPU(); label = "CPU")
        test_frames(CPU(); label = "CPU")
        test_gold2020(CPU(); res = 48, N = 500, tol = 0.03, label = "CPU coarse")
    end
    if CUDA.functional()
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
            @test CUDA.limit(CUDA.LIMIT_STACK_SIZE) >= Geodesics.cuda_stack_bytes(Float64)
        end
    else
        @warn "CUDA is not functional on this machine; GPU tests skipped"
    end
end
