using Test
using KerrSplat
using KerrSplat.Geodesics
using KernelAbstractions
using CUDA

include("reference.jl")
include("highprec_reference.jl")
include("test_geodesics.jl")
include("test_recurrence.jl")
include("test_quadrature.jl")

const GATE2_N = 130          # two re-anchoring intervals of the default Recurrence(64)
gate2_refs = gate2_references()

@testset "KerrSplat" begin
    test_jacobi()
    @testset "Geodesics on the CPU backend" begin
        test_geodesics(CPU(); res = 40, N = 24, tol = 1e-12, label = "CPU")
        test_recurrence(CPU(), gate2_refs; N = GATE2_N, M = 64, label = "CPU")
        test_quadrature(CPU(); N = 1000, M = 64, tol_ϕ = 2e-8, tol_t = 3e-7, tol_hp_ϕ = 2e-9, tol_hp_t = 2e-8, label = "CPU")
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
            test_quadrature(CUDABackend(); N = 1000, M = 64, tol_ϕ = 2e-8, tol_t = 3e-7, tol_hp_ϕ = 2e-9, tol_hp_t = 2e-8, label = "CUDA")
            @test CUDA.limit(CUDA.LIMIT_STACK_SIZE) >= Geodesics.CUDA_STACK_BYTES
        end
    else
        @warn "CUDA is not functional on this machine; GPU tests skipped"
    end
end
