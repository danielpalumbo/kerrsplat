using Test
using KerrSplat
using KerrSplat.Geodesics
using KernelAbstractions
using CUDA

include("reference.jl")
include("test_geodesics.jl")

@testset "KerrSplat" begin
    @testset "Geodesics on the CPU backend" begin
        test_geodesics(CPU(); res = 40, N = 24, tol = 1e-12, label = "CPU")
    end
    if CUDA.functional()
        @testset "Geodesics on CUDA" begin
            # Measured GPU/CPU agreement: r, θ to 1e-12 and t̃ to 1e-10 everywhere. The looser
            # tolerances cover rounding differences amplified by Krang's internal cancellation in
            # the polar constants of axis-grazing rays (every ray, for the θo = 1° observer): up
            # to 1.3e-9 in the constants and 6.5e-9 in the position through the azimuth. See the
            # header of test/reference.jl and docs/notes/2026-09-03_direct_azimuth_conditioning.md.
            test_geodesics(CUDABackend(); res = 64, N = 32, tol = 2e-9, tol_pos = 1e-8, label = "CUDA")
            @test CUDA.limit(CUDA.LIMIT_STACK_SIZE) >= Geodesics.CUDA_STACK_BYTES
        end
    else
        @warn "CUDA is not functional on this machine; GPU tests skipped"
    end
end
