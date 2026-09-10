# The diffractive scattering kernel of Sgr A* against ehtim's `sgra_kernel_uv` (test/data/sgra_kernel_ehtim.csv, made by
# validation/uvfits/dump_scattering.py at 230 GHz for ehtim's constants and for Johnson et al. 2018's), and the kernel's
# place in the time-resolved likelihood.
using DelimitedFiles

"""
    test_scattering()

`ScatteringKernel` reproduces ehtim's kernel for both parameter sets at 45 (u, v) points, is one at the origin, and
`taper` leaves model visibilities alone without a kernel.
"""
function test_scattering()
    @testset "scattering kernel vs ehtim" begin
        rows = readdlm(joinpath(@__DIR__, "data", "sgra_kernel_ehtim.csv"), ','; comments = true, comment_char = '#')
        u = Float64.(rows[:, 1]); v = Float64.(rows[:, 2])
        bower = ScatteringKernel(230e9; fwhm_maj = 1.309, fwhm_min = 0.640, pa = 78.0)
        johnson = ScatteringKernel(230e9)
        eb = maximum(abs.(bower.(u, v) .- rows[:, 3])); ej = maximum(abs.(johnson.(u, v) .- rows[:, 4]))
        @test eb < 1e-12 && ej < 1e-12
        @test johnson(0.0, 0.0) == 1 && bower(0.0, 0.0) == 1
        @test minimum(rows[:, 4]) < 0.05                         # the fixture reaches the long baselines where the kernel bites
        model = [SVector{4}(complex.(randn(4), randn(4))) for _ in 1:5]
        @test taper(nothing, model, u[1:5], v[1:5]) === model
        t = taper(johnson, model, u[1:5], v[1:5])
        @test all(t[i] ≈ model[i] .* johnson(u[i], v[i]) for i in 1:5)
        @info "scattering kernel vs ehtim: ehtim's constants to $eb, Johnson et al. 2018 to $ej; kernel range $(extrema(rows[:, 4]))"
    end
end
