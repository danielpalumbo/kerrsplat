# What one and two bands can separate in the fluid-frame coefficients: the log-derivatives of the thermal
# synchrotron coefficients with respect to (ln nₑ, ln Θe, ln B, θ_B) at 230 GHz and at 230 + 345 GHz for the
# plasma of the large-N self-fits, and the singular values of those Jacobians (a small singular value is a
# direction the coefficients do not see). Usage: julia --project=../.. coefficient_degeneracy.jl
using KerrSplat, KerrSplat.Transfer, ForwardDiff, LinearAlgebra, Printf
coeffs(x, ν) = (c = thermal_synchrotron(exp(x[1]), exp(x[2]), exp(x[3]), ν, x[4]); [c.jI, c.jQ, c.jV, c.αI, c.ρQ, c.ρV])
names = ["jI", "jQ", "jV", "αI", "ρQ", "ρV"]
params = ["ln ne", "ln Θe", "ln B", "θB"]
for (ne, Θe, B, θ) in ((3e5, 30.0, 20.0, deg2rad(60)), (3e5, 30.0, 20.0, deg2rad(30)), (1e6, 20.0, 30.0, deg2rad(60)))
    x0 = [log(ne), log(Θe), log(B), θ]
    @printf("\nplasma: ne %.1e, Θe %.0f, B %.0f G, pitch angle %.0f°\n", ne, Θe, B, rad2deg(θ))
    J = Dict{Float64,Matrix{Float64}}()
    for ν in (230e9, 345e9)
        c = coeffs(x0, ν)
        Jν = ForwardDiff.jacobian(x -> coeffs(x, ν), x0) ./ abs.(c)      # ∂ ln|c| / ∂ x
        J[ν] = Jν
        @printf("  %.0f GHz: jQ/jI %.3f  jV/jI %.4f  αI/jI %.2e  ρV/αI %.2f  ρQ/αI %.2f\n", ν / 1e9, c[2] / c[1], c[3] / c[1], c[4] / c[1], c[6] / c[4], c[5] / c[4])
        for (k, nm) in enumerate(names)
            @printf("    ∂ln|%s|/∂(%s) = %s\n", nm, join(params, ", "), join([@sprintf("%7.3f", Jν[k, m]) for m in 1:4], " "))
        end
    end
    # the emissivities alone (jI, jQ, jV: what an optically thin polarized image sees at one sample), one band and two
    for (label, rows, bands) in (("emissivities, 230 GHz", 1:3, (230e9,)), ("emissivities, 230 + 345 GHz", 1:3, (230e9, 345e9)),
                                 ("all six coefficients, 230 GHz", 1:6, (230e9,)), ("all six coefficients, 230 + 345 GHz", 1:6, (230e9, 345e9)))
        A = vcat((J[ν][rows, :] for ν in bands)...)
        F = svd(A)
        weak = F.V[:, end]
        @printf("  %-36s singular values %s; weakest direction (%s) = %s\n", label, join([@sprintf("%.3f", s) for s in F.S], ", "), join(params, ", "), join([@sprintf("%.2f", v) for v in weak], " "))
    end
end
