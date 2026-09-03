# Gate for KerrSplat.Transfer's coefficient fits: every value is compared with the tables that
# validation/symphony/symphony_table.c generates from the symphony fitting formulae shipped with
# ipole (Dexter 2016 / Shcherbakov 2008 thermal fits, Pandya+ 2016 power-law fits), in ipole's sign
# convention. The remaining differences are the Bessel-function implementation (Bessels.jl vs GSL)
# and libm rounding, so the tolerance is tight. A second part evaluates the same functions inside
# a KernelAbstractions kernel on the requested backend (Bessels.jl, gamma and pow on CUDA).

using DelimitedFiles
using KerrSplat.Transfer
using KerrSplat.Transfer: ME, CL, EE, HPL

const SYMPHONY_DIR = joinpath(@__DIR__, "..", "validation", "symphony")

function read_table(name)
    data, header = readdlm(joinpath(SYMPHONY_DIR, name), ','; header = true)
    cols = Dict(String(strip(h)) => data[:, i] for (i, h) in enumerate(vec(header)))
    return cols, size(data, 1)
end

"Relative difference that tolerates values underflowing to the same tiny magnitudes."
relerr(a, b) = abs(a - b) / max(abs(b), 1e-280)

@kernel function thermal_table_kernel!(out, @Const(ne), @Const(Θe), @Const(B), @Const(ν), @Const(θ))
    i = @index(Global)
    @inbounds begin
        c = thermal_synchrotron(ne[i], Θe[i], B[i], ν[i], θ[i])
        out[i, 1] = c.jI; out[i, 2] = c.jQ; out[i, 3] = c.jV; out[i, 4] = c.αI
        out[i, 5] = c.αQ; out[i, 6] = c.αV; out[i, 7] = c.ρQ; out[i, 8] = c.ρV
        p = powerlaw_synchrotron(ne[i], 3.0, 10.0, 1e5, B[i], ν[i], θ[i])
        out[i, 9] = p.jI; out[i, 10] = p.αI
        out[i, 11] = thermal_emissivity_leung(ne[i], Θe[i], B[i], ν[i], θ[i])
    end
end

function test_coefficients(backend; tol = 1e-12, label = "")
    @testset "Transfer coefficients vs symphony tables ($label)" begin
        th, nth = read_table("thermal_table.csv")
        worst = zeros(13)
        for i in 1:nth
            ne, Θe, B, ν, θ = th["ne"][i], th["Thetae"][i], th["B"][i], th["nu"][i], th["theta"][i]
            c = thermal_synchrotron(ne, Θe, B, ν, θ)
            vals = (c.jI, c.jQ, c.jV, c.αI, c.αQ, c.αV, c.ρQ, c.ρV)
            refs = (th["jI"][i], th["jQ"][i], th["jV"][i], th["aI"][i], th["aQ"][i], th["aV"][i], th["rhoQ"][i], th["rhoV"][i])
            for k in 1:8
                worst[k] = max(worst[k], relerr(vals[k], refs[k]))
            end
            worst[9] = max(worst[9], relerr(thermal_emissivity_leung(ne, Θe, B, ν, θ), th["jI_leung"][i]))
            pj = thermal_synchrotron_pandya(ne, Θe, B, ν, θ)
            worst[10] = max(worst[10], relerr(pj[1], th["jI_pandya"][i]))
            worst[11] = max(worst[11], relerr(pj[2], th["jQ_pandya"][i]))
            worst[12] = max(worst[12], relerr(pj[3], th["jV_pandya"][i]))
            cd = thermal_synchrotron(ne, Θe, B, ν, θ; dexter_rhoV = true)
            worst[13] = max(worst[13], relerr(cd.ρV, th["rhoV_dexter"][i]))
        end
        names = ("jI", "jQ", "jV", "αI", "αQ", "αV", "ρQ", "ρV", "jI (Leung)", "jI (Pandya)", "jQ (Pandya)", "jV (Pandya)", "ρV (Dexter)")
        for k in eachindex(names)
            @test worst[k] < tol
        end
        @info "thermal coefficients vs symphony ($nth rows): worst relative errors" pairs = collect(zip(names, round.(worst, sigdigits = 2)))

        pl, npl = read_table("powerlaw_table.csv")
        worstp = zeros(6)
        for i in 1:npl
            c = powerlaw_synchrotron(pl["ne"][i], pl["p"][i], pl["gamma_min"][i], pl["gamma_max"][i], pl["B"][i], pl["nu"][i], pl["theta"][i])
            vals = (c.jI, c.jQ, c.jV, c.αI, c.αQ, c.αV)
            refs = (pl["jI"][i], pl["jQ"][i], pl["jV"][i], pl["aI"][i], pl["aQ"][i], pl["aV"][i])
            for k in 1:6
                worstp[k] = max(worstp[k], relerr(vals[k], refs[k]))
            end
        end
        for k in 1:6
            @test worstp[k] < tol
        end
        @info "power-law coefficients vs symphony ($npl rows): worst relative errors" jI = worstp[1] jQ = worstp[2] jV = worstp[3] αI = worstp[4] αQ = worstp[5] αV = worstp[6]

        # physical consistency: Kirchhoff's law, sign conventions, the field-aligned limit, invariants
        c = thermal_synchrotron(1e5, 10.0, 30.0, 230e9, 1.0)
        @test c.αI ≈ c.jI / planck(230e9, 10.0)
        @test c.jQ > 0 && c.jI > c.jQ                     # partially linearly polarized along the field-aligned basis
        @test c.jV > 0 && thermal_synchrotron(1e5, 10.0, 30.0, 230e9, π - 1.0).jV < 0   # V flips across θ = 90°
        @test c.ρV > 0 && thermal_synchrotron(1e5, 10.0, 30.0, 230e9, π - 1.0).ρV < 0
        c0 = thermal_synchrotron(1e5, 10.0, 30.0, 230e9, 0.0)
        @test c0.jI == 0 && c0.αI == 0 && c0.ρQ == 0 && c0.ρV != 0
        inv = invariants(c, 230e9)
        @test inv.jI == c.jI / 230e9^2 && inv.αQ == 230e9 * c.αQ && inv.ρV == 230e9 * c.ρV
        capped = cap_polarization(StokesCoefficients(1.0, 0.9, 0.9, 1.0, 0.1, 0.1, 0.0, 0.0))
        @test sqrt(capped.jQ^2 + capped.jV^2) ≈ 0.99 && capped.αQ == 0.1
        x = HPL * 230e9 / (ME * CL^2 * 100.0)   # Planck expansion branch continuity
        @test planck_invariant(230e9, 100.0) ≈ (2 * HPL / CL^2) / expm1(x) rtol = 1e-9   # x ≈ 2e-11: the series branch vs expm1

        # the same functions inside a kernel on the backend
        n = nth
        ne = adapt_to(backend, th["ne"]); Θe = adapt_to(backend, th["Thetae"]); B = adapt_to(backend, th["B"])
        ν = adapt_to(backend, th["nu"]); θ = adapt_to(backend, th["theta"])
        out = KernelAbstractions.zeros(backend, Float64, n, 11)
        thermal_table_kernel!(backend, 64)(out, ne, Θe, B, ν, θ; ndrange = n)
        KernelAbstractions.synchronize(backend)
        O = Array(out)
        worstk = 0.0
        for i in 1:n
            c = thermal_synchrotron(th["ne"][i], th["Thetae"][i], th["B"][i], th["nu"][i], th["theta"][i])
            p = powerlaw_synchrotron(th["ne"][i], 3.0, 10.0, 1e5, th["B"][i], th["nu"][i], th["theta"][i])
            host = (c.jI, c.jQ, c.jV, c.αI, c.αQ, c.αV, c.ρQ, c.ρV, p.jI, p.αI, thermal_emissivity_leung(th["ne"][i], th["Thetae"][i], th["B"][i], th["nu"][i], th["theta"][i]))
            for k in 1:11
                worstk = max(worstk, relerr(O[i, k], host[k]))
            end
        end
        @test worstk < 1e-13
        @info "coefficients in a kernel on $label vs host: worst relative error $worstk"
    end
end

adapt_to(backend, x) = (y = KernelAbstractions.allocate(backend, eltype(x), size(x)); copyto!(y, x); y)
