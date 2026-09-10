# Gate for the instrument model (Comrade's structure): the coherency basis, the RIME with ehtim's
# Jones matrices (test/data/jones_*.csv, made by validation/uvfits/dump_jones.py from the noiseless
# synthetic EHT 2017 observation with seeded gains, R/L ratios, phases and d-terms, for
# feed-rotation-corrected and raw data), the Jones assembly from its components in both forms,
# the reader's correlation products against ehtim's, and the model's χ², gauge, priors and duals.

using DelimitedFiles
using ForwardDiff

function _jones_fixture(dir, label)
    rows, hr = readdlm(joinpath(dir, "jones_$(label)_rows.csv"), ','; header = true)
    mats, hm = readdlm(joinpath(dir, "jones_$(label)_matrices.csv"), ','; header = true)
    col(raw, hdr, name) = raw[:, findfirst(==(name), vec(hdr))]
    c(name) = col(rows, hr, name); m(name) = col(mats, hm, name)
    cplx(a, b) = complex.(Float64.(c(a)), Float64.(c(b)))
    clean = [SVector(cplx("RRre", "RRim")[k], cplx("LLre", "LLim")[k], cplx("RLre", "RLim")[k], cplx("LRre", "LRim")[k]) for k in 1:size(rows, 1)]
    corrupt = [SVector(cplx("cRRre", "cRRim")[k], cplx("cLLre", "cLLim")[k], cplx("cRLre", "cRLim")[k], cplx("cLRre", "cLRim")[k]) for k in 1:size(rows, 1)]
    J = Dict{Tuple{String,Float64},SMatrix{2,2,ComplexF64,4}}(); φ = Dict{Tuple{String,Float64},Float64}()
    for k in 1:size(mats, 1)
        key = (strip(string(m("site")[k])), round(Float64(m("time_h")[k]); digits = 6))
        J[key] = SMatrix{2,2}(complex(m("J11re")[k], m("J11im")[k]), complex(m("J21re")[k], m("J21im")[k]), complex(m("J12re")[k], m("J12im")[k]), complex(m("J22re")[k], m("J22im")[k]))
        φ[key] = Float64(m("phi")[k])
    end
    return (time = Float64.(c("time_h")), t1 = strip.(string.(c("t1"))), t2 = strip.(string.(c("t2"))), clean = clean, corrupt = corrupt, J = J, φ = φ)
end

function test_instrument()
    dir = joinpath(@__DIR__, "data")
    rng = Random.MersenneTwister(3)
    @testset "coherency basis" begin
        s = SVector{4}(randn(rng, ComplexF64, 4))
        C = coherency(s)
        @test stokes(C) ≈ s
        @test products(C) == SVector(C[1, 1], C[2, 2], C[1, 2], C[2, 1]) && coherency_of_products(products(C)) == C
        @test C[1, 1] == s[1] + s[4] && C[2, 2] == s[1] - s[4] && C[1, 2] == s[2] + im * s[3] && C[2, 1] == s[2] - im * s[3]
    end
    obs = read_uvfits(joinpath(dir, "synth_eht2017_noiseless.uvfits"))
    rowof(f, k) = findfirst(i -> abs(obs.time[i] - f.time[k]) < 1e-6 && obs.stations[obs.s1[i]] == f.t1[k] && obs.stations[obs.s2[i]] == f.t2[k], 1:length(obs))
    @testset "the reader's products equal ehtim's" begin
        f = _jones_fixture(dir, "corrected")
        worst = 0.0
        for k in eachindex(f.time)
            r = rowof(f, k)
            @test r !== nothing
            worst = max(worst, maximum(abs.(obs.coh[r] .- f.clean[k])) / maximum(abs.(f.clean[k])))
            @test all(isfinite, obs.σ_coh[r])
        end
        @test worst < 2e-7
        @info "correlation products vs ehtim's circular parse: worst relative difference $worst (float32 file)"
    end
    for label in ("corrected", "raw")
        @testset "Jones corruption vs ehtim ($label)" begin
            f = _jones_fixture(dir, label)
            corrected = label == "corrected"
            worst = 0.0
            for k in eachindex(f.time)
                key1 = (f.t1[k], round(f.time[k]; digits = 6)); key2 = (f.t2[k], round(f.time[k]; digits = 6))
                V = products(apply_jones(coherency_of_products(f.clean[k]), f.J[key1], f.J[key2]))
                worst = max(worst, maximum(abs.(V .- f.corrupt[k])) / maximum(abs.(f.corrupt[k])))
            end
            @test worst < 1e-9
            worstJ = 0.0
            for (key, J) in f.J
                φ = f.φ[key]
                if corrected
                    gR = J[1, 1]; gL = J[2, 2]; dR = J[1, 2] / (gR * cis(2φ)); dL = J[2, 1] / (gL * cis(-2φ))
                else
                    gR = J[1, 1] * cis(φ); gL = J[2, 2] * cis(-φ); dR = J[1, 2] / (cis(φ) * gR); dL = J[2, 1] / (cis(-φ) * gL)
                end
                worstJ = max(worstJ, maximum(abs.(jones(gR, gL, dR, dL, φ; corrected) .- J)) / maximum(abs.(J)))
                @test abs(dR) < 0.5 && abs(dL) < 0.5                   # ehtim's d-terms: 0.1 per part
            end
            @test worstJ < 1e-10
            @info "Jones RIME vs ehtim ($label): products to $worst, the assembled matrices to $worstJ over $(length(f.J)) station-times"
        end
    end
    @testset "instrument model: χ², gauge, priors, duals" begin
        im = InstrumentModel(obs)
        @test im.polarized && im.leakage && im.corrected && im.nseg == maximum(scan_index(obs))
        gains, dterms = zero_instrument(im)
        rows = 1:length(obs)
        model = [obs.vis[r] .* (1 + 0.1im) .+ SVector(0.01, 0.0, 0.0, 0.0) for r in rows]
        χ = chi2_instrument(model, obs, rows, im, gains, dterms)
        χs = sum(sum(abs2, (model[k] .- obs.vis[r]) ./ obs.σ[r]) for (k, r) in enumerate(rows))      # σ_V = σ_I, σ_U = σ_Q here
        @test abs(χ - χs) <= 1e-10 * χs
        @test abs(sum(abs2, instrument_residuals(model, obs, rows, im, gains, dterms)) - χ) <= 1e-10 * χ
        g2 = copy(gains)
        for g in 1:im.nseg
            c = gain_column(im, im.ref[g], g); g2[2, c] = 1.0; g2[4, c] = 0.5
        end
        @test chi2_instrument(model, obs, rows, im, g2, dterms) ≈ χ                  # the reference phases are the gauge
        gm, dm = free_mask(im, obs)
        @test all(.!gm[2, [gain_column(im, im.ref[g], g) for g in 1:im.nseg]]) && all(.!gm[4, [gain_column(im, im.ref[g], g) for g in 1:im.nseg]]) && all(dm)
        other = findfirst(s -> s != im.ref[1] && (s in obs.s1[im.seg .== 1] || s in obs.s2[im.seg .== 1]), 1:nstations(im))
        g3 = copy(gains); g3[2, gain_column(im, other, 1)] = 0.3
        @test chi2_instrument(model, obs, rows, im, g3, dterms) != χ
        gains .= 0.1; dterms .= 0.05
        pen = penalty_instrument(im, gains, dterms)
        expected = sum((0.1 / im.σ_lg[s])^2 + (0.1 / im.σ_lgrat)^2 + (0.1 / im.σ_gprat)^2 for s in 1:nstations(im)) * im.nseg + 4 * nstations(im) * (0.05 / im.σ_d)^2
        @test pen ≈ expected
        @test im.σ_lg[findfirst(==("LM"), obs.stations)] == 1.0 && im.σ_lg[findfirst(==("AA"), obs.stations)] == 0.2
        # Stokes I only: the scalar gain g1 conj(g2) of `apply_gains`
        im1 = InstrumentModel(obs; polarized = false)
        g1 = 0.2 .* randn(rng, 2, nstations(im1) * im1.nseg)
        for g in 1:im1.nseg
            g1[2, gain_column(im1, im1.ref[g], g)] = 0.0
        end
        t1 = [gain_column(im1, obs.s1[r], im1.seg[r]) for r in rows]; t2 = [gain_column(im1, obs.s2[r], im1.seg[r]) for r in rows]
        gained = apply_gains([m[1] for m in model], g1, t1, t2)
        χ1 = chi2_instrument(model, obs, rows, im1, g1, zeros(4, nstations(im1)))
        χ1s = sum(abs2(gained[k] - obs.vis[r][1]) / obs.σ[r][1]^2 for (k, r) in enumerate(rows))
        @test abs(χ1 - χ1s) <= 1e-10 * χ1s
        # duals through the residuals (the Levenberg–Marquardt polish and the Laplace covariance of the instrument)
        gains .= 0.0; dterms .= 0.0
        Jg = ForwardDiff.jacobian(x -> instrument_residuals(model, obs, rows, im, reshape(x, size(gains)), dterms), vec(gains))
        Jd = ForwardDiff.jacobian(x -> instrument_residuals(model, obs, rows, im, gains, reshape(x, size(dterms))), vec(dterms))
        @test all(isfinite, Jg) && all(isfinite, Jd) && any(!=(0), Jg) && any(!=(0), Jd)
        @info "instrument model: $(im.nseg) scans, $(nstations(im)) stations, $(count(gm)) free gain parameters and $(count(dm)) d-term parts; χ² of the unit instrument matches the Stokes χ² to $(abs(χ - χs) / χs)"
    end
end
