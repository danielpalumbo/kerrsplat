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
    # ehtim's circular representation lists a baseline with its stations in the array table's order, so some rows are the
    # reverse of the file's (and the reader's): the reversed baseline has the conjugate-transposed coherency, (RR*, LL*, LR*, RL*)
    rowof(f, k) = (r = findfirst(i -> abs(obs.time[i] - f.time[k]) < 1e-6 && obs.stations[obs.s1[i]] == f.t1[k] && obs.stations[obs.s2[i]] == f.t2[k], 1:length(obs));
                   r !== nothing ? (r, false) : (findfirst(i -> abs(obs.time[i] - f.time[k]) < 1e-6 && obs.stations[obs.s1[i]] == f.t2[k] && obs.stations[obs.s2[i]] == f.t1[k], 1:length(obs)), true))
    reversed(p) = SVector(conj(p[1]), conj(p[2]), conj(p[4]), conj(p[3]))
    @testset "the reader's products equal ehtim's" begin
        f = _jones_fixture(dir, "corrected")
        worst = 0.0; nrev = 0
        for k in eachindex(f.time)
            r, rev = rowof(f, k)
            @test r !== nothing
            ref = rev ? reversed(f.clean[k]) : f.clean[k]
            nrev += rev
            worst = max(worst, maximum(abs.(obs.coh[r] .- ref)) / maximum(abs.(ref)))
            @test all(isfinite, obs.σ_coh[r])
        end
        @test worst < 2e-7
        @info "correlation products vs ehtim's circular parse: worst relative difference $worst (float32 file; $nrev of $(length(f.time)) baselines listed reversed by ehtim)"
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
        inst = InstrumentModel(obs)
        @test inst.polarized && inst.leakage && inst.corrected && inst.nseg == maximum(scan_index(obs))
        gains, dterms = zero_instrument(inst)
        rows = 1:length(obs)
        model = [obs.vis[r] .* (1 + 0.1im) .+ SVector(0.01, 0.0, 0.0, 0.0) for r in rows]
        χ = chi2_instrument(model, obs, rows, inst, gains, dterms)
        χs = sum(sum(abs2, (model[k] .- obs.vis[r]) ./ obs.σ[r]) for (k, r) in enumerate(rows))      # σ_V = σ_I, σ_U = σ_Q here
        @test abs(χ - χs) <= 1e-10 * χs
        @test abs(sum(abs2, instrument_residuals(model, obs, rows, inst, gains, dterms)) - χ) <= 1e-10 * χ
        g2 = copy(gains)
        for g in 1:inst.nseg
            c = gain_column(inst, inst.ref[g], g); g2[2, c] = 1.0; g2[4, c] = 0.5
        end
        @test chi2_instrument(model, obs, rows, inst, g2, dterms) ≈ χ                  # the reference phases are the gauge
        gm, dm = free_mask(inst, obs)
        @test all(.!gm[2, [gain_column(inst, inst.ref[g], g) for g in 1:inst.nseg]]) && all(.!gm[4, [gain_column(inst, inst.ref[g], g) for g in 1:inst.nseg]]) && all(dm)   # every station of this file is observed
        other = findfirst(s -> s != inst.ref[1] && (s in obs.s1[inst.seg .== 1] || s in obs.s2[inst.seg .== 1]), 1:nstations(inst))
        g3 = copy(gains); g3[2, gain_column(inst, other, 1)] = 0.3
        @test chi2_instrument(model, obs, rows, inst, g3, dterms) != χ
        gains .= 0.1; dterms .= 0.05
        pen = penalty_instrument(inst, gains, dterms)
        expected = sum((0.1 / inst.σ_lg[s])^2 + (0.1 / inst.σ_lgrat)^2 + (0.1 / inst.σ_gprat)^2 for s in 1:nstations(inst)) * inst.nseg + 4 * nstations(inst) * (0.05 / inst.σ_d)^2
        @test pen ≈ expected
        @test inst.σ_lg[findfirst(==("LM"), obs.stations)] == 1.0 && inst.σ_lg[findfirst(==("AA"), obs.stations)] == 0.2
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
        Jg = ForwardDiff.jacobian(x -> instrument_residuals(model, obs, rows, inst, reshape(x, size(gains)), dterms), vec(gains))
        Jd = ForwardDiff.jacobian(x -> instrument_residuals(model, obs, rows, inst, gains, reshape(x, size(dterms))), vec(dterms))
        @test all(isfinite, Jg) && all(isfinite, Jd) && any(!=(0), Jg) && any(!=(0), Jd)
        @info "instrument model: $(inst.nseg) scans, $(nstations(inst)) stations, $(count(gm)) free gain parameters and $(count(dm)) d-term parts; χ² of the unit instrument matches the Stokes χ² to $(abs(χ - χs) / χs)"
    end
end

"""
    test_feed_rotation()

The feed rotation angles against ehtim's: the antenna positions of the fixture file against the
station table ehtim used; the elevation, parallactic angle and feed angle of every station and
time against the values ehtim applied (test/data/jones_*_matrices.csv), with the mount parameters
of the dump; `feed_angles` over the observation's rows.
"""
function test_feed_rotation()
    dir = joinpath(@__DIR__, "data")
    st, hs = readdlm(joinpath(dir, "jones_stations.csv"), ','; header = true)
    scol(name) = st[:, findfirst(==(name), vec(hs))]
    names = strip.(string.(scol("site")))
    xyz_ref = Dict(names[i] => SVector(Float64(scol("x")[i]), Float64(scol("y")[i]), Float64(scol("z")[i])) for i in eachindex(names))
    mounts = Dict(names[i] => Mount(Float64(scol("fr_par")[i]), Float64(scol("fr_elev")[i]), Float64(scol("fr_off_deg")[i])) for i in eachindex(names))   # f_off in degrees, as dumped
    obs = read_uvfits(joinpath(dir, "synth_eht2017_noiseless.uvfits"))
    @testset "feed rotation vs ehtim" begin
        xyz = antenna_positions(joinpath(dir, "synth_eht2017_noiseless.uvfits"))
        @test sort(collect(keys(xyz))) == sort(names)
        @test maximum(norm(xyz[n] - xyz_ref[n]) for n in names) < 1e-3
        mats, hm = readdlm(joinpath(dir, "jones_raw_matrices.csv"), ','; header = true)
        mcol(name) = mats[:, findfirst(==(name), vec(hm))]
        worst_el = 0.0; worst_par = 0.0; worst_φ = 0.0
        for k in 1:size(mats, 1)
            site = strip(string(mcol("site")[k])); t = Float64(mcol("time_h")[k])
            e, p = station_angles(xyz[site], obs.ra, obs.dec, t, obs.mjd)
            worst_el = max(worst_el, abs(e - Float64(mcol("elev")[k])))
            worst_par = max(worst_par, abs(rem(p - Float64(mcol("parang")[k]), 2π, RoundNearest)))
            worst_φ = max(worst_φ, abs(rem(feed_angle(mounts[site], e, p) - Float64(mcol("phi")[k]), 2π, RoundNearest)))
        end
        @test worst_el < 1e-4 && worst_par < 5e-4 && worst_φ < 5e-4
        φ1, φ2 = feed_angles(obs, xyz, mounts)
        keyof(s, t) = (s, round(t; digits = 6))
        ref = Dict(keyof(strip(string(mcol("site")[k])), Float64(mcol("time_h")[k])) => Float64(mcol("phi")[k]) for k in 1:size(mats, 1))
        worst_rows = maximum(max(abs(rem(φ1[r] - ref[keyof(obs.stations[obs.s1[r]], obs.time[r])], 2π, RoundNearest)),
                                 abs(rem(φ2[r] - ref[keyof(obs.stations[obs.s2[r]], obs.time[r])], 2π, RoundNearest))) for r in 1:length(obs))
        @test worst_rows < 5e-4
        @test haskey(EHT_MOUNTS, "AA") && EHT_MOUNTS["SM"].f_off_deg == 45 && feed_angle(EHT_MOUNTS["SM"], 0.0, 0.0) ≈ deg2rad(45)
        @info "feed rotation vs ehtim over $(size(mats, 1)) station-times: elevation to $worst_el, parallactic angle to $worst_par, feed angle to $worst_φ rad (rows: $worst_rows); sidereal time from the IAU 1982 formula against astropy's"
    end
end

"""
    test_selfcal(backend; res = 6, N = 16, tol = 1e-9, label = "CPU backend")

Self-calibration through the time-resolved likelihood: a synthetic observation of the two-parcel
truth (three scans, two frames, five stations) corrupted by known gains, R/L ratios, d-terms
and feed rotation through the model's own Jones matrices plus noise; the χ² with the true
instrument at the noise level; the sky gradient on `backend` and the instrument gradient on
the host against host Enzyme and ForwardDiff of `chi2_timeresolved`; the joint residuals
against the χ²; the instrument recovered by the Levenberg–Marquardt core from unit gains with
the sky fixed, and a joint sky-plus-instrument polish that lowers χ² with a finite covariance.
"""
function test_selfcal(backend; res = 6, N = 16, tol = 1e-9, label = "CPU backend")
    rng = Random.MersenneTwister(23)
    a = 0.9; θo = deg2rad(60.0)
    fov = 18.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cpu, a, θo; marcher = Fused(64))
    M_solar = 6.5e9; D = 16.8e6 * Transfer.PC; L = gravitational_radius(M_solar); ν = 230e9
    p = polarized_test_params()
    stations = ["AA", "AP", "LM", "PV", "SM"]
    # three scans at two frame times on the baselines among four of the five stations, as an Observation
    time = Float64[]; s1 = Int[]; s2 = Int[]; u = Float64[]; v = Float64[]; frame = Float64[]
    for (k, (t_ut, t_M)) in enumerate(((0.0, 0.0), (0.5, 0.0), (1.0, 20.0)))
        sts = sort(randperm(rng, 5)[1:4])
        for i in 1:4, j in i+1:4
            push!(time, t_ut); push!(s1, sts[i]); push!(s2, sts[j]); push!(u, 3e9 * randn(rng)); push!(v, 3e9 * randn(rng)); push!(frame, t_M)
        end
    end
    n = length(time)
    frames = Dict(t => polarized_image(cpu, p, t, ν, L) for t in unique(frame))
    vis = [visibilities(frames[frame[k]], Δα, L, D, [u[k]], [v[k]])[1] for k in 1:n]
    flux = real(visibilities(frames[0.0], Δα, L, D, [0.0], [0.0])[1][1])
    σ = 0.01 * flux
    obs = Observation{Float64}(time, fill(10.0, n), s1, s2, stations, u, v, vis, fill(SVector(σ, σ, σ, σ), n), ν, 2e9, 12.5, 12.4, 57854, "SYNTH")
    # the true instrument: per-scan R/L gains, d-terms per station, feed angles per row
    φ1 = 0.3 .* randn(rng, n); φ2 = 0.3 .* randn(rng, n)
    inst = InstrumentModel(obs; feedangles = (φ1, φ2), reference = SingleReference("AA"))
    gm, dm = free_mask(inst, obs)
    gains_true, dterms_true = zero_instrument(inst)
    gains_true[gm] .= vcat(0.1 .* randn(rng, count(gm)))
    dterms_true[dm] .= 0.05 .* randn(rng, count(dm))
    for k in 1:n
        g = inst.seg[k]
        J1 = station_jones(inst, gains_true, dterms_true, s1[k], g, φ1[k]); J2 = station_jones(inst, gains_true, dterms_true, s2[k], g, φ2[k])
        V = products(apply_jones(coherency(vis[k]), J1, J2))
        obs.coh[k] = V .+ σ .* SVector{4}(complex.(randn(rng, 4), randn(rng, 4)))
        obs.σ_coh[k] = SVector(σ, σ, σ, σ)
    end
    tr = observed_scans(obs, [0.0, 0.0, 20.0])
    @testset "$label self-calibration through the time-resolved likelihood" begin
        @test frame_times(tr) == [0.0, 20.0] && ndata(tr) == 8n
        instrument = (inst, gains_true, dterms_true)
        χ_true = chi2_timeresolved(p, tr, cpu, L, Δα, D, ν; instrument)
        @test 0.5 * ndata(tr) < χ_true - penalty_instrument(inst, gains_true, dterms_true) < 1.6 * ndata(tr)
        # the sky gradient on the backend and the instrument gradient on the host, against host Enzyme and ForwardDiff
        q = p .+ 0.05 .* randn(rng, size(p))
        gains = gains_true .+ 0.02 .* (gm .* randn(rng, size(gains_true))); dterms = dterms_true .+ 0.01 .* randn(rng, size(dterms_true))
        instrument = (inst, gains, dterms)
        χ_host = chi2_timeresolved(q, tr, cpu, L, Δα, D, ν; instrument)
        g_host = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(x -> chi2_timeresolved(x, tr, cpu, L, Δα, D, ν; instrument)), q)[1]
        gg_ref = ForwardDiff.gradient(g -> chi2_timeresolved(q, tr, cpu, L, Δα, D, ν; instrument = (inst, g, dterms)), gains)
        gd_ref = ForwardDiff.gradient(d -> chi2_timeresolved(q, tr, cpu, L, Δα, D, ν; instrument = (inst, gains, d)), dterms)
        cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
        regenerate!(cache, a, θo; marcher = Recurrence(64))
        params = adapt_to(backend, q); dparams = adapt_to(backend, zeros(size(q)))
        dinst = (zeros(size(gains)), zeros(size(dterms)))
        χ = timeresolved_gradient!(dparams, params, tr, cache, L, Δα, D, ν; instrument, dinstrument = dinst)
        @test abs(χ - χ_host) <= 1e-10 * χ_host
        e = maximum(abs.(Array(dparams) .- g_host)) / maximum(abs.(g_host))
        eg = maximum(abs.(dinst[1] .- gg_ref)) / maximum(abs.(gg_ref)); ed = maximum(abs.(dinst[2] .- gd_ref)) / maximum(abs.(gd_ref))
        @test e <= tol && eg <= 1e-10 && ed <= 1e-10
        # the joint residuals square to the χ²
        r = timeresolved_residuals(q, tr, cpu, L, Δα, D, ν; instrument)
        @test abs(sum(abs2, r) - χ_host) <= 1e-12 * χ_host
        # the instrument recovered from unit gains with the sky fixed at the truth
        x0 = pack(p, falses(size(p)), zero_instrument(inst)[1], gm, zero_instrument(inst)[2], dm)
        x, hist, cov = levenberg_marquardt!(x0, x -> (t = unpack(p, falses(size(p)), zeros(size(gains_true)), gm, zeros(size(dterms_true)), dm, x); timeresolved_residuals(p, tr, cpu, L, Δα, D, ν; instrument = (inst, t[2], t[3]))); iterations = 8, chunk = 12)
        xt = pack(p, falses(size(p)), gains_true, gm, dterms_true, dm)
        σx = sqrt.(max.(diag(cov), 0))
        @test hist[end] < hist[1] && hist[end] <= χ_true * 1.05
        @test maximum(abs.(x .- xt) ./ max.(σx, 1e-3)) < 5
        # the instrument-only solve with the model visibilities held is the same LM problem without the renders
        models = scan_models(p, tr, cpu, L, Δα, D, ν)
        gC, dC = zero_instrument(inst)
        gC, dC, histC, covC = calibrate!(gC, dC, tr, models, inst; masks = (gm, dm), iterations = 8, chunk = 12, phases = false, phase_first = false)
        xC = pack(p, falses(size(p)), gC, gm, dC, dm)
        @test length(histC) == length(hist) && maximum(abs.(histC .- hist) ./ hist) < 1e-8
        @test maximum(abs.(xC .- x)) < 1e-8 && maximum(abs.(covC .- cov)) <= 1e-8 * maximum(abs.(cov))
        @test all(gC[.!gm] .== 0) && all(dC[.!dm] .== 0)
        # the reference-baseline phase start: from the true amplitudes and d-terms with the phases scrambled, the chained
        # phases land within the noise of the truth, and the default solve (phase start, phases first) from a scrambled
        # start reaches the same solution as from zero phases
        gS = copy(gains_true); gS[2, :] .+= 2.5 .* (gm[2, :] .* randn(rng, size(gS, 2))); gS[4, :] .+= 0.7 .* (gm[4, :] .* randn(rng, size(gS, 2)))
        reference_phases!(gS, tr, models, inst; dterms = dterms_true)
        wrap(x) = rem(x, 2π, RoundNearest)
        @test maximum(abs.(wrap.(gS[2, gm[2, :]] .- gains_true[2, gm[2, :]]))) < 0.3
        @test maximum(abs.(wrap.(gS[4, gm[4, :]] .- gains_true[4, gm[4, :]]))) < 0.3
        gD, dD = zero_instrument(inst); gD, dD, histD, _ = calibrate!(gD, dD, tr, models, inst; masks = (gm, dm), iterations = 8, chunk = 12)
        gE = zeros(size(gD)); gE[2, :] .= 2.5 .* (gm[2, :] .* randn(rng, size(gE, 2))); dE = zeros(size(dD))
        gE, dE, histE, _ = calibrate!(gE, dE, tr, models, inst; masks = (gm, dm), iterations = 8, chunk = 12)
        @test histD[end] <= histC[end] * (1 + 1e-6) && abs(histE[end] - histD[end]) <= 1e-6 * histD[end]
        @test length(histD) == 18 && abs(histD[10] - histD[9]) <= 1e-8 * histD[9]      # the full solve starts where the phase solve ended
        @test maximum(abs.(wrap.(gE[2, :] .- gD[2, :]))) < 1e-4 && maximum(abs.(gE[1, :] .- gD[1, :])) < 1e-4
        # the per-product split sums to the instrument χ² and counts every finite product twice (real, imaginary)
        sC = first(s for s in tr.scans if s.data isa ObservedScan); kC = findfirst(s -> s.data isa ObservedScan, tr.scans)
        χP, nP = chi2_products(models[kC], sC.data.obs, sC.data.rows, inst, gC, dC)
        @test sum(χP) ≈ chi2_instrument(models[kC], sC.data.obs, sC.data.rows, inst, gC, dC) && sum(nP) == ndata(sC)
        # a joint sky-plus-instrument polish from a perturbed sky and unit instrument lowers χ² with a finite covariance
        free = freeze(p, (:x, :y, :logne, :logB))
        xj0 = pack(q, free, zeros(size(gains_true)), gm, zeros(size(dterms_true)), dm)
        xj, hj, covj = levenberg_marquardt!(xj0, x -> (t = unpack(q, free, zeros(size(gains_true)), gm, zeros(size(dterms_true)), dm, x); timeresolved_residuals(t[1], tr, cpu, L, Δα, D, ν; instrument = (inst, t[2], t[3]))); iterations = 3, chunk = 12)
        @test hj[end] <= hj[1] && all(isfinite, covj)                 # LM never raises χ²; at the CI size (4² × 12) no step is accepted in 3 iterations
        res >= 6 && @test hj[end] < hj[1]
        # the joint Adam loop lowers χ² from the perturbed sky and unit instrument
        skyA = copy(q); gA, dA = zero_instrument(inst)
        skyA, gA, dA, hA = selfcal!(skyA, gA, dA, tr, cache, L, Δα, D, ν; inst, masks = (gm, dm), free, iterations = 6, η = 0.02, η_inst = 0.05)
        @test length(hA) == 6 && hA[end] < hA[1] && all(isfinite, gA) && all(gA[.!gm] .== 0)
        # a per-row step multiplier: the first update of the scaled row is the factor times the plain one
        s1, _, _, _ = selfcal!(copy(q), zero_instrument(inst)..., tr, cache, L, Δα, D, ν; inst, masks = (gm, dm), free, iterations = 1, η = 0.02, η_inst = 0.05)
        s2, _, _, _ = selfcal!(copy(q), zero_instrument(inst)..., tr, cache, L, Δα, D, ν; inst, masks = (gm, dm), free, iterations = 1, η = 0.02, η_inst = 0.05, steps = (; logB = 0.25))
        r = Fit.prow(:logB); others = setdiff(1:size(q, 1), r)
        @test s2[others, :] == s1[others, :] && any(s1[r, :] .!= q[r, :])
        @test maximum(abs.((s2[r, :] .- q[r, :]) .- 0.25 .* (s1[r, :] .- q[r, :]))) < 1e-12
        @info "$label self-calibration: χ² at the truth $(round(χ_true; digits = 1)) for $(ndata(tr)) values; sky gradient vs host Enzyme $e, gain gradient $eg, d-term gradient $ed; instrument recovered from unit gains: χ² $(round(hist[1]; digits = 1)) → $(round(hist[end]; digits = 1)), worst |Δ|/σ $(round(maximum(abs.(x .- xt) ./ max.(σx, 1e-3)); digits = 2)); joint polish χ² $(round(hj[1]; digits = 1)) → $(round(hj[end]; digits = 1)); joint Adam (6 iterations) $(round(hA[1]; digits = 1)) → $(round(hA[end]; digits = 1))"
    end
end

"""
    test_crosshand_rotation()

`rotate_crosshands`: the products and the Stokes visibilities transform together (RL e^{iθ},
LR e^{−iθ}, Q + iU rotated), and the instrument χ² is invariant under rotating the data by θ
together with the sky's cross-hands by θ and the d-terms by d_R e^{iθ}, d_L e^{−iθ}: the
transformation that a missing global R−L phase (ALMA's 45° feed offset, Paper VII) imprints on
fitted d-terms.
"""
function test_crosshand_rotation()
    dir = joinpath(@__DIR__, "data")
    obs = read_uvfits(joinpath(dir, "synth_eht2017_noiseless.uvfits"))
    rng = Random.MersenneTwister(31)
    @testset "cross-hand phase rotation" begin
        θ = 0.7
        r = rotate_crosshands(obs, θ)
        @test r.vis != obs.vis && all(r.vis[k][1] == obs.vis[k][1] && r.vis[k][4] == obs.vis[k][4] for k in 1:length(obs))
        @test all(r.coh[k][3] ≈ obs.coh[k][3] * cis(θ) && r.coh[k][4] ≈ obs.coh[k][4] * cis(-θ) for k in 1:length(obs))
        @test all(r.coh[k][3] ≈ r.vis[k][2] + im * r.vis[k][3] for k in 1:length(obs))
        @test rotate_crosshands(r, -θ).vis ≈ obs.vis
        inst = InstrumentModel(obs; feedangles = (0.4 .* randn(rng, length(obs)), 0.4 .* randn(rng, length(obs))))
        gains, dterms = zero_instrument(inst)
        gains .= 0.1 .* randn(rng, size(gains)); dterms .= 0.05 .* randn(rng, size(dterms))
        rows = 1:length(obs)
        model = [v .* (1 + 0.05im) for v in obs.vis]
        χa = chi2_instrument(model, obs, rows, inst, gains, dterms)
        model_r = [(rl = (m[2] + im * m[3]) * cis(θ); lr = (m[2] - im * m[3]) * cis(-θ); SVector(m[1], (rl + lr) / 2, (rl - lr) / (2im), m[4])) for m in model]
        dterms_r = copy(dterms)
        for s in 1:nstations(inst)
            dR = complex(dterms[1, s], dterms[2, s]) * cis(θ); dL = complex(dterms[3, s], dterms[4, s]) * cis(-θ)
            dterms_r[:, s] = [real(dR), imag(dR), real(dL), imag(dL)]
        end
        χb = chi2_instrument(model_r, r, rows, inst, gains, dterms_r)
        @test abs(χa - χb) <= 1e-10 * χa
        @info "cross-hand rotation by $θ rad: χ² invariant with the sky's cross-hands and the d-terms rotated along ($(abs(χa - χb) / χa))"
    end
end

"""
    test_dterm_recovery()

ehtim's D-terms recovered through the instrument fit: the corrupted products of the fixture
observation (seeded gains with uniformly random phases, R/L amplitude ratios and d-terms, feed
rotation corrected; no noise) against the clean products as the sky, with gains per time stamp
and d-terms per station fitted by `levenberg_marquardt!` from the gains of ehtim's Jones
matrices (relative to the reference station: the calibration state after a gain solve) and
zero d-terms; the recovered d-terms equal ehtim's and the gains stay theirs.
"""
function test_dterm_recovery()
    dir = joinpath(@__DIR__, "data")
    f = _jones_fixture(dir, "corrected")
    obs0 = read_uvfits(joinpath(dir, "synth_eht2017_noiseless.uvfits"))
    n = length(f.time)
    stations = obs0.stations
    s1 = [findfirst(==(f.t1[k]), stations) for k in 1:n]; s2 = [findfirst(==(f.t2[k]), stations) for k in 1:n]
    σ = 1e-3
    obs = Observation{Float64}(f.time, fill(10.0, n), s1, s2, stations, zeros(n), zeros(n), [stokes(coherency_of_products(c)) for c in f.clean], fill(SVector(σ, σ, σ, σ), n),
                               obs0.freq, obs0.bandwidth, obs0.ra, obs0.dec, obs0.mjd, "SYNTH", copy(f.corrupt), fill(SVector(σ, σ, σ, σ), n))
    key(s, k) = (stations[s], round(f.time[k]; digits = 6))
    φ1 = [f.φ[key(s1[k], k)] for k in 1:n]; φ2 = [f.φ[key(s2[k], k)] for k in 1:n]
    inst = InstrumentModel(obs; feedangles = (φ1, φ2), segmentation = IntegSeg(), reference = SingleReference("AA"))
    gm, dm = free_mask(inst, obs)
    model = [stokes(coherency_of_products(c)) for c in f.clean]
    rows = 1:n
    resid(x) = (t = unpack(zeros(1, 1), falses(1, 1), zeros(size(gm)), gm, zeros(size(dm)), dm, x); instrument_residuals(model, obs, rows, inst, t[2], t[3]))
    # ehtim's gains in our parameterization, relative to the reference station of each time stamp
    true_gains, _ = zero_instrument(inst)
    jones_of(s, g) = f.J[key(s, findfirst(==(g), inst.seg))]
    for g in 1:inst.nseg
        Jref = jones_of(inst.ref[g], g)
        for s in 1:nstations(inst)
            c = gain_column(inst, s, g)
            gm[1, c] || continue
            J = jones_of(s, g)
            gR = J[1, 1]; gL = J[2, 2]
            true_gains[1, c] = log(abs(gR)); true_gains[2, c] = angle(gR / Jref[1, 1])
            true_gains[3, c] = log(abs(gL / gR)); true_gains[4, c] = angle((gL / gR) / (Jref[2, 2] / Jref[1, 1]))
        end
    end
    @testset "ehtim's D-terms recovered through the instrument fit" begin
        x0 = pack(zeros(1, 1), falses(1, 1), true_gains, gm, zeros(size(dm)), dm)
        x, hist, cov = levenberg_marquardt!(x0, resid; iterations = 8, chunk = 12)
        _, gains, dterms = unpack(zeros(1, 1), falses(1, 1), zeros(size(gm)), gm, zeros(size(dm)), dm, x)
        @test hist[end] < 1e-8 * hist[1]
        worst_d = 0.0; worst_g = 0.0
        for (k, J) in f.J
            s = findfirst(==(k[1]), stations)
            r = findfirst(i -> abs(obs.time[i] - k[2]) < 1e-6 && (s1[i] == s || s2[i] == s), 1:n)
            r === nothing && continue                                      # the station is absent at that time
            φ = f.φ[k]
            gR = J[1, 1]; gL = J[2, 2]; dR = J[1, 2] / (gR * cis(2φ)); dL = J[2, 1] / (gL * cis(-2φ))
            worst_d = max(worst_d, abs(complex(dterms[1, s], dterms[2, s]) - dR), abs(complex(dterms[3, s], dterms[4, s]) - dL))
            gRf, gLf = station_gains(inst, gains, s, inst.seg[r])
            worst_g = max(worst_g, abs(abs(gRf) - abs(gR)) / abs(gR), abs(abs(gLf / gRf) - abs(gL / gR)))
        end
        @test worst_d < 1e-6 && worst_g < 1e-6
        @info "ehtim's d-terms recovered from its gains and zero leakage: χ² $(hist[1]) → $(hist[end]); worst d-term error $worst_d, worst gain amplitude / ratio error $worst_g"
    end
end
