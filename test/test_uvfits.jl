# Gate for the uvfits reader (real-data path): the parse of an ehtim-written polarized
# observation (the EHT 2017 stations observing a two-component Gaussian, thermal noise from
# realistic SEFDs; test/data/synth_eht2017*.uvfits, with the source image) must reproduce
# ehtim's own parse of the same files (test/data/*_ehtim.csv) row by row, and the direct
# transform of the source image must reproduce the noiseless data to the float32 precision of
# the file, which pins the sky conventions of `visibilities` (east = −α, north = +β) and of the
# FITS image reader against an independent code. The noisy file gives a reduced χ² of one.

using DelimitedFiles

function _ehtim_csv(path)
    raw, hdr = readdlm(path, ','; header = true)
    col(name) = raw[:, findfirst(==(name), vec(hdr))]
    return (time = Float64.(col("time_h")), t1 = String.(strip.(string.(col("t1")))), t2 = String.(strip.(string.(col("t2")))),
            u = Float64.(col("u")), v = Float64.(col("v")),
            vis = [SVector(complex(Float64(col("Ire")[k]), Float64(col("Iim")[k])), complex(Float64(col("Qre")[k]), Float64(col("Qim")[k])),
                           complex(Float64(col("Ure")[k]), Float64(col("Uim")[k])), complex(Float64(col("Vre")[k]), Float64(col("Vim")[k]))) for k in 1:size(raw, 1)],
            σ = [SVector(Float64(col("sigI")[k]), Float64(col("sigQ")[k]), Float64(col("sigU")[k]), Float64(col("sigV")[k])) for k in 1:size(raw, 1)],
            tint = Float64.(col("tint")))
end

function test_uvfits()
    dir = joinpath(@__DIR__, "data")
    @testset "uvfits reader vs ehtim" begin
        for name in ("synth_eht2017", "synth_eht2017_noiseless")
            obs = read_uvfits(joinpath(dir, "$name.uvfits"))
            ref = _ehtim_csv(joinpath(dir, "$(name)_ehtim.csv"))
            @test length(obs) == length(ref.time) == 386
            @test obs.mjd == 57854 && obs.freq == 230e9 && obs.bandwidth == 1.856e9 && obs.source == "SYNTH"
            @test abs(obs.ra - 12.513728717168174) < 1e-9 && abs(obs.dec - 12.39112323919932) < 1e-9
            @test sort(obs.stations) == ["AA", "AP", "AZ", "JC", "LM", "PV", "SM", "SR"]
            @test maximum(abs.(obs.time .- ref.time)) < 1e-6
            @test all(obs.stations[obs.s1] .== ref.t1) && all(obs.stations[obs.s2] .== ref.t2)
            # ehtim multiplies the float32 UU, VV by the frequency in float32 (numpy scalar casting)
            @test maximum(abs.(obs.u .- ref.u) ./ (abs.(ref.u) .+ 1)) < 2e-7 && maximum(abs.(obs.v .- ref.v) ./ (abs.(ref.v) .+ 1)) < 2e-7
            @test maximum(maximum(abs.(obs.vis[k] .- ref.vis[k])) for k in 1:386) < 1e-7
            @test maximum(maximum(abs.(obs.σ[k] .- ref.σ[k])) for k in 1:386) < 1e-9
            @test obs.tint == ref.tint
        end
    end
    @testset "source image transform vs ehtim's observation" begin
        img, hdr = read_stokes_fits(joinpath(dir, "synth_eht2017_image.fits"))
        psize = hdr.psize_deg * π / 180
        @test size(img) == (32, 32) && abs(psize - 200e-6 / 206264.806247 / 32) < 1e-13   # 200 μas field over 32 pixels
        @test abs(sum(getindex.(img, 1)) - 1.4) < 1e-9                                   # Jy/pixel image: total flux
        stokes = map(s -> s .* (Transfer.JY / psize^2), img)                             # cgs intensity as `visibilities` expects
        clean = read_uvfits(joinpath(dir, "synth_eht2017_noiseless.uvfits"))
        model = visibilities(stokes, psize, 1.0, 1.0, clean.u, clean.v)
        err = maximum(maximum(abs.(model[k] .- clean.vis[k])) for k in 1:length(clean))
        @test err < 5e-7                                                                # float32 storage of a 1.4 Jy signal
        @info "uvfits: source image vs ehtim's noiseless visibilities, worst Stokes difference $err Jy over $(length(clean)) rows; zero spacing $(real(model[1][1])) Jy"
        # a mirrored sky would break the phases at the long baselines: check the gate has teeth
        mirrored = visibilities(reverse(stokes; dims = 1), psize, 1.0, 1.0, clean.u, clean.v)
        @test maximum(maximum(abs.(mirrored[k] .- clean.vis[k])) for k in 1:length(clean)) > 1e-2
        noisy = read_uvfits(joinpath(dir, "synth_eht2017.uvfits"))
        χ2 = chi2_visibilities(stokes, psize, 1.0, 1.0, VisibilityData(noisy))
        ndata = 8 * length(noisy)
        @test 0.85 < χ2 / ndata < 1.15
        @info "uvfits: reduced χ² of the noisy observation against its source image $(χ2 / ndata)"
        # closure quantities from the scan structure: consistent with the model on the noiseless data
        tri = scan_triangles(clean); quad = scan_quadrangles(clean)
        @test !isempty(tri) && !isempty(quad) && any(t -> any(<(0), t), tri)
        @test maximum(abs.(rem.(closure_phases(clean.vis, tri) .- closure_phases(model, tri), 2π, RoundNearest))) < 1e-5
        @test maximum(abs.(log_closure_amplitudes(clean.vis, quad) .- log_closure_amplitudes(model, quad))) < 1e-5
        @info "uvfits: $(length(tri)) triangles and $(length(quad)) quadrangles over $(length(unique(round.(clean.time; digits = 6)))) scans"
    end
end

"""
    test_scan_average()

`average_scans` on the synthetic fixture (one integration per scan) is the identity, and on a
copy whose rows are doubled 20 s later with perturbed visibilities it applies ehtim's rules:
earliest time, summed integration time, mean u, v, mean visibility, noise √(Σσ²)/n, absent
Stokes parameters staying absent.
"""
function test_scan_average()
    dir = joinpath(@__DIR__, "data")
    obs = read_uvfits(joinpath(dir, "synth_eht2017.uvfits"))
    @testset "scan averaging" begin
        same = average_scans(obs)
        @test length(same) == length(obs) && same.time == obs.time && same.vis == obs.vis && same.σ == obs.σ && same.u == obs.u
        n = length(obs)
        δ = [SVector{4}(0.01 * cis(k), 0.0im, 0.002im, 0.0im) for k in 1:n]
        dbl = Fit.Observation{Float64}(vcat(obs.time, obs.time .+ 20 / 3600), vcat(obs.tint, obs.tint), vcat(obs.s1, obs.s1), vcat(obs.s2, obs.s2), obs.stations,
                                       vcat(obs.u, obs.u .+ 1e3), vcat(obs.v, obs.v .- 1e3), vcat(obs.vis, obs.vis .+ δ),
                                       vcat(obs.σ, [SVector(2σ[1], σ[2], Inf, σ[4]) for σ in obs.σ]), obs.freq, obs.bandwidth, obs.ra, obs.dec, obs.mjd, obs.source)
        av = average_scans(dbl)
        @test length(av) == n && av.time == obs.time && av.tint == 2 .* obs.tint && all(av.u .≈ obs.u .+ 500) && all(av.v .≈ obs.v .- 500)
        @test maximum(abs(av.vis[k][1] - (obs.vis[k][1] + δ[k][1] / 2)) for k in 1:n) < 1e-15
        @test maximum(abs(av.σ[k][1] - sqrt(obs.σ[k][1]^2 + 4obs.σ[k][1]^2) / 2) for k in 1:n) < 1e-15
        @test all(av.vis[k][3] == obs.vis[k][3] && av.σ[k][3] == obs.σ[k][3] for k in 1:n)   # U present in one row only: that row
        @test all(abs(av.σ[k][2] - obs.σ[k][2] / sqrt(2)) < 1e-15 for k in 1:n)
        # a gap larger than the scan threshold separates the rows again
        far = Fit.Observation{Float64}(vcat(obs.time, obs.time .+ 0.1), dbl.tint, dbl.s1, dbl.s2, obs.stations, dbl.u, dbl.v, dbl.vis, dbl.σ, obs.freq, obs.bandwidth, obs.ra, obs.dec, obs.mjd, obs.source)
        @test length(average_scans(far)) == 2n
    end
end
