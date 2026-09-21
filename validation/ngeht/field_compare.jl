# The fields of a triband self-fit state against the truth's, weighted by where the truth emits: on a voxel grid of the
# emitting region, every voxel weighted by the truth's 230 GHz thermal emissivity (or its density), the emission-weighted
# means, the distribution of ln(fit/truth) per field, the field direction's angle, and the fit's local fields at the truth
# parcels' centres. CPU only; a minute for the campaign's state.
#
#     julia -t 4 --project=. validation/ngeht/field_compare.jl [--tag triband_lsqr] [--voxel 0.25]
#
# Reads validation/ngeht/output/<tag>_params.csv and <tag>_truth.csv; prints to stdout (docs/notes/2026-09-14_ngeht_triband.md
# quotes the numbers of the 1.11 and 1.045 states).
using KerrSplat, KerrSplat.Splats, KerrSplat.Transfer, DelimitedFiles, LinearAlgebra, Printf
getopt(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : parse(typeof(default), ARGS[i+1]))
getstr(flag, default) = (i = findfirst(==(flag), ARGS); i === nothing ? default : ARGS[i+1])
tag = getstr("--tag", "triband_lsqr"); voxel = getopt("--voxel", 0.25)
datadir = joinpath(@__DIR__, "output")
fit = readdlm(joinpath(datadir, "$(tag)_params.csv"), ','); truth = readdlm(joinpath(datadir, "$(tag)_truth.csv"), ',')
println("fit $(tag): $(size(fit, 2)) parcels; truth $(size(truth, 2)) parcels")
xs = -6:voxel:6; ys = xs; zs = -1.5:voxel:1.5
"the density, the density-weighted temperature and field strength, and the density-weighted unit field vector (tetrad components) at a point"
function local_fields(p, t, x, y, z)
    ne = 0.0; wΘ = 0.0; wB = 0.0; v = zeros(3)
    for i in 1:size(p, 2)
        n = exp(p[13, i]) * Splats.splat_weight(p, i, t, x, y, z)
        n > 1e-30 || continue
        ne += n; wΘ += n * exp(p[14, i]); wB += n * exp(p[15, i])
        th = p[16, i]; ph = p[17, i]
        v .+= n .* (sin(th) * cos(ph), sin(th) * sin(ph), cos(th))
    end
    return ne > 0 ? (ne, wΘ / ne, wB / ne, v ./ max(norm(v), eps())) : (0.0, 0.0, 0.0, zeros(3))
end
emis(ne, Θ, B) = ne > 0 ? Transfer.thermal_synchrotron(ne, Θ, B, 230e9, π / 2).jI : 0.0
function wquantile(x, w, q)
    o = sortperm(x); c = cumsum(w[o]) ./ sum(w)
    return x[o[something(findfirst(>=(q), c), length(x))]]
end
for t in (0.0, 6.6)
    nf = zeros(length(xs), length(ys), length(zs)); Θf = similar(nf); Bf = similar(nf); vf = zeros(3, size(nf)...)
    nt = similar(nf); Θt = similar(nf); Bt = similar(nf); vt = similar(vf); jt = similar(nf)
    for (k, z) in enumerate(zs), (j, y) in enumerate(ys), (i, x) in enumerate(xs)
        nf[i, j, k], Θf[i, j, k], Bf[i, j, k], vf[:, i, j, k] = local_fields(fit, t, x, y, z)
        nt[i, j, k], Θt[i, j, k], Bt[i, j, k], vt[:, i, j, k] = local_fields(truth, t, x, y, z)
        jt[i, j, k] = emis(nt[i, j, k], Θt[i, j, k], Bt[i, j, k])
    end
    wj = vec(jt) ./ sum(jt); wn = vec(nt) ./ sum(nt)
    println("\n== t = $t M   (grid $(length(xs))×$(length(ys))×$(length(zs)) of $voxel M over |x,y| ≤ 6, |z| ≤ 1.5)")
    for (name, ft, ff) in (("ne [cm^-3]", nt, nf), ("Θe", Θt, Θf), ("B [G]", Bt, Bf))
        mt = sum(wj .* vec(ft)); mf = sum(wj .* vec(ff))
        @printf("  %-11s emission-weighted mean: truth %9.3g  fit %9.3g  (fit/truth %.3f)\n", name, mt, mf, mf / mt)
    end
    @printf("  emission weight where the fit has no density: %.2e\n", sum(wj[vec(nf) .<= 0]))
    both = (vec(nf) .> 0) .& (vec(nt) .> 0)
    for (name, ft, ff) in (("ne", nt, nf), ("Θe", Θt, Θf), ("B", Bt, Bf))
        r = log.(vec(ff)[both] ./ vec(ft)[both])
        for (wname, w) in (("emission", wj), ("density", wn))
            ww = w[both] ./ sum(w[both])
            @printf("  %-3s %-8s-weighted: mean |ln(fit/truth)| %.3f   bias %+.3f   16/50/84%% of ln ratio %+.3f/%+.3f/%+.3f\n", name, wname, sum(ww .* abs.(r)), sum(ww .* r),
                    wquantile(r, ww, 0.16), wquantile(r, ww, 0.5), wquantile(r, ww, 0.84))
        end
    end
    Vf = reshape(vf, 3, :); Vt = reshape(vt, 3, :); dots = [dot(Vf[:, i], Vt[:, i]) for i in 1:length(nf)][both]
    ang = acosd.(clamp.(dots, -1, 1)); axis = acosd.(clamp.(abs.(dots), 0, 1)); ww = wj[both] ./ sum(wj[both])
    @printf("  B direction, emission-weighted: mean angle %.1f°, median %.1f°; as an axis (sign ignored) mean %.1f°, median %.1f°; weight with the sign flipped %.2f\n",
            sum(ww .* ang), wquantile(ang, ww, 0.5), sum(ww .* axis), wquantile(axis, ww, 0.5), sum(ww[dots .< 0]))
    m = Splats.recovery_metrics(fit, truth, t, range(-6, 6; length = 25), range(-6, 6; length = 25), range(-1.5, 1.5; length = 7))
    @printf("  the driver's recovery metrics (density PSNR %.1f dB, relative density error %.3f, density-weighted temperature error %.3f, field error %.3f)\n", m.psnr_density, m.rel_density, m.temperature, m.field)
end
println("\n== at the truth parcels' centres (t = 0): the parcel's own values, then the fit's local density-weighted fields")
println("  parcel   r [M]   ne_truth  ne_fit    Θ_truth Θ_fit    B_truth B_fit    B-direction angle")
for i in 1:size(truth, 2)
    x, y, z = truth[1, i], truth[2, i], truth[3, i]
    nT, ΘT, BT, vT = local_fields(truth, 0.0, x, y, z); nF, ΘF, BF, vF = local_fields(fit, 0.0, x, y, z)
    @printf("  %d      %5.2f   %8.3g  %8.3g  %6.1f  %6.1f   %6.1f  %6.1f   %5.1f°\n", i, hypot(x, y), nT, nF, ΘT, ΘF, BT, BF, acosd(clamp(dot(vF, vT), -1, 1)))
end
