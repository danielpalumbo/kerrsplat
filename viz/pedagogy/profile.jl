# Where the time goes: the measured cost of the pipeline's pieces on the RTX 2080 SUPER at the triband fit's size, and their
# projection to other NVIDIA cards from peak throughput (an optimistic ceiling: the kernels are latency- and
# occupancy-bound as much as compute-bound, and the host-side pieces do not scale with the card). A pedagogical summary
# figure (docs/pedagogy/README.md); regenerate when the timings or the cards change.
#
#     julia --project=viz viz/pedagogy/profile.jl
using CairoMakie, Printf
outdir = joinpath(@__DIR__, "..", "..", "docs", "pedagogy"); mkpath(outdir)
# ---- measured on the 2080 SUPER (64² × 160 samples, 90 frame–band pairs, 70–300 parcels; docs/notes/2026-09-14_ngeht_triband.md, 2026-09-12_fp32_transport.md)
# (piece, seconds, where it runs, precision)
pieces = [("Adam iteration: the dual-sweep gradient of all frames (Float32)", 60.0, :device, :f32),
          ("Adam iteration: the same in Float64", 6 * 60.0, :device, :f64),
          ("one LSQR polish step (forty solves: J·v + Jᵀw each, Float32)", 20 * 60.0, :device, :f32),
          ("one explicit Jacobian (184 tails passes on eight-partial duals)", 55 * 60.0, :device, :f32),
          ("its normal matrix JᵀJ in Float64 on the host", 60.0, :host, :f64),
          ("a chord step on it (Cholesky + trial residuals)", 3.0, :mixed, :f64),
          ("the spacetime block of a joint iteration (dual renders of every third frame, three LM steps, Float64)", 8 * 60.0, :device, :f64),
          ("self-calibration of the three bands (per-scan LM on the host)", 40.0, :host, :f64),
          ("a checkpoint, a 64² frame's transform on the device", 0.5, :device, :f32)]
# ---- the cards: (name, FP32 TFLOPS, FP64 TFLOPS, memory GB); public peak numbers, dense
cards = [("RTX 2080 SUPER (this workstation)", 11.2, 0.35, 8), ("RTX 4090", 82.6, 1.29, 24), ("RTX 5090", 105.0, 1.64, 32), ("L40S", 91.6, 1.43, 48),
         ("A100 SXM", 19.5, 9.7, 80), ("H100 SXM", 67.0, 34.0, 80), ("H200", 67.0, 34.0, 141)]
fig = Figure(size = (1600, 1000), fontsize = 14)
# left: the measured pieces
ax1 = Axis(fig[1, 1], title = "Measured on the RTX 2080 SUPER at the triband fit's size", xlabel = "seconds (log)", xscale = log10, yticks = (1:length(pieces), [p[1] for p in pieces]),
           yreversed = true, ylabelsize = 12, yticklabelsize = 11)
cols = [p[3] == :device ? (p[4] == :f32 ? :darkorange : :firebrick) : p[3] == :host ? :steelblue : :gray50 for p in pieces]
barplot!(ax1, 1:length(pieces), [p[2] for p in pieces], direction = :x, color = cols)
for (i, p) in enumerate(pieces)
    text!(ax1, p[2] * 1.15, i, text = p[2] >= 60 ? @sprintf("%.0f min", p[2] / 60) : @sprintf("%.1f s", p[2]), align = (:left, :center), fontsize = 11)
end
xlims!(ax1, 0.2, 2e4)
Legend(fig[2, 1], [PolyElement(color = :darkorange), PolyElement(color = :firebrick), PolyElement(color = :steelblue), PolyElement(color = :gray50)],
       ["device, Float32", "device, Float64", "host", "mixed"], orientation = :horizontal, tellwidth = false, framevisible = false)
# right: projections per card for the three routine costs, from the peak-rate ratios (device parts) with the host parts fixed
ax2 = Axis(fig[1, 2], title = "Projected from peak throughput (a ceiling; host parts fixed)", xlabel = "minutes (log)", xscale = log10,
           yticks = (1:length(cards), [@sprintf("%s\nFP32 %.0f / FP64 %.2g TFLOPS, %d GB", c[1], c[2], c[3], c[4]) for c in cards]), yreversed = true, yticklabelsize = 10)
ref32 = cards[1][2]; ref64 = cards[1][3]
routine = [("Adam iteration, Float32 sweep", 60.0, 0.0, :f32), ("LSQR polish step", 20 * 60.0, 0.0, :f32), ("explicit Jacobian + normal matrix", 55 * 60.0, 60.0, :f32),
           ("joint iteration (Float32 sweep + Float64 spacetime block)", 60.0 + 0.0, 0.0, :joint)]
rcols = [:darkorange, :goldenrod, :sienna, :firebrick]
w = 0.2
for (k, (name, dev, host, prec)) in enumerate(routine)
    vals = Float64[]
    for c in cards
        if prec == :joint
            push!(vals, (60.0 * ref32 / c[2] + 8 * 60.0 * ref64 / c[3]) / 60)
        else
            push!(vals, (dev * ref32 / c[2] + host) / 60)
        end
    end
    barplot!(ax2, (1:length(cards)) .+ (k - 2.5) * w, vals, direction = :x, width = w, color = rcols[k], label = name)
    for (i, v) in enumerate(vals)
        text!(ax2, v * 1.1, i + (k - 2.5) * w, text = v >= 1 ? @sprintf("%.0f", v) : @sprintf("%.1f", v), align = (:left, :center), fontsize = 8)
    end
end
xlims!(ax2, 0.05, 200)
axislegend(ax2, position = :rb, labelsize = 10)
Label(fig[3, 1:2], "The bottlenecks: on a consumer card the Float32 transport (nine mechanisms, docs/notes/2026-09-12_fp32_transport.md) is what makes the polishes viable and Float64 is what the joint fit's spacetime block still needs; on a datacenter card Float64 throughout is the right choice. " *
      "Memory: 5 GB on the device for the campaign (the scans, the tails of eight frames, the parcel lists); the explicit Jacobian is 2.8 GB on the host. The Enzyme device gates need a free card; the dual sweep does not.",
      fontsize = 11, tellwidth = false, justification = :left, word_wrap = true)
save(joinpath(outdir, "profile.png"), fig, px_per_unit = 2)
println("written docs/pedagogy/profile.png")
