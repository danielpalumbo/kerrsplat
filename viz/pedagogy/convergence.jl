# The triband ngEHT self-fit's descent across every stage of the fitting procedure, against wall-clock time on the
# RTX 2080 SUPER: the stages coloured and labelled, the truth's χ²/N as the floor. A pedagogical summary figure
# (docs/pedagogy/README.md); regenerate after any change of the stages.
#
#     julia --project=viz viz/pedagogy/convergence.jl
using CairoMakie, DelimitedFiles, Printf
datadir = joinpath(@__DIR__, "..", "..", "validation", "ngeht", "output"); outdir = joinpath(@__DIR__, "..", "..", "docs", "pedagogy"); mkpath(outdir)
N = 241264; truth = 0.998
# (tag, label, minutes, kind): the stages in order, each history's first entry being the previous stage's last
stages = [("triband", "Adam with hygiene\n(300 → 74 parcels)", 293.8, :adam), ("triband_resumed", "Adam resumed\n(hygiene spikes)", 581.6, :adam),
          ("triband_cont", "Adam, hygiene off", 564.1, :adam), ("triband_polish", "plain CG polish", 62.9, :polish), ("triband_pcg", "preconditioned CG", 92.7, :polish),
          ("triband_pcg2", "guarded CG", 104.2, :polish), ("triband_dense", "dense LM (no step)", 317.0, :dense), ("triband_dense2", "dense LM, floored", 158.6, :dense),
          ("triband_pcg3", "preconditioned CG ×40", 471.8, :polish), ("triband_pcg4", "×60", 762.6, :polish), ("triband_lsqr", "LSQR ×60", 1242.4, :polish),
          ("triband_dense3", "dense LM, chord steps", 277.2, :dense)]
colors = Dict(:adam => (:steelblue, 0.18), :polish => (:darkorange, 0.18), :dense => (:seagreen, 0.18))
fig = Figure(size = (1600, 900), fontsize = 15)
ax = Axis(fig[1, 1], xlabel = "wall-clock hours on the RTX 2080 SUPER (Float32 transport, 64² pixels × 160 samples, three bands, 241,264 visibility values)",
          ylabel = "χ² / N", yscale = log10, title = "The triband ngEHT self-fit from the over-complete shell to χ²/N 1.0135: every stage of the procedure",
          xgridvisible = false)
t = 0.0; starts = Float64[]
for (i, (tag, label, minutes, kind)) in enumerate(stages)
    h = vec(readdlm(joinpath(datadir, "$(tag)_history.csv"), ',')) ./ N
    n = length(h)
    ts = t .+ range(0, minutes / 60; length = n)
    vspan!(ax, t, t + minutes / 60, color = colors[kind])
    lines!(ax, ts, h, color = kind == :adam ? :navy : kind == :polish ? :darkorange3 : :darkgreen, linewidth = 2.2)
    text!(ax, t + minutes / 120, 6e4, text = string(i), align = (:center, :top), fontsize = 13, font = :bold)
    push!(starts, t)
    global t += minutes / 60
end
hlines!(ax, [truth], color = :black, linestyle = :dash, linewidth = 1.5)
text!(ax, 0.5, truth * 0.82, text = "the truth scores χ²/N 0.998", align = (:left, :top), fontsize = 12)
ylims!(ax, 0.6, 1e5); xlims!(ax, 0, t)
Legend(fig[2, 1], [PolyElement(color = colors[:adam]), PolyElement(color = colors[:polish]), PolyElement(color = colors[:dense])],
       ["Adam on the dual-sweep gradient", "Gauss–Newton polish, matrix-free (J·v by dual tails, Jᵀw by the adjoint sweep)", "Levenberg–Marquardt on the explicit Jacobian (chunked duals)"],
       orientation = :horizontal, tellwidth = false, framevisible = false, labelsize = 12)
names = ["1 Adam with prune/merge/densify hygiene, 300 → 74 parcels (5 h)", "2 Adam resumed; each hygiene event throws the χ² up by a decade (10 h)", "3 Adam with hygiene off (9 h)",
         "4 plain conjugate-gradient polish: the linear solve is the limit (1 h)", "5 preconditioned CG (Hutchinson diagonal), gain-ratio damping (1.5 h)", "6 guarded CG, twelve steps (2 h)",
         "7 dense LM on the explicit Jacobian: no step (flat columns) (5 h)", "8 dense LM with the floored diagonal (2.6 h)", "9 preconditioned CG, forty steps (8 h)", "10 sixty more (13 h)",
         "11 LSQR on the scaled Jacobian, sixty steps (21 h)", "12 dense LM with chord steps on five reused Jacobians (4.7 h)"]
Label(fig[3, 1], join((join(names[3k-2:3k], "   ·   ") for k in 1:4), "\n"), fontsize = 11, tellwidth = false, justification = :left)
Label(fig[4, 1], @sprintf("1,500 Adam iterations and 228 Gauss–Newton steps over %.0f hours; the fields at the end: |B| to 17%%, Θe to 13%%, the field direction to 8° in the emitting region (docs/notes/2026-09-14_ngeht_triband.md)", t),
      fontsize = 12, tellwidth = false)
save(joinpath(outdir, "fitting_convergence.png"), fig, px_per_unit = 2)
println("written docs/pedagogy/fitting_convergence.png")
