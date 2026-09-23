# The fitting procedure for VLBI data, as a flowchart: the data's path, the likelihood on the backend, the gradient by the
# dual sweep, the optimizer's stages, the polishes, the errors, and the two side loops (the spacetime block of a joint fit,
# self-calibration). A pedagogical summary figure (docs/pedagogy/README.md); regenerate when a stage changes.
#
#     julia --project=viz viz/pedagogy/pipeline.jl
using CairoMakie
outdir = joinpath(@__DIR__, "..", "..", "docs", "pedagogy"); mkpath(outdir)
fig = Figure(size = (1900, 1500), fontsize = 13)
ax = Axis(fig[1, 1], limits = (0, 120, 0, 92), aspect = DataAspect()); hidedecorations!(ax); hidespines!(ax)
function node!(x, y, w, h, title, body; color = (:aliceblue, 1.0), stroke = :steelblue, tfs = 15, bfs = 12.5)
    poly!(ax, Rect(x, y, w, h), color = color, strokecolor = stroke, strokewidth = 1.6)
    text!(ax, x + w / 2, y + h - 0.5, text = title, align = (:center, :top), fontsize = tfs, font = :bold)
    text!(ax, x + 0.8, y + h - 2.8, text = body, align = (:left, :top), fontsize = bfs, justification = :left)
end
arrow!(x1, y1, x2, y2; color = :gray20, label = "", lw = 2) = (arrows!(ax, [x1], [y1], [x2 - x1], [y2 - y1], color = color, linewidth = lw, arrowsize = 13); isempty(label) || text!(ax, (x1 + x2) / 2 + 0.8, (y1 + y2) / 2, text = label, align = (:left, :center), fontsize = 12, color = :gray20))
# ---- the main chain, top to bottom (x 2–56)
H = 10.4; G = 1.6; ys = [88 - 2.5 - k * (H + G) for k in 1:7]           # the top of row k is ys[k] + H
node!(2, ys[1], 54, H, "1. Data",
      "VLBI scans per band: (u, v), Stokes IQUV visibilities, thermal σ, station indices (uvfits through FITSIO; closures or a\nscattering kernel when the source asks for them). Synthetic campaigns: the coverage of real or simulated arrays (ngehtsim)\nlends its sampling and noise to a rendered truth (synthetic_scans), with per-scan station gains when the exercise asks.")
node!(2, ys[2], 54, H, "2. Model and likelihood (on the backend)",
      "The parcels (21 parameters each) are rendered at every scan time: geodesics from the GeodesicCache, full-Stokes\ntransport along each ray with slow light, n ≤ 2 half-orbits, per-ray parcel lists. Each frame's image goes through the direct\nFourier transform onto its scans' baselines (vis_kernel!), the gains applied as g₁ conj(g₂), and χ² = Σ |V_model − V_data|² / σ²\nover the four Stokes parameters; frames are batched eight to a launch.")
node!(2, ys[3], 54, H, "3. Gradient: the dual sweep",
      "The seed of every frame is the adjoint transform of the weighted residuals (seed_kernel!). The reverse over the compositing\nis written by hand from 4-vectors: a backward pass stores the tails behind each sample, a forward pass carries the adjoint;\nthe per-sample derivatives are ForwardDiff duals inside the kernels (no tape, no Enzyme on the device). Four forward transports.")
node!(2, ys[4], 54, H, "4. Adam with hygiene, then without",
      "From an over-complete shell (hundreds of small parcels, a common density, Keplerian pattern rates). Stage 1: geometry and\ndensities; stage 2: everything. Every `every` iterations: prune faint parcels, merge duplicates, densify where the gradient is\nlarge. Hygiene is useful early and harmful late (each event throws the χ² up by a decade), so the last decade runs with it off.")
node!(2, ys[5], 54, H, "5. Gauss–Newton polish, matrix-free",
      "The residual vector r (real and imaginary parts of the four Stokes visibilities of every scan) and its Jacobian J, never\nformed: J·v by one tails pass with the parameters as one-partial duals seeded along v, Jᵀw by the adjoint sweep seeded by the\nadjoint transform of w. LSQR on the Jacobian scaled by a Hutchinson estimate of diag(JᵀJ) gives the damped step\n(JᵀJ + λD) p = −Jᵀr; the damping follows the gain ratio of the actual to the model's decrease.")
node!(2, ys[6], 54, H, "6. Levenberg–Marquardt on the explicit Jacobian",
      "For a few thousand unknowns J fits in host memory: tails passes on duals with eight partials give eight columns each;\nJᵀJ and Jᵀr in Float64; Cholesky of the damped system with its diagonal floored (flat columns); many chord steps on one\nJacobian at seconds apiece, a fresh Jacobian when a step is rejected. The last decade above the floor, where the matrix-free\nsolves stall on the over-complete basis's null directions.")
node!(2, ys[7], 54, H, "7. Errors and reports",
      "The normal matrix's spectrum: the constrained modes and the null ones (an over-complete basis has many; their count is part\nof the answer). Marginal errors from the pseudo-inverse over the kept modes. The fields on a voxel grid against the truth,\nweighted by its emissivity; movies of the fit and of the fields in three dimensions; checkpoints throughout, resumable.")
labels = ["", "", "", "", "χ²/N a few times the floor", "the matrix-free solves stall", ""]
for k in 1:6
    arrow!(29, ys[k], 29, ys[k + 1] + H + 0.2; label = labels[k + 1])
end
# ---- the side loops (x 64–118), beside the stages they enter
node!(64, ys[4] - 1, 54, H + 2, "Joint fit: the spacetime block (enters 4 and 2)",
      "Every k iterations, three Levenberg–Marquardt steps on the spin and inclination: the residuals of every third frame\nrendered on a dual cache (the geodesics differentiated by duals through Krang's march), the splats' pattern and Keplerian\npriors tied to the current spin. The block that cannot be mimicked leads from the first iteration (no warmup of the splats\non a wrong spacetime); the splats' sweep may run in Float32 while the geodesics and this block stay Float64.",
      color = (:honeydew, 1.0), stroke = :seagreen)
node!(64, ys[6] + 1, 54, H + 2, "Self-calibration (enters 2, before every Adam block and every polish step)",
      "With the sky held: the frames rendered once on the backend, then per scan a Levenberg–Marquardt on the log-amplitudes\nand phases of the stations present, with an amplitude prior, one reference phase per connected component of the scan's\nbaselines (a scan at the high bands can hold baselines sharing no station), and the phases started along a spanning tree of\nthe strongest baselines. The gain factors stay on the device, so the gained fit keeps the device path and the polishes.",
      color = (:honeydew, 1.0), stroke = :seagreen)
arrow!(64, ys[4] + H / 2, 56.3, ys[4] + H / 2; color = :seagreen, lw = 1.8)
arrow!(64, ys[6] + H / 2 + 1, 56.3, ys[6] + H / 2 + 1; color = :seagreen, lw = 1.8)
text!(ax, 60, 91, text = "Fitting the parcel model to VLBI data", align = (:center, :top), fontsize = 18, font = :bold)
text!(ax, 60, 87.6, text = "on the RTX 2080 SUPER at the triband campaign's size (Float32): an iteration of 4 costs a minute, a step of 5 twenty minutes, a Jacobian of 6 an hour and a chord step on it three seconds", align = (:center, :top), fontsize = 13, color = :gray30)
save(joinpath(outdir, "fitting_pipeline.png"), fig, px_per_unit = 2)
println("written docs/pedagogy/fitting_pipeline.png")
