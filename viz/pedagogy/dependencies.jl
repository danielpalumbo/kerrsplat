# The library's modules and what they depend on: KerrSplat's four modules layered from the geodesics up to the fits, the
# visualization scripts above them, and the external packages grouped by role, with an edge from each module to the
# packages it uses. A pedagogical summary figure (docs/pedagogy/README.md); regenerate when a dependency changes
# (Project.toml, the `using` lines of src/*/ and viz/).
#
#     julia --project=viz viz/pedagogy/dependencies.jl
using CairoMakie
outdir = joinpath(@__DIR__, "..", "..", "docs", "pedagogy"); mkpath(outdir)
fig = Figure(size = (1700, 1000), fontsize = 14)
ax = Axis(fig[1, 1], limits = (-2, 100, 0, 63), aspect = DataAspect()); hidedecorations!(ax); hidespines!(ax)
box!(x, y, w, h; color, text, fs = 13, stroke = :gray30) = (poly!(ax, Rect(x, y, w, h), color = color, strokecolor = stroke, strokewidth = 1.2); text!(ax, x + w / 2, y + h / 2, text = text, align = (:center, :center), fontsize = fs))
# ---- the modules (left column, bottom-up)
modules = [("Geodesics", 4, "Kerr geodesics on the screen: Krang's closed forms marched\nper pixel, samples stored on the backend (GeodesicCache);\nthe spacetime's dual caches for its derivatives", (:lightsteelblue, 1.0), "uses Krang, JacobiElliptic, KernelAbstractions, CUDA, Adapt, ForwardDiff, StaticArrays"),
           ("Transfer", 15, "the fluid frame, thermal synchrotron coefficients (Leung, Dexter;\nBessel and hypergeometric functions), the polarized step operator,\nFloat32-safe invariants (ν̂), the per-ray element loop", (:lightsteelblue, 1.0), "uses Krang, Bessels, HypergeometricFunctions, ForwardDiff, StaticArrays, Adapt"),
           ("Splats", 26, "the parcel model (21 parameters, pattern motion, per-ray lists),\nthe fused image, the dual sweep (tails + adjoints) on the backend,\npopulations (thermal, power-law, κ), the hygiene routines", (:lightsteelblue, 1.0), "uses KernelAbstractions, ForwardDiff, Enzyme, StaticArrays, Random"),
           ("Fit", 37, "movies and VLBI data (uvfits, closures, scattering), the time-resolved\nlikelihood on the backend, Adam with hygiene, the joint spacetime fit,\nthe Gauss–Newton polishes (LSQR, explicit Jacobian), Laplace errors,\nstation gains and self-calibration, the instrument model", (:lightsteelblue, 1.0), "uses KernelAbstractions, ForwardDiff, Enzyme, Optimisers, FITSIO, StaticArrays, LinearAlgebra"),
           ("viz/ and validation/", 49, "movies and stills (CairoMakie), the pedagogical figures, the drivers of\nevery experiment (shell self-fits, triband ngEHT campaign, M87 2017),\nreference comparisons (ipole, symphony, Krang, ehtim)", (:papayawhip, 1.0), "uses CairoMakie, Meshing, CUDA; Python: ngehtsim, ehtim; C: ipole, symphony")]
for (name, y, desc, col, uses) in modules
    box!(2, y, 12, 9; color = col, text = name, fs = 17)
    box!(14, y, 40, 9; color = (:white, 1.0), text = desc, fs = 11, stroke = :gray70)
    text!(ax, 15, y + 0.5, text = uses, align = (:left, :bottom), fontsize = 9.5, color = :gray30)
end
for k in 1:4
    arrows!(ax, [8.0], [modules[k][2] + 9.3], [0.0], [1.4], color = :gray30, linewidth = 2, arrowsize = 12)
end
text!(ax, 8, 60, text = "KerrSplat.jl (Julia 1.10, Apache-2.0)", align = (:center, :bottom), fontsize = 16, font = :bold)
# ---- the external packages (right column), grouped by role
groups = [("geodesics", ["Krang.jl (git f36f43a)", "JacobiElliptic.jl"], 50, (:honeydew, 1.0)),
          ("device", ["CUDA.jl 5 (runtime pinned to the driver)", "KernelAbstractions.jl (CPU backend for tests, CUDA for real)", "Adapt.jl"], 39, (:honeydew, 1.0)),
          ("differentiation", ["ForwardDiff.jl (duals in the kernels: the sweep, J·v, the spacetime)", "Enzyme.jl 0.13 (reverse mode on the host paths and gates)"], 30, (:honeydew, 1.0)),
          ("numerics", ["StaticArrays.jl (4-vectors, 4×4 operators)", "LinearAlgebra, Random", "Bessels.jl, HypergeometricFunctions.jl", "Optimisers.jl (Adam)"], 18, (:honeydew, 1.0)),
          ("data and figures", ["FITSIO.jl (uvfits)", "CairoMakie.jl, Meshing.jl (viz only)", "Python, separate environment: ngehtsim (campaigns), ehtim, ipole, symphony (references)"], 7, (:honeydew, 1.0))]
for (name, items, y, col) in groups
    h = 2.6 * length(items) + 1.8
    poly!(ax, Rect(60, y, 38, h), color = col, strokecolor = :gray50, strokewidth = 1)
    text!(ax, 61, y + h - 0.4, text = name, align = (:left, :top), fontsize = 12, font = :bold)
    for (i, item) in enumerate(items)
        text!(ax, 62, y + h - 2.6 * i - 0.2, text = item, align = (:left, :center), fontsize = 10.5)
    end
end
text!(ax, 79, 60, text = "external packages, by role (the Manifest pins every version)", align = (:center, :bottom), fontsize = 14, font = :bold)
Label(fig[2, 1], "Every module runs on the CPU backend for its tests and on CUDA for the fits; the kernels are written once in KernelAbstractions and differentiated by ForwardDiff duals inside them, so nothing in the hot path depends on the card.", fontsize = 12, tellwidth = false)
save(joinpath(outdir, "dependencies.png"), fig, px_per_unit = 2)
println("written docs/pedagogy/dependencies.png")
