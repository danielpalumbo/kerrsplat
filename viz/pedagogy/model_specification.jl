# The specification of the model as a table: the 21 parameters of a parcel in their groups (symbol, meaning, units, the
# self-fits' truth and the shell start), and the physics and conventions the rendering follows. A pedagogical summary
# figure (docs/pedagogy/README.md); regenerate when a row or a rule changes (src/Splats/polarized.jl, the plans).
#
#     julia --project=viz viz/pedagogy/model_specification.jl
using CairoMakie
outdir = joinpath(@__DIR__, "..", "..", "docs", "pedagogy"); mkpath(outdir)
rows = [("geometry", "x, y, z", "the centre at the reference time t₀, Cartesian Boyer–Lindquist (spin axis z)", "M", "2.5–5 M on the midplane ± 0.3", "a shell 2.2–5.5 M, |z| < 0.6"),
        ("", "s₁, s₂, s₃", "log of the principal Gaussian scales", "ln M", "ln 0.7, ln 0.7, ln 0.5", "ln 0.25 (three hundred small parcels)"),
        ("", "q₁…q₄", "the orientation quaternion of the principal axes", "–", "near the identity", "random"),
        ("envelope", "t₀, ln w", "the centre time and log width of a Gaussian temporal envelope", "M", "0, ln 1e9 (always on)", "0, ln 1e9"),
        ("plasma", "ln nₑ", "log of the peak electron density (the Gaussian weight scales it)", "ln cm⁻³", "ln 3e5 ± 0.3 (scaled to 0.6 Jy at 230 GHz)", "a common value: the truth's flux"),
        ("", "ln Θe", "log of the dimensionless electron temperature kT/mc²", "–", "ln 30 ± 0.2", "ln 30"),
        ("", "ln B", "log of the magnetic field strength in the fluid frame", "ln G", "ln 20 ± 0.2", "ln 20"),
        ("", "θ_B, φ_B", "the field's direction in the fluid frame (polar and azimuthal angles about the ZAMO axes r̂, φ̂, −θ̂)", "rad", "π/2 ± 0.3, ± 0.5", "random on the sphere"),
        ("kinematics", "u₁, u₂, u₃", "the fluid's ZAMO three-velocity γβ along r̂, φ̂ and the vertical", "–", "0, 0.3, ± 0.05", "at rest"),
        ("", "ω", "the pattern angular velocity of the centre about the spin axis", "rad/M", "Keplerian, 1/(r^{3/2} + a)", "Keplerian at the start radius")]
physics = ["Spacetime: Kerr with spin a and observer inclination θo (fitted jointly when free); Krang's closed-form geodesics from a screen of pixels, stored as samples along each ray.",
           "Slow light: every sample carries its own coordinate time; a parcel is evaluated where the ray is at that time, on its pattern orbit.",
           "Emission: thermal synchrotron (Leung 2011 with Dexter 2016's rotativities), full Stokes: jI,Q,V, αI,Q,V, ρQ,V, in the fluid frame, boosted and rotated to the screen (Walker–Penrose).",
           "Transport: the polarized step operator per sample (the exact 4×4 exponential of the absorption–rotation matrix), invariants formed with ν̂ = ν/10¹¹ Hz so that Float32 stays finite.",
           "Overlap: parcels are fluid elements whose coefficients add at a sample; per-ray lists keep only the parcels a ray crosses (above sixteen parcels).",
           "Truncation: rays end after n ≤ 2 crossings of the slab |z| < 0.5 M (the direct image and the first two lensed ones).",
           "Observables: images at each frequency → the direct Fourier transform onto (u, v) → Stokes visibilities, closures, or the correlation products through the instrument model (gains, leakage, feed rotation).",
           "Priors and hygiene: Gaussian priors on the pattern and Keplerian rates about the current spin; prune faint parcels, merge duplicates, densify where the gradient is large."]
fig = Figure(size = (1900, 1150), fontsize = 14)
ax = Axis(fig[1, 1], limits = (0, 100, 0, 70), aspect = DataAspect()); hidedecorations!(ax); hidespines!(ax)
text!(ax, 50, 69, text = "The specification of a parcel: 21 parameters", align = (:center, :top), fontsize = 20, font = :bold)
cols = [1, 10, 22, 56, 64, 82]; heads = ["group", "parameters", "meaning", "units", "the self-fits' truth", "the over-complete start"]
y = 64.5
for (c, h) in zip(cols, heads); text!(ax, c, y, text = h, align = (:left, :center), fontsize = 13, font = :bold); end
lines!(ax, [0.5, 99.5], [y - 1.3, y - 1.3], color = :gray40)
for (k, r) in enumerate(rows)
    yk = y - 1.3 - 3.0 * k + 0.6
    k % 2 == 0 && poly!(ax, Rect(0.5, yk - 1.4, 99, 2.9), color = (:gray90, 0.6))
    for (c, v) in zip(cols, r)
        text!(ax, c, yk, text = v, align = (:left, :center), fontsize = 12)
    end
end
ytop = y - 1.3 - 3.0 * length(rows) - 2.5
text!(ax, 50, ytop, text = "The physics and conventions of the rendering", align = (:center, :top), fontsize = 18, font = :bold)
for (k, line) in enumerate(physics)
    text!(ax, 1, ytop - 2.6 - 2.9 * k + 0.3, text = "•  " * line, align = (:left, :center), fontsize = 12.5)
end
Label(fig[2, 1], "src/Splats/polarized.jl (POLARIZED_SPLAT_PARAMS), src/Transfer/ (coefficients, step, frames), src/Geodesics/ (the cache); the plans in docs/plans/ and the notes in docs/notes/ carry the derivations and the validations against Krang, ipole and symphony.", fontsize = 12, tellwidth = false)
save(joinpath(outdir, "model_specification.png"), fig, px_per_unit = 2)
println("written docs/pedagogy/model_specification.png")
