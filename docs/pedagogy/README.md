# Pedagogical figures

High-level summaries of the model, the fitting procedure, the library and its cost, for talks
and for orientation. Every figure is made by a script in `viz/pedagogy/` from the code and the
recorded results, so it can be regenerated (`viz/pedagogy/make_all.sh`, a few minutes on the
CPU) whenever development changes what it shows; the rule is that a change to the model, a
fitting stage, a dependency or a measured timing regenerates the figures it touches and copies
them to Daniel's Dropbox folder (`figures_and_animations/pedagogy/`).

| figure | shows | script | regenerate when |
|---|---|---|---|
| `model_structure.png` | the six-parcel truth in three dimensions around the horizon with a dozen geodesics from the observer's screen, the spin axis and the line of sight; the images the observer sees at 86, 230 and 345 GHz with EVPA ticks, the drawn rays' pixels marked | `model_structure.jl` | the model, the rendering or the truth changes |
| `model_structure_riaf.png` | the model at the scale of a RIAF: a puffy disk (Broderick & Loeb's profile, h/r = 0.7) and a parabolic jet sheath on a lattice of 2,124 parcels whose scale grows with radius, the near half cut away, with geodesics from the screen; the images at 86, 230 and 345 GHz at a tenth of Sgr A*'s density (thin: the ring and shadow at every band) and at Sgr A*'s (thick at 86 GHz: the shadow hidden by the foreground flow, opening as the frequency rises) | `model_structure_riaf.jl` (the card, about fifteen minutes) | the RIAF or jet prescription, the rendering or the regimes change |
| `model_specification.png` | the 21 parameters of a parcel in their groups (symbol, meaning, units, the self-fits' truth, the over-complete start) and the physics and conventions of the rendering | `model_specification.jl` | a parameter row or a rendering rule changes |
| `fitting_pipeline.png` | the fitting procedure for VLBI data as a flowchart: data, the likelihood on the backend, the dual-sweep gradient, Adam with hygiene, the matrix-free and explicit-Jacobian polishes, errors and reports; the joint spacetime block and self-calibration as side loops | `fitting_pipeline.jl` | a stage is added, removed or changes its method |
| `fitting_convergence.png` | the triband ngEHT self-fit's χ²/N against wall-clock time through every stage, from the shell start to 1.0135 (the truth 0.998) | `convergence.jl` | the stages of a reference fit change (reads the histories in `validation/ngeht/output/`) |
| `dependencies.png` | KerrSplat's four modules layered from the geodesics to the fits, what each does and uses, and the external packages by role | `dependencies.jl` | `Project.toml` or a module's `using` lines change |
| `profile.png` | the measured cost of the pipeline's pieces on the RTX 2080 SUPER at the triband fit's size, and their projection to other NVIDIA cards from peak throughput (a ceiling; the host parts fixed) | `profile.jl` | a timing is measured anew or a piece is added |

The numbers in the figures come from `docs/notes/2026-09-14_ngeht_triband.md` and
`docs/notes/2026-09-12_fp32_transport.md`; the animations that go with them are listed in the
Dropbox folder's README.
