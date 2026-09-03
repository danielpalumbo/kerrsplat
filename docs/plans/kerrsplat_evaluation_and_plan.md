# Gaussian Splatting of Plasma into the Kerr Spacetime: Code Evaluation and Project Plan

> **Update 2026-07-10:** superseded in part by `kerrsplat_maximal_generality_addendum.md`, which replaces the global velocity/B-field models of §7 with fully self-contained "one-zone splats" (per-splat electron distribution, B, and velocity), reduces the globals to the spacetime block (M, a, θ_o, PA fitted in parallel), and adds a quantitative identifiability ("Fisher audit") program. The code evaluation (§§3–5), feasibility results (§6), and validation gates (§7.5) here remain current.

**Date:** 2026-07-09
**Goal:** A differentiable inference code that represents time-dependent plasma near a black hole as a set of Gaussian primitives ("splats") in the Kerr spacetime, renders them through full general-relativistic polarized radiative transfer, and fits the splat parameters to observer-frequency-dependent movies of Stokes *I, Q, U, V* — a spectro-polarimetric, plasma-level generalization of the neural-field emission tomography of Feng et al. (2026, PI-DEF, arXiv:2602.08029).

---

## 1. Executive summary

**Recommendation: build on Krang.jl.** After a detailed review of both codebases (Krang.jl read in full locally; Jipole cloned from GitHub and reviewed, plus its ApJ paper), Krang.jl is the clearly better foundation, for one architectural reason and several practical ones:

1. **Reverse-mode differentiability is the binding constraint, and only Krang satisfies it.** Fitting ~10³–10⁵ splat parameters requires reverse-mode (adjoint) gradients: one backward pass per loss evaluation regardless of parameter count. Krang is written in a pure, non-mutating, StaticArrays style explicitly designed for Enzyme.jl, with AD rules provided all the way down to its elliptic-function library. Jipole's differentiability — the headline feature of its paper — is *forward-mode* tangent propagation (ForwardDiff) with respect to a handful of scalars (spin, inclination, R_high). Its cost scales linearly with the number of parameters, which is unusable for splat fitting, and its mutation-and-globals code style blocks the reverse-mode tools (Zygote outright; Enzyme untested and likely painful).
2. **Krang's geodesics are analytic and point-wise evaluable.** `emission_coordinates(pixel, τ)` returns (t, r, θ, φ) plus momentum-sign flags at *any* Mino time τ in closed form (Gralla & Lupsasca 2020 formalism, Jacobi elliptic functions). There is no sequential ODE integration to differentiate through, no step-size control entangled with the parameters, and sample placement along rays can be chosen freely — exactly the access pattern a splatting renderer wants. Jipole integrates geodesics with a hand-rolled RK2 midpoint stepper (an ipole port) storing the full trajectory per pixel.
3. **Krang natively supports slow light with a regularized time integral**, so 4-D (time-dependent) rendering of movies is well-posed: relative photon arrival delays between pixels and between winding orders are finite and analytic. Jipole also supports slow light (a genuine strength), but on top of its GRMHD-dump interpolation machinery.
4. **Neither code provides the full polarized transfer solver we need, so that module is new work either way.** Krang provides optically-thin polarization via the analytic Walker–Penrose constant (no Faraday effects, no absorption, V = 0). Jipole — despite the ipole lineage — is currently **unpolarized (Stokes I only)**; its paper explicitly defers polarization to future work. Since the polarized integrator must be written regardless, it should be written once, in an AD-first style, on top of the backend whose geodesic and frame machinery is analytic.
5. **Feasibility is demonstrated, not assumed.** As part of this evaluation I ran an end-to-end smoke test (`krang_enzyme_smoketest.jl`, this directory): a toy Gaussian splat volume-rendered along Krang slow-light geodesics, with Enzyme reverse-mode gradients w.r.t. the splat parameters matching finite differences to **~8×10⁻¹³**, at a warm-gradient cost approximately equal to one forward render. A reverse-mode derivative **through the pixel/geodesic construction itself** (dI/d spin, exercising the radial roots and elliptic integrals) also succeeded, matching finite differences to ~3×10⁻¹¹. Krang's own `examples/neural-net-example.jl` already fits a Lux.jl NeRF through the raytracer with `Optimization.AutoEnzyme()` — a miniature (unpolarized, fast-light) version of this project.

Jipole remains valuable to the project as a *validation asset*: it reproduces ipole to ~15 significant digits on the Gold et al. (2020) analytic test suite, so it (and C ipole) become the reference implementations against which the new renderer's Stokes I is verified, and ipole proper the reference for polarized output.

---

## 2. The target problem, relative to Feng et al. (2026)

PI-DEF (Feng et al. 2026) reconstructs a 4-D *emissivity* field e(t, x) plus a 3-D velocity field around Sgr A*-like black holes from simulated EHT data, with heavy simplifications inherited from BH-NeRF:

- unpolarized (Stokes I only), no absorption, no Faraday effects;
- frequency-independent emissivity (single observing band);
- **fast light** (light-travel time across the domain neglected);
- geodesics precomputed with `kgeo`; redshift enters only as a g² weight (they drop one power by absorbing it into e, cf. their Eq. 3);
- emissivity is an abstract field — not tied to plasma quantities;
- spin and inclination fixed (a spin sensitivity study is a proof of concept).

This project generalizes along every one of these axes: the unknowns become *plasma* fields (electron density n_e, electron temperature Θ_e, magnetic field **B**, bulk four-velocity u^μ) represented by Gaussian primitives; the forward model becomes full polarized GRRT (synchrotron emission, self-absorption, Faraday rotation and conversion) so that the model predicts Stokes (I, Q, U, V)(x, y; ν, t) cubes; light is slow; and the fit is to multi-frequency movies. That is precisely the regime where the ν-dependence (Faraday rotation ∝ ν⁻², self-absorption turnover, spectral index) and the polarization carry the information that breaks the n_e–Θ_e–B degeneracies a Stokes-I-only fit cannot.

Notably, Feng et al. explicitly *dismiss* Gaussian splatting for their setting: "the number of Gaussians needs to be pre-defined, so it is unsuitable for our setting, in which hotspots of emission may appear and disappear over time" (§2.2). This is a solvable objection, and answering it is part of the project's novelty (§7.2): give each splat a smooth temporal envelope (birth time, duration) so splats can fade in/out differentiably, and adopt the standard 3DGS densify/prune outer loop so the primitive count adapts during fitting.

---

## 3. Krang.jl — review

**Source:** local at `/home/daniel/local_scripts/Krang.jl` (git checkout of https://github.com/dchang10/Krang.jl, v0.4.1). Author: Dominic Chang (Harvard/BHI). MIT license. Registered Julia package (`add Krang`), JOSS submission badge, Zenodo DOI 10.5281/zenodo.13936258. Julia ≥ 1.10. The `src/` tree is compact (~4,300 lines) and was reviewed in full.

### 3.1 What it does

- **Analytic Kerr null geodesics** following Gralla & Lupsasca (Phys. Rev. D 101, 044032), parameterized by Mino time. A pixel object (`SlowLightIntensityPixel`, `src/cameras/SlowLightIntensityCamera.jl`) precomputes conserved quantities (η, λ), the four radial roots, and all radial/angular antiderivative constants at construction. After that, `emission_coordinates(pix, τ)` (`src/metrics/Kerr/emission_coordinates.jl:316`) returns `(t_reg, r, θ, φ, νr, νθ, success)` at any Mino time in closed form — O(1) per query, arbitrary sampling order, no accumulated integration error. Turning points, multiple windings, and sub-images are handled analytically.
- **Slow light via a regularized time integral** (`docs/src/time_regularization.md`): the logarithmic/linear divergence of coordinate time at the asymptotic observer is subtracted analytically (following Cárdenas-Avendaño, Lupsasca & Zhu 2023), so relative arrival times — all a movie needs — are exact.
- **Polarization via the Walker–Penrose constant** (`src/materials/physicsUtils.jl`, `docs/src/polarization.md`, `docs/src/newmann_penrose.md`): parallel transport of the polarization plane is *algebraic* in Kerr (type-D spacetime), so no transport ODE is needed. The provided material (`ElectronSynchrotronPowerLawPolarization`) implements the Gelles et al. (2021) optically-thin screen-polarization model: fluid-frame B̂×k̂ polarization, boosted ZAMO→fluid frames via closed-form Jacobians, transported to the Bardeen screen through κ_PW. Stokes V is not produced; there is no absorption and no Faraday mixing.
- **Rendering paradigm**: geodesics are intersected with *surfaces* (cones, equatorial disk, triangle meshes, level sets), and a "material" functor is evaluated at intersections. Volumetric use is supported by the primitives (`generate_ray!` in `src/schemes/RayTrace.jl` marches uniform Mino-time samples), but a volumetric radiative-transfer integrator is **not** part of the package — that is the main thing the new project writes.
- **Momentum and frames**: p_μ at any sample is analytic (`p_bl_d`), and BL↔ZAMO↔fluid frame Jacobians are provided as closed-form StaticArrays matrices — everything needed for redshift factors g = 1/(−p_μ u^μ), pitch angles, and fluid-frame coefficient evaluation.
- **Coordinates**: Boyer–Lindquist plus quasi-Cartesian Kerr–Schild transforms (`boyer_lindquist_to_quasi_cartesian_kerr_schild[_fast_light]`, Enzyme-tested in `test/enzyme_raytracer_tests.jl`) — the natural coordinates in which to define Cartesian Gaussian splats without φ-periodicity headaches.

### 3.2 Differentiability and hardware story

- Pure functional style: no mutation in the hot path, StaticArrays throughout, branches written to "return 0 when emission coordinates do not exist … to play nice with Enzyme's AD" (comment at the top of `emission_coordinates.jl`).
- The elliptic-function dependency **JacobiElliptic.jl** (same author) ships dedicated AD extensions for **Enzyme, ForwardDiff, Zygote, and Reactant**, plus Metal/CUDA compatibility. The AD story goes down to the special functions.
- Extensions: `KrangKernelAbstractionsExt` (portable GPU kernels), `KrangMetalExt`, and `KrangReactantExt` (~1,800 lines) which rewrites the branchy root/integral code into `ifelse` form via `@reactant_overlay` so the whole raytracer can be traced and compiled to XLA. This gives a credible long-term GPU + compiled-autodiff path beyond native Enzyme.
- CI-tested Enzyme forward and reverse rules; `examples/neural-net-example.jl` trains an MLP emission model through the raytracer end-to-end with `Optimization.AutoEnzyme()` and Adam.
- Types interoperate with the EHT VLBI ecosystem: Stokes vectors are `PolarizedTypes.StokesParams` — the same types used by Comrade.jl, easing a later transition from image-domain to visibility-domain fitting.

### 3.3 Gaps (what the new project must add)

- No volumetric polarized radiative transfer (emission/absorption/Faraday along the ray) — the core new module.
- No plasma/synchrotron coefficient library beyond the simple power-law profile material (no thermal j_Q/j_V, no α_S, no ρ_Q/ρ_V; Marszewski et al. 2021 not implemented).
- Kerr only, observer at infinity (fine for this application).
- Single active maintainer; mitigate by pinning/vendoring the dependency (it is already vendored locally) and keeping the coupling surface small (§7.3).

## 4. Jipole — review

**Source:** https://github.com/pedronaethe/Jipole (cloned and reviewed; MIT license, © Motta, Cárdenas-Avendaño & Prather). Lead: Pedro Naethe Motta (IAG–USP), with Alejandro Cárdenas-Avendaño and Ben Prather. **Code paper:** "Jipole: A Differentiable ipole-based Code for Radiative Transfer in Curved Spacetimes," arXiv:2509.07065, ApJ 995, 56 (2025). A follow-up application (pixel-wise image sensitivities of GRMHD images, arXiv:2604.11869) appeared in April 2026. Actively developed (slow-light merged to master 2026-05; a "brisk-light" acceleration branch had commits five days ago).

### 4.1 What it is

A faithful Julia port of ipole (Mościbrodzka & Gammie 2018) for *unpolarized* imaging:

- **Geodesics:** hand-rolled RK2 midpoint stepper traced backward from an ipole-style tetrad camera at r ≈ 1000 M (`push_photon`, `src/geodesics.jl`), with closed-form MKS/FMKS Christoffels and ipole's adaptive step; full trajectory stored per pixel, then intensity integrated forward.
- **Radiative transfer:** ipole's invariant-intensity `approximate_solve` (endpoint-averaged j, α with exact exponential local solution) — **Stokes I only**. The coherency-tensor field in its structs is an unused placeholder; `grep` finds no ρ_Q/ρ_V, no Marszewski/Pandya coefficients, no Q/U/V anywhere. The paper states polarization is left for future work.
- **Plasma models:** iharm3d/KHARMA HDF5 GRMHD dumps (thermal Maxwell–Jüttner only, Leung et al. 2011 emissivity fit + Kirchhoff absorption, R_high/R_low temperature prescription); the Gold et al. (2020) analytic disk; a Novikov–Thorne thin disk. Model selection is a compile-time string constant with if-chains — pluggable only by editing the source, though the "analytic" model is a good template for a closed-form fluid function.
- **Slow light:** yes — geodesics traced once, a rolling window of 3 GRMHD dumps held in memory, fluid quantities linearly interpolated in time, several movie frames integrated concurrently. This is a genuinely useful design pattern (§7.6).
- **AD:** a hybrid tangent-linear scheme: ForwardDiff Jacobians of the geodesic RHS and of the local transfer step, chained by hand along the ray (`src/autodiff.jl`), propagating sensitivities to θ_o, a, and R_high. Validated carefully against central finite differences in the paper (with a thoughtful analysis of why AD and FD disagree in photon-ring pixels, where FD perturbs the geodesics themselves).
- **Style/infra:** `const` globals for configuration, mutable StaticArrays in hot loops, `Threads.@threads` over pixels, no GPU, unregistered (deps-only Project.toml, pinned to Julia 1.12), single frequency and Stokes I per run.

### 4.2 Assessment for this project

Jipole is a solid, verified piece of work whose goals differ from ours in the two ways that matter most:

1. **Wrong AD mode for many-parameter inference.** Forward-mode cost grows linearly with parameter count; a splat model with 10⁴ parameters would cost ~10⁴ renders per gradient. Retrofitting reverse mode would require refactoring away the globals, the mutation, and the per-pixel trajectory storage — effectively rewriting the code in Krang's style.
2. **No polarization.** The single largest physics module we need does not exist there either, so ipole-lineage gives no head start beyond what the (C) ipole reference already provides for validation.

What we *should* take from Jipole: its validation methodology (bit-level comparison against ipole on the Gold et al. 2020 analytic suite and the thin disk), its slow-light windowing design, its `approximate_solve` step form (the unpolarized limit of our integrator), and its AD-vs-FD photon-ring caveat — which, pleasingly, the analytic Krang formulation sidesteps (our smoke test's spin derivative matches FD *through* the geodesic construction, because nothing is discretized).

---

## 5. Head-to-head for Gaussian-splatting coupling

| Criterion | Krang.jl | Jipole |
|---|---|---|
| Geodesics | Analytic (Gralla–Lupsasca, elliptic functions); O(1) evaluation at any Mino time | Numerical RK2 midpoint (ipole port); sequential, trajectory stored per pixel |
| Reverse-mode AD over many parameters | **Yes** — Enzyme by design, CI-tested; demonstrated here on a splat toy problem to ~1e-12 | No — ForwardDiff tangent chains w.r.t. ~3 scalars; mutation + globals block Zygote/Enzyme |
| Gradient w.r.t. spin/inclination | Analytic, exact, consistent (verified vs FD, 3e-11) | Supported (its main feature), FD-validated, with photon-ring subtleties |
| Polarization infrastructure | Walker–Penrose analytic transport, frame Jacobians, screen EVPA; optically thin only | None (Stokes I only); polarization deferred to future work |
| Full IQUV transfer with absorption + Faraday | Must be written (new module) | Must be written (same amount of new physics) |
| Slow light | Native, regularized analytic time integral | Yes, via GRMHD dump window interpolation |
| Arbitrary Julia plasma function at (t,r,θ,φ) | Natural (materials are functors; volumetric sampler is trivial) | Editing-the-source pluggable (string-constant model dispatch) |
| Frequency dependence | Coefficients are ours to write → any ν grid | Single ν per run |
| GPU / compiled-AD path | KernelAbstractions, Metal, **Reactant/XLA** extensions | None (CPU threads) |
| Packaging | Registered package, semver, tests | Unregistered include-style environment |
| Ecosystem | PolarizedTypes/Comrade (EHT VLBI) types | Comrade used in notebooks (analysis side) |
| License / maintenance | MIT; single maintainer, mature | MIT; single maintainer, very active |

**Conclusion:** use **Krang.jl** as the geodesic + frame-transport backbone; write the polarized volumetric renderer, plasma-splat representation, and inference stack as a new package; use **ipole (C) and Jipole as external validation references**, and imitate Jipole's validation discipline.

The one scenario in which Jipole would have been preferable — needing GRMHD-dump-driven forward modeling with derivatives w.r.t. a few global parameters — is not this project.

---

## 6. Feasibility demonstration (run today, this machine)

`krang_enzyme_smoketest.jl` (copy in this directory; environment in the session scratchpad) builds three `SlowLightIntensityPixel`s (scattering, plunging, and near-critical rays; a = 0.94, θ_o = 60°), marches 200 Mino-time samples per ray, converts each sample to quasi-Cartesian Kerr–Schild coordinates, evaluates a Gaussian splat (center, log-width, log-amplitude) times a ZAMO redshift factor g³ obtained from the analytic momentum, and sums — a minimal but structurally faithful optically-thin splat render. Results (Julia 1.10.10, Krang v0.4.1 dev, Enzyme v0.13.179):

| Quantity | Result |
|---|---|
| Warm forward render (3 rays × 200 samples) | 2.1 ms |
| Enzyme **reverse** gradient w.r.t. 5 splat params, warm | 1.9 ms (≈ 1× forward — ideal adjoint scaling) |
| Max relative error vs 5-point central finite differences | **7.9×10⁻¹³** |
| Enzyme reverse dI/da *through pixel construction* (roots + elliptic integrals) | works; rel. err vs FD **3.1×10⁻¹¹** |
| Enzyme compile time (first gradient) | ~18 s |

Implications: (i) the core coupling — Gaussian splats → Krang geodesics → Enzyme reverse mode — works today with machine-precision gradients; (ii) gradient cost is independent of parameter count, so scaling to 10³–10⁵ splat parameters changes nothing structurally; (iii) even spin/inclination can be fit by gradient descent through the analytic raytracer (or, more cheaply, by forward mode over those 2–3 parameters while reverse mode handles the splats).

---

## 7. Project plan: `KerrSplat.jl`

A new Julia package (working name matching this directory) with Krang.jl as its geodesic backend.

### 7.1 Forward model, precisely

For each observed frame time T_obs, observer frequency ν_obs, and pixel (α, β) of a Bardeen screen at inclination θ_o around a Kerr black hole of spin a, mass M, distance D:

1. **Sample the ray.** Construct (once) the `SlowLightIntensityPixel`; draw N samples in Mino time τ_i ∈ (0, τ_total). At each sample Krang gives analytically: coordinates (t̃_i, r_i, θ_i, φ_i), momentum signs (νr, νθ), and hence the exact photon momentum p_μ (from η, λ and the radial/angular potentials). The regularized time t̃_i (negative lookback along the ray) sets the **emission coordinate time** t_i = T_obs + t̃_i (a single global constant fixes the zero point) — this is slow light, exact.
2. **Evaluate the plasma** at (t_i, r_i, θ_i, φ_i) — the splat model of §7.2 returns (n_e, Θ_e, **B** in the fluid orthonormal frame or lab b^μ, u^μ).
3. **Local radiation quantities.** Redshift g = 1/(−p_μ u^μ) (E_obs ≡ 1); fluid-frame frequency ν_i = ν_obs/g; pitch angle cos θ_B = p̂·b̂ in the fluid frame (Krang's ZAMO/fluid Jacobians). Evaluate the **Marszewski et al. (2021)** fitting formulae — j_I, j_Q, j_V; α_I, α_Q, α_V; ρ_Q, ρ_V (j_U = α_U = 0 in the B-aligned Stokes frame) — for the thermal distribution first (power-law and κ later; the paper provides all fits, and they are smooth elementary functions plus modified Bessel functions K_0, K_1, K_2 — analytically differentiable). Convert to relativistic invariants (j/ν², να, νρ).
4. **Polarized transport.** Rotate the fluid-frame Stokes basis to a **parallel-transported screen basis** using the Walker–Penrose constant (Krang's `synchrotronPolarization` machinery generalized to return the frame rotation angle χ_i at each sample rather than only thin emission). This is the grtrans (Dexter 2016) formulation: because transport is algebraic in Kerr, the transfer equation along the ray becomes an ODE for the Stokes vector in a *fixed* frame with position-dependent 4×4 Mueller matrix — no tetrad transport ODE.
5. **Integrate** from the far end of the ray toward the observer over affine parameter (dλ_affine = Σ dτ_Mino): per step, apply the closed-form constant-coefficient solution of the polarized transfer equation (Landi Degl'Innocenti & Landi Degl'Innocenti 1985 give the analytic 4×4 exponential; ipole uses the equivalent split), with substep control bounding the per-step optical depth and Faraday rotation angle. This is unconditionally stable in Faraday-thick regions and reduces exactly to Jipole/ipole's `approximate_solve` in the unpolarized limit — a built-in cross-check.
6. **Camera assembly.** Repeat over pixels, frequencies, frames → model cube S(x, y; ν, t) ∈ {I, Q, U, V}; rotate EVPA to sky position angle; scale to Jy via (M, D); optionally convolve with the instrument beam.

Everything in steps 2–6 is smooth, pure, static-array arithmetic; step 1 is Krang. Two AD regimes:

- **(a) Fixed spacetime + camera (the common case).** The ray samples, p_μ, and PW transport data are *constant* w.r.t. all plasma parameters — precompute them once per (pixel, τ-grid) and reuse across the entire optimization, exactly as PI-DEF precomputes kgeo geodesics. The differentiable graph is then only plasma → coefficients → compositing: small, fast, and AD-able by Enzyme (or even Zygote, since it can be written mutation-free).
- **(b) Spacetime parameters free** (a, θ_o, M/D; Phase 5): differentiate through Krang as demonstrated in the smoke test, or use forward mode over these ≤ 4 parameters alongside reverse mode for the splats (mixed-mode is the right tool: forward for few, reverse for many).

### 7.2 The splat representation (and the answer to Feng's objection)

Each splat k carries:

- **Position** μ_k ∈ ℝ³ in quasi-Cartesian Kerr–Schild coordinates (Krang provides the BL↔KS transforms, Enzyme-tested). Cartesian Gaussians avoid φ-wrapping and polar-axis pathologies.
- **Shape**: covariance Σ_k = R(q_k) diag(exp(2s_k)) R(q_k)ᵀ — unit quaternion q_k + log-scales s_k ∈ ℝ³ (the 3DGS parameterization; always SPD, no constraints).
- **Temporal envelope**: birth time t_k, log-duration w_k, with amplitude a_k(t) = exp(−(t−t_k)²/2e^{2w_k}) (or a compact C² bump). *This is what lets emission appear and disappear* — directly addressing the PI-DEF critique that splat counts are fixed: a splat "not yet born" contributes nothing, differentiably.
- **Amplitudes**: log n_e,k (density is the primary splatted field); optionally δlog Θ_e,k and fluid-frame B-direction perturbations in later phases.
- **Kinematics**: either (i) static centers with the temporal envelope (baseline), or (ii) **advected centers**: μ_k(t) obtained by integrating dx^i/dt = u^i/u^t of the global velocity model from (t_k, μ_k) — a Lagrangian, exactly-consistent version of PI-DEF's soft "dynamics loss," and cheap because it is one small ODE per splat per frame (fixed-step RK4, differentiable), not per ray sample.

Global (non-splat) parameters: an axisymmetric background plasma profile (RIAF-like power laws in r for n_e, Θ_e — possibly zero if the splats carry everything); a parametric magnetic field (e.g., B̂ specified by two angles η, ι in the fluid frame plus radial power-law magnitude — the parameterization used in EHT ring-model fitting); and the **AART/Cunningham sub-Keplerian + infall velocity model** (ξ, β_r, β_φ) exactly as written out in Feng et al. Appendix D.3 — deliberately identical, for comparability. Electron distribution: thermal first; κ and power-law later via the same Marszewski fits.

Model count guidance: hundreds to a few thousand splats × ~15 parameters ≈ 10³–10⁵ parameters — comparable to PI-DEF's MLPs (2×10⁵ weights), and well within Adam-over-Enzyme territory given per-(pixel, frame, frequency) loss separability (mini-batch over rays).

**Densification/pruning outer loop** (non-differentiable, standard 3DGS practice): periodically clone/split splats with large positional-gradient norms, prune splats whose peak contribution is negligible, optionally seed new splats where the data residual is large. Combined with temporal envelopes this fully answers the "fixed number of Gaussians" objection.

### 7.3 Package architecture

```
KerrSplat.jl
├── src/
│   ├── splats.jl        # Gaussian4D type, parameter (re)parameterizations, advection
│   ├── plasma.jl        # AbstractPlasmaModel: splats + background; (t,r,θ,φ) → local fluid state
│   ├── velocity.jl      # AART/Cunningham sub-Keplerian + infall u^μ (Feng App. D.3 formulas)
│   ├── bfield.jl        # parametric B models; b^μ construction from lab B and u^μ
│   ├── coefficients.jl  # Marszewski+2021 fits: thermal (then κ, power-law); Bessels.jl-based
│   ├── transport.jl     # WP frame rotation angles; 4×4 analytic constant-coefficient step;
│   │                    #   invariant-form compositing along Mino-time samples
│   ├── camera.jl        # wraps Krang SlowLightIntensityCamera; sample caching; (ν, t) loops
│   ├── render.jl        # movie-cube assembly; units (M, D, Jy); beam convolution
│   ├── likelihood.jl    # image/cube Gaussian likelihood; masks; (later) Comrade visibilities
│   └── fit.jl           # Enzyme adjoints, Optimisers.jl loop, schedules, densify/prune
├── test/                # unit + gradient + convergence tests (see §7.5)
└── validation/          # ipole/Jipole comparison harness (Gold+2020 suite, RIAF IQUV)
```

Dependencies: Krang.jl (pinned; keep the coupling surface to `SlowLightIntensityPixel`, `emission_coordinates`, `p_bl_d`, frame Jacobians, KS transforms, WP utilities — small enough to vendor if upstream stalls), JacobiElliptic (transitive), StaticArrays, PolarizedTypes, **Bessels.jl** (pure-Julia K_ν; add trivial `EnzymeRules` via K_n′ = −(K_{n−1}+K_{n+1})/2 if needed), Enzyme, Optimisers/Optimization.jl, ComponentArrays (parameter organization), HDF5/JLD2 (I/O). Float64 throughout (photon-ring elliptic-function conditioning; accuracy priority).

### 7.4 Inference strategy

- **Loss**: L = Σ_{t,ν,pix} ‖S_model − S_data‖²/2σ² over the Stokes cube (per-Stokes σ; V typically noisier), plus priors/regularizers: weak priors on Θ_e, n_e ranges (log-space Gaussians), splat-size floors (avoid sub-resolution spikes), temporal-envelope smoothness, optional advection-consistency penalty when using static-center mode (the PI-DEF dynamics loss, Eq. 10, adapted), total-flux/rotation-measure sanity terms.
- **Optimizer**: Adam with cosine/exponential LR decay; regularization weights annealed as in PI-DEF (strong early guidance → data-dominated late). Mini-batch over (frame, frequency, pixel tiles); gradients are embarrassingly parallel over rays (`Threads.@threads`, later KernelAbstractions/Reactant).
- **Staged unfreezing**: fit splat amplitudes/positions first with fixed global physics; then unfreeze B-model and velocity parameters; spin/inclination last (Phase 5), via mixed-mode AD or a small grid/profile likelihood (the PI-DEF Fig. 9 experiment, done properly with gradients).
- **Uncertainty** (stretch): Laplace approximation from Enzyme Hessian-vector products; or HMC (AdvancedHMC) on a reduced parameter set — differentiability makes this available "for free."

### 7.5 Validation plan (gates between phases)

1. **Coefficients**: unit tests of `coefficients.jl` against published Marszewski/Dexter values and symphony output tables; sign conventions audited against IEEE/IAU (Marszewski's stated convention) — EVPA sign errors are the classic GRRT bug.
2. **Geodesics/timing**: spot-check Krang samples (r, θ, φ, Δt) against Jipole's RK2 trajectories for identical pixels (independent implementations, independent formalisms).
3. **Unpolarized images**: reproduce the **Gold et al. (2020) §3.2 analytic test suite** and match ipole/Jipole Stokes I images (Jipole ↔ ipole agree to ~1e-28 NMSE; we should land within quadrature/sampling tolerance, quantified by an N-samples convergence study).
4. **Polarized images**: render the same analytic RIAF/thick-disk models used in the ipole paper (Mościbrodzka & Gammie 2018) and the EHT polarized-transfer comparisons with C ipole as reference; compare I, Q, U, V maps, EVPA patterns, and net/resolved polarization fractions across ν (Faraday-thin → thick).
5. **Slow light**: an analytic flaring hotspot (Gaussian in t) — verify photon-ring echo delays against the known ~photon-orbit period structure and against fast-light rendering in the appropriate limit.
6. **Gradients**: every module and the end-to-end model checked Enzyme-vs-FD (pattern established in the smoke test); gradient tests run in CI.
7. **Recovery tests**: (i) self-consistency — fit data generated by the model itself (PSNR/MSE of recovered n_e, Θ_e, B fields on a voxel grid, à la Feng Table 1); (ii) cross-model — render a GRMHD movie with ipole (IQUV, multi-ν, slow light) and fit splats to it: the honest test that the representation, not just the pipeline, works.

### 7.6 Roadmap (phased; each phase gated by §7.5 items)

- **Phase 0 — scaffolding (short).** Package skeleton, pinned environment, Krang wrapper + cached ray sampling, CI with gradient tests. Reproduce Krang's polarization example and the smoke test inside the package.
- **Phase 1 — optically-thin unpolarized splats, fast light, single ν.** Volumetric g³-weighted emission of splats (the smoke-test physics, productionized); fit synthetic hotspot images; verify densify/prune loop. *This already reproduces PI-DEF-class capability with splats instead of MLPs — a publishable comparison on its own.*
- **Phase 2 — full polarized transfer.** `coefficients.jl` (thermal), WP frame-rotation extraction, analytic 4×4 stepping, invariant compositing; validation gates 1, 3, 4. This is the largest physics work item (~ the effort of the rest combined; budget accordingly).
- **Phase 3 — slow light + time dependence.** Regularized-time bookkeeping, temporal envelopes, advected splats, movie rendering; gate 5. Multi-frequency cubes (loop + cache-friendly layout; coefficients re-evaluated per ν, samples shared).
- **Phase 4 — inference at scale.** Full loss, minibatching, annealing, densification; recovery tests (gate 7): first self-consistent, then ipole-rendered GRMHD ground truth; metrics mirroring Feng et al. for direct comparison.
- **Phase 5 — physics parameters + real-data path.** Spin/inclination/M-D fitting (mixed-mode AD); κ-distribution electrons; visibility-domain likelihood via Comrade.jl (types already shared through PolarizedTypes) for eventual EHT/ngEHT application.
- **Continuous:** performance passes only after correctness gates (KernelAbstractions kernels, Reactant compilation, per-splat ray-interval culling — see risks).

### 7.7 Risks and mitigations

| Risk | Assessment / mitigation |
|---|---|
| Enzyme fragility on the full model (compile times, unsupported patterns) | Highest-likelihood annoyance. Mitigations: regime (a) precomputed-geodesic split makes the differentiated graph small and mutation-free (Zygote-compatible as fallback); function barriers; Reactant path as the long-term compiled alternative; JacobiElliptic already ships rules for all four AD systems. |
| Faraday-thick low-ν channels (huge ρ_V → rotation stiffness) | Analytic constant-coefficient step is unconditionally stable; substep on rotation angle; validate in gate 4 across ν. |
| Uniform Mino-time sampling misses compact splats (quadrature noise → noisy gradients) | Accuracy-first default: large N (10³–10⁴/ray is affordable — smoke test: 2 ms per 600 samples single-threaded, and speed is explicitly secondary). Later: per-(ray, splat) τ-interval localization via a few Newton steps on the analytic coordinates, then Gauss quadrature per interval — sample *placement* need not be differentiated (it's a quadrature choice), only integrand values. |
| Degeneracies (n_e–B–Θ_e; splat multiplicity) | Multi-ν + IQUV is the point — it breaks the classic one-zone degeneracies; plus priors, staged unfreezing, and densify/prune hygiene. Report Laplace uncertainties. |
| Near-critical pixels (τ_total → large, many windings) | Finite sample budget truncates high-n windings (exponentially demagnified anyway); quantify in the convergence study; optional dedicated high-n handling via Krang's sub-image indexing. |
| Emission inside/near horizon | g → 0 kills it physically; clamp r > r_h(1+ε) as in the smoke test. |
| Krang single-maintainer risk | MIT; already vendored locally; coupling surface small (≈ 10 functions); we can pin or fork. |
| Conventions (EVPA sign, Stokes frame, IEEE/IAU) | Dedicated convention tests against ipole maps before any physics conclusions (gate 4). |

---

## 8. Key references

- Feng, Chael, Bromley, Levis, Freeman, Bouman 2026, *Dynamic Black-hole Emission Tomography with Physics-informed Neural Fields* (PI-DEF), arXiv:2602.08029 — `feng_2026.pdf` here.
- Marszewski, Prather, Joshi, Pandya, Gammie 2021, *Updated Transfer Coefficients for Magnetized Plasmas*, ApJ 921, 17 — `Marszewski_2021.pdf` here; the coefficient module's specification.
- Krang.jl — https://github.com/dchang10/Krang.jl (local: `/home/daniel/local_scripts/Krang.jl`, v0.4.1); Gralla & Lupsasca 2020, PRD 101, 044032 (geodesic formalism); Cárdenas-Avendaño, Lupsasca & Zhu 2023, PRD 107, 043030 (time regularization); Gelles et al. 2021 (polarization material).
- Jipole — https://github.com/pedronaethe/Jipole; Motta, Prather & Cárdenas-Avendaño 2025, ApJ 995, 56 (arXiv:2509.07065); application: arXiv:2604.11869.
- ipole — Mościbrodzka & Gammie 2018, MNRAS 475, 43; https://github.com/AFD-Illinois/ipole (polarized validation reference).
- grtrans — Dexter 2016, MNRAS 462, 115 (Walker–Penrose-based polarized transfer formulation to emulate).
- Gold et al. 2020, ApJ 897, 148 (GRRT code-comparison analytic test suite — validation gate 3).
- Cárdenas-Avendaño et al. 2023 (AART velocity model; as transcribed in Feng et al. App. D.3).
- Kerbl et al. 2023 (3D Gaussian splatting: shape parameterization, densify/prune); Levis et al. 2022/2024 (BH-NeRF, orbital polarimetric tomography — lineage).
- Smoke test: `krang_enzyme_smoketest.jl` (this directory); Jipole clone reviewed at the session scratchpad (`.../scratchpad/jipole`).
