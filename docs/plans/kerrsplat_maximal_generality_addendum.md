# Addendum: Maximally General ("One-Zone Splat") Formulation

**Date:** 2026-07-10. Extends `kerrsplat_evaluation_and_plan.md` (v1) in response to three design requirements:

1. **No global flow or magnetic-field model.** Each splat is a self-contained one-zone plasma parcel: its own electron distribution (thermal or power-law), its own magnetic field orientation and strength, its own fluid velocity.
2. **No advection assumption.** Splats either sit in space (motion represented by births/deaths of overlapping splats) or carry their own fitted velocity as a function of time — no Keplerian/AART/infall prior baked into the forward model.
3. **Spacetime parameters (M, a, θ_o) fit in parallel** with all plasma parameters, against ideal I, Q, U, V movies as a function of time and frequency, with well-posedness expected in the high-resolution/high-sensitivity limit from strong lensing (multi-view) plus full spectral sampling (optically thick and thin views of every splat).

**Verdict up front:** this formulation is not only compatible with the Krang-based architecture — it is *cleaner* than v1, because the radiative transfer of independent one-zone parcels is exactly additive at the level of invariant transfer coefficients (§2), which removes the need to ever construct a single global "plasma state" field. The globals collapse from ~20 parameters across four models to a **5–6 parameter spacetime/calibration block** (§4). The costs are: a heavier coefficient module (per-splat distributions, §3), a per-sample cost that scales with local splat multiplicity (mitigated by culling, §6.3), a genuinely harder optimization landscape (§6.4), and an identifiability question that should be *measured*, not asserted — the plan now includes the machinery to do so (§5.4). A second smoke test run today (`krang_enzyme_polarized_smoketest.jl`, this directory) de-risks the key new AD requirement: Enzyme reverse-mode gradients through Krang's polarization pipeline — per-splat fluid velocity (ZAMO boost), per-splat B orientation (fluid frame), complex Walker–Penrose screen transport — for all 10 one-zone parameters simultaneously, matching finite differences to **2.3×10⁻¹¹** at warm-gradient cost ≈ 1× forward.

---

## 1. The one-zone splat

Each splat k is a complete, self-contained emitting parcel:

| Block | Parameters | Count | Parameterization notes |
|---|---|---|---|
| Position | μ_k ∈ ℝ³ | 3 | quasi-Cartesian Kerr–Schild, **geometric units** (r_g) — see §4.2 |
| Shape | quaternion q_k, log-scales s_k | 3+3 | Σ_k = R(q)diag(e^{2s})R(q)ᵀ, always SPD (3DGS standard) |
| Temporal envelope | birth t_k, log-duration w_k | 2 | smooth bump in *coordinate time* (units of t_g = GM/c³); C² compact support optional |
| Density | log n_e,k | 1 | physical cm⁻³ once (M, D) fixed |
| Electron distribution | thermal: log Θ_e,k · power-law: log w̃_k (energy norm), p_k, log γ_min,k [, log γ_max,k] | 1–4 | see §3; both populations may coexist (additive) |
| Magnetic field | log |B|_k, direction (ϑ_B, φ_B)_k in the fluid frame | 3 | direction enters pitch angle + EVPA; |B| enters ν_c; optional low-order time dependence |
| Fluid velocity | ũ_k ∈ ℝ³ in the ZAMO (normal-observer) frame | 3 (static) or 3K (K time knots) | unconstrained ℝ³ → always subluminal/timelike: γ = √(1+g_ij ũ^i ũ^j) (exactly Feng et al. §3.1.2, Eqs. 26–31) |
| (Optional) center trajectory | knots of μ_k(t) | 3K′ | *pattern* motion, distinct from fluid velocity — see §1.1 |
| (Optional) field-order fraction | f_B,k ∈ (0,1) via logit | 1 | multiplies polarized (Q,U,V) coefficients only; models sub-splat field tangling. Introduces an LP↔|B| degeneracy that V/Faraday partially breaks — off by default, enable deliberately |

≈ **16–25 parameters per splat** (static velocity, single population) — 10³ splats ≈ 2×10⁴ parameters, comfortably in reverse-mode/Adam territory, and each ray's gradient touches only the splats it intersects (natural sparsity).

### 1.1 Pattern velocity vs fluid velocity — keep both, tie neither

Two physically distinct velocities exist and the maximally general model must not conflate them:

- The **fluid velocity** u^μ_k enters the *radiation physics*: redshift g_k = 1/(−p_μ u_k^μ), Doppler beaming, aberration of the pitch angle, and the fluid frame in which B̂_k is defined. Every splat needs one at all times — even a "sitting" splat must declare what its emitting plasma is doing, because I, Q, U, V are unintelligible without it. "Sitting in space" = ũ_k free but the *center* static.
- The **pattern velocity** μ̇_k moves the envelope. In real flows they differ routinely (orbiting hotspot patterns, jet pattern speeds ≠ bulk Γ, standing shocks with fast flow through them).

The three motion modes requested are then simply:
- **Mode A (birth/death):** static centers, temporal envelopes, densify/prune creates the illusion of motion the way movie frames do. Fluid velocity still fitted per splat. Maximally assumption-free; leans hardest on the outer densification loop and on temporal-envelope resolution.
- **Mode B (ballistic/spline drift):** μ_k(t) low-order polynomial or K-knot spline; fluid velocity fitted separately.
- **Mode C (self-consistent):** μ̇^i_k tied to u^i_k/u^t_k by a *soft penalty* (weight → 0 recovers Mode B). This is PI-DEF's dynamics loss recast per-splat and made optional, not structural.

All three coexist in one code path (Mode A is Mode B with zero drift; Mode C is a loss term). No advection onto the hole is ever assumed; infall, outflow, corona, and jet are all expressible because ũ_k is unconstrained in the ZAMO frame (any sub-photon-sphere or outflowing velocity that is subluminal is representable).

---

## 2. Radiative transfer of overlapping one-zone parcels is exactly additive

This is the physics fact that makes the whole design clean. The polarized transfer equation is linear in the *coefficients*: emissivities j_S, absorptivities α_S, and rotativities ρ_S are integrals of the single-particle response over the local electron population, so for multiple co-located populations (each with its own distribution, B, and even its own bulk velocity) the coefficients **sum**:

  j = Σ_k j_k,  α = Σ_k α_k,  ρ = Σ_k ρ_k  (per Stokes component, in any single frame).

There is no need to construct a blended "local plasma state" (density-weighted Θ_e, mean B, mean u) — which would be nonlinear, unphysical, and gradient-entangling. Each splat is evaluated entirely in *its own* comoving frame and contributes invariants:

1. At ray sample (t_i, r_i, θ_i, φ_i) with photon momentum p_μ (analytic from Krang), loop over splats whose support covers the point.
2. For splat k: g_k = 1/(−p_μ u_k^μ) → fluid-frame frequency ν_k = ν_obs/g_k; aberrated pitch angle cos θ_B,k = p̂·B̂_k in splat k's frame (Krang's `jac_fluid_u_zamo_d` boost, already per-(velocity, B) as verified in the smoke test); evaluate Marszewski coefficients for splat k's distribution at (ν_k, θ_B,k, n_e,k·G_k(x,t), Θ_e,k or (p, γ_min)_k, |B|_k), where G_k is the Gaussian×envelope weight.
3. Convert to relativistic invariants (j/ν², ν α, ν ρ) — invariants computed from each splat's own frame are frame-independent numbers, so **summing invariants from splats with different bulk velocities is exact**, the same statement as interpenetrating beams in kinetic theory.
4. Each splat's Stokes basis is aligned with *its* B̂_k projected on the local polarization plane; rotate its coefficient vector into the single parallel-transported screen basis by its Walker–Penrose angle χ_k (Krang's `synchrotronPolarization` machinery, per-splat). The local Mueller matrix is assembled as
   M(x) = Σ_k R(χ_k) M_k R(χ_k)ᵀ,
   with R the Stokes rotation about the I–V axis. All smooth, all StaticArrays.
5. One analytic constant-coefficient 4×4 step per sample interval (unchanged from v1 §7.1 step 5).

Emergent bonus: **Faraday screens come for free.** A cold, dense splat (low Θ_e) has negligible j but large ρ_V — internal and external Faraday rotation/conversion, "dark" rotating plasma between the emitter and the observer, and the corona's depolarizing effect are all representable without any new machinery. This matters for realism at 86–230 GHz and is a capability neither PI-DEF nor any splatting code has.

Consequence for `transport.jl`: the sample loop becomes (samples × local splat multiplicity); with compact splats and interval culling (§6.3) multiplicity is O(1–10), and the per-splat inner block is exactly what the smoke test timed (2 ms warm for 600 sample-evaluations single-threaded, gradient included).

---

## 3. Electron distributions per splat, differentiably

The requirement is "thermal **or** power-law" per splat. A discrete per-splat switch is non-differentiable; three clean resolutions, in recommended order:

1. **Two-population additive splat (recommended baseline).** Every splat carries a thermal component *and* a power-law component sharing its B and u, each with its own normalization (log n_th, log n_pl); coefficients add (§2). Fitting drives the irrelevant one to zero — a *continuous* embedding of the discrete choice, and physically standard (thermal core + nonthermal tail is exactly how EHT/jet hybrid models are built). All coefficients are closed-form Marszewski (2021) fits: thermal Eqs. 30–37, power-law Eqs. 38–42 (with their stated validity windows: ρ fits accurate for γ_min ≲ 10², ν/ν_c ≫ 1 — enforce by clamped priors and log validity diagnostics). Derivatives w.r.t. p pass through Γ-functions (SpecialFunctions has AD rules); w.r.t. everything else through elementary functions and Bessel K_n (Bessels.jl; K_n′ = −(K_{n−1}+K_{n+1})/2 as a trivial custom rule if needed).
2. **κ-distribution splat (upgrade path).** Single continuous family nesting thermal (κ→∞) and power-law (p = κ−1) with width w — the most elegant answer, but the Marszewski ρ_Q/ρ_V fits exist only at κ ∈ {3.5, 4, 4.5, 5} and the low-frequency absorptivity bridge uses ₂F₁ with κ in its parameters (parameter-derivatives of hypergeometrics are not off-the-shelf). Making κ a continuous fitted parameter therefore requires **building a differentiable coefficient surrogate**: run symphony over a (ν/ν_c, θ_B, κ, w) grid, fit tensor-product Chebyshev/spline surfaces with a stated error budget, and register the surrogate's analytic derivatives. Genuine but bounded new work (order weeks, not months); scheduled Phase 5.
3. Not recommended: a soft-max blend of separately-normalized thermal/power-law *states* (redundant given option 1).

---

## 4. Global parameters: what actually remains

### 4.1 The reduction

v1 carried four global model blocks (background profile ~6, parametric B ~4, AART velocity ~3, distribution ~2, spacetime ~5). With one-zone splats, **the only structural globals are the spacetime/observer block**:

| Global | Meaning | Constrained by |
|---|---|---|
| θ_g = GM/(c²D) | angular scale (μas per r_g) | apparent sizes, ring diameter |
| t_g = GM/c³ | temporal scale (s per geometric time) | variability timescales, photon-ring echo delays |
| a | spin | shadow shape, sub-image structure, frame dragging in delays/EVPA |
| θ_o | inclination | asymmetry, sub-image arrangement |
| PA | camera roll (spin-axis position angle) | overall EVPA/Q-U rotation and image orientation |
| (t_0) | time zero-point convention | fixed by convention, not fitted |

M and D are then *derived*: M = c³t_g/G from timing, D = GM/(c²θ_g) from the ratio. **The movie is what breaks the classic M/D ring-size degeneracy** — a static image constrains only θ_g; time-resolved data adds t_g independently. This deserves to be listed as a headline capability of the movie-fitting formulation.

A background floor (e.g., one very large splat) is optional and within-representation, not a separate model class. The former velocity/B-field models (AART etc.) are demoted to **optional initializers and soft priors** in `fit.jl`; they never enter the forward model.

### 4.2 Units discipline (important, subtle)

Parameterize *all* splat geometry and times in geometric units (r_g, t_g), and apply (θ_g, t_g, PA) only at the camera/data interface. Then gradients w.r.t. M do not spuriously drag every splat through space (a pure relabeling direction is removed from the Fisher matrix), and the spacetime gradients carry only *physical* lensing/timing information. |B| and n_e stay in physical (Gauss, cm⁻³) units; note that the ν_c ∝ B and opacity scalings give them well-defined meaning only jointly with (θ_g, t_g) — part of what the spectral coverage pins down (§5).

### 4.3 AD strategy for the spacetime block

Changing (a, θ_o) invalidates the per-pixel geodesic cache that makes plasma-gradient evaluation cheap. Handle with **mixed-mode AD**, which is the textbook-correct tool here:

- **Reverse mode (Enzyme)** over the 10⁴–10⁵ splat parameters with geodesics treated as constants (cache valid).
- **Forward mode** over the ≤ 5 spacetime parameters *through* the analytic pixel construction (roots, elliptic integrals, WP transport) — 5 extra forward passes per iteration, each ≈ 1 render. Verified feasible in both smoke tests (reverse dI/da worked too; forward is simply cheaper bookkeeping for so few parameters).
- Cache rebuild per spacetime update; update spacetime on a slower cadence (e.g., every 10–50 splat steps) or jointly with a small learning rate. Fisher-block preconditioning (spacetime block vs splat blocks) if the scales separate badly.

Practical staging: fit splats at fixed plausible (a, θ_o) first, unfreeze PA + θ_g + t_g next (nearly linear effects), then (a, θ_o) (the PI-DEF Fig. 9 experiment, but with exact gradients rather than a grid).

---

## 5. Is it well-posed? An honest assessment

The user's conjecture: in the limit of perfect resolution, sensitivity, and spectral sampling, strong lensing provides multiple viewing angles of every splat, and the thick↔thin frequency sweep provides interior/surface views, so the full one-zone state of every splat is pinned. Assessment: **largely correct, with a specific and enumerable set of residual degeneracies** — and the plan should include machinery to *measure* identifiability rather than assume it.

### 5.1 The information channels, itemized

Per splat, ideal data provide:

1. **Multi-view geometry (lensing).** Every emission event is imaged at n = 0, 1, 2, … screen positions; the n-th image views the splat along a *different* photon direction k̂ through the fluid. Different k̂ ⇒ different Doppler factor g_k (velocity projection), different pitch angle θ_B (field projection), different WP transport (EVPA). Consequences:
   - **B̂_k is triangulated in 3-D**: a single view constrains B̂ only up to the classic synchrotron plane ambiguity; two-plus views of the *same parcel* at known k̂'s make field-direction tomography solvable. (This is the Himwich et al. multi-image EVPA idea pushed to volumetric primitives.)
   - **u_k is triangulated**: g measured along ≥ 2 directions (via per-image spectral shifts of the turnover and beaming ratios) determines the velocity vector, separating it from the pattern motion measured astrometrically across frames.
   - Photon-ring image multiplicity also stamps **exact time-delay echoes** (≈ photon-orbit period spacing), which pin t_g and a nearly geometrically.
2. **The spectral sweep through the turnover.** A one-zone synchrotron sphere observed from ν ≪ ν_turnover to ν ≫ ν_turnover yields the classic solvable system (Marscher-style one-zone inversion): thick-side brightness temperature → Θ_e (thermal) or effective temperature (PL); turnover frequency and flux → combinations of n_e, |B|, size; thin-side slope → distribution shape (Θ_e vs p); thin/thick LP fractions and EVPA vs ν → field order and orientation; **V and Faraday rotation/conversion (ρ_V ∝ n_e B cosθ_B/ν², ρ_Q)** → line-of-sight field *sign* and magnitude, breaking the B̂ → −B̂ ambiguity that all linear-polarization-only data possess. Splat size is measured directly (resolved), removing the usual size–B–n_e degeneracy of unresolved cores.
3. **Time structure.** Envelope shape in coordinate time; slow-light delays across the splat's own extent; echo trains. Distinguishes pattern from fluid velocity (§1.1) because beaming modulates *amplitude vs viewing angle* while pattern motion moves *position vs time*.
4. **Opacity tomography.** At thick frequencies, only the τ ≈ 1 shell of a splat (or of a crowd of splats) is visible; sweeping ν moves that shell smoothly through the volume. Formally a mildly ill-conditioned Abel-type inversion, not a null space — conditioning degrades gracefully with optical depth. Rear-side splats occulted at low ν are seen directly at high ν and via lensed sub-images around the shadow at all ν.

### 5.2 Degeneracies that survive ideal data (the honest list)

1. **Near-horizon information death.** As emission approaches the horizon, g → 0: fluxes are killed as g³⁺, delays diverge, and infalling-parcel signatures pile up at the shadow edge. Recovery of anything inside r ≲ 1.2–1.5 r_h will be poor at *any* data quality (Feng et al. observe exactly this in the much easier problem, their §5.5). Well-posedness fails smoothly, not catastrophically — report posterior variance maps vs r.
2. **Sub-resolution field topology (representation error, not data degeneracy).** A one-zone splat asserts a uniform B̂ inside its support. Real tangled fields at sub-splat scales masquerade as (lower LP + same I), which the optional f_B parameter absorbs — but then (f_B, |B|, θ_B near the ρ_Q null) have partially overlapping effects; V and the ν-dependence of Faraday effects break most but not all of this. Mitigation: densification shrinks splats until the uniform-field assumption is locally adequate; the residual is representation error, diagnosed by fit residuals rather than posterior width.
3. **Partition non-uniqueness.** Two half-amplitude overlapping splats ↔ one splat: exactly flat likelihood directions in the *splat* parameterization. Harmless for the reconstructed *fields* (n_e, Θ_e, B, u evaluated on a grid) — which is what should carry scientific claims and uncertainties — but fatal to naive per-splat interpretation. Deliverables must be field-level; parsimony (prune + merge) keeps the representation tame.
4. **Vortical/polar geodesics and pole-on splats**: sparse sub-image coverage for emission very near the spin axis viewed near pole-on; expect anisotropic uncertainty there.
5. **Global gauge freedoms**: t_0 convention; overall EVPA zero vs PA (broken only by knowing the instrument's absolute EVPA calibration — for synthetic data, fixed by convention).
6. **Slow, smooth ambient plasma** (a uniform Faraday screen far from the hole) is degenerate with per-splat internal rotation only at a single frequency — the ν² lever plus lensed-path differences break it with full spectral sampling; residual leakage expected at finite sampling.

None of these overturn the conjecture; they bound it. The correct scientific posture is:

### 5.3 Identifiability as a measured quantity — "Fisher audit" (new deliverable)

Because the whole forward model is differentiable, local identifiability is *computable*: Enzyme Hessian-vector products give the Gauss–Newton/Fisher operator J ᵀΣ⁻¹J; Lanczos on HVPs yields its spectrum and the eigenvectors of the flattest directions — i.e., an automatic, quantitative catalog of the actual degeneracies of a given configuration + data spec (resolution, ν grid, cadence, Stokes noise). Deliverables:

- Fisher spectra vs data quality: resolution ladder, ν-coverage ladder (single band → thick+thin → dense sweep), I-only vs IQUV, image vs movie. This *proves or disproves the well-posedness conjecture quantitatively*, and identifies which channel (V? multi-ν? sub-images?) supplies which constraint — a publishable result by itself, independent of any real-data application.
- Per-splat marginal uncertainties (Laplace) projected onto field-level maps.
- Flat-direction monitors during fitting (freeze or prior-anchor parameters the current data provably cannot see, e.g., V-block parameters when V ≈ 0).

### 5.4 Practical corollary: the frequency curriculum

Even a well-posed problem can be unfindable by local optimization. The spectral structure suggests a natural annealing: fit optically-thin high-ν channels first (emission ≈ linear in n_e, landscape mildest), then extend downward through the turnover (opacity + Faraday effects switch on progressively), then unfreeze spacetime. Mirrors multi-frequency RML imaging practice; implemented as a data-weighting schedule, no code structure impact.

---

## 6. Changes to the v1 plan

### 6.1 Module deltas

- `splats.jl` → **one-zone splat type** of §1: adds per-splat (log|B|, ϑ_B, φ_B), ũ ∈ ℝ³ ZAMO velocity (+ optional time knots), two-population normalizations, optional f_B, optional trajectory knots. ComponentArrays layout so parameter blocks can be frozen/unfrozen by name.
- `plasma.jl` → thin: splat set + optional background splat; **no global field construction at all** — it returns *per-splat* local states at a query point (list, not blend).
- `velocity.jl` → per-splat ZAMO-frame parameterization + u^μ assembly (Feng Eqs. 26–31); AART/Cunningham model moves to `priors.jl` as an optional initializer/regularizer only.
- `bfield.jl` → absorbed into `splats.jl` (per-splat B); deleted as a global model.
- `coefficients.jl` → thermal + power-law complete Marszewski sets from the start (both needed by the two-population splat); validity-window guards; κ surrogate deferred (§3).
- `transport.jl` → restructured around **per-splat invariant-coefficient summation** and per-splat WP rotation angles χ_k with Mueller assembly M = Σ R(χ_k)M_kR(χ_k)ᵀ (§2). This replaces v1's single-plasma-state evaluation — a simplification in coupling, a widening in the inner loop.
- `fit.jl` → mixed-mode AD (reverse splats / forward spacetime), geodesic-cache invalidation, staged unfreezing, frequency curriculum, Fisher-audit tooling (HVP + Lanczos), densify/prune/merge (merge is new — the §5.2.3 partition hygiene).

### 6.2 Phase deltas

- **Phase 1** unchanged (optically-thin I-only splats) but with the one-zone parameter layout from day one, exercising per-splat velocity (already smoke-tested).
- **Phase 2** (polarized transfer) now implements per-splat summation + both distributions; validation gates unchanged (Gold et al. 2020 suite; ipole IQUV references) plus a new **two-splat overlap unit test**: two co-located parcels with different (u, B) vs an equivalent single-frame hand calculation; and a Faraday-screen test (cold splat in front of a hot one) against an analytic slab solution.
- **Phase 3** (slow light/time) adds motion Modes A/B/C and the pattern-vs-fluid separation test: synthetic orbiting hotspot fit in each mode.
- **Phase 4** (inference at scale) gains the **Fisher audit** as a first-class deliverable (§5.3) and the frequency curriculum.
- **Phase 5** becomes: continuous-κ surrogate; spacetime-block fitting to full precision; visibility-domain likelihood (Comrade). Spacetime fitting *starts* earlier (Phase 4) at coarse precision since it is now structural, not an afterthought.

### 6.3 Cost note

Per-sample cost multiplies by local splat multiplicity and by the two-population coefficient evaluation (~2× a dozen special-function calls). With interval culling (per-ray splat interval lists from bounding ellipsoids in KS coordinates — computable analytically from Krang's coordinates without differentiating the culling decision) the multiplicity is O(few). The smoke tests put the scale at ~3–4 μs per (sample × splat) including gradient, single-threaded; a 128² image × 10³ samples × 5 splats/sample × 20 frames × 8 frequencies ≈ 10¹² sample-splat gradient evaluations per full-data pass — large but embarrassingly parallel, minibatchable over (pixel, frame, ν), and squarely what the KernelAbstractions/Reactant path is for. Accuracy-first remains the ordering: correctness gates precede all performance work.

### 6.4 Risk-table updates

| Risk (new/changed) | Mitigation |
|---|---|
| Harder nonconvexity: per-splat velocity/B can fit noise or lock into wrong basins (e.g., B̂ sign before V data constrain it) | Frequency curriculum; staged unfreezing (amplitudes → geometry → velocity/B → distribution shape → spacetime); optional *hierarchical shrinkage* — per-splat parameters drawn around fitted population hyper-means with fitted spread ("generality without anarchy": hyperprior weight → 0 recovers full independence, so no flow model is imposed, only shared statistics if the data like them); multi-start on seeds; Fisher-guided freezing of unidentified blocks |
| Partition non-uniqueness contaminates interpretation | Field-level deliverables + merge/prune hygiene (§5.2.3) |
| Marszewski power-law ρ fits outside validity (γ_min ≳ 10², low ν/ν_c) | clamped priors, runtime validity flags, symphony spot checks in gate 1 |
| Continuous-κ derivative wall (₂F₁ parameters, discrete-κ ρ fits) | deferred to Phase 5 behind the symphony-grid Chebyshev surrogate with error budget |
| Spacetime updates invalidate geodesic caches | mixed-mode AD + slow-cadence spacetime updates (§4.3) |
| M-direction gauge dragging splats | geometric-units parameterization (§4.2) |

---

## 7. Updated feasibility evidence

`krang_enzyme_polarized_smoketest.jl` (this directory; run 2026-07-10, Julia 1.10.10, Krang v0.4.1, Enzyme v0.13.179): one Gaussian splat carrying its own ZAMO-frame fluid velocity (speed + 2 angles) and its own fluid-frame B̂ (2 angles), rendered to (I, Q, U) along three slow-light rays through `Krang.synchrotronPolarization` — i.e., through the ZAMO→fluid boost Jacobian, the fluid-frame k̂×B̂ construction, and the complex Walker–Penrose screen transport. Enzyme reverse-mode gradient over all 10 parameters:

| Quantity | Result |
|---|---|
| Warm forward (3 rays × 200 polarized samples) | 2.1 ms |
| Warm reverse gradient, 10 params incl. velocity + B angles | 2.2 ms (≈ 1× forward) |
| Max rel. error vs 5-point central FD | **2.3×10⁻¹¹** |

Together with the v1 smoke test (splat params to 8×10⁻¹³; dI/da through the geodesic construction to 3×10⁻¹¹), every AD-critical ingredient of the maximal design — per-splat plasma, per-splat kinematics, per-splat field orientation, polarized transport, spacetime parameters — now has a passing end-to-end gradient demonstration on this machine.
