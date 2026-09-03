# GPU-resident Kerr geodesics for KerrSplat: plan

**Date:** 2026-09-02. Companion to `kerrsplat_evaluation_and_plan.md` (2026-07-09) and `kerrsplat_maximal_generality_addendum.md` (2026-07-10). Scope: the geodesic layer only, i.e. everything that changes when spin `a`, inclination `θo`, or the camera changes. Inference is out of scope here.

> **Status 2026-09-03** (`KerrSplat.Geodesics`, PRs #2–#4): steps 2–6 of §9 are implemented and
> gates 1–5 of §8 pass on the CPU backend and on CUDA. Measured on the RTX 2080 SUPER for
> 256² × 1000 samples in Float64: K1 18 ms; K2 with Krang's direct evaluation 8.2 s (125 ns/sample);
> the recurrence for r, θ alone 0.14 s (2 ns/sample); the full recurrence + anchored quadrature for
> (t̃, r, θ, φ) 0.55 s (8.5 ns/sample); duals for (a, θo) flow through everything. Against a
> BigFloat evaluation of the closed forms the recurrence reaches 1e-11 in r and θ where Krang's
> inputs allow it, and the quadrature 1e-9 in t̃ and φ inside 50 M; where it differs from Krang's
> closed forms by more, Krang is the one off (`docs/notes/`). The design change relative to §4:
> t̃ and φ are integrated after three singular pieces are removed in closed form (large r, both
> horizon poles, the polar-axis part of the third-kind integral), which is what makes one Simpson
> panel per sample enough; no pixel needs the §5 direct fallback (the near-critical flag exists for
> 1 − k < 1e-7, where JacobiElliptic's amplitude breaks and Krang's direct evaluation is unusable too).

**Requirement.** After changing `(a, θo)` the full set of per-pixel, per-sample geodesic data (t̃, r, θ, φ, momentum signs, and hence p_μ) must be regenerated quickly, on the GPU, using only the CUDA stack (CUDA.jl + KernelAbstractions kernels, no XLA/Reactant), with derivatives with respect to `(a, θo)` available at small extra cost.

**Verdict.** Krang's analytic (Gralla–Lupsasca) formulation is the right GPU formulation: no ODE, every sample is an O(1) closed-form evaluation, perfectly data-parallel. All of it already runs on this machine's GPU in Float64 after four small fixes (§2). But a *direct* port is not fast on a consumer card: the per-sample evaluation calls six to eight incomplete elliptic integrals, and at 1/32-rate FP64 that is only 2× faster than the 16-thread CPU (§3). The plan therefore changes the per-sample algorithm: the Jacobi functions that give r(τ) and θ(τ) are advanced by their **addition theorems** (a rational recurrence, no transcendental calls), and t(τ), φ(τ) come from **cumulative quadrature of the Mino-time rates** anchored to Krang's exact values every few dozen samples (§4). Expected cost is a few nanoseconds per sample in Float64, i.e. a 256²×1000-sample recompute in ~0.1–0.3 s on this RTX 2080 SUPER, versus 14 s on the CPU today. Spacetime derivatives come from ForwardDiff dual numbers propagated through the same kernels, verified today against finite differences to 1e-8 (§6).

---

## 1. What "fast" has to mean

| Configuration | Samples | CPU, Krang direct, 16 threads (measured) | GPU, Krang direct, Float64 (measured) | GPU, recurrence design (target) |
|---|---|---|---|---|
| 256² pixels × 1000 samples | 6.6×10⁷ | 14 s | 8.2 s | 0.1–0.3 s |
| 128² × 1000 | 1.6×10⁷ | 3.6 s | 2.1 s | < 0.1 s |
| 512² × 2000 | 5.2×10⁸ | 115 s | 66 s | 1–2 s |

Per-pixel constants (roots, elliptic constants, τ_total) are already fast on the GPU: 65,536 slow-light pixels in **15 ms** (0.33 s single-thread CPU), agreeing to 2.5×10⁻¹².

## 2. State of the toolchain on this machine (verified today)

Hardware and software: RTX 2080 SUPER (Turing, sm_75, 8 GB, FP64 at 1/32 rate), driver 570.133.07 (CUDA 12.8), Julia 1.10.11, CUDA.jl 6.3.1, KernelAbstractions 0.9.42, Krang.jl main at commit `f36f43a` (2026-08-15), JacobiElliptic 0.3.10.

Fixes needed to run Krang inside CUDA kernels. All are small; the first four are reproduced in `gpu_probes/`.

1. **Pin the CUDA runtime to the driver.** CUDA.jl 6.3 picks a CUDA 13.3 toolchain through its forward-compatibility driver shim; GeForce cards do not support that path and the resulting kernels fail to load ("device kernel image is invalid"). `CUDA.set_runtime_version!(v"12.8")` fixes it (stored in `LocalPreferences.toml`).
2. **Use a concrete pixel type.** `SlowLightIntensityPixel` has 17 type parameters (added for Reactant tracing). Krang's KernelAbstractions extension allocates `Matrix{SlowLightIntensityPixel{T}}`, leaving 16 of them abstract, which CUDA.jl rejects. Allocating `CuArray{typeof(SlowLightIntensityPixel(met, α, β, θo))}` works: the concrete struct is 232 bytes in Float64, 116 in Float32, and isbits.
3. **Raise the per-thread stack.** Krang's Float64 radial-integral code needs more than CUDA's default 1 KB stack; the symptom is an illegal memory access during pixel construction. `CUDA.limit!(CUDA.LIMIT_STACK_SIZE, 4096)` suffices (2 KB was enough in tests).
4. **Own ray kernel.** `Krang.generate_rays(pixels, N; A=CuArray)` calls `first(pixels)` on the device array (scalar indexing error). A 10-line replacement kernel is in the probes.
5. **One method for duals.** `_θs` calls `unsafe_trunc(Int, τ/τ̂)`; with ForwardDiff duals that is a dynamic dispatch. Define `unsafe_trunc(::Type{I}, d::Dual) = unsafe_trunc(I, value(d))`.
6. Debug builds (`julia -g2`) fail because Krang's Unicode identifiers reach the PTX debug info, which ptxas rejects. Cosmetic; do not use `-g2` with Krang kernels.

All of JacobiElliptic's Carlson-algorithm functions (F, E, Π, K, sn, am) run correctly inside CUDA kernels in both precisions (Float64 agreement with CPU 1e-15). Items 2, 4 and 5 should go upstream as pull requests so the project does not carry patches.

## 3. Measured baseline: Krang's direct evaluation on the GPU

RTX 2080 SUPER, a = 0.94, θo = 60°, screen ±10 M, 256² pixels, uniform Mino-time samples.

| Stage | Float64 GPU | Float32 GPU | CPU 16 threads (Float64) |
|---|---|---|---|
| Per-pixel constants, 65,536 px | 15 ms | 1.5 ms | 200 ms (335 ms single thread) |
| Per-sample coordinates, stored (200/px) | 110 ns/sample | 8 ns/sample | – |
| Per-sample march, fused, no storage (1000/px) | 126 ns/sample (8.2 s) | 16.5 ns/sample (1.1 s) | 219 ns/sample (14.4 s) |
| Agreement with CPU Float64 (image through a test splat) | 1.2×10⁻¹² | see below | – |
| Duals for ∂/∂a, ∂/∂θo through everything | 2.7× the plain kernel | – | matches FD to 1.3×10⁻⁸, 7.8×10⁻⁸ |

Why Float64 is slow here: each `emission_coordinates(pix, τ)` evaluates the polar Jacobi amplitude, the radial Jacobi function for the root case, and then incomplete elliptic integrals of the first, second and third kind for the radial (I₀, I₁, I₂, I₊, I₋) and angular (G_φ, G_t) pieces of t and φ. That is roughly a dozen Carlson-iteration transcendental evaluations per sample, plus large complex-valued temporaries that spill to local memory. On a card with 1/32-rate FP64 that costs ~126 ns; the 8-core CPU does it in ~219 ns per thread-sample.

Why Float32 is not the answer: it is 7.6× faster, but the per-sample path is numerically unstable in single precision on both CPU and GPU. Against the Float64 image, the Float32 image has a median relative error of 9×10⁻⁶ but 8.7 % of pixels exceed 10⁻³ and the worst are wrong by orders of magnitude (near-critical pixels, elliptic modulus → 1). Upstream has started "F32-stable" reformulations (case 3, May 2026) but the τ path is not there. The plan keeps Float64 everywhere and makes the per-sample work cheap enough that FP64 rate no longer matters.

## 4. Design: cheap per-sample marching by addition theorems

The expensive part of Krang's per-sample evaluation is not needed at every sample, because the samples are uniformly spaced in Mino time and every Jacobi argument is linear in τ.

**Radial motion.** Krang's own formulas (`misc.jl`, `_rs_case*`) are, with X linear in τ:

- case 1/2 (four real roots, scattering): r = (r₃₁ r₄ − r₃ r₄₁ sn²(X₂,k)) / (r₃₁ − r₄₁ sn²(X₂,k)), X₂ = ½√(r₃₁r₄₂)(I₀ − τ);
- case 3 (two complex roots, plunging): r = (−A r₁ + B r₂ + (A r₁ + B r₂) cn(X₃,k)) / (−A + B + (A + B) cn(X₃,k)), X₃ = √(AB)(I₀ − τ);
- case 4 (four complex roots): r = −(a₂ (g₀ − sc(X₄,k₄))/(1 + g₀ sc(X₄,k₄)) + b₁), X₄ = ½(C + D)(I₀ − τ), sc = sn/cn.

**Polar motion.** cos²θ = u₊ sn²(X_θ, k_θ) (ordinary) or the vortical analogue, X_θ linear in τ.

**Recurrence.** For a fixed step Δ in the argument, (sn, cn, dn)(u + Δ) follow from (sn, cn, dn)(u) by the Jacobi addition theorems:

```
den = 1 − m sn²(u) sn²(Δ)
sn(u+Δ) = (sn u · cn Δ · dn Δ + sn Δ · cn u · dn u) / den
cn(u+Δ) = (cn u · cn Δ − sn u · dn u · sn Δ · dn Δ) / den
dn(u+Δ) = (dn u · dn Δ − m sn u · cn u · sn Δ · cn Δ) / den
```

Fifteen multiplications and one division per step, no transcendental functions, and the three step constants (sn, cn, dn)(Δ, k) are per-pixel quantities computed once in K1. Prototype (`gpu_probes/jacobi_recurrence_prototype.jl`, CPU, Float64): marching 1000–4000 steps over a full period with k² from 0.1 to 0.9999, the error against direct evaluation is ≤ 1.5×10⁻¹¹ without any correction and ≤ 9×10⁻¹⁵ when the state is re-anchored to a direct evaluation every 64 steps. Turning points need no special handling: r depends on sn² (or cn, sc) and the radial sign ν_r is the sign of d(sn²)/du = 2 sn·cn·dn; winding counts come from sign changes. Rays that hit the horizon simply stop (r < r_h(1+ε)), exactly as now.

**Time and azimuth.** Instead of the incomplete Π integrals, integrate the Mino-time rates, which are rational in r and cos²θ and regular at turning points (no square roots appear in Mino time):

```
dφ/dτ = a (2 r − a λ) / Δ(r) + λ / sin²θ
dt/dτ = (r²+a²)(r²+a² − a λ) / Δ(r) + a (λ − a sin²θ)
```

Accumulate them along the ray with composite Simpson (or 3-point Gauss–Lobatto between consecutive samples, which reuses the sample values). Anchor the running sums to Krang's exact analytic t̃(τ), φ(τ) every M samples (M ≈ 64): the anchor residual is a per-ray error estimate that the kernel can write out, and the observer-end regularization of t is inherited from Krang's anchor value at the first sample. The only elliptic-integral evaluations left are N/M per ray instead of ~10 N.

**Kernels.**

- **K1, per pixel.** Krang's pixel construction (already on GPU, 15 ms for 256²), plus the recurrence constants: root case, k_r, k_θ, the step values (sn, cn, dn)(Δ_r) and (sn, cn, dn)(Δ_θ), the initial state at the first sample, and the anchor values. Output as structure-of-arrays, not the 232-byte struct, so K2 reads coalesced. Sort pixels by root case (a permutation computed once per spacetime) so each warp runs one branch of the case logic.
- **K2, per ray.** One thread per ray marches its N samples with the recurrences and the running quadrature, emitting (t̃, r, θ, φ, ν_r, ν_θ) and optionally p_μ (which is closed-form in η, λ, r, θ and the signs). N is a compile-time constant (`Val(N)`); this is also what Enzyme reverse mode inside kernels requires (no dynamic tape). Two modes with identical device code: **stored** (write the sample arrays; consumers read them) and **fused** (a consumer kernel such as the transport step calls the marcher inline and never stores samples). Choose by memory (§7).
- **K3, consumers.** Transport and splat kernels from the main plan; unchanged by this document except that they may run in fused mode.

Cost estimate for K2: ~40 FP64 flops and one division per sample for both recurrences, ~20 flops for the two rates and the quadrature update, plus one exact anchor per 64 samples. On this card's ~170 GFLOP/s FP64 that is a few nanoseconds per sample even with poor efficiency, so the memory write of ~40 bytes per stored sample (2.6 GB for 256²×1000, ~10 ms at 300 GB/s) is not the bottleneck either. The target numbers in §1 assume 2–5 ns per sample; the validation gate in §8 will replace the estimate with a measurement.

## 5. Precision policy

Float64 throughout the geodesic layer. The recurrence makes FP64 affordable on this card, and the measured Float32 instability (§3) rules single precision out for the per-sample path regardless of hardware. Re-anchor every 64 samples. Pixels whose elliptic modulus is extremely close to 1 (near the critical curve, τ_total large) are flagged in K1 and evaluated with Krang's direct code on the GPU instead of the recurrence; the flag threshold is set by the validation study, not guessed. A data-center GPU with full-rate FP64 (A100/H100) would run even the direct evaluation quickly, but the design does not depend on one.

## 6. Derivatives with respect to spin, inclination and camera

- **Forward mode with ForwardDiff duals through K1 and K2.** Verified today on the GPU with Krang's direct evaluation: `Dual{2}` for (a, θo) through pixel construction and 200 samples per ray, 2.7× the plain kernel's cost, matching CPU central finite differences to 1.3×10⁻⁸ (∂/∂a) and 7.8×10⁻⁸ (∂/∂θo), which is the FD noise floor. With the recurrence, dual propagation is ordinary rational arithmetic; the anchors use Krang's dual-capable code.
- Consequence for the main plan: the geodesic cache no longer needs an invalidation strategy (addendum §4.3). Recomputing K1+K2 per optimizer iteration costs a fraction of a second, so `(a, θo)` can be updated at the same cadence as the splats, with their gradient blocks from forward mode and the splat blocks from Enzyme reverse mode.
- Mass/distance scale θ_g and camera roll PA never touch K1/K2 (pure relabelings of the screen), so their derivatives are free.
- Reverse mode through the marcher (Enzyme in-kernel) is possible later for the same code path because N is static; not needed for ≤ 5 spacetime parameters.

## 7. Memory plan for 8 GB

| Item | Bytes | 256² × 1000 | 512² × 1000 |
|---|---|---|---|
| Per-pixel constants (SoA, Float64) | ~300 / pixel | 20 MB | 80 MB |
| Stored samples (t̃, r, θ, φ, 2 flags) | 34–40 / sample | 2.6 GB | 10.5 GB |
| Stored samples + p_μ | +32 / sample | 4.7 GB | too large |

Stored mode fits 256²×1000 on this card with room for the transport working set; 512² or 2000 samples must use fused mode or pixel tiles (K2 is cheap enough to rerun per tile). p_μ should be recomputed in the consumer from (η, λ, r, θ, signs) rather than stored.

## 8. Validation gates

1. K1 on GPU vs Krang CPU: done today, 2.5×10⁻¹² (τ_total), 7×10⁻¹⁴ (η).
2. Recurrence marcher vs Krang's direct `emission_coordinates` at **every** sample for 10⁴ random pixels per root case, spins 0, 0.5, 0.94, 0.999, inclinations 1°–89°, including a band of near-critical pixels. Pass criterion: relative error < 10⁻¹⁰ in r, θ; sets the near-critical flag threshold of §5.
3. Quadrature t̃, φ vs Krang's analytic values; anchor residual statistics; check the slow-light delays between sub-images against Krang.
4. Duals vs finite differences for (a, θo) with the recurrence (today: 10⁻⁸ with the direct path).
5. End-to-end image through a test splat, GPU vs CPU Float64: done for the direct path (1.2×10⁻¹²), repeat for the recurrence.
6. Cross-code check of coordinates and time delays against Jipole/ipole trajectories (validation gate 2 of the main plan) — this validates Krang itself, which has changed materially since the registered v0.4.1 (root fixes in May and August 2026, polarization fix in July 2026).

## 9. Work plan

1. **Done today:** toolchain fixes, probes and measurements (`gpu_probes/`).
2. **Geodesic module skeleton** (`KerrSplat.Geodesics`): K1 with concrete SoA types and case sorting; stored-mode K2 using Krang's direct evaluation as the reference implementation (it works now at 110 ns/sample); the API `regenerate!(cache, a, θo, camera)`. Gate 1.
3. **Recurrence marcher** for radial cases 1–4 and polar ordinary/vortical motion, with re-anchoring. Gate 2. This is the step that delivers the speed.
4. **t, φ by anchored quadrature.** Gate 3. Measure the real per-sample cost; update §1.
5. **Duals** through the recurrence path. Gate 4.
6. **Fused mode** and the consumer interface for the transport kernel; tile scheduler for large screens.
7. **Upstream PRs to Krang**: concrete pixel array type in the KA extension, `first(pixels)` in `generate_rays`, `unsafe_trunc` for duals, a note on the stack size. Pin the project to a Krang commit until they land.

## 10. Risks

- **Case 3/4 and vortical formulas.** The recurrence must reproduce Krang's exact branch logic (which root is the turning point, sign conventions of ν_r, ν_θ, winding index). Mitigation: gate 2 compares at every sample; the direct evaluation stays in the code as the reference and as the fallback for flagged pixels.
- **Near-critical pixels.** The recurrence error grows toward k → 1 (10⁻⁹ at k² = 0.9999 over 4000 unanchored steps). Anchoring cures the measured cases; the flag-and-fallback covers the rest.
- **Quadrature accuracy for t near the horizon.** dt/dτ ∝ 1/Δ grows as r → r_h; samples inside r_h(1+ε) are discarded anyway (g → 0), and the anchor residuals report where the error is.
- **Krang drift.** Main moves; the registered release is stale and numerically different. Pin the commit; vendor if needed; keep the coupling to the ~12 functions listed in §4.
- **CUDA.jl toolchain selection.** Keep the 12.8 runtime preference in every environment on this machine until the driver is CUDA-13 capable.
- **Occupancy.** The direct-evaluation kernels are register- and stack-heavy (needed a 2 KB stack). The recurrence kernel is small; keep the SoA layout and static N so the compiler can hold the state in registers.

## 11. Files

- `gpu_probes/gpu_geodesics_bench.jl` — the measurements of §3 (screen, stored rays, fused march, Float32 comparison).
- `gpu_probes/gpu_spacetime_duals.jl` — ∂/∂a and ∂/∂θo on the GPU via duals, checked against finite differences.
- `gpu_probes/jacobi_recurrence_prototype.jl` — accuracy of the addition-theorem recurrence.
- `gpu_probes/README.md` — environment setup (Krang checkout path, runtime pin, stack limit).
