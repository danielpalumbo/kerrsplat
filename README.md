# kerrsplat

Differentiable Gaussian splatting of time-dependent plasma into the Kerr spacetime, fitted to
spectro-polarimetric (Stokes I, Q, U, V; multi-frequency) movies of the near-horizon region.
Julia, built on [Krang.jl](https://github.com/dchang10/Krang.jl) analytic geodesics, with the
hot path on NVIDIA GPUs via CUDA.jl + KernelAbstractions and gradients from Enzyme (reverse mode,
splat parameters) and ForwardDiff duals (forward mode, spacetime parameters).

Status (2026-09-02): planning and feasibility. No package code yet.

- `docs/plans/` — the project plan (2026-07-09), the one-zone-splat addendum (2026-07-10), and
  the GPU geodesic plan (2026-09-02). Read them in that order.
- `smoketests/` — Enzyme reverse-mode gradients through Krang geodesics and polarization
  (the feasibility evidence for the plan).
- `gpu_probes/` — CUDA measurements behind the GPU geodesic plan, plus the environment fixes
  needed to run Krang inside CUDA kernels (see its README).

See `CLAUDE.md` for the conventions that apply to all work in this repository.
