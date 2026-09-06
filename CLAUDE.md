# KerrSplat: working conventions

## What this is
Gaussian splatting of plasma into Kerr spacetime, fitted to Stokes IQUV movies. Plans live in
`docs/plans/`; the newest (`kerrsplat_gpu_geodesics_plan.md`) supersedes earlier text where
they disagree. Accuracy and correctness matter far more than speed: every physics or numerics
change is validated (usually against Krang's direct evaluation, finite differences, or ipole)
before any performance work.

## Environment facts (this workstation)
- Julia 1.10 (juliaup `lts`). GPU: RTX 2080 SUPER (sm_75, 8 GB, FP64 at 1/32 rate),
  driver 570 (CUDA 12.8), 16 CPU threads, 31 GB RAM.
- Krang.jl must be the git `main` branch pinned to commit `f36f43a` (2026-08-15), never the
  registered v0.4.1 (stale: no GPU extensions, missing root fixes and a 2026-07-06
  Walker–Penrose polarization fix; numerically different). Environments here add it with
  `Pkg.add(url="https://github.com/dchang10/Krang.jl", rev="f36f43a")`.
- CUDA.jl needs `CUDA.set_runtime_version!(v"12.8")` in every environment on this machine
  (its compat shim otherwise selects a CUDA 13 toolchain that GeForce cards cannot load).
- Krang code inside CUDA kernels needs `CUDA.limit!(CUDA.LIMIT_STACK_SIZE, 4096)` and fully
  concrete pixel types; never build kernels with `julia -g2` (Unicode identifiers break ptxas).
- Enzyme reverse mode inside KernelAbstractions kernels requires compile-time trip counts
  (`Val(N)`); runtime loop bounds crash with an illegal memory access.
- Float32 is numerically unstable in Krang's per-sample geodesic path. Geodesics are Float64.
- Enzyme inside a CUDA kernel (reverse mode, one ray per thread) works only over stored
  samples (the march's special functions get compiled through checked host code otherwise) and
  with device-safe code: no `sincos` (Enzyme has no rule for `__nv_sincos`; use
  `Geodesics.sincos_pair`), no `evalpoly`/`@horner` tables of nine or more coefficients (they
  are outlined into calls Enzyme cannot cache; use `Transfer.@muladd_chain`), no mutually
  recursive helpers, and no loops in the differentiated device function (Enzyme keeps a loop's
  per-iteration cache in device malloc; unroll with `ntuple(f, Val(N))` and tuple reductions,
  as `Transfer.accumulate_elements` does for a `StaticCount` model — a recursion on the index
  is not inferable, a `@generated` unroll hit a JIT error on the CPU backend, a closure that
  captures a `Type` is a dynamic dispatch on the device, and ptxas runs out of memory on more
  than two unrolled polarized samples). Enzyme's device reverse pass of the polarized transfer
  step is ~50× its forward cost (3–5× on the CPU): the GPU sweep is exact but not faster than
  the CPU on this card. The per-thread stack tops out at 64 KB on this card (eight polarized samples per
  kernel); longer tapes spill to device malloc, so `prepare_backend!` raises the malloc heap to
  1 GB at the first gradient kernel (CUDA refuses to change it after a kernel has used malloc). Newer Enzyme (0.13.200) with CUDA.jl 6.3.1
  is worse, not better. The fast path is the dual sweep (`Splats.polarized_gradient!`, default
  `method = :dual`): the reverse over the compositing is written by hand from 4-vectors (tails
  and adjoints) and the per-sample derivatives are ForwardDiff duals, which run in CUDA
  kernels without any of Enzyme's constraints; four times the forward transport per gradient.
- ForwardDiff ≥ 1: `x == 0` on a dual also requires zero partials, and `sqrt` at a zero value
  has NaN partials. Guard removable singularities on the value (`Transfer.vanishes`,
  `Transfer.safe_sqrt`), never with `==` against a literal.
- Enzyme's reverse pass over the KernelAbstractions CPU kernel tapes the whole screen: about
  1 GB per 8e4 pixel-samples (12k pixels × 160 samples reached 25 GB and was OOM-killed).
  Differentiate large screens tile by tile (`Geodesics.tiles`, a cache and a movie slice per
  tile, gradients summed).
- Enzyme with non-`const` globals captured in closures hits an internal error; make them
  `const` or pass them as arguments. A variable assigned both inside a closure Enzyme
  differentiates and elsewhere in the enclosing function is boxed by Julia, and Enzyme then
  treats the box captured by the `Const` closure as constant memory and silently returns a
  zero gradient: never reuse a closure's local names in the enclosing function.
- Enzyme compilation inside a KernelAbstractions CPU kernel deadlocks when several worker tasks
  hit the first call at once (`julia -t 8`; never single-threaded): launch such a kernel once
  with `ndrange = 1` (scratch outputs and a copy of the adjoint seed) before the real launch.
- The full test suite writes its progress to stderr and takes over an hour; run it through a
  pipe (`… 2>&1 | tee log`), since output redirected to a file is buffered until exit.

## Git workflow
- `main` is changed through pull requests. Work on a branch named `<topic>` (e.g.
  `geodesics-skeleton`), commit in small steps with descriptive messages, push the branch and
  open a PR with `gh pr create`. Claude may merge pull requests into `main` itself (Daniel,
  2026-09-03); when merging a stacked PR, retarget the next PR to `main`
  (`gh pr edit N --base main`) before deleting the merged branch, otherwise GitHub closes it.
- Never force-push `main`. Never commit papers (`*.pdf`), logs, or credentials.
- Run the relevant tests/smoke tests before opening a PR and state in the PR what was run.

## Code conventions
- Pure, allocation-free, StaticArrays-style functions in the hot path (Enzyme- and GPU-safe).
- Kernels via KernelAbstractions so they run on the CPU backend for testing and on CUDA for real.
- Every new numerical routine ships with a test comparing against an independent reference.
