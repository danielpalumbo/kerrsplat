# GPU probes for the KerrSplat geodesic layer

Scripts behind the measurements in `../docs/plans/kerrsplat_gpu_geodesics_plan.md` (2026-09-02).

## Setup (once)

The Manifests in this directory and in `../smoketests` pin Krang.jl to the git commit
`f36f43a` of `main` (the registered v0.4.1 lacks the GPU extensions, three geodesic root fixes
and a Walker–Penrose polarization fix). Instantiate and pin the CUDA runtime to the driver's
CUDA version (12.8 for driver 570); without the pin, CUDA.jl 6.3 selects a CUDA 13.3 toolchain
through its forward-compatibility shim and the kernels fail to load on GeForce cards:

    julia --project=. -e 'import Pkg; Pkg.instantiate(); using CUDA; CUDA.set_runtime_version!(v"12.8")'

`LocalPreferences.toml` in this directory already carries that pin. If Pkg hangs while cloning
Krang, set `JULIA_PKG_USE_CLI_GIT=true` so it uses the system git (and gh's credential helper).

## Running

    julia -t 16 --project=. gpu_geodesics_bench.jl        # screen / stored rays / fused march, Float64 and Float32
    julia --project=. gpu_spacetime_duals.jl              # d/da, d/dθo on the GPU vs finite differences
    julia --project=. jacobi_recurrence_prototype.jl      # addition-theorem recurrence accuracy (CPU)

Two things every Krang-in-CUDA kernel needs, both already in the scripts:

- `CUDA.limit!(CUDA.LIMIT_STACK_SIZE, 4096)` (or larger): Krang's Float64 radial integrals
  overflow the default 1 KB per-thread stack, which shows up as an illegal memory access.
- Allocate pixel arrays with the fully concrete type, e.g.
  `CuArray{typeof(Krang.SlowLightIntensityPixel(met, α, β, θo))}`; `SlowLightIntensityPixel{T}`
  alone is abstract and rejected by CUDA.jl.

Do not use `julia -g2` with these kernels: Krang's Unicode identifiers end up in the PTX debug
info and ptxas rejects them.
