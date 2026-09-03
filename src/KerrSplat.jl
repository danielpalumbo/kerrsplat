"""
    KerrSplat

Differentiable Gaussian splatting of plasma into the Kerr spacetime, fitted to Stokes IQUV
movies. The design lives in `docs/plans/`. Submodules:

- [`Geodesics`](@ref KerrSplat.Geodesics): GPU-resident per-pixel and per-sample geodesic
  data (Krang's analytic geodesics evaluated inside KernelAbstractions kernels).
- [`Splats`](@ref KerrSplat.Splats): Gaussian plasma splats and their optically-thin rendering
  through the fused marcher (Phase 1 of the roadmap).
- [`Transfer`](@ref KerrSplat.Transfer): synchrotron transfer coefficients and (Phase 2)
  polarized radiative transfer along the cached rays.
"""
module KerrSplat

include("Geodesics/Geodesics.jl")
using .Geodesics
include("Transfer/Transfer.jl")
using .Transfer
include("Splats/Splats.jl")
using .Splats

export Geodesics, Splats, Transfer

end
