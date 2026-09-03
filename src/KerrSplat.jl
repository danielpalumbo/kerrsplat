"""
    KerrSplat

Differentiable Gaussian splatting of plasma into the Kerr spacetime, fitted to Stokes IQUV
movies. The design lives in `docs/plans/`. Submodules:

- [`Geodesics`](@ref KerrSplat.Geodesics): GPU-resident per-pixel and per-sample geodesic
  data (Krang's analytic geodesics evaluated inside KernelAbstractions kernels).
"""
module KerrSplat

include("Geodesics/Geodesics.jl")
using .Geodesics

export Geodesics

end
