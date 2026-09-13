"""
    KerrSplat.Geodesics

The geodesic layer of the GPU plan (`docs/plans/kerrsplat_gpu_geodesics_plan.md`, §4 and §9):
everything that has to be regenerated when the spin `a`, the observer inclination `θo` or the
camera change.

* `camera.jl` — [`Camera`](@ref): the Bardeen screen coordinates of every pixel.
* `pixel_constants.jl` — kernels K0/K1: Krang's per-pixel constants (conserved quantities,
  radial roots, elliptic antiderivatives at infinity, total Mino time) computed on the device
  and stored as a structure of arrays ([`PixelConstants`](@ref)) in root-case-sorted order.
* `direct_march.jl` — kernel K2 in stored mode: one thread per ray marches `N` uniform
  Mino-time samples and writes (t̃, r, θ, φ, ν_r, ν_θ, ok) into [`GeodesicSamples`](@ref).
  This is the *reference* marcher: it calls Krang's closed-form `emission_coordinates` at
  every sample.
* `jacobi.jl`, `recurrence.jl` — K2 in recurrence mode (plan §4): r(τ) and θ(τ) advanced by
  the Jacobi addition theorems with periodic re-anchoring, validated against the direct
  marcher and a BigFloat reference at every sample.
* `quadrature.jl` — t̃(τ) and φ(τ) by Simpson quadrature of the Mino-time rates between
  Krang anchors, with the large-r, horizon and polar-axis singular parts integrated in closed
  form. This is what `Recurrence(M)` runs; each ray's largest anchor residual is kept as an
  error estimate.
* `cache.jl` — [`GeodesicCache`](@ref) owning the device buffers, and
  [`regenerate!`](@ref)`(cache, a, θo[, camera])`.

All kernels are KernelAbstractions kernels, so they run on the CPU backend for testing and on
CUDA for production. Float64 throughout (plan §5).
"""
module Geodesics

using Adapt
using CUDA
using ForwardDiff
using JacobiElliptic
using KernelAbstractions
using Krang
using StaticArrays

const KA = KernelAbstractions

export Camera, PixelConstants, GeodesicSample, GeodesicSamples, GeodesicCache, Binning, bin, binned_grid, binned_polar, concatenate
export regenerate!, npixels, nsamples, build_pixel, direct_sample, mino_step, mino_times
export unsort, to_screen, prepare_backend!, case_permutation, host, ConcretePixel, sincos_pair, ENZYME_STACK_BYTES, ENZYME_HEAP_BYTES
export pack_flags, unpack_flags, SAMPLE_OK, SAMPLE_NUR, SAMPLE_NUTH
export Direct, Recurrence, Fused, JacobiState, jacobi_state, jacobi_step, jacobi_step_constants
export march_ray, fused_march!, has_samples, tiles, precision
export Case2, Case3, Case4, radial_marcher, PolarMarcher, radius, polar_angle, polar_angle_cos
export radial_parameter, near_critical, NEAR_CRITICAL_ONE_MINUS_K
export QuadratureConstants, quadrature_march!, recurrence_march!, quasi_cartesian_kerr_schild

include("camera.jl")
include("pixel_constants.jl")
include("direct_march.jl")
include("jacobi.jl")
include("recurrence.jl")
include("quadrature.jl")
include("coordinates.jl")
include("cache.jl")

# Krang's `_θs` evaluates `unsafe_trunc(Int, τ / τ̂)`. ForwardDiff provides no such method, so
# with dual numbers (spacetime derivatives, plan §6) this becomes a dynamic dispatch, which is
# fatal inside a GPU kernel (plan §2, item 5). To be upstreamed; delete this once it lands.
Base.unsafe_trunc(::Type{I}, d::ForwardDiff.Dual) where {I<:Integer} =
    unsafe_trunc(I, ForwardDiff.value(d))

"""
    host(x)

Copy a structure of arrays (a `PixelConstants`, `GeodesicSamples`, or any array) to fresh host
`Array`s, field by field. Always a copy, also for host arrays, so that the result survives a
later `regenerate!` of the cache it came from.
"""
host(x) = Adapt.adapt(HostCopy(), x)

struct HostCopy end
Adapt.adapt_storage(::HostCopy, x::AbstractArray) = Array(x)

end
