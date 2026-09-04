"""
    KerrSplat.Transfer

Polarized radiative transfer: synchrotron transfer coefficients (this file set), later the
frame rotation angles and the analytic constant-coefficient step. Everything is pure,
allocation-free Float64 arithmetic that runs inside KernelAbstractions kernels on the CPU and
CUDA backends and differentiates with Enzyme and ForwardDiff.

Conventions follow ipole (Mościbrodzka & Gammie 2018): cgs units, the electron temperature as
Θe = kT/(mₑc²), Stokes parameters in the field-aligned basis of ipole's plasma tetrad, whose Q
axis is perpendicular to the magnetic field projected on the plane of the sky (so that
synchrotron emission, polarized perpendicular to the field, has j_Q > 0 and j_U = 0), and
Stokes V in the IEEE/IAU sense (ipole's sign corrections to the Pandya+ 2016 fits included).
"""
module Transfer

using Adapt
using Bessels
using ForwardDiff
using Krang
using StaticArrays
using ..Geodesics: GeodesicSample

include("constants.jl")
include("coefficients.jl")
include("kappa.jl")
include("step.jl")
include("frames.jl")
include("transport.jl")

export StokesCoefficients, thermal_synchrotron, powerlaw_synchrotron, powerlaw_rotativities, powerlaw_rotativities_valid, thermal_emissivity_leung,
       kappa_synchrotron, kappa_hypergeometric, kappa_rotativities,
       thermal_synchrotron_pandya, planck, planck_invariant, invariants, cap_polarization
export transfer_step, RadiativeState, advance, rotate_to_screen
export LocalFrame, local_frame, boost_zamo_to_fluid, walker_penrose, screen_direction
export UnpolarizedState, UnpolarizedTransport, unpolarized_coefficients, unpolarized_step, observed_intensity,
       gravitational_radius, pixel_solid_angle
export RadiativeTransport, nelements, element, observed_stokes, CompositeModel

end
