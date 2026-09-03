"""
    KerrSplat.Transfer

Polarized radiative transfer: synchrotron transfer coefficients (this file set), later the
frame rotation angles and the analytic constant-coefficient step. Everything is pure,
allocation-free Float64 arithmetic that runs inside KernelAbstractions kernels on the CPU and
CUDA backends and differentiates with Enzyme and ForwardDiff.

Conventions follow ipole (Mościbrodzka & Gammie 2018): cgs units, the electron temperature as
Θe = kT/(mₑc²), Stokes parameters in a basis aligned with the magnetic field projected on the
plane of the sky (Q along the projected field, so that synchrotron emission has j_Q > 0 and
j_U = 0), and Stokes V in the IEEE/IAU sense (ipole's sign corrections to the Pandya+ 2016
fits included).
"""
module Transfer

using Bessels
using StaticArrays

include("constants.jl")
include("coefficients.jl")
include("step.jl")

export StokesCoefficients, thermal_synchrotron, powerlaw_synchrotron, thermal_emissivity_leung,
       thermal_synchrotron_pandya, planck, planck_invariant, invariants, cap_polarization
export transfer_step, RadiativeState, advance, rotate_to_screen

end
