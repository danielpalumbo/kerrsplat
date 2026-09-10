# Interstellar scattering of Sgr A*: the diffractive (ensemble-average) kernel as a Gaussian in
# the visibility domain, multiplied into the model visibilities of every scan that carries it.
# The refractive substructure and the non-Gaussian core of the Johnson et al. (2018) kernel are
# not modelled.
"""
    ScatteringKernel(ν; fwhm_maj = 1.380, fwhm_min = 0.703, pa = 81.9)

The diffractive scattering kernel of Sgr A* at the frequency `ν` (Hz) as a Gaussian in the
visibility domain: FWHM `fwhm_maj` mas × λ_cm² along the position angle `pa` (degrees east
of north) and `fwhm_min` mas × λ_cm² across it. The defaults are Johnson et al. (2018), the
values of the EHT's 2017 Sgr A* analysis; ehtim's own constants (Bower et al. 2006) are
1.309, 0.640 and 78. The kernel's value at `(u, v)` in wavelengths is
exp(−2π² (a u² + 2c uv + b v²)) with ehtim's covariance form (`sgra_kernel_uv`; the gate
`test_scattering`); `taper` multiplies model visibilities by it.
"""
struct ScatteringKernel{T}
    a::T
    b::T
    c::T
end
function ScatteringKernel(ν; fwhm_maj = 1.380, fwhm_min = 0.703, pa = 81.9)
    λcm = 2.99792458e10 / ν
    mas = 1e-3 * π / (180 * 3600)
    σmaj = fwhm_maj * λcm^2 * mas / (2 * sqrt(2 * log(2)))
    σmin = fwhm_min * λcm^2 * mas / (2 * sqrt(2 * log(2)))
    θ = -deg2rad(pa)                                          # ehtim: the angle enters negated in this convention
    a = (σmin * cos(θ))^2 + (σmaj * sin(θ))^2
    b = (σmaj * cos(θ))^2 + (σmin * sin(θ))^2
    c = (σmin^2 - σmaj^2) * cos(θ) * sin(θ)
    return ScatteringKernel(promote(a, b, c)...)
end
(k::ScatteringKernel)(u, v) = exp(-2 * oftype(float(u), π)^2 * (k.a * u^2 + 2 * k.c * u * v + k.b * v^2))

"Model visibilities through the scattering kernel (`nothing`: unchanged)."
taper(::Nothing, model, u, v) = model
taper(k::ScatteringKernel, model, u, v) = [model[i] .* k(u[i], v[i]) for i in eachindex(model)]

export ScatteringKernel, taper
