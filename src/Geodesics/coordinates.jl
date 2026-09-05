# GPU-safe coordinate transforms for consumers of the samples.

"""
    sincos_pair(x) -> (sin(x), cos(x))

The fused `sincos` lowers to CUDA's `__nv_sincos` intrinsic, for which Enzyme has no reverse
rule (an in-kernel gradient then throws "No augmented forward pass found for __nv_sincos");
the hot path uses this pair instead, which differentiates on every backend.
"""
@inline sincos_pair(x) = (sin(x), cos(x))

"""
    quasi_cartesian_kerr_schild(met, r, θ, ϕ) -> (x, y, z)

Boyer–Lindquist (r, θ, φ) to the quasi-Cartesian Kerr–Schild coordinates
x = r sin θ cos φ_KS, y = r sin θ sin φ_KS, z = r cos θ with
φ_KS = φ + a/(2√(1−a²)) ln|(r − r₊)/(r − r₋)| − arctan(a/r), the same map as Krang's
`boyer_lindquist_to_quasi_cartesian_kerr_schild_fast_light` (Phys. Rev. D 86, 084049) without
the host-side warning on the horizon that keeps Krang's version out of GPU kernels. The
transform is singular at the horizon; callers discard those samples.
"""
@inline function quasi_cartesian_kerr_schild(met::Krang.Kerr{T}, r, θ, ϕ) where {T}
    a = met.spin
    temp = sqrt(one(T) - a^2)
    rp = one(T) + temp
    rm = one(T) - temp
    ϕks = ϕ + a / (2 * temp) * log(abs((r - rp + eps(T)) / (r - rm + eps(T)))) - atan(a, r)
    sθ, cθ = sincos_pair(θ)
    sϕ, cϕ = sincos_pair(ϕks)
    return r * sθ * cϕ, r * sθ * sϕ, r * cθ
end
