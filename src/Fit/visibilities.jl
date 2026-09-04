# Visibility-domain likelihood without an external imaging package: complex visibilities of a
# Stokes image by direct Fourier transform at arbitrary (u, v) points (differentiable in the
# image, hence in the splat parameters through Enzyme), and the χ² of a set of visibilities. The
# Comrade.jl route (closure quantities, gains) can be layered on this later.

"""
    visibilities(image, Δα, L, D, u, v) -> Vector{Complex}

Complex visibilities V(u, v) = Σ_pixels I(x, y) ΔΩ exp(−2πi (u x + v y)) of a Stokes image
(matrix of 4-vectors, cgs intensity, x toward +α = west, y toward north) for baselines `u`, `v`
in wavelengths, with the pixel size `Δα` in M, the length unit `L` and distance `D` in cm.
Returns the visibilities of the four Stokes parameters as a vector of `SVector{4}` in Jy. The
zero-spacing values are the flux densities.
"""
function visibilities(image::AbstractMatrix{<:SVector{4}}, Δα, L, D, u::AbstractVector, v::AbstractVector)
    nx, ny = size(image)
    psize = Δα * L / D                           # radians; east is −α, so the RA offset of pixel i is −x
    Ω = psize^2
    xs = [-(i - (nx + 1) / 2) * psize for i in 1:nx]
    ys = [(j - (ny + 1) / 2) * psize for j in 1:ny]
    T = eltype(first(image))
    out = Vector{SVector{4,Complex{T}}}(undef, length(u))
    for k in eachindex(u)
        acc = zero(SVector{4,Complex{T}})
        for j in 1:ny, i in 1:nx
            ph = -2 * T(π) * (u[k] * xs[i] + v[k] * ys[j])
            acc += image[i, j] .* (Ω / Transfer.JY * cis(ph))
        end
        out[k] = acc
    end
    return out
end

"""
    VisibilityData(u, v, vis, σ)

Observed Stokes visibilities (`vis`: vector of complex `SVector{4}` in Jy) at baselines `u`, `v`
(wavelengths) with thermal noise `σ` per Stokes parameter (a real `SVector{4}` in Jy, or one
per visibility).
"""
struct VisibilityData{U,V,W,S}
    u::U
    v::V
    vis::W
    σ::S
end

"χ² of a model image against visibility data: Σ |V_model − V_data|² / σ² over baselines and Stokes parameters."
function chi2_visibilities(image, Δα, L, D, data::VisibilityData)
    model = visibilities(image, Δα, L, D, data.u, data.v)
    total = zero(real(eltype(first(model))))
    for k in eachindex(model)
        r = (model[k] .- data.vis[k]) ./ noise(data.σ, k)
        total += sum(abs2, r)
    end
    return total
end

export visibilities, VisibilityData, chi2_visibilities
