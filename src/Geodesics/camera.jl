"""
    Camera{T}

The observer's screen: the Bardeen coordinates `(α, β)` of every pixel, in units of GM/c²,
as two flat vectors in screen (column-major) order, plus the screen shape `(nα, nβ)`.

The spacetime parameters (spin, inclination) are deliberately *not* part of the camera: they
are passed to [`regenerate!`](@ref) so that the geodesic cache can be rebuilt whenever the
optimizer moves them. Mass/distance scale and camera roll never enter the geodesic layer (plan
§6).

    Camera((αmin, αmax), (βmin, βmax), res)            # res × res grid, like Krang's screens
    Camera((αmin, αmax), (βmin, βmax), (nα, nβ))
    Camera(αs, βs)                                      # arbitrary pixel set, shape (npix, 1)

Pixel `(i, j)` of a grid has `α = αmin + (αmax - αmin) (i - 1)/(nα - 1)` and likewise for β
(Krang's convention), at linear index `i + (j - 1) nα`.
"""
struct Camera{T,V<:AbstractVector{T}}
    αs::V
    βs::V
    size::Tuple{Int,Int}
    function Camera(αs::V, βs::V, size::Tuple{Int,Int}) where {T,V<:AbstractVector{T}}
        length(αs) == length(βs) == prod(size) ||
            throw(DimensionMismatch("camera pixel vectors do not match the screen size"))
        return new{T,V}(αs, βs, size)
    end
end

Camera(αs::AbstractVector{T}, βs::AbstractVector{T}) where {T} = Camera(αs, βs, (length(αs), 1))

function Camera(αrange::NTuple{2,Real}, βrange::NTuple{2,Real}, size::Tuple{Int,Int})
    T = float(promote_type(typeof.(αrange)..., typeof.(βrange)...))
    nα, nβ = size
    αv = grid_axis(T, αrange..., nα)
    βv = grid_axis(T, βrange..., nβ)
    αs = [αv[i] for i in 1:nα, j in 1:nβ]
    βs = [βv[j] for i in 1:nα, j in 1:nβ]
    return Camera(vec(αs), vec(βs), size)
end
Camera(αrange::NTuple{2,Real}, βrange::NTuple{2,Real}, res::Integer) =
    Camera(αrange, βrange, (Int(res), Int(res)))

# Krang's screen formula, evaluated the same way so that pixels agree bit for bit.
grid_axis(::Type{T}, lo, hi, n) where {T} =
    n == 1 ? [T(lo)] : [T(lo) + (T(hi) - T(lo)) * (T(i) - 1) / (n - 1) for i in 1:n]

npixels(c::Camera) = length(c.αs)
Base.size(c::Camera) = c.size
Base.eltype(::Camera{T}) where {T} = T

# ---- pixel integration: cameras whose points are sub-samples of pixels -----------------------------
"""
    Binning(pixel, size)

Pixel integration of a camera: `pixel[m]` is the pixel (linear index into a screen of shape
`size`) that point `m` of the camera belongs to, and a pixel's value is the mean over its
points ([`bin`](@ref)). Every pixel needs at least one point. Built by [`binned_grid`](@ref),
[`binned_polar`](@ref) and [`concatenate`](@ref), which also build the matching camera (a point
list, screen shape `(npoints, 1)`).
"""
struct Binning
    pixel::Vector{Int}
    count::Vector{Int}
    size::Tuple{Int,Int}
end
function Binning(pixel::AbstractVector{<:Integer}, size::Tuple{Int,Int})
    npix = prod(size)
    count = zeros(Int, npix)
    for q in pixel
        1 <= q <= npix || throw(ArgumentError("pixel index $q outside 1:$npix"))
        count[q] += 1
    end
    all(>(0), count) || throw(ArgumentError("every pixel needs at least one point"))
    return Binning(collect(Int, pixel), count, size)
end
npixels(b::Binning) = prod(b.size)
npoints(b::Binning) = length(b.pixel)
Base.size(b::Binning) = b.size

"""
    bin(binning, values) -> Array of shape size(binning)
    bin(binning, cube)   -> (nα, nβ, nt, nν)

Mean of `values` (one per camera point, screen order) over the points of every pixel; the
element type is whatever supports `+` and division by an integer (Stokes 4-vectors, duals).
For a movie cube `(npoints, 1, nt, nν)` every frame and frequency is binned.
"""
function bin(b::Binning, values::AbstractVector)
    length(values) == npoints(b) || throw(DimensionMismatch("expected $(npoints(b)) values, got $(length(values))"))
    acc = zeros(eltype(values), npixels(b))
    for m in eachindex(values)
        @inbounds acc[b.pixel[m]] += values[m]
    end
    return reshape(acc ./ b.count, b.size)
end
function bin(b::Binning, cube::AbstractArray{<:Any,4})
    out = Array{eltype(cube)}(undef, b.size..., size(cube, 3), size(cube, 4))
    for l in axes(cube, 4), k in axes(cube, 3)
        out[:, :, k, l] = bin(b, vec(view(cube, :, :, k, l)))
    end
    return out
end

"""
    binned_grid((αlo, αhi), (βlo, βhi), (nα, nβ); subsamples = 1) -> (camera, binning)

A screen of `nα × nβ` square pixels covering the given edges, each pixel integrated over
`subsamples²` points at the centres of its sub-cells (`subsamples = 1`: one point at the pixel
centre). Points are listed pixel by pixel in screen (column-major) order.
"""
function binned_grid(αrange::NTuple{2,Real}, βrange::NTuple{2,Real}, size::Tuple{Int,Int}; subsamples::Integer = 1)
    T = float(promote_type(typeof.(αrange)..., typeof.(βrange)...))
    nα, nβ = size; K = Int(subsamples)
    K >= 1 || throw(ArgumentError("subsamples must be at least 1"))
    Δα = (T(αrange[2]) - T(αrange[1])) / nα; Δβ = (T(βrange[2]) - T(βrange[1])) / nβ
    αs = T[]; βs = T[]; pixel = Int[]
    for j in 1:nβ, i in 1:nα, sj in 1:K, si in 1:K
        push!(αs, T(αrange[1]) + (i - 1 + (si - T(0.5)) / K) * Δα)
        push!(βs, T(βrange[1]) + (j - 1 + (sj - T(0.5)) / K) * Δβ)
        push!(pixel, i + (j - 1) * nα)
    end
    return Camera(αs, βs), Binning(pixel, size)
end
binned_grid(αrange::NTuple{2,Real}, βrange::NTuple{2,Real}, res::Integer; kwargs...) = binned_grid(αrange, βrange, (Int(res), Int(res)); kwargs...)

"""
    binned_polar(ρedges, nψ; subsamples = (1, 1), ψ0 = 0) -> (camera, binning)

An annulus of polar pixels: radial cells between consecutive `ρedges` and `nψ` equal azimuthal
cells from `ψ0`, pixel `(iρ, iψ)` at linear index `iρ + (iψ - 1) nρ`, each integrated over
`subsamples[1] × subsamples[2]` points at the centres of its (ρ, ψ) sub-cells (α = ρ cos ψ,
β = ρ sin ψ).
"""
function binned_polar(ρedges::AbstractVector{<:Real}, nψ::Integer; subsamples::Tuple{Integer,Integer} = (1, 1), ψ0::Real = 0)
    T = float(eltype(ρedges))
    nρ = length(ρedges) - 1; Kρ, Kψ = Int.(subsamples)
    nρ >= 1 || throw(ArgumentError("need at least two radial edges"))
    Δψ = 2 * T(π) / nψ
    αs = T[]; βs = T[]; pixel = Int[]
    for iψ in 1:nψ, iρ in 1:nρ, sψ in 1:Kψ, sρ in 1:Kρ
        ρ = T(ρedges[iρ]) + (sρ - T(0.5)) / Kρ * (T(ρedges[iρ+1]) - T(ρedges[iρ]))
        ψ = T(ψ0) + (iψ - 1 + (sψ - T(0.5)) / Kψ) * Δψ
        push!(αs, ρ * cos(ψ)); push!(βs, ρ * sin(ψ))
        push!(pixel, iρ + (iψ - 1) * nρ)
    end
    return Camera(αs, βs), Binning(pixel, (nρ, nψ))
end

"""
    concatenate((camera, binning), (camera, binning), ...) -> (camera, binning)

One camera and binning from several: the points and the pixels are concatenated in order (the
pixels of the second screen follow those of the first), with screen shape `(npixels, 1)`.
"""
function concatenate(parts::Tuple{Camera,Binning}...)
    αs = reduce(vcat, (c.αs for (c, _) in parts)); βs = reduce(vcat, (c.βs for (c, _) in parts))
    pixel = Int[]; offset = 0
    for (_, b) in parts
        append!(pixel, b.pixel .+ offset)
        offset += npixels(b)
    end
    return Camera(αs, βs), Binning(pixel, (offset, 1))
end
