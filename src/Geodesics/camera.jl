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
