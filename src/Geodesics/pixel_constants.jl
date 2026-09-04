# K0/K1: per-pixel constants (plan §4 "K1, per pixel").

"""
    ConcretePixel{T}

The fully concrete type of `Krang.SlowLightIntensityPixel` for scalar type `T`. Krang's own
`SlowLightIntensityPixel{T}` leaves 16 of the 17 type parameters abstract, which device arrays
reject (plan §2, item 2); this alias is what a device array of pixels has to be typed with.
"""
const ConcretePixel{T} = Krang.SlowLightIntensityPixel{T,T,Complex{T},T,T,T,T,T,T,T,T,T,T,T,T,T,T}

"""
    PixelConstants{T}

Structure of arrays holding, for every pixel, what Krang precomputes in
`SlowLightIntensityPixel`: the screen coordinates `α, β`; the conserved quantities `η, λ`;
the four radial roots `r1…r4` (complex); the radial antiderivatives at infinity `I0_inf`,
`Iϕ_inf`, `It_inf` and the I0-subtracted pieces `I1_inf, I2_inf, Ip_inf, Im_inf`; the total
Mino time `τ_total`; the angular antiderivatives `(Gθo, Gθhat)`, `(Gϕo, Gϕhat)`,
`(Gto, Gthat)`; `k_r`, the elliptic parameter of the radial closed form (1 − k_r → 0 at the
critical curve); and `numreals`, the number of real radial roots (4: Krang's cases 1 and 2,
scattering orbits; 2: case 3; 0: case 4).

Pixels are stored in root-case-sorted order (see [`GeodesicCache`](@ref)`.perm`) so that each
warp of the per-ray kernels executes a single branch of Krang's case logic, and one field read
across consecutive pixels is contiguous. [`build_pixel`](@ref) reassembles Krang's struct for
one slot.
"""
struct PixelConstants{T,VT<:AbstractVector{T},VC<:AbstractVector{Complex{T}},VI<:AbstractVector{Int8}}
    α::VT
    β::VT
    η::VT
    λ::VT
    r1::VC
    r2::VC
    r3::VC
    r4::VC
    I0_inf::VT
    τ_total::VT
    Iϕ_inf::VT
    It_inf::VT
    I1_inf::VT
    I2_inf::VT
    Ip_inf::VT
    Im_inf::VT
    Gθo::VT
    Gθhat::VT
    Gϕo::VT
    Gϕhat::VT
    Gto::VT
    Gthat::VT
    k_r::VT
    numreals::VI
end
Adapt.@adapt_structure PixelConstants

"""
    PixelConstants{T}(backend, npix)

Allocate (uninitialized) per-pixel constants for `npix` pixels on `backend`.
"""
function PixelConstants{T}(backend::KA.Backend, npix::Integer) where {T}
    v() = KA.allocate(backend, T, npix)
    c() = KA.allocate(backend, Complex{T}, npix)
    return PixelConstants(
        v(), v(), v(), v(),
        c(), c(), c(), c(),
        v(), v(), v(), v(), v(), v(), v(), v(),
        v(), v(), v(), v(), v(), v(),
        v(),
        KA.allocate(backend, Int8, npix),
    )
end

npixels(pc::PixelConstants) = length(pc.η)
Base.eltype(::PixelConstants{T}) where {T} = T

"""
    build_pixel(pc, j, met, θo) -> Krang.SlowLightIntensityPixel

Reassemble Krang's pixel struct for sorted slot `j`. Pure field copies: no elliptic function
is evaluated, so this is cheap enough to call at the top of every per-ray kernel.
"""
@inline function build_pixel(pc::PixelConstants, j::Integer, met::Krang.Kerr, θo)
    @inbounds return Krang.SlowLightIntensityPixel(
        met,
        (pc.α[j], pc.β[j]),
        (pc.r1[j], pc.r2[j], pc.r3[j], pc.r4[j]),
        pc.I0_inf[j],
        pc.τ_total[j],
        pc.Iϕ_inf[j],
        pc.It_inf[j],
        pc.I1_inf[j],
        pc.I2_inf[j],
        pc.Ip_inf[j],
        pc.Im_inf[j],
        (pc.Gθo[j], pc.Gθhat[j]),
        (pc.Gϕo[j], pc.Gϕhat[j]),
        (pc.Gto[j], pc.Gthat[j]),
        θo,
        pc.η[j],
        pc.λ[j],
    )
end

@inline num_real_roots(roots) = Int8(sum(Krang._isreal2, roots))

# K0: the root case of every pixel in screen order. Only the (cheap) roots are computed here;
# the result decides the case-sorted layout that K1 then fills directly.
@kernel function root_case_kernel!(numreals, met::Krang.Kerr, θo, @Const(αs), @Const(βs))
    i = @index(Global, Linear)
    @inbounds begin
        ηi = Krang.η(met, αs[i], βs[i], θo)
        λi = Krang.λ(met, αs[i], θo)
        numreals[i] = num_real_roots(Krang.get_radial_roots(met, ηi, λi))
    end
end

# K1: Krang's full pixel construction for the pixel in sorted slot j, written field by field
# into the structure of arrays.
@kernel function pixel_constants_kernel!(pc, met::Krang.Kerr, θo, @Const(αs), @Const(βs), @Const(perm))
    j = @index(Global, Linear)
    @inbounds begin
        i = perm[j]
        α = αs[i]
        β = βs[i]
        pix = Krang.SlowLightIntensityPixel(met, α, β, θo)
        pc.α[j] = α
        pc.β[j] = β
        pc.η[j] = pix.η
        pc.λ[j] = pix.λ
        r1, r2, r3, r4 = pix.roots
        pc.r1[j] = r1
        pc.r2[j] = r2
        pc.r3[j] = r3
        pc.r4[j] = r4
        pc.I0_inf[j] = pix.I0_inf
        pc.τ_total[j] = pix.total_mino_time
        pc.Iϕ_inf[j] = pix.Iϕ_inf
        pc.It_inf[j] = pix.It_inf
        pc.I1_inf[j] = pix.I1_inf_m_I0_terms
        pc.I2_inf[j] = pix.I2_inf_m_I0_terms
        pc.Ip_inf[j] = pix.Ip_inf_m_I0_terms
        pc.Im_inf[j] = pix.Im_inf_m_I0_terms
        Gθo, Gθhat = pix.absGθo_Gθhat
        pc.Gθo[j] = Gθo
        pc.Gθhat[j] = Gθhat
        Gϕo, Gϕhat = pix.absGϕo_Gϕhat
        pc.Gϕo[j] = Gϕo
        pc.Gϕhat[j] = Gϕhat
        Gto, Gthat = pix.absGto_Gthat
        pc.Gto[j] = Gto
        pc.Gthat[j] = Gthat
        pc.k_r[j] = radial_parameter(pix)
        pc.numreals[j] = num_real_roots(pix.roots)
    end
end

"""
    case_permutation(numreals) -> (perm, ranges)

Permutation that groups pixels by number of real radial roots in the order 4, 2, 0, and the
slot ranges of the three groups as a NamedTuple `(case2, case3, case4)` (Krang's case
numbering; `case2` covers cases 1 and 2, which share one set of formulas). `perm[j]` is the
screen index of the pixel stored in sorted slot `j`.
"""
function case_permutation(numreals::AbstractVector{Int8})
    perm = sortperm(numreals; rev = true)
    sorted = numreals[perm]
    ranges = (
        case2 = searchsorted(sorted, Int8(4); rev = true),
        case3 = searchsorted(sorted, Int8(2); rev = true),
        case4 = searchsorted(sorted, Int8(0); rev = true),
    )
    return perm, ranges
end
