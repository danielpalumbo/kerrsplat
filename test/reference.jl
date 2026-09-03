# Host-side reference (Krang's own CPU evaluation) and comparison helpers for the geodesic tests.
#
# Criterion. The device result of a quantity q must agree with the host result to within
#     |Δq| ≤ tol + C · sens(q),
# where sens(q) is the change of the host result when any one of the inputs (a, α, β, θo) is
# moved by one ulp. sens measures the conditioning of Krang's closed-form expressions at that
# pixel (the azimuth of a ray grazing the polar axis has parameter n → 1 in the third-kind
# elliptic integral and moves by ~1e-7 per ulp), and the GPU's transcendental functions round
# differently from the CPU's by a few ulps, which propagates exactly like an input perturbation.
# Where the formulas are well conditioned the criterion reduces to |Δq| ≤ tol.
#
# Input perturbations do not exercise one further source of rounding noise: the cancellation
# u₊ = Δθ + √(Δθ² + η/a²) in Krang's polar constants (Δθ ~ −(η + λ²)/(2a²) is large and
# negative for rays grazing the polar axis), whose rounding differs between GPU and CPU through
# FMA contraction and is amplified by Π(n = u₊ → 1). That is why the CUDA base tolerance in
# runtests.jl is 2e-9 rather than the 1e-10 seen elsewhere; see docs/notes/.

using Krang
using KerrSplat.Geodesics

const SENS_FACTOR = 8.0

"""
    scaled_err(x, y)

|x − y| / max(|y|, 1): relative error for |y| ≥ 1, absolute error below. Non-finite values
count as agreeing only if both are non-finite in the same way.
"""
function scaled_err(x::Real, y::Real)
    if isfinite(x) && isfinite(y)
        return abs(x - y) / max(abs(y), one(y))
    elseif isnan(x) && isnan(y)
        return 0.0
    elseif isinf(x) && isinf(y) && sign(x) == sign(y)
        return 0.0
    else
        return Inf
    end
end
scaled_err(x::Complex, y::Complex) = max(scaled_err(real(x), real(y)), scaled_err(imag(x), imag(y)))

const ROOT_PERMUTATIONS = let p = Vector{NTuple{4,Int}}()
    for a in 1:4, b in 1:4, c in 1:4, d in 1:4
        length(unique((a, b, c, d))) == 4 && push!(p, (a, b, c, d))
    end
    p
end

"Best-assignment scaled error between two 4-tuples of roots (order-insensitive)."
root_set_err(got, want) = minimum(ROOT_PERMUTATIONS) do σ
    maximum(scaled_err(got[σ[i]], want[i]) for i in 1:4)
end

make_pixel(a, α, β, θo) = Krang.SlowLightIntensityPixel(Krang.Kerr(a), α, β, θo)

"Reference pixels for `camera` in the cache's sorted order."
reference_pixels(cache::GeodesicCache, camera::Camera, a, θo) =
    [make_pixel(a, camera.αs[i], camera.βs[i], θo) for i in cache.perm_host]

"The four one-ulp input perturbations of a pixel, in the cache's sorted order."
function perturbed_pixels(cache::GeodesicCache, camera::Camera, a, θo)
    return map(cache.perm_host) do i
        α, β = camera.αs[i], camera.βs[i]
        (make_pixel(nextfloat(a), α, β, θo), make_pixel(a, nextfloat(α), β, θo),
         make_pixel(a, α, nextfloat(β), θo), make_pixel(a, α, β, nextfloat(θo)))
    end
end

const CONSTANT_FIELDS = (
    (:α, pix -> pix.screen_coordinate[1]),
    (:β, pix -> pix.screen_coordinate[2]),
    (:η, pix -> pix.η),
    (:λ, pix -> pix.λ),
    (:I0_inf, pix -> pix.I0_inf),
    (:τ_total, pix -> pix.total_mino_time),
    (:Iϕ_inf, pix -> pix.Iϕ_inf),
    (:It_inf, pix -> pix.It_inf),
    (:I1_inf, pix -> pix.I1_inf_m_I0_terms),
    (:I2_inf, pix -> pix.I2_inf_m_I0_terms),
    (:Ip_inf, pix -> pix.Ip_inf_m_I0_terms),
    (:Im_inf, pix -> pix.Im_inf_m_I0_terms),
    (:Gθo, pix -> pix.absGθo_Gθhat[1]),
    (:Gθhat, pix -> pix.absGθo_Gθhat[2]),
    (:Gϕo, pix -> pix.absGϕo_Gϕhat[1]),
    (:Gϕhat, pix -> pix.absGϕo_Gϕhat[2]),
    (:Gto, pix -> pix.absGto_Gthat[1]),
    (:Gthat, pix -> pix.absGto_Gthat[2]),
)

"""
    compare_constants(pc_host, refpix, pertpix) -> Dict(field => (; err, sens, excess))

Per field: the maximum scaled error of a host copy of `PixelConstants` against the reference
pixels, the maximum one-ulp sensitivity, and the maximum over pixels of
`err − SENS_FACTOR · sens` (the quantity that must stay below `tol`). The roots are compared
as a set: which member of a complex-conjugate pair lands in which slot depends on rounding,
and every Krang formula uses the pair symmetrically.
"""
function compare_constants(pc::PixelConstants, refpix::AbstractVector, pertpix::AbstractVector)
    out = Dict{Symbol,NamedTuple{(:err, :sens, :excess),NTuple{3,Float64}}}()
    for (name, getter) in CONSTANT_FIELDS
        col = getfield(pc, name)
        err = sens = excess = 0.0
        for j in eachindex(refpix)
            e = scaled_err(col[j], getter(refpix[j]))
            s = maximum(scaled_err(getter(p), getter(refpix[j])) for p in pertpix[j])
            err = max(err, e)
            sens = max(sens, s)
            excess = max(excess, e - SENS_FACTOR * s)
        end
        out[name] = (err = err, sens = sens, excess = excess)
    end
    err = sens = excess = 0.0
    for j in eachindex(refpix)
        e = root_set_err((pc.r1[j], pc.r2[j], pc.r3[j], pc.r4[j]), refpix[j].roots)
        s = maximum(root_set_err(p.roots, refpix[j].roots) for p in pertpix[j])
        err = max(err, e)
        sens = max(sens, s)
        excess = max(excess, e - SENS_FACTOR * s)
    end
    out[:roots] = (err = err, sens = sens, excess = excess)
    nr = maximum(abs(Int(pc.numreals[j]) - sum(Krang._isreal2, refpix[j].roots)) for j in eachindex(refpix))
    out[:numreals] = (err = Float64(nr), sens = 0.0, excess = Float64(nr))
    return out
end

spherical_position(s::GeodesicSample) = s.r .* (sin(s.θ) * cos(s.ϕ), sin(s.θ) * sin(s.ϕ), cos(s.θ))

"Error in the spherical position r (sin θ cos φ, sin θ sin φ, cos θ), scaled by max(r, 1)."
function position_err(s::GeodesicSample, ref::GeodesicSample)
    (isfinite(s.r) && isfinite(ref.r)) || return scaled_err(s.r, ref.r)
    d = spherical_position(s) .- spherical_position(ref)
    return sqrt(sum(abs2, d)) / max(abs(ref.r), 1.0)
end

sample_errs(s::GeodesicSample, ref::GeodesicSample) =
    (t = scaled_err(s.t, ref.t), r = scaled_err(s.r, ref.r), θ = scaled_err(s.θ, ref.θ),
     ϕ = scaled_err(s.ϕ, ref.ϕ), pos = position_err(s, ref))

ref_samples(pix, ::Val{N}) where {N} =
    (Δτ = mino_step(Krang.total_mino_time(pix), Val(N)); ntuple(k -> direct_sample(pix, k * Δτ), N))

"""
    compare_samples(S_host, refpix, pertpix, Val(N)) -> (; err, sens, excess, flag_mismatches, nvalid)

Stored samples against Krang's direct `emission_coordinates` on the reference pixels' own
Mino-time grids. `err`, `sens`, `excess` are NamedTuples over the metrics `(t, r, θ, ϕ, pos)`
(scaled errors of t̃, r, θ, the unwrapped azimuth, and the spherical position; the azimuth
alone is ill-conditioned for rays grazing the polar axis, the position is what consumers use).
`flag_mismatches` counts samples whose (ok, νr, νθ) differ from the reference.
"""
function compare_samples(S::GeodesicSamples, refpix::AbstractVector, pertpix::AbstractVector, ::Val{N}) where {N}
    z = (t = 0.0, r = 0.0, θ = 0.0, ϕ = 0.0, pos = 0.0)
    err = sens = excess = z
    mism = 0
    nvalid = 0
    for j in eachindex(refpix)
        base = ref_samples(refpix[j], Val(N))
        perts = map(p -> ref_samples(p, Val(N)), pertpix[j])
        for k in 1:N
            ref = base[k]
            s = S[j, k]
            (s.ok, s.νr, s.νθ) == (ref.ok, ref.νr, ref.νθ) || (mism += 1)
            ref.ok || continue
            nvalid += 1
            e = sample_errs(s, ref)
            # a perturbed ray whose sample k is invalid contributes no sensitivity information
            ps = [sample_errs(p[k], ref) for p in perts if p[k].ok]
            sk = isempty(ps) ? z : (t = maximum(x.t for x in ps), r = maximum(x.r for x in ps),
                                    θ = maximum(x.θ for x in ps), ϕ = maximum(x.ϕ for x in ps),
                                    pos = maximum(x.pos for x in ps))
            err = map(max, err, e)
            sens = map(max, sens, sk)
            excess = map((ex, ee, ss) -> max(ex, ee - SENS_FACTOR * ss), excess, e, sk)
        end
    end
    return (err = err, sens = sens, excess = excess, flag_mismatches = mism, nvalid = nvalid)
end
