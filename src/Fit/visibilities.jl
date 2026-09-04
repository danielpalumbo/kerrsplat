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

# ---- closure quantities ---------------------------------------------------------------------------
"""
    closure_phases(vis, triangles) -> Vector

Closure phases arg(V₁V₂V₃) of Stokes I for `triangles` given as triples of visibility indices
whose baselines (ij, jk, ki) close; Stokes I is the first component of each visibility.
"""
function closure_phases(vis::AbstractVector, triangles)
    return [angle(vis[t[1]][1] * vis[t[2]][1] * vis[t[3]][1]) for t in triangles]
end

"""
    log_closure_amplitudes(vis, quadrangles) -> Vector

Log closure amplitudes ln(|V₁₂||V₃₄|/(|V₁₃||V₂₄|)) of Stokes I for `quadrangles` given as
4-tuples of visibility indices (12, 34, 13, 24).
"""
function log_closure_amplitudes(vis::AbstractVector, quadrangles)
    return [log(abs(vis[q[1]][1]) * abs(vis[q[2]][1]) / (abs(vis[q[3]][1]) * abs(vis[q[4]][1]))) for q in quadrangles]
end

"""
    ClosureData(u, v, triangles, phases, σ_phase, quadrangles, logamps, σ_logamp)

Observed closure phases (radians) on `triangles` and log closure amplitudes on `quadrangles`
(index tuples into the baselines `u`, `v`) with their standard deviations (any of the two sets
may be empty).
"""
struct ClosureData{U,V,TR,P,SP,Q,A,SA}
    u::U
    v::V
    triangles::TR
    phases::P
    σ_phase::SP
    quadrangles::Q
    logamps::A
    σ_logamp::SA
end

"""
    chi2_closures(image, Δα, L, D, data::ClosureData)

χ² of a model image against closure phases (with the phase difference wrapped to (−π, π])
and log closure amplitudes; gain-independent, so the usual likelihood for calibrated-free fits.
"""
function chi2_closures(image, Δα, L, D, data::ClosureData)
    model = visibilities(image, Δα, L, D, data.u, data.v)
    total = zero(real(eltype(first(model))))
    if !isempty(data.triangles)
        cp = closure_phases(model, data.triangles)
        for k in eachindex(cp)
            d = rem(cp[k] - data.phases[k], 2 * oftype(cp[k], π), RoundNearest)
            total += (d / data.σ_phase[k])^2
        end
    end
    if !isempty(data.quadrangles)
        la = log_closure_amplitudes(model, data.quadrangles)
        for k in eachindex(la)
            total += ((la[k] - data.logamps[k]) / data.σ_logamp[k])^2
        end
    end
    return total
end

export closure_phases, log_closure_amplitudes, ClosureData, chi2_closures

# ---- station gains ----------------------------------------------------------------------------------
"""
    apply_gains(vis, gains, s1, s2) -> Vector

Visibilities corrupted by complex station gains: V′ₖ = g_{s1[k]} conj(g_{s2[k]}) Vₖ, with `gains`
given as a matrix of size 2 × nstations holding log-amplitude and phase per station (a
parameter block that a fit can carry as nuisance parameters), and `s1`, `s2` the station indices
of baseline k.
"""
function apply_gains(vis::AbstractVector, gains::AbstractMatrix, s1::AbstractVector{<:Integer}, s2::AbstractVector{<:Integer})
    return [vis[k] .* (exp(gains[1, s1[k]] + gains[1, s2[k]]) * cis(gains[2, s1[k]] - gains[2, s2[k]])) for k in eachindex(vis)]
end

"""
    chi2_visibilities(image, Δα, L, D, data::VisibilityData, gains, s1, s2; σ_logamp = 0.1, σ_phase = Inf)

χ² of a gained model against visibility data plus Gaussian priors on the gain log-amplitudes
(spread `σ_logamp`, centred on zero) and phases (spread `σ_phase`; `Inf` leaves them free), the
usual self-calibration likelihood; closure quantities need no gains.
"""
function chi2_visibilities(image, Δα, L, D, data::VisibilityData, gains::AbstractMatrix, s1, s2; σ_logamp = 0.1, σ_phase = Inf)
    model = apply_gains(visibilities(image, Δα, L, D, data.u, data.v), gains, s1, s2)
    total = zero(real(eltype(first(model))))
    for k in eachindex(model)
        r = (model[k] .- data.vis[k]) ./ noise(data.σ, k)
        total += sum(abs2, r)
    end
    for j in 1:size(gains, 2)
        total += (gains[1, j] / σ_logamp)^2
        isfinite(σ_phase) && (total += (gains[2, j] / σ_phase)^2)
    end
    return total
end

export apply_gains
