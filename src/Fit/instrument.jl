# The instrument model, with the structure and assumptions of Comrade.jl (Tiede 2022; the
# polarized-imaging tutorial of its documentation) and the conventions ehtim's simulator uses,
# against which it is gated (test_instrument, test/data/jones_*.csv):
#
#   • the sky enters as the coherency matrix in the circular basis, C = [RR RL; LR LL] with
#     RR = I + V, LL = I − V, RL = Q + iU, LR = Q − iU (Stokes I = (RR + LL)/2, …);
#   • the RIME: the observed matrix of baseline ij is V_ij = J_i C_ij J_j†, with one Jones matrix
#     per station and time segment, J = G D R for raw data (G = diag(g_R, g_L) the complex feed
#     gains, D = [1 d_R; d_L 1] the leakage, R = diag(e^{−iφ}, e^{iφ}) the feed rotation by the
#     angle φ) and J = R† G D R for data whose feed rotation was corrected in calibration with
#     the leakage still in (the EHT pipelines' products; Comrade's `JonesSandwich(adjoint(R), G,
#     D, R)`, ehtim's `frcal = true, dcal = false`);
#   • the gains as g_R = exp(lg_R + i gp_R) and g_L = g_R exp(lg_rat + i gp_rat) (the L feed as a
#     ratio to R), one set per station and segment (scans by default; integrations or the whole
#     track otherwise), the leakage d = d_x + i d_y per station and constant over the track;
#   • the priors: lg_R ~ N(0, 0.2) (1.0 for the LMT), lg_rat ~ N(0, 0.1), gp_rat ~ N(0, 0.1),
#     d_x, d_y ~ N(0, 0.2); the phases gp_R flat, their gauge fixed by a reference station whose
#     phase is zero in every segment (Comrade's `SingleReference`; its SEFD-ordered default is
#     stood in for by the lowest station index present, `FirstReference`);
#   • Stokes-I-only data take a scalar gain g = exp(lg + i gp) per station and segment, applied as
#     g_i conj(g_j) (Comrade's `SingleStokesGain`), with the same priors and gauge.
# Parameters are two matrices, `gains` (4 × nstations·nsegments: lg_R, gp_R, lg_rat, gp_rat; or
# 2 × … for single Stokes: lg, gp) and `dterms` (4 × nstations: d_Rx, d_Ry, d_Lx, d_Ly), plain
# arrays so that the optimizers and ForwardDiff carry them.

"Coherency matrix [RR RL; LR LL] of a Stokes visibility (I, Q, U, V)."
@inline coherency(s::SVector{4}) = @SMatrix [s[1]+s[4]  s[2]+im*s[3]; s[2]-im*s[3]  s[1]-s[4]]
"Stokes visibility (I, Q, U, V) of a coherency matrix."
@inline stokes(C::SMatrix{2,2}) = SVector((C[1, 1] + C[2, 2]) / 2, (C[1, 2] + C[2, 1]) / 2, (C[1, 2] - C[2, 1]) / (2im), (C[1, 1] - C[2, 2]) / 2)
"The products (RR, LL, RL, LR) of a coherency matrix, the order of `Observation.coh`."
@inline products(C::SMatrix{2,2}) = SVector(C[1, 1], C[2, 2], C[1, 2], C[2, 1])
"A coherency matrix from the products (RR, LL, RL, LR)."
@inline coherency_of_products(p::SVector{4}) = @SMatrix [p[1] p[3]; p[4] p[2]]

"Feed rotation by the angle φ (circular feeds): diag(e^{−iφ}, e^{iφ})."
@inline feed_rotation(φ) = @SMatrix [cis(-φ) 0; 0 cis(φ)]
@inline jones_gain(gR, gL) = @SMatrix [gR 0; 0 gL]
@inline jones_leakage(dR, dL) = @SMatrix [1 dR; dL 1]
"""
    jones(gR, gL, dR, dL, φ; corrected = true)

The Jones matrix of a station: `R† G D R` for feed-rotation-corrected data (`corrected = true`),
`G D R` for raw data. See the header of this file.
"""
@inline function jones(gR, gL, dR, dL, φ; corrected::Bool = true)
    GD = jones_gain(gR, gL) * jones_leakage(dR, dL)
    R = feed_rotation(φ)
    return corrected ? R' * GD * R : GD * R
end
"The RIME: the corrupted coherency matrix `J1 * C * J2'` of a baseline."
@inline apply_jones(C::SMatrix{2,2}, J1::SMatrix{2,2}, J2::SMatrix{2,2}) = J1 * C * J2'

abstract type Segmentation end
"Gains constant over a scan (`scan_index`)."
struct ScanSeg <: Segmentation end
"Gains free at every time stamp."
struct IntegSeg <: Segmentation end
"Gains constant over the whole track."
struct TrackSeg <: Segmentation end
segments(::ScanSeg, o::Observation) = scan_index(o)
segments(::TrackSeg, o::Observation) = ones(Int, length(o))
function segments(::IntegSeg, o::Observation; tol = 1e-6)
    ts = sort(unique(round.(o.time ./ tol))) .* tol
    return [searchsortedfirst(ts, t - tol / 2) for t in o.time]
end

abstract type Reference end
"The phase of the named station is zero in every segment where it is present (the lowest-index station present otherwise)."
struct SingleReference <: Reference
    station::String
end
"The phase of the lowest-index station present in each segment is zero."
struct FirstReference <: Reference end

"""
    InstrumentModel(o::Observation; polarized = true, leakage = polarized, corrected = true, segmentation = ScanSeg(),
                    reference = FirstReference(), σ_lg = 0.2, σ_lg_station = Dict("LM" => 1.0), σ_lgrat = 0.1, σ_gprat = 0.1, σ_d = 0.2,
                    feedangles = nothing)

The instrument model of an observation (see the header of this file): dual-feed gains with
leakage on the coherency products, or a scalar gain on Stokes I (`polarized = false`); gains
per `segmentation`; the phase gauge by `reference`; the priors' widths (`σ_lg` per station,
overridden by `σ_lg_station`); `feedangles` the feed rotation angles φ of every row's two
stations as a pair of vectors (zero when absent: no feed rotation, or data with it corrected).
"""
struct InstrumentModel{T}
    stations::Vector{String}
    nseg::Int
    seg::Vector{Int}
    ref::Vector{Int}
    polarized::Bool
    leakage::Bool
    corrected::Bool
    φ1::Vector{T}
    φ2::Vector{T}
    σ_lg::Vector{T}
    σ_lgrat::T
    σ_gprat::T
    σ_d::T
end
function InstrumentModel(o::Observation{T}; polarized::Bool = true, leakage::Bool = polarized, corrected::Bool = true, segmentation::Segmentation = ScanSeg(),
                         reference::Reference = FirstReference(), σ_lg = 0.2, σ_lg_station = Dict("LM" => 1.0), σ_lgrat = 0.1, σ_gprat = 0.1, σ_d = 0.2,
                         feedangles = nothing) where {T}
    seg = segments(segmentation, o); nseg = maximum(seg)
    nst = length(o.stations)
    ref = zeros(Int, nseg)
    want = reference isa SingleReference ? findfirst(==(reference.station), o.stations) : nothing
    for g in 1:nseg
        present = sort(unique(vcat(o.s1[seg .== g], o.s2[seg .== g])))
        ref[g] = (want !== nothing && want in present) ? want : first(present)
    end
    φ1 = feedangles === nothing ? zeros(T, length(o)) : collect(T, feedangles[1])
    φ2 = feedangles === nothing ? zeros(T, length(o)) : collect(T, feedangles[2])
    σlg = [T(get(σ_lg_station, s, σ_lg)) for s in o.stations]
    return InstrumentModel{T}(o.stations, nseg, seg, ref, polarized, leakage, corrected, φ1, φ2, σlg, T(σ_lgrat), T(σ_gprat), T(σ_d))
end
nstations(im::InstrumentModel) = length(im.stations)
ngainrows(im::InstrumentModel) = im.polarized ? 4 : 2
"Column of the gain matrix for station `s` in segment `g`."
@inline gain_column(im::InstrumentModel, s::Integer, g::Integer) = s + (g - 1) * nstations(im)
"Zero instrument parameters: unit gains, no leakage."
zero_instrument(im::InstrumentModel{T}) where {T} = (gains = zeros(T, ngainrows(im), nstations(im) * im.nseg), dterms = zeros(T, 4, nstations(im)))
"""
    free_mask(im) -> (gains, dterms)

Which instrument parameters a fit moves: the reference station's phases (gp_R and gp_rat) in
each segment are held at zero (the gauge), the d-terms only with `leakage` and only for
stations that appear in the data; columns of stations absent from a segment are held too.
"""
function free_mask(im::InstrumentModel, o::Observation)
    gm = falses(ngainrows(im), nstations(im) * im.nseg)
    present = falses(nstations(im))
    for r in eachindex(im.seg)
        g = im.seg[r]
        for s in (o.s1[r], o.s2[r])
            present[s] = true
            c = gain_column(im, s, g)
            gm[1, c] = true
            gm[2, c] = s != im.ref[g]
            if im.polarized
                gm[3, c] = true
                gm[4, c] = s != im.ref[g]
            end
        end
    end
    dm = falses(4, nstations(im))
    im.leakage && (dm[:, present] .= true)
    return gm, dm
end

"The complex gains (g_R, g_L) of station `s` in segment `g` (the reference station's phases forced to zero)."
@inline function station_gains(im::InstrumentModel, gains::AbstractMatrix, s::Integer, g::Integer)
    c = gain_column(im, s, g)
    isref = s == im.ref[g]
    @inbounds gR = exp(complex(gains[1, c], isref ? zero(gains[2, c]) : gains[2, c]))
    im.polarized || return gR, gR
    @inbounds gL = gR * exp(complex(gains[3, c], isref ? zero(gains[4, c]) : gains[4, c]))
    return gR, gL
end
"The leakage (d_R, d_L) of station `s`."
@inline function station_leakage(im::InstrumentModel, dterms::AbstractMatrix, s::Integer)
    im.leakage || return zero(complex(eltype(dterms))), zero(complex(eltype(dterms)))
    @inbounds return complex(dterms[1, s], dterms[2, s]), complex(dterms[3, s], dterms[4, s])
end
"The Jones matrix of station `s` in segment `g` with the feed rotation angle `φ`."
@inline function station_jones(im::InstrumentModel, gains::AbstractMatrix, dterms::AbstractMatrix, s::Integer, g::Integer, φ)
    gR, gL = station_gains(im, gains, s, g)
    dR, dL = station_leakage(im, dterms, s)
    return jones(gR, gL, dR, dL, φ; corrected = im.corrected)
end

"""
    instrument_residuals(model, o::Observation, rows, im, gains, dterms) -> Vector

The scaled residuals of the model Stokes visibilities `model[k]` of the rows `rows` of `o`
against the data through the instrument: for dual-feed data the real and imaginary parts of
every product (RR, LL, RL, LR) with finite noise, `(J1 C J2† − data)/σ`; for Stokes I only
the gained I, `(g1 conj(g2) I − data)/σ`. Generic in the element types of `model`, `gains`
and `dterms` (ForwardDiff duals).
"""
function instrument_residuals(model::AbstractVector{<:SVector{4}}, o::Observation, rows, im::InstrumentModel, gains::AbstractMatrix, dterms::AbstractMatrix)
    S = promote_type(real(eltype(first(model))), eltype(gains), eltype(dterms))
    res = S[]
    for (k, r) in enumerate(rows)
        g = im.seg[r]; s1 = o.s1[r]; s2 = o.s2[r]
        if im.polarized
            J1 = station_jones(im, gains, dterms, s1, g, im.φ1[r]); J2 = station_jones(im, gains, dterms, s2, g, im.φ2[r])
            V = products(apply_jones(coherency(model[k]), J1, J2))
            for p in 1:4
                σ = o.σ_coh[r][p]
                isfinite(σ) || continue
                d = (V[p] - o.coh[r][p]) / σ
                push!(res, real(d)); push!(res, imag(d))
            end
        else
            g1, _ = station_gains(im, gains, s1, g); g2, _ = station_gains(im, gains, s2, g)
            d = (g1 * conj(g2) * model[k][1] - o.vis[r][1]) / o.σ[r][1]
            push!(res, real(d)); push!(res, imag(d))
        end
    end
    return res
end

"""
    chi2_instrument(model, o::Observation, rows, im, gains, dterms) -> χ²

The squared norm of [`instrument_residuals`](@ref) (no priors; see [`penalty_instrument`](@ref)).
"""
function chi2_instrument(model::AbstractVector{<:SVector{4}}, o::Observation, rows, im::InstrumentModel, gains::AbstractMatrix, dterms::AbstractMatrix)
    S = promote_type(real(eltype(first(model))), eltype(gains), eltype(dterms))
    total = zero(S)
    for (k, r) in enumerate(rows)
        g = im.seg[r]; s1 = o.s1[r]; s2 = o.s2[r]
        if im.polarized
            J1 = station_jones(im, gains, dterms, s1, g, im.φ1[r]); J2 = station_jones(im, gains, dterms, s2, g, im.φ2[r])
            V = products(apply_jones(coherency(model[k]), J1, J2))
            for p in 1:4
                σ = o.σ_coh[r][p]
                isfinite(σ) || continue
                d = V[p] - o.coh[r][p]
                total += (real(d)^2 + imag(d)^2) / σ^2
            end
        else
            g1, _ = station_gains(im, gains, s1, g); g2, _ = station_gains(im, gains, s2, g)
            d = g1 * conj(g2) * model[k][1] - o.vis[r][1]
            total += (real(d)^2 + imag(d)^2) / o.σ[r][1]^2
        end
    end
    return total
end

"""
    chi2_products(model, o::Observation, rows, im, gains, dterms) -> (χ², n)

[`chi2_instrument`](@ref) split by product: the χ² and the number of real values of RR, LL,
RL and LR as two `SVector{4}`s (dual-feed data; for Stokes I only the first entry). The
diagnostic of a self-calibration: closures constrain Stokes I alone, so a sky whose
polarization is wrong misfits the cross-hands while the parallel hands fit.
"""
function chi2_products(model::AbstractVector{<:SVector{4}}, o::Observation, rows, im::InstrumentModel, gains::AbstractMatrix, dterms::AbstractMatrix)
    S = promote_type(real(eltype(first(model))), eltype(gains), eltype(dterms))
    total = zeros(S, 4); n = zeros(Int, 4)
    for (k, r) in enumerate(rows)
        g = im.seg[r]; s1 = o.s1[r]; s2 = o.s2[r]
        if im.polarized
            J1 = station_jones(im, gains, dterms, s1, g, im.φ1[r]); J2 = station_jones(im, gains, dterms, s2, g, im.φ2[r])
            V = products(apply_jones(coherency(model[k]), J1, J2))
            for p in 1:4
                σ = o.σ_coh[r][p]
                isfinite(σ) || continue
                d = V[p] - o.coh[r][p]
                total[p] += (real(d)^2 + imag(d)^2) / σ^2; n[p] += 2
            end
        else
            g1, _ = station_gains(im, gains, s1, g); g2, _ = station_gains(im, gains, s2, g)
            d = g1 * conj(g2) * model[k][1] - o.vis[r][1]
            total[1] += (real(d)^2 + imag(d)^2) / o.σ[r][1]^2; n[1] += 2
        end
    end
    return SVector{4}(total), SVector{4}(n)
end

"The prior residuals of the instrument (see the header): `lg/σ_lg[s]`, `lg_rat/σ_lgrat`, `gp_rat/σ_gprat` per station and segment, the d-term parts over `σ_d`; the phases gp_R carry no prior."
function instrument_prior_residuals(im::InstrumentModel, gains::AbstractMatrix, dterms::AbstractMatrix)
    S = promote_type(eltype(gains), eltype(dterms))
    res = S[]
    for g in 1:im.nseg, s in 1:nstations(im)
        c = gain_column(im, s, g)
        push!(res, gains[1, c] / im.σ_lg[s])
        if im.polarized
            push!(res, gains[3, c] / im.σ_lgrat); push!(res, gains[4, c] / im.σ_gprat)
        end
    end
    if im.leakage
        for s in 1:nstations(im), p in 1:4
            push!(res, dterms[p, s] / im.σ_d)
        end
    end
    return res
end
"The prior penalty of the instrument parameters, the squared norm of [`instrument_prior_residuals`](@ref)."
penalty_instrument(im::InstrumentModel, gains, dterms) = sum(abs2, instrument_prior_residuals(im, gains, dterms); init = zero(promote_type(eltype(gains), eltype(dterms))))

export chi2_products
export coherency, stokes, products, coherency_of_products, feed_rotation, jones, apply_jones,
       Segmentation, ScanSeg, IntegSeg, TrackSeg, segments, Reference, SingleReference, FirstReference,
       InstrumentModel, nstations, ngainrows, gain_column, zero_instrument, free_mask, station_gains, station_leakage, station_jones,
       instrument_residuals, chi2_instrument, instrument_prior_residuals, penalty_instrument
