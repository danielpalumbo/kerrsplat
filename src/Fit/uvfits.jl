# Reading interferometric data from uvfits files (the EHT's distribution format) without an
# external imaging package: the random-groups primary HDU through CFITSIO's group routines, the
# station table from AIPS AN, the frequency setup from AIPS FQ. The parsing rules follow
# ehtim's `load_uvfits` (validated against it in test_uvfits): visibilities are averaged over
# channels and IFs without weights, the noise of the average is sqrt(Σσ²)/n, and circular
# correlation products are converted to Stokes as I = (RR + LL)/2, Q = (RL + LR)/2,
# U = (RL − LR)/(2i), V = (RR − LL)/2 with the noise propagated in quadrature.

"""
    Observation

Visibilities read from a uvfits file (see [`read_uvfits`](@ref)). Per row: `time` (UT hours
of day `mjd`), `tint` (s), the station indices `s1`, `s2` into `stations`, the baseline `u`,
`v` (wavelengths, as stored: no sign flip), the Stokes visibilities `vis` (complex 4-vectors
in Jy) and their noise `σ` (real 4-vectors in Jy; `Inf` for a Stokes parameter the row lacks).
`freq` is the reference frequency (Hz), `bandwidth` the channel width, `ra` (hours) and `dec`
(degrees) the phase centre, `source` the object name.
"""
struct Observation{T}
    time::Vector{T}
    tint::Vector{T}
    s1::Vector{Int}
    s2::Vector{Int}
    stations::Vector{String}
    u::Vector{T}
    v::Vector{T}
    vis::Vector{SVector{4,Complex{T}}}
    σ::Vector{SVector{4,T}}
    freq::T
    bandwidth::T
    ra::T
    dec::T
    mjd::Int
    source::String
    coh::Vector{SVector{4,Complex{T}}}      # the circular correlation products (RR, LL, RL, LR)
    σ_coh::Vector{SVector{4,T}}             # their thermal noise (Inf where a product is absent)
end

"The products (RR, LL, RL, LR) and their noise from Stokes visibilities and noise (RR = I + V, LL = I − V, RL = Q + iU, LR = Q − iU; the noise adds in quadrature)."
function _products_from_stokes(vis::SVector{4,Complex{T}}, σ::SVector{4,T}) where {T}
    sP = sqrt(σ[1]^2 + σ[4]^2); sX = sqrt(σ[2]^2 + σ[3]^2)
    return SVector(vis[1] + vis[4], vis[1] - vis[4], vis[2] + im * vis[3], vis[2] - im * vis[3]), SVector(sP, sP, sX, sX)
end

"An observation from Stokes visibilities alone: the correlation products derived from them (`Observation(...; coh, σ_coh)` keeps measured products)."
function Observation{T}(time, tint, s1, s2, stations, u, v, vis, σ, freq, bandwidth, ra, dec, mjd, source) where {T}
    pr = [_products_from_stokes(vis[k], σ[k]) for k in eachindex(vis)]
    return Observation{T}(time, tint, s1, s2, stations, u, v, vis, σ, freq, bandwidth, ra, dec, mjd, source, first.(pr), last.(pr))
end

Base.length(o::Observation) = length(o.u)

"The [`VisibilityData`](@ref) of an observation (the likelihood input)."
VisibilityData(o::Observation) = VisibilityData(o.u, o.v, o.vis, o.σ)

const _libcfitsio = FITSIO.CFITSIO.libcfitsio

function _group_parameters!(f, group::Integer, out::Vector{Float64})
    status = Ref{Cint}(0)
    ccall((:ffggpd, _libcfitsio), Cint, (Ptr{Cvoid}, Clong, Clong, Clong, Ptr{Float64}, Ref{Cint}),
          f.ptr, group, 1, length(out), out, status)
    status[] == 0 || FITSIO.CFITSIO.fits_assert_ok(status[])
    return out
end

function _group_data!(f, group::Integer, out::Vector{Float64})
    status = Ref{Cint}(0)
    anynul = Ref{Cint}(0)
    ccall((:ffgpvd, _libcfitsio), Cint, (Ptr{Cvoid}, Clong, Int64, Int64, Float64, Ptr{Float64}, Ref{Cint}, Ref{Cint}),
          f.ptr, group, 1, length(out), NaN, out, anynul, status)
    status[] == 0 || FITSIO.CFITSIO.fits_assert_ok(status[])
    return out
end

_hkey(h, k, default) = haskey(h, k) ? h[k] : default

"""
    read_uvfits(path) -> Observation

Read a uvfits file (AIPS random-groups layout as written by ehtim, HOPS/EHT-HOPS or CASA
exports): the UU, VV parameters scaled to wavelengths with the reference frequency of the
antenna table (or of the FREQ axis when the table lacks one), the baseline codes decoded as
256 a₁ + a₂, the two DATE parameters summed to a Julian date, the correlation products (RR,
LL, RL, LR, or a Stokes basis when the STOKES axis says so) averaged over channels and IFs
where their weights are positive and finite, and converted to Stokes visibilities; the
circular products themselves are kept in `coh` (RR, LL, RL, LR) with their own noise `σ_coh`
(infinite where a product is absent), which the instrument model compares against. Rows with
no parallel-hand data are dropped; a row with only one parallel hand takes it as Stokes I
without V; missing cross hands leave Q and U with infinite noise.
"""
function read_uvfits(path::AbstractString)
    f = FITS(path)
    try
        hdr = read_header(f[1])
        _hkey(hdr, "GROUPS", false) == true || throw(ArgumentError("$path is not a random-groups uvfits file"))
        ngroups = Int(hdr["GCOUNT"]); npar = Int(hdr["PCOUNT"]); naxis = Int(hdr["NAXIS"])
        dims = [Int(hdr["NAXIS$i"]) for i in 2:naxis]
        ctypes = [uppercase(strip(string(_hkey(hdr, "CTYPE$i", "")))) for i in 2:naxis]
        ptypes = [uppercase(strip(string(hdr["PTYPE$i"]))) for i in 1:npar]
        pscal = [Float64(_hkey(hdr, "PSCAL$i", 1.0)) for i in 1:npar]
        pzero = [Float64(_hkey(hdr, "PZERO$i", 0.0)) for i in 1:npar]
        icomplex = findfirst(==("COMPLEX"), ctypes); istokes = findfirst(==("STOKES"), ctypes)
        ifreq = findfirst(==("FREQ"), ctypes); iif = findfirst(==("IF"), ctypes)
        (icomplex == 1 && istokes == 2 && ifreq !== nothing) || throw(ArgumentError("unsupported axis order $(ctypes) in $path"))
        dims[1] == 3 || throw(ArgumentError("expected (re, im, weight) triples on the COMPLEX axis of $path"))
        nstokes = dims[2]; nchan = dims[ifreq]; nif = iif === nothing ? 1 : dims[iif]
        stokes0 = Int(round(hdr["CRVAL$(istokes + 1)"])); dstokes = Int(round(_hkey(hdr, "CDELT$(istokes + 1)", 1.0)))
        codes = [stokes0 + (k - 1) * dstokes for k in 1:nstokes]      # −1..−4 = RR, LL, RL, LR; 1..4 = I, Q, U, V
        circular = all(<(0), codes)
        circular || all(>(0), codes) || throw(ArgumentError("mixed STOKES axis $(codes) in $path"))
        slot(code) = findfirst(==(code), codes)
        ra = Float64(_hkey(hdr, "CRVAL$(ifreq + 3)", NaN)) * 12 / 180          # RA axis follows FREQ (and IF)
        dec = Float64(_hkey(hdr, "CRVAL$(ifreq + 4)", NaN))
        for i in 2:naxis
            ctypes[i - 1] == "RA" && (ra = Float64(hdr["CRVAL$i"]) * 12 / 180)
            ctypes[i - 1] == "DEC" && (dec = Float64(hdr["CRVAL$i"]))
        end
        source = strip(string(_hkey(hdr, "OBJECT", "")))
        ch1 = Float64(hdr["CRVAL$(ifreq + 1)"]); bandwidth = abs(Float64(_hkey(hdr, "CDELT$(ifreq + 1)", NaN)))
        # station table
        stations = String[]; nosta = Int[]; rf = ch1
        for k in 2:length(f)
            h = read_header(f[k])
            extname = uppercase(strip(string(_hkey(h, "EXTNAME", ""))))
            if extname == "AIPS AN"
                stations = [strip(String(s)) for s in read(f[k], "ANNAME")]
                nosta = Int.(read(f[k], "NOSTA"))
                haskey(h, "FREQ") && (rf = Float64(h["FREQ"]))
            end
        end
        isempty(stations) && throw(ArgumentError("no AIPS AN table in $path"))
        order = sortperm(nosta); stations = stations[order]; nosta = nosta[order]
        station_index(n) = (i = findfirst(==(n), nosta); i === nothing ? throw(ArgumentError("baseline refers to station $n absent from the AN table")) : i)
        iu = findfirst(p -> startswith(p, "UU"), ptypes); iv = findfirst(p -> startswith(p, "VV"), ptypes)
        ibl = findfirst(==("BASELINE"), ptypes); idates = findall(==("DATE"), ptypes)
        itint = findfirst(==("INTTIM"), ptypes)
        (iu !== nothing && iv !== nothing && ibl !== nothing && !isempty(idates)) || throw(ArgumentError("missing UU/VV/BASELINE/DATE parameters in $path"))
        # rows
        fptr = f.fitsfile
        FITSIO.CFITSIO.fits_movabs_hdu(fptr, 1)
        pars = zeros(npar); block = zeros(prod(dims))
        nprod = nstokes
        time = Float64[]; tint = Float64[]; s1 = Int[]; s2 = Int[]; u = Float64[]; v = Float64[]
        vis = SVector{4,ComplexF64}[]; σ = SVector{4,Float64}[]; jds = Float64[]
        coh = SVector{4,ComplexF64}[]; σ_coh = SVector{4,Float64}[]
        mean = zeros(ComplexF64, nprod); sig = zeros(nprod)
        stride_chan = 3 * nstokes; stride_if = iif === nothing ? 0 : 3 * nstokes * prod(dims[3:iif-1])
        for g in 1:ngroups
            _group_parameters!(fptr, g, pars)
            _group_data!(fptr, g, block)
            for p in 1:nprod
                acc = zero(ComplexF64); var = 0.0; n = 0
                for m in 1:nif, c in 1:nchan
                    base = 3 * (p - 1) + stride_chan * (c - 1) + stride_if * (m - 1)
                    w = block[base + 3]
                    (isfinite(w) && w > 0) || continue
                    acc += complex(block[base + 1], block[base + 2]); var += 1 / w; n += 1
                end
                mean[p] = n == 0 ? NaN : acc / n
                sig[p] = n == 0 ? Inf : sqrt(var) / n
            end
            jd = sum(pscal[i] * pars[i] + pzero[i] for i in idates)
            if circular
                rr = slot(-1); ll = slot(-2); rl = slot(-3); lr = slot(-4)
                hasrr = rr !== nothing && isfinite(sig[rr]); hasll = ll !== nothing && isfinite(sig[ll])
                (hasrr || hasll) || continue
                if hasrr && hasll
                    I = (mean[rr] + mean[ll]) / 2; V = (mean[rr] - mean[ll]) / 2
                    sI = sqrt(sig[rr]^2 + sig[ll]^2) / 2; sV = sI
                else
                    I = hasrr ? mean[rr] : mean[ll]; V = 0.0im
                    sI = hasrr ? sig[rr] : sig[ll]; sV = Inf
                end
                hascross = rl !== nothing && lr !== nothing && isfinite(sig[rl]) && isfinite(sig[lr])
                Q = hascross ? (mean[rl] + mean[lr]) / 2 : 0.0im
                U = hascross ? (mean[rl] - mean[lr]) / (2im) : 0.0im
                sQ = hascross ? sqrt(sig[rl]^2 + sig[lr]^2) / 2 : Inf
                push!(vis, SVector(I, Q, U, V)); push!(σ, SVector(sI, sQ, sQ, sV))
                prod(j) = j === nothing || !isfinite(sig[j]) ? (0.0im, Inf) : (mean[j], sig[j])
                pRR, sRR = prod(rr); pLL, sLL = prod(ll); pRL, sRL = prod(rl); pLR, sLR = prod(lr)
                push!(coh, SVector(pRR, pLL, pRL, pLR)); push!(σ_coh, SVector(sRR, sLL, sRL, sLR))
            else
                si = slot(1)
                (si !== nothing && isfinite(sig[si])) || continue
                comp(k) = (j = slot(k); j === nothing || !isfinite(sig[j]) ? (0.0im, Inf) : (mean[j], sig[j]))
                Q, sQ = comp(2); U, sU = comp(3); V, sV = comp(4)
                push!(vis, SVector(mean[si], Q, U, V)); push!(σ, SVector(sig[si], sQ, sU, sV))
                pr = _products_from_stokes(vis[end], σ[end]); push!(coh, pr[1]); push!(σ_coh, pr[2])
            end
            bl = Int(floor(pscal[ibl] * pars[ibl] + pzero[ibl]))
            a1 = bl ÷ 256; a2 = bl - 256 * a1
            push!(s1, station_index(a1)); push!(s2, station_index(a2))
            push!(u, (pscal[iu] * pars[iu] + pzero[iu]) * rf); push!(v, (pscal[iv] * pars[iv] + pzero[iv]) * rf)
            push!(jds, jd)
            push!(tint, itint === nothing ? 0.0 : pscal[itint] * pars[itint] + pzero[itint])
        end
        isempty(jds) && throw(ArgumentError("no unflagged parallel-hand data in $path"))
        mjd = Int(floor(minimum(jds) - 2400000.5))
        time = (jds .- 2400000.5 .- mjd) .* 24
        return Observation{Float64}(time, tint, s1, s2, stations, u, v, vis, σ, rf, bandwidth, ra, dec, mjd, String(source), coh, σ_coh)
    finally
        close(f)
    end
end

"""
    scan_triangles(o::Observation; tol = 1e-6) -> Vector{NTuple{3,Int}}
    scan_quadrangles(o::Observation; tol = 1e-6) -> Vector{NTuple{4,Int}}

Index tuples for [`closure_phases`](@ref) and [`log_closure_amplitudes`](@ref) from the
simultaneous baselines of an observation: rows within `tol` hours of each other form a scan,
and every triple (i < j < k) of stations with all three baselines present gives the triangle
(ij, jk, ki), every quadruple (i < j < k < l) with all six baselines present the quadrangle
(ij, kl, ik, jl). A negative index means the conjugate of that row's visibility (the stored
baseline runs the other way). Only the minimal sets of an Nₛ-station scan are not enforced:
the tuples are all triangles and quadrangles, whose closure quantities are correlated.
"""
function scan_triangles(o::Observation; tol = 1e-6)
    out = NTuple{3,Int}[]
    for rows in _scans(o, tol)
        bl = _baseline_lookup(o, rows)
        sts = sort!(unique!(vcat(o.s1[rows], o.s2[rows])))
        for (a, i) in enumerate(sts), (b, j) in enumerate(sts), (c, k) in enumerate(sts)
            a < b < c || continue
            ij = _signed(bl, i, j); jk = _signed(bl, j, k); ki = _signed(bl, k, i)
            (ij == 0 || jk == 0 || ki == 0) && continue
            push!(out, (ij, jk, ki))
        end
    end
    return out
end

function scan_quadrangles(o::Observation; tol = 1e-6)
    out = NTuple{4,Int}[]
    for rows in _scans(o, tol)
        bl = _baseline_lookup(o, rows)
        sts = sort!(unique!(vcat(o.s1[rows], o.s2[rows])))
        n = length(sts)
        for a in 1:n, b in a+1:n, c in b+1:n, d in c+1:n
            i, j, k, l = sts[a], sts[b], sts[c], sts[d]
            ij = _signed(bl, i, j); kl = _signed(bl, k, l); ik = _signed(bl, i, k); jl = _signed(bl, j, l)
            (ij == 0 || kl == 0 || ik == 0 || jl == 0) && continue
            push!(out, (ij, kl, ik, jl))
        end
    end
    return out
end

function _scans(o::Observation, tol)
    order = sortperm(o.time)
    scans = Vector{Int}[]
    for r in order
        if isempty(scans) || abs(o.time[r] - o.time[scans[end][1]]) > tol
            push!(scans, [r])
        else
            push!(scans[end], r)
        end
    end
    return scans
end

_baseline_lookup(o::Observation, rows) = Dict((o.s1[r], o.s2[r]) => r for r in rows)

function _signed(bl, i, j)
    haskey(bl, (i, j)) && return bl[(i, j)]
    haskey(bl, (j, i)) && return -bl[(j, i)]
    return 0
end

"""
    scan_index(o::Observation; gap = 0.0165) -> Vector{Int}

Scan number of every row: scans are the runs of distinct time stamps separated by less than
`gap` hours (ehtim's `add_scans` rule), numbered from 1 in time order.
"""
function scan_index(o::Observation{T}; gap = 0.0165) where {T}
    stamps = sort(unique(o.time))
    scan_of = Dict{T,Int}()
    id = 1
    for (k, t) in enumerate(stamps)
        scan_of[t] = id
        k < length(stamps) && stamps[k + 1] - t > gap && (id += 1)
    end
    return [scan_of[t] for t in o.time]
end

"""
    average_scans(o::Observation; gap = 0.0165) -> Observation

Coherent scan averaging with ehtim's rules (`add_scans` and `avg_coherent(0, scan_avg = true)`
with predicted errors): scans are the runs of distinct time stamps separated by less than `gap`
hours (59.4 s by default); within a scan every baseline's rows are averaged, the visibility as
the plain mean over the rows where the Stokes parameter is present, its noise as √(Σσ²)/n, `u`
and `v` as means, `tint` as the sum and `time` as the earliest stamp. Rows keep their order of
first appearance.
"""
function average_scans(o::Observation{T}; gap = 0.0165) where {T}
    scans = scan_index(o; gap)
    groups = Dict{Tuple{Int,Int,Int},Vector{Int}}()
    order = Tuple{Int,Int,Int}[]
    for r in eachindex(o.time)
        key = (scans[r], o.s1[r], o.s2[r])
        haskey(groups, key) || (groups[key] = Int[]; push!(order, key))
        push!(groups[key], r)
    end
    time = T[]; tint = T[]; s1 = Int[]; s2 = Int[]; u = T[]; v = T[]
    vis = SVector{4,Complex{T}}[]; σ = SVector{4,T}[]
    for key in order
        rows = groups[key]
        push!(time, minimum(o.time[r] for r in rows)); push!(tint, sum(o.tint[r] for r in rows))
        push!(s1, key[2]); push!(s2, key[3])
        push!(u, sum(o.u[r] for r in rows) / length(rows)); push!(v, sum(o.v[r] for r in rows) / length(rows))
        vk = zeros(Complex{T}, 4); sk = fill(T(Inf), 4)
        for k in 1:4
            present = [r for r in rows if isfinite(o.σ[r][k])]
            isempty(present) && continue
            vk[k] = sum(o.vis[r][k] for r in present) / length(present)
            sk[k] = sqrt(sum(o.σ[r][k]^2 for r in present)) / length(present)
        end
        push!(vis, SVector{4}(vk)); push!(σ, SVector{4}(sk))
    end
    return Observation{T}(time, tint, s1, s2, o.stations, u, v, vis, σ, o.freq, o.bandwidth, o.ra, o.dec, o.mjd, o.source)
end

export Observation, read_uvfits, scan_index, average_scans, scan_triangles, scan_quadrangles
