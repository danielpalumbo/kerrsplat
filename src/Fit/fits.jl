# Stokes images and movies as FITS files in the layout ehtim writes (one HDU per Stokes parameter
# with a STOKES keyword, BUNIT = JY/PIXEL, CDELT in degrees with RA increasing to the left, FREQ in
# Hz, MJD), and the conversions to the model's camera (pixels in M) and cgs intensities.

using FITSIO

"Length unit L = GM/c² [cm] and time unit GM/c³ [s] for a mass in solar masses."
time_unit(M_solar) = Splats.gravitational_radius(M_solar) / Transfer.CL

"""
    read_stokes_fits(path) -> (stokes, header)

Read an ehtim-style FITS image: the HDUs carrying `STOKES` = I, Q, U, V (a missing one counts
as zero) as a matrix of Stokes 4-vectors indexed (x, y) in the file's pixel units, plus a named
tuple of the header quantities used here: `psize_deg` (|CDELT2|), `freq` (FREQ, Hz), `mjd`,
`bunit`.
"""
function read_stokes_fits(path::AbstractString)
    f = FITS(path, "r")
    imgs = Dict{Char,Matrix{Float64}}()
    hdr = nothing
    for hdu in f
        hdu isa ImageHDU || continue
        h = read_header(hdu)
        hdr === nothing && (hdr = h)
        st = haskey(h, "STOKES") ? uppercase(strip(string(h["STOKES"])))[1] : (isempty(imgs) ? 'I' : ' ')
        st in ('I', 'Q', 'U', 'V') || continue
        imgs[st] = Float64.(read(hdu))
    end
    close(f)
    haskey(imgs, 'I') || throw(ArgumentError("no Stokes I image in $path"))
    n = size(imgs['I'])
    get0(c) = get(imgs, c, zeros(n))
    S = [SVector(imgs['I'][i, j], get0('Q')[i, j], get0('U')[i, j], get0('V')[i, j]) for i in 1:n[1], j in 1:n[2]]
    header = (psize_deg = abs(haskey(hdr, "CDELT2") ? hdr["CDELT2"] : hdr["CDELT1"]),
              freq = haskey(hdr, "FREQ") ? Float64(hdr["FREQ"]) : NaN,
              mjd = haskey(hdr, "MJD") ? Float64(hdr["MJD"]) : NaN,
              bunit = haskey(hdr, "BUNIT") ? uppercase(strip(string(hdr["BUNIT"]))) : "JY/PIXEL")
    return S, header
end

"""
    read_stokes_movie(paths; M_solar, D_pc, mjd0 = nothing, σ, freq = nothing) -> (movie, camera, L)

Assemble a `StokesMovie` from ehtim-style FITS images (one file per frame and frequency, any
order): the frames are grouped by MJD and FREQ (or the given `freq` when the files lack it),
intensities converted from Jy/pixel to cgs with the pixel solid angle, times to M from
`mjd0` (default: the earliest frame), the pixel grid to a `Geodesics.Camera` in M for a black
hole of `M_solar` solar masses at `D_pc` parsecs. `σ` is the noise per Stokes parameter in
Jy/pixel (a 4-vector; converted like the data).
"""
function read_stokes_movie(paths::AbstractVector{<:AbstractString}; M_solar, D_pc, mjd0 = nothing, σ, freq = nothing)
    L = Splats.gravitational_radius(M_solar)
    D = D_pc * Transfer.PC
    frames = [(read_stokes_fits(p)..., p) for p in paths]
    hdr1 = frames[1][2]
    psize_rad = hdr1.psize_deg * π / 180
    nx, ny = size(frames[1][1])
    Ω = psize_rad^2
    tojy = hdr1.bunit == "JY/PIXEL" ? Transfer.JY / Ω : throw(ArgumentError("unsupported BUNIT $(hdr1.bunit)"))
    mjds = [fr[2].mjd for fr in frames]
    freqs = [isnan(fr[2].freq) ? (freq === nothing ? throw(ArgumentError("no FREQ in $(fr[3])")) : Float64(freq)) : fr[2].freq for fr in frames]
    t0 = mjd0 === nothing ? minimum(mjds) : mjd0
    times = sort(unique(mjds)); νs = sort(unique(freqs))
    data = Array{SVector{4,Float64}}(undef, nx, ny, length(times), length(νs))
    filled = falses(length(times), length(νs))
    for (fr, mjd, ν) in zip(frames, mjds, freqs)
        k = findfirst(==(mjd), times); l = findfirst(==(ν), νs)
        size(fr[1]) == (nx, ny) || throw(DimensionMismatch("image sizes differ across the movie"))
        data[:, :, k, l] = fr[1] .* tojy
        filled[k, l] = true
    end
    all(filled) || throw(ArgumentError("the movie does not cover every (time, frequency) pair"))
    tM = (times .- t0) .* 86400 ./ time_unit(M_solar)
    psize_M = psize_rad * D / L
    αs = [(i - (nx + 1) / 2) * psize_M for i in 1:nx]; βs = [(j - (ny + 1) / 2) * psize_M for j in 1:ny]
    camera = Geodesics.Camera(vec([αs[i] for i in 1:nx, j in 1:ny]), vec([βs[j] for i in 1:nx, j in 1:ny]), (nx, ny))
    movie = StokesMovie(data, tM, νs, SVector{4,Float64}(σ) .* tojy)
    return movie, camera, L
end

"""
    write_stokes_fits(path, image, psize_M; M_solar, D_pc, freq, mjd, source = "KerrSplat")

Write a Stokes image (matrix of 4-vectors in cgs intensity, indexed (x, y) with x toward west
= +α and y toward north) as an ehtim-style FITS file in Jy/pixel with one HDU per Stokes
parameter, for a pixel size `psize_M` (M) of a black hole of `M_solar` solar masses at `D_pc`
parsecs.
"""
function write_stokes_fits(path::AbstractString, image::AbstractMatrix{<:SVector{4}}, psize_M; M_solar, D_pc, freq, mjd, source = "KerrSplat")
    L = Splats.gravitational_radius(M_solar); D = D_pc * Transfer.PC
    psize_rad = psize_M * L / D
    Ω = psize_rad^2
    nx, ny = size(image)
    f = FITS(path, "w")
    for (k, st) in enumerate(("I", "Q", "U", "V"))
        img = [image[i, j][k] * Ω / Transfer.JY for i in 1:nx, j in 1:ny]
        h = FITSHeader(["OBJECT", "CTYPE1", "CTYPE2", "CDELT1", "CDELT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CUNIT1", "CUNIT2", "FREQ", "MJD", "BUNIT", "STOKES", "TELESCOP"],
                       Any[source, "RA---SIN", "DEC--SIN", -psize_rad * 180 / π, psize_rad * 180 / π, (nx + 1) / 2, (ny + 1) / 2, 0.0, 0.0, "deg", "deg", Float64(freq), Float64(mjd), "JY/PIXEL", st, "KerrSplat"],
                       fill("", 16))
        write(f, img; header = h)
    end
    close(f)
    return path
end

export read_stokes_fits, read_stokes_movie, write_stokes_fits, time_unit
