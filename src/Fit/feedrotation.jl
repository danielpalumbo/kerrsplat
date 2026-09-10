# Feed rotation angles of the stations, as ehtim computes them (obs_simulate.make_jones with
# obs_helpers.elev, hr_angle, par_angle): for a station at geocentric (x, y, z) observing the
# source (ra, dec) at UTC hour `t` of day `mjd`, the Greenwich mean sidereal time θ_G, the hour
# angle H = θ_G + λ − α (λ the geocentric longitude), the elevation e from the rotated station
# vector against the source direction, the parallactic angle
#     ψ = atan(sin H cos φ, sin φ cos δ − cos φ sin δ cos H)
# (φ the geocentric latitude), and the feed rotation of a mount with parameters (f_par, f_elev,
# f_off): χ = f_par ψ + f_elev e + f_off. The sidereal time is the IAU 1982 mean sidereal time
# of UTC taken as UT1 (ehtim uses astropy's IAU 2006 model with the IERS UT1 − UTC, up to a
# second of time): against ehtim's angles the elevation agrees to 3e-5 rad and the parallactic
# angle to 2.4e-4 rad (0.014°, the hour-angle offset amplified near transit), which is far below
# anything a calibration model resolves (test_feed_rotation).

"Mount parameters of a station: the feed rotation angle is `f_par ψ + f_elev e + f_off_deg°` (ehtim's FR_PAR_ANGLE, FR_ELEV_ANGLE, FR_OFFSET)."
struct Mount{T}
    f_par::T
    f_elev::T
    f_off_deg::T
end

"""
    EHT_MOUNTS

The mount parameters of the EHT stations from ehtim's `arrays/EHT2017.txt` (parallactic-angle
factor, elevation factor, offset in degrees), under both the array names and the two-letter
codes of the uvfits files.
"""
const EHT_MOUNTS = Dict(
    "ALMA" => Mount(1, 0, 0), "AA" => Mount(1, 0, 0),
    "APEX" => Mount(1, 1, 0), "AP" => Mount(1, 1, 0),
    "SMT" => Mount(1, 1, 0), "AZ" => Mount(1, 1, 0),
    "JCMT" => Mount(1, 0, 0), "JC" => Mount(1, 0, 0),
    "LMT" => Mount(1, -1, 0), "LM" => Mount(1, -1, 0),
    "PV" => Mount(1, -1, 0),
    "SMA" => Mount(1, -1, 45), "SM" => Mount(1, -1, 45), "SR" => Mount(1, -1, 45),      # SR: the SMA's reference antenna, the same mount
    "SPT" => Mount(1, 0, 0), "SP" => Mount(1, 0, 0),
)

"Greenwich mean sidereal time in hours (IAU 1982) for the UTC hour `t` of day `mjd` (UTC taken as UT1)."
function gmst(t, mjd::Integer)
    jd = mjd + 2400000.5 + t / 24
    d = jd - 2451545.0
    T = d / 36525
    θ = 280.46061837 + 360.98564736629 * d + 0.000387933 * T^2 - T^3 / 38710000
    return mod(θ, 360) / 15
end

"Geocentric latitude and longitude (radians) of a station at (x, y, z)."
latlon(xyz) = (atan(xyz[3], hypot(xyz[1], xyz[2])), atan(xyz[2], xyz[1]))

"""
    station_angles(xyz, ra, dec, t, mjd) -> (elevation, parallactic angle)

Elevation and parallactic angle (radians) of the source at `ra` (hours), `dec` (degrees) from a
station at geocentric `xyz` (metres) at UTC hour `t` of day `mjd`, ehtim's formulas.
"""
function station_angles(xyz, ra, dec, t, mjd::Integer)
    lat, lon = latlon(xyz)
    θG = gmst(t, mjd) * (π / 12)
    α = ra * (π / 12); δ = deg2rad(dec)
    θ = mod(θG - α, 2π)                                              # ehtim rotates the station by (gst − ra) and keeps the source in the x–z plane
    rot = SVector(cos(θ) * xyz[1] - sin(θ) * xyz[2], sin(θ) * xyz[1] + cos(θ) * xyz[2], xyz[3])
    src = SVector(cos(δ), 0.0, sin(δ))
    elev = π / 2 - acos(clamp(dot(rot, src) / norm(rot), -1, 1))
    H = mod(θG + lon - α, 2π)
    par = atan(sin(H) * cos(lat), sin(lat) * cos(δ) - cos(lat) * sin(δ) * cos(H))
    return elev, par
end

"The feed rotation angle (radians) of a station with mount `m` at elevation `elev` and parallactic angle `par`."
feed_angle(m::Mount, elev, par) = m.f_par * par + m.f_elev * elev + deg2rad(m.f_off_deg)

"""
    feed_angles(o::Observation, xyz, mounts) -> (φ1, φ2)

The feed rotation angles of both stations of every row of `o`: `xyz` maps station names to
geocentric positions (`antenna_positions`), `mounts` station names to `Mount`s (`EHT_MOUNTS`);
the input of `InstrumentModel(o; feedangles = feed_angles(o, xyz, mounts))`.
"""
function feed_angles(o::Observation{T}, xyz::AbstractDict, mounts::AbstractDict) where {T}
    φ1 = zeros(T, length(o)); φ2 = zeros(T, length(o))
    for r in 1:length(o)
        for (k, s) in enumerate((o.s1[r], o.s2[r]))
            name = o.stations[s]
            haskey(xyz, name) || throw(ArgumentError("no position for station $name"))
            haskey(mounts, name) || throw(ArgumentError("no mount parameters for station $name"))
            e, p = station_angles(xyz[name], o.ra, o.dec, o.time[r], o.mjd)
            (k == 1 ? φ1 : φ2)[r] = feed_angle(mounts[name], e, p)
        end
    end
    return φ1, φ2
end

export Mount, EHT_MOUNTS, gmst, latlon, station_angles, feed_angle, feed_angles
