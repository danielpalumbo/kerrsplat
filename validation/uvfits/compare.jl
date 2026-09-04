# Compare Fit.read_uvfits with ehtim's parse (dump_ehtim.py) of the same uvfits file.
using Test, KerrSplat, KerrSplat.Fit, StaticArrays, DelimitedFiles
include(joinpath(@__DIR__, "..", "..", "test", "test_uvfits.jl"))   # _ehtim_csv
obs = read_uvfits(ARGS[1]); ref = _ehtim_csv(ARGS[2])
n = length(obs)
println("rows $n (ehtim $(length(ref.time))); mjd $(obs.mjd); freq $(obs.freq) Hz; source $(obs.source); stations $(obs.stations)")
n == length(ref.time) || error("row count differs")
println("stations match: ", all(obs.stations[obs.s1] .== ref.t1) && all(obs.stations[obs.s2] .== ref.t2))
println("time max |Δ| h: ", maximum(abs.(obs.time .- ref.time)), "; tint max |Δ|: ", maximum(abs.(obs.tint .- ref.tint)))
println("u, v max relative |Δ|: ", maximum(abs.(obs.u .- ref.u) ./ (abs.(ref.u) .+ 1)), " ", maximum(abs.(obs.v .- ref.v) ./ (abs.(ref.v) .+ 1)))
for (k, name) in enumerate(("I", "Q", "U", "V"))
    present = [isfinite(obs.σ[r][k]) for r in 1:n]
    any(present) || (println("$name: absent in all rows (ehtim σ = $(maximum(getindex.(ref.σ, k))))"); continue)
    println("$name: $(count(present)) rows; max |Δvis| ", maximum(abs(obs.vis[r][k] - ref.vis[r][k]) for r in 1:n if present[r]),
            " Jy; max |Δσ| ", maximum(abs(obs.σ[r][k] - ref.σ[r][k]) for r in 1:n if present[r]), " Jy")
end
