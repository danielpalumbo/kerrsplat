import Pkg; Pkg.activate(@__DIR__)
using JacobiElliptic, Printf
# Jacobi addition theorems: advance (sn,cn,dn) by a fixed step Δ using precomputed (sΔ,cΔ,dΔ)
@inline function step(s, c, d, sΔ, cΔ, dΔ, m)
    den = 1 - m * s * s * sΔ * sΔ
    (s * cΔ * dΔ + sΔ * c * d) / den, (c * cΔ - s * d * sΔ * dΔ) / den, (d * dΔ - m * s * c * sΔ * cΔ) / den
end
for m in (0.1, 0.5, 0.9, 0.99, 0.9999), N in (1000, 4000), anchor in (typemax(Int), 64)
    K = JacobiElliptic.K(m); u0 = 0.37; Δ = 4K / N          # one full period over the ray
    sΔ = JacobiElliptic.sn(Δ, m); cΔ = JacobiElliptic.cn(Δ, m); dΔ = JacobiElliptic.dn(Δ, m)
    s = JacobiElliptic.sn(u0, m); c = JacobiElliptic.cn(u0, m); d = JacobiElliptic.dn(u0, m)
    maxerr = 0.0
    for i in 1:N
        if i % anchor == 0
            u = u0 + i * Δ; s = JacobiElliptic.sn(u, m); c = JacobiElliptic.cn(u, m); d = JacobiElliptic.dn(u, m)
        else
            s, c, d = step(s, c, d, sΔ, cΔ, dΔ, m)
        end
        maxerr = max(maxerr, abs(s - JacobiElliptic.sn(u0 + i * Δ, m)))
    end
    @printf("m=%.4f N=%4d anchor=%-6s  max |sn err| = %.1e\n", m, N, anchor == typemax(Int) ? "none" : string(anchor), maxerr)
end
# per-step cost (CPU, single thread)
function runrec(N, s, c, d, sΔ, cΔ, dΔ, m)
    acc = 0.0
    for i in 1:N
        s, c, d = step(s, c, d, sΔ, cΔ, dΔ, m); acc += s
    end
    acc
end
function rundirect(N, Δ, m)
    acc = 0.0
    for i in 1:N; acc += JacobiElliptic.sn(0.37 + i * Δ, m); end
    acc
end
let m = 0.9, Δ = 1e-3, N = 10^7
    sΔ = JacobiElliptic.sn(Δ, m); cΔ = JacobiElliptic.cn(Δ, m); dΔ = JacobiElliptic.dn(Δ, m)
    runrec(10, 0.1, sqrt(1 - 0.01), sqrt(1 - m * 0.01), sΔ, cΔ, dΔ, m); t = @elapsed runrec(N, 0.1, sqrt(1 - 0.01), sqrt(1 - m * 0.01), sΔ, cΔ, dΔ, m)
    @printf("recurrence step: %.2f ns/step (CPU, 1 thread)\n", t / N * 1e9)
    rundirect(10, Δ, m); t2 = @elapsed rundirect(10^5, Δ, m)
    @printf("direct JacobiElliptic.sn: %.1f ns/eval (CPU, 1 thread)\n", t2 / 1e5 * 1e9)
end
println("REC_DONE")
