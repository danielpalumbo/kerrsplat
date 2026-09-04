# Gate for the analytic constant-coefficient transfer step: the operator exp(−KΔ) and the emission
# integral ∫₀^Δ exp(−Ku) du j against BigFloat references (Taylor series with scaling and
# doubling), across the regimes the closed forms switch between (tiny and huge optical depths and
# rotation angles, near-degenerate eigenvalues, the nilpotent case, pure rotation, pure absorption,
# unpolarized), plus physical checks (Kirchhoff equilibrium, Faraday rotation as a rotation about
# ρ⃗, the semigroup property), and the same step inside a kernel on the backend.

using LinearAlgebra
using StaticArrays
using ForwardDiff
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Transfer: transfer_step, RadiativeState, advance, rotate_to_screen

function bigK(α, ρ)
    αI, αQ, αU, αV = big.(α)
    ρQ, ρU, ρV = big.(ρ)
    return [αI αQ αU αV; αQ αI ρV -ρU; αU -ρV αI ρQ; αV ρU -ρQ αI]
end

"exp(−KΔ) and ∫₀^Δ exp(−Ku) du in BigFloat: Taylor series after scaling Δ by 2^k, then doubling."
function big_step(K::Matrix{BigFloat}, Δ::BigFloat)
    k = max(0, ceil(Int, log2(max(norm(K, Inf) * Δ, big"1e-30"))) + 4)
    d = Δ / big(2)^k
    n = size(K, 1)
    O = Matrix{BigFloat}(I, n, n); term = Matrix{BigFloat}(I, n, n)
    Eop = d * Matrix{BigFloat}(I, n, n); eterm = d * Matrix{BigFloat}(I, n, n)
    for m in 1:60
        term = term * (-K * d) / m
        O += term
        eterm = eterm * (-K * d) / (m + 1)
        Eop += eterm
    end
    for _ in 1:k
        Eop = Eop + O * Eop       # ∫₀^{2d} = ∫₀^d + O(d) ∫₀^d
        O = O * O
    end
    return O, Eop
end

relnorm(x, y) = norm(x - y) / max(norm(y), floatmin(Float64))

@kernel function step_kernel!(out, @Const(jα), @Const(ρs), @Const(Δs))
    i = @index(Global)
    @inbounds begin
        j = SVector(jα[i, 1], jα[i, 2], jα[i, 3], jα[i, 4])
        α = SVector(jα[i, 5], jα[i, 6], jα[i, 7], jα[i, 8])
        ρ = SVector(ρs[i, 1], ρs[i, 2], ρs[i, 3])
        O, E = transfer_step(j, α, ρ, Δs[i])
        for k in 1:16
            out[i, k] = O[k]
        end
        for k in 1:4
            out[i, 16 + k] = E[k]
        end
    end
end

function step_cases(rng)
    cases = Tuple{SVector{4,Float64},SVector{4,Float64},SVector{3,Float64},Float64}[]
    # random physical sets: |α⃗| < αI, ρ of any size, Δ spanning tiny to huge depths
    for _ in 1:400
        j = SVector{4}(rand(rng, 4) .* [1.0, 0.5, 0.5, 0.3])
        αI = 10.0^(rand(rng) * 8 - 4)
        dir = normalize(randn(rng, 3)); fp = 0.99 * rand(rng)
        α = SVector(αI, (αI * fp * dir)...)
        ρ = SVector{3}(10.0^(rand(rng) * 10 - 5) * normalize(randn(rng, 3)))
        Δ = 10.0^(rand(rng) * 6 - 5)
        push!(cases, (j, α, ρ, Δ))
    end
    j = SVector(1.0, 0.3, -0.2, 0.1)
    push!(cases, (j, SVector(0.0, 0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), 0.7))                    # no absorption, no rotation
    push!(cases, (j, SVector(2.0, 0.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0), 0.7))                    # unpolarized absorption
    push!(cases, (j, SVector(0.0, 0.0, 0.0, 0.0), SVector(3.0, -1.0, 2.0), 0.7))                   # pure Faraday rotation
    push!(cases, (j, SVector(2.0, 1.2, -0.6, 0.9), SVector(0.0, 0.0, 0.0), 0.7))                   # pure polarized absorption
    push!(cases, (j, SVector(2.0, 1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), 0.7))                    # nilpotent: |α⃗| = |ρ⃗|, α⃗ ⟂ ρ⃗
    push!(cases, (j, SVector(2.0, 1.0, 0.0, 0.0), SVector(0.0, 1.0 + 1e-9, 0.0), 0.7))             # nearly nilpotent
    push!(cases, (j, SVector(2.0, 1.0, 0.0, 0.0), SVector(1e-6, 1.0, 0.0), 0.7))                   # tiny α⃗·ρ⃗
    push!(cases, (j, SVector(50.0, 40.0, 10.0, 5.0), SVector(0.5, 0.2, 0.1), 30.0))                # optical depth 1500, τ_P 1200
    push!(cases, (j, SVector(1.0, 0.5, 0.1, 0.1), SVector(1e4, 2e3, 5e3), 1.0))                    # 10⁴ radians of rotation
    push!(cases, (j, SVector(1e-9, 5e-10, 1e-10, 1e-10), SVector(1e-8, 2e-9, 5e-9), 1e-3))         # everything tiny
    push!(cases, (j, SVector(1.0, 0.99, 0.0, 0.0), SVector(0.0, 0.0, 0.0), 5.0))                   # α_P at the cap
    return cases
end

function test_transfer_step(backend; tol = 1e-13, label = "")
    Geodesics.prepare_backend!(backend)      # the 4×4 products need more than CUDA's default 1 KB stack
    @testset "transfer step vs BigFloat ($label)" begin
        rng = Random.MersenneTwister(7)
        cases = step_cases(rng)
        setprecision(BigFloat, 256) do
            # A rotation angle φ = |ρ|Δ evaluated in Float64 carries an absolute error ~ eps φ, and the
            # bounded-matrix assembly adds a few eps per radian, so the attainable accuracy of the step
            # is ~eps (1 + φ); the criterion scales accordingly (measured: ≤ 1.4e-15 per radian for E,
            # 3e-16 for O; the bound below allows 2e-15 per radian).
            worstO = 0.0; worstE = 0.0
            for (j, α, ρ, Δ) in cases
                O, E = transfer_step(j, α, ρ, Δ)
                Ob, Eop = big_step(bigK(α, ρ), big(Δ))
                Eb = Eop * big.(j)
                eO = relnorm(Matrix(O), Float64.(Ob))
                eE = relnorm(Vector(E), Float64.(Eb))
                # exp(−KΔ) at optical depth 1500 is 1e-650: compare it in absolute terms there
                if norm(Float64.(Ob)) < 1e-300
                    eO = norm(Matrix(O))
                end
                scale = 1 + 0.02 * norm(ρ) * Δ
                worstO = max(worstO, eO / scale); worstE = max(worstE, eE / scale)
                if !(eO < tol * scale && eE < tol * scale)
                    αv = SVector(α[2], α[3], α[4])
                    @info "transfer_step case off" a = α[1] * Δ αP = norm(αv) * Δ ρ = norm(ρ) * Δ αρ = dot(αv, ρ) * Δ^2 eO eE
                end
            end
            @test worstO < tol
            @test worstE < tol
            @info "transfer_step vs BigFloat over $(length(cases)) cases: worst angle-scaled relative errors" O = worstO E = worstE
        end

        # Kirchhoff equilibrium: thermal j = α B_ν drives every Stokes vector to (B_ν, 0, 0, 0)
        c = thermal_synchrotron(1e7, 20.0, 50.0, 230e9, 1.1)
        j, α, ρ = rotate_to_screen(c, 0.4)
        Bν = planck(230e9, 20.0)
        @test j ≈ Bν * α
        O, E = transfer_step(j, α, ρ, 400 / c.αI)     # the slowest decay is e^{−(αI − Λ₁)Δ}; Λ₁ < α_P ≈ 0.8 αI here
        S = O * SVector(3Bν, Bν, -Bν, 0.5Bν) + E
        @test S ≈ SVector(Bν, 0, 0, 0) rtol = 1e-12 atol = 1e-12 * Bν

        # pure Faraday rotation is a rotation of (Q, U, V) about ρ⃗ by |ρ|Δ
        ρ = SVector(0.3, -1.1, 0.7); Δ = 2.5
        O, E = transfer_step(zero(SVector{4,Float64}), zero(SVector{4,Float64}), ρ, Δ)
        S0 = SVector(1.0, 0.4, -0.2, 0.3)
        S = O * S0
        n = ρ / norm(ρ); φ = norm(ρ) * Δ
        v = SVector(S0[2], S0[3], S0[4])
        vrot = v * cos(φ) + cross(n, v) * sin(φ) + n * dot(n, v) * (1 - cos(φ))
        @test S[1] == S0[1] && E == zero(E)
        @test SVector(S[2], S[3], S[4]) ≈ vrot rtol = 1e-13

        # semigroup: O(Δ₁+Δ₂) = O(Δ₂)O(Δ₁), E(Δ₁+Δ₂) = E(Δ₂) + O(Δ₂)E(Δ₁)
        j = SVector(1.0, 0.3, -0.2, 0.1); α = SVector(0.8, 0.3, -0.2, 0.4); ρ = SVector(2.0, -1.0, 0.5)
        O1, E1 = transfer_step(j, α, ρ, 0.4); O2, E2 = transfer_step(j, α, ρ, 1.1); O12, E12 = transfer_step(j, α, ρ, 1.5)
        @test O12 ≈ O2 * O1 rtol = 1e-13
        @test E12 ≈ E2 + O2 * E1 rtol = 1e-13
        # and the front-to-back accumulator composes the same way: observer-side interval first
        st = advance(advance(zero(RadiativeState{Float64}), O2, E2), O1, E1)      # O2 nearer the observer
        @test st.P ≈ O2 * O1 && st.S ≈ E2 + O2 * E1

        # unpolarized limit against the scalar solution
        O, E = transfer_step(SVector(2.0, 0.0, 0.0, 0.0), SVector(0.7, 0.0, 0.0, 0.0), zero(SVector{3,Float64}), 1.3)
        @test O[1, 1] ≈ exp(-0.7 * 1.3) && E[1] ≈ 2.0 / 0.7 * (1 - exp(-0.7 * 1.3)) && E[2] == 0

        # differentiable: ForwardDiff through the step vs finite differences
        f(x) = sum(transfer_step(SVector(x[1], 0.3, -0.2, 0.1), SVector(0.8, x[2], -0.2, 0.4), SVector(x[3], -1.0, 0.5), x[4])[2])
        x0 = [1.0, 0.3, 2.0, 0.9]
        g = ForwardDiff.gradient(f, x0)
        for k in 1:4
            h = 1e-6 * max(1, abs(x0[k])); xp = copy(x0); xm = copy(x0); xp[k] += h; xm[k] -= h
            @test g[k] ≈ (f(xp) - f(xm)) / 2h rtol = 1e-6
        end

        # the same step inside a kernel on the backend
        n = length(cases)
        jα = zeros(n, 8); ρs = zeros(n, 3); Δs = zeros(n)
        for (i, (j, α, ρ, Δ)) in enumerate(cases)
            jα[i, 1:4] = j; jα[i, 5:8] = α; ρs[i, :] = ρ; Δs[i] = Δ
        end
        out = KernelAbstractions.zeros(backend, Float64, n, 20)
        step_kernel!(backend, 64)(out, adapt_to(backend, jα), adapt_to(backend, ρs), adapt_to(backend, Δs); ndrange = n)
        KernelAbstractions.synchronize(backend)
        Oa = Array(out)
        worstk = 0.0
        for (i, (j, α, ρ, Δ)) in enumerate(cases)
            O, E = transfer_step(j, α, ρ, Δ)
            scale = 1 + 0.02 * norm(ρ) * Δ        # the backends' sin/cos and contraction differ by ~eps per radian
            worstk = max(worstk, relnorm(Oa[i, 1:16], vec(Matrix(O))) / scale, relnorm(Oa[i, 17:20], Vector(E)) / scale)
        end
        @test worstk < 1e-12
        @info "transfer_step in a kernel on $label vs host: worst relative error $worstk"
    end
end
