import Pkg; Pkg.activate(@__DIR__)
using CUDA, KernelAbstractions, Krang, StaticArrays, ForwardDiff, FiniteDifferences, Printf
const KA = KernelAbstractions
CUDA.limit!(CUDA.LIMIT_STACK_SIZE, 32 * 1024)
# Krang calls unsafe_trunc(Int, τ/τhat) inside _θs; ForwardDiff has no such method -> dynamic dispatch in the kernel
Base.unsafe_trunc(::Type{I}, d::ForwardDiff.Dual) where {I<:Integer} = unsafe_trunc(I, ForwardDiff.value(d))
@kernel function dual_kernel!(out, met, @Const(αs), @Const(βs), θo, p, ::Val{N}) where {N}
    I = @index(Global)
    pix = Krang.SlowLightIntensityPixel(met, αs[I], βs[I], θo)
    T = typeof(Krang.total_mino_time(pix))
    rh = Krang.horizon(met); τf = Krang.total_mino_time(pix); dτ = τf / (N + 1)
    σ = exp(p[4]); A = exp(p[5]); acc = zero(T)
    for k in 1:N
        ts, rs, θs, ϕs, νr, νθ, ok = Krang.emission_coordinates(pix, dτ * k)
        sθ = sin(θs); x = rs * sθ * cos(ϕs); y = rs * sθ * sin(ϕs); z = rs * cos(θs)
        d2 = ((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2) / (2σ^2)
        acc += ifelse(ok & (rs > rh * (1 + 1e-3)) & (rs < 1e3), A * exp(-d2) * dτ, zero(T))
    end
    out[I] = acc
end
res = 64; N = 200
αs = [x for x in range(-10.0, 10.0, res), y in range(-10.0, 10.0, res)]; βs = [y for x in range(-10.0, 10.0, res), y in range(-10.0, 10.0, res)]
p = [6.0, 4.0, 0.5, log(1.5), log(2.0)]
a = 0.94; θo = 60 * pi / 180
D = ForwardDiff.Dual{Nothing,Float64,2}
dual(v, d1, d2) = D(v, ForwardDiff.Partials((d1, d2)))
metD = Krang.Kerr(dual(a, 1.0, 0.0)); θoD = dual(θo, 0.0, 1.0)
αD = CuArray(dual.(αs, 0.0, 0.0)); βD = CuArray(dual.(βs, 0.0, 0.0)); pD = CuArray(dual.(p, 0.0, 0.0)); outD = CUDA.zeros(D, res * res)
backend = KA.get_backend(outD)
t0 = time(); dual_kernel!(backend, 64)(outD, metD, αD, βD, θoD, pD, Val(N); ndrange = res * res); KA.synchronize(backend); t1 = time() - t0
t0 = time(); dual_kernel!(backend, 64)(outD, metD, αD, βD, θoD, pD, Val(N); ndrange = res * res); KA.synchronize(backend); t2 = time() - t0
# plain Float64 kernel timing for the ratio
out0 = CUDA.zeros(Float64, res * res)
dual_kernel!(backend, 64)(out0, Krang.Kerr(a), CuArray(αs), CuArray(βs), θo, CuArray(p), Val(N); ndrange = res * res); KA.synchronize(backend)
t0 = time(); dual_kernel!(backend, 64)(out0, Krang.Kerr(a), CuArray(αs), CuArray(βs), θo, CuArray(p), Val(N); ndrange = res * res); KA.synchronize(backend); t3 = time() - t0
h = Array(outD); Isum = sum(ForwardDiff.value.(h)); dIda = sum(ForwardDiff.partials.(h, 1)); dIdθ = sum(ForwardDiff.partials.(h, 2))
@printf("GPU Dual{2} kernel %d px x %d samples: %.3f s (first %.1f s); plain Float64 kernel %.3f s -> Dual overhead %.1fx\n", res^2, N, t2, t1, t3, t2 / t3)
@printf("GPU: I = %.10e  dI/da = %.10e  dI/dθo = %.10e\n", Isum, dIda, dIdθ)
function Icpu(a, θo)
    met = Krang.Kerr(a); s = 0.0
    for i in eachindex(αs)
        pix = Krang.SlowLightIntensityPixel(met, αs[i], βs[i], θo); rh = Krang.horizon(met); τf = Krang.total_mino_time(pix); dτ = τf / (N + 1)
        σ = exp(p[4]); A = exp(p[5]); acc = 0.0
        for k in 1:N
            ts, rs, θs, ϕs, νr, νθ, ok = Krang.emission_coordinates(pix, dτ * k)
            sθ = sin(θs); x = rs * sθ * cos(ϕs); y = rs * sθ * sin(ϕs); z = rs * cos(θs)
            d2 = ((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2) / (2σ^2)
            acc += ifelse(ok & (rs > rh * (1 + 1e-3)) & (rs < 1e3), A * exp(-d2) * dτ, 0.0)
        end
        s += acc
    end
    s
end
fdm = central_fdm(5, 1)
dIda_fd = fdm(x -> Icpu(x, θo), a); dIdθ_fd = fdm(x -> Icpu(a, x), θo)
@printf("CPU FD: I = %.10e  dI/da = %.10e (rel err %.1e)  dI/dθo = %.10e (rel err %.1e)\n", Icpu(a, θo), dIda_fd, abs(dIda - dIda_fd) / abs(dIda_fd), dIdθ_fd, abs(dIdθ - dIdθ_fd) / abs(dIdθ_fd))
println("DUAL_DONE")
