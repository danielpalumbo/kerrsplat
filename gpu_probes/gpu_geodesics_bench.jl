import Pkg; Pkg.activate(@__DIR__)
using CUDA, KernelAbstractions, Krang, StaticArrays, ForwardDiff, FiniteDifferences, Printf
const KA = KernelAbstractions
println("threads = ", Threads.nthreads()); flush(stdout)
CUDA.versioninfo(); flush(stdout)

CUDA.limit!(CUDA.LIMIT_STACK_SIZE, 32 * 1024); println("stack limit = ", CUDA.limit(CUDA.LIMIT_STACK_SIZE), " bytes")
sec(f) = (CUDA.synchronize(); t0 = time(); r = f(); CUDA.synchronize(); (time() - t0, r))

# ---------------------------------------------------------------- 1. screen (per-pixel constants): own kernel into a CONCRETE pixel type
# (Krang's KA ext allocates Matrix{SlowLightIntensityPixel{T}}, which leaves 16 type params abstract -> not isbits -> CuArray refuses)
@kernel function screen_kernel!(pixels, met, αmin, αmax, βmin, βmax, θo, res)
    I, J = @index(Global, NTuple)
    T = typeof(met.spin)
    α = αmin + (αmax - αmin) * (T(I) - 1) / (res - 1)
    β = βmin + (βmax - βmin) * (T(J) - 1) / (res - 1)
    @inbounds pixels[I, J] = Krang.SlowLightIntensityPixel(met, α, β, θo)
end
function make_screen(met, ρ, θo, res, A)
    T = typeof(met.spin)
    PixT = typeof(Krang.SlowLightIntensityPixel(met, T(0.1), T(0.2), θo))
    pixels = A{PixT}(undef, res, res)
    backend = KA.get_backend(pixels)
    screen_kernel!(backend, (16, 16))(pixels, met, -ρ, ρ, -ρ, ρ, θo, res; ndrange = (res, res))
    KA.synchronize(backend)
    return (pixels = pixels,)
end
function bench_screen(T, res; ρ = T(10), a = T(0.94), θo = T(60 * pi / 180))
    met = Krang.Kerr(a)
    PixT = typeof(Krang.SlowLightIntensityPixel(met, T(0.1), T(0.2), θo))
    @printf("[%s] sizeof(pixel) = %d bytes (isbits=%s)\n", T, sizeof(PixT), isbitstype(PixT))
    # single-thread CPU reference
    αs = range(-ρ, ρ, res); βs = range(-ρ, ρ, res)
    t0 = time(); cpu1 = [Krang.SlowLightIntensityPixel(met, T(α), T(β), θo) for α in αs, β in βs]; tc1 = time() - t0
    tck, cpu = sec(() -> make_screen(met, ρ, θo, res, Array))      # KA CPU backend, multithreaded
    tg1, _ = sec(() -> make_screen(met, ρ, θo, res, CuArray))       # compile
    tg, gpu = sec(() -> make_screen(met, ρ, θo, res, CuArray))
    gp = Array(gpu.pixels); cp = cpu1
    dτ = maximum(abs.(Krang.total_mino_time.(gp) .- Krang.total_mino_time.(cp)) ./ (abs.(Krang.total_mino_time.(cp)) .+ eps(T)))
    dη = maximum(abs.(Krang.η.(gp) .- Krang.η.(cp)) ./ (abs.(Krang.η.(cp)) .+ eps(T)))
    @printf("[%s] screen %d^2: CPU 1 thread %.3f s | CPU %d threads %.3f s | GPU %.4f s (first %.1f s) | max rel diff GPU vs CPU: τ_total %.1e, η %.1e\n",
            T, res, tc1, Threads.nthreads(), tck, tg, tg1, dτ, dη); flush(stdout)
    return (pixels = cpu1,), gpu
end

# ---------------------------------------------------------------- 2. per-(pixel,sample) coordinates stored on GPU
# (own kernel: Krang.generate_rays(...; A=CuArray) does `first(pixels)` on the device array -> scalar indexing error)
@kernel function rays_kernel!(rays, @Const(pixels), N)
    I, J, K = @index(Global, NTuple)
    pix = pixels[I, J]
    T = typeof(Krang.total_mino_time(pix))
    actual = unsafe_trunc(Int, sum(Krang._isreal2.(Krang.roots(pix)))) == 4 ? N + 1 : N
    Δτ = Krang.total_mino_time(pix) / actual
    ts, rs, θs, ϕs, νr, νθ, _ = Krang.emission_coordinates(pix, Δτ * K)
    @inbounds rays[I, J, K] = Krang.Intersection(T(ts), T(rs), T(θs), T(ϕs % T(2π)), νr, νθ)
end
function make_rays(pixels, N)
    T = typeof(Krang.total_mino_time(pixels isa CuArray ? Array(pixels[1:1])[1] : first(pixels)))
    A = pixels isa CuArray ? CuArray : Array
    rays = A{Krang.Intersection{T}}(undef, size(pixels)..., N)
    backend = KA.get_backend(rays)
    rays_kernel!(backend, (8, 8, 4))(rays, pixels, N; ndrange = size(rays))
    KA.synchronize(backend)
    return rays
end
function bench_rays(cpu, gpu, N)
    T = typeof(Krang.metric(first(cpu.pixels)).spin)
    tg1, _ = sec(() -> make_rays(gpu.pixels, N))
    tg, rays = sec(() -> make_rays(gpu.pixels, N))
    npix = length(gpu.pixels)
    @printf("[%s] stored rays %d px x %d samples on GPU: %.4f s (first %.1f s) -> %.1f ns/sample; %.2f GB stored\n",
            T, npix, N, tg, tg1, tg / (npix * N) * 1e9, sizeof(rays) / 1e9); flush(stdout)
    rh = Array(rays); pix = cpu.pixels; maxd = zero(T)
    for (i, j) in ((1, 1), (size(pix, 1) ÷ 2, size(pix, 2) ÷ 2), (size(pix, 1) ÷ 3, 2 * size(pix, 2) ÷ 3), (size(pix, 1), size(pix, 2)), (size(pix, 1) ÷ 2 + 3, size(pix, 2) ÷ 2 - 2))
        p = pix[i, j]
        actual = sum(Krang._isreal2.(Krang.roots(p))) == 4 ? N + 1 : N
        Δτ = Krang.total_mino_time(p) / actual
        for k in (1, N ÷ 2, N)
            ts, rs, θs, ϕs, νr, νθ, _ = Krang.emission_coordinates(p, Δτ * k)
            g = rh[i, j, k]
            isfinite(rs) && isfinite(g.rs) && (maxd = max(maxd, abs(rs - g.rs) / (abs(rs) + eps(T)), abs(θs - g.θs), abs(ts - g.ts) / (abs(ts) + 1)))
        end
    end
    @printf("       spot-check GPU vs CPU coordinates: max rel diff %.1e\n", maxd); flush(stdout)
    return rays
end

# ---------------------------------------------------------------- 3. fused: march N samples per pixel inside one kernel, no sample storage
@kernel function fused_kernel!(out, @Const(pixels), p, ::Val{N}) where {N}
    I = @index(Global)
    pix = pixels[I]
    T = typeof(Krang.total_mino_time(pix))
    met = Krang.metric(pix)
    rh = Krang.horizon(met)
    τf = Krang.total_mino_time(pix)
    dτ = τf / (N + 1)
    σ = exp(p[4]); A = exp(p[5])
    acc = zero(T)
    for k in 1:N
        ts, rs, θs, ϕs, νr, νθ, ok = Krang.emission_coordinates(pix, dτ * k)
        sθ = sin(θs); x = rs * sθ * cos(ϕs); y = rs * sθ * sin(ϕs); z = rs * cos(θs)
        d2 = ((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2) / (2σ^2)
        acc += ifelse(ok & (rs > rh * (1 + T(1e-3))) & (rs < T(1e3)), A * exp(-d2) * dτ, zero(T))
    end
    out[I] = acc
end
function fused!(out, pixels, p, ::Val{N}) where {N}
    backend = KA.get_backend(out)
    fused_kernel!(backend, 128)(out, pixels, p, Val(N); ndrange = length(out))
    KA.synchronize(backend)
    return out
end
function bench_fused(cpu, gpu, ::Val{N}) where {N}
    T = typeof(Krang.metric(first(cpu.pixels)).spin)
    p = T[6.0, 4.0, 0.5, log(1.5), log(2.0)]
    pg = CuArray(p); out = CUDA.zeros(T, length(gpu.pixels))
    t1, _ = sec(() -> fused!(out, gpu.pixels, pg, Val(N)))
    tg, _ = sec(() -> fused!(out, gpu.pixels, pg, Val(N)))
    npix = length(gpu.pixels)
    @printf("[%s] FUSED march %d px x %d samples on GPU (no storage): %.4f s (first %.1f s) -> %.1f ns/sample\n",
            T, npix, N, tg, t1, tg / (npix * N) * 1e9); flush(stdout)
    # CPU multithreaded reference on a subset of pixels
    sub = vec(cpu.pixels)[1:min(end, 4096)]
    outc = zeros(T, length(sub))
    f = () -> Threads.@threads for i in eachindex(sub)
        pix = sub[i]; met = Krang.metric(pix); rh = Krang.horizon(met); τf = Krang.total_mino_time(pix); dτ = τf / (N + 1)
        σ = exp(p[4]); A = exp(p[5]); acc = zero(T)
        for k in 1:N
            ts, rs, θs, ϕs, νr, νθ, ok = Krang.emission_coordinates(pix, dτ * k)
            sθ = sin(θs); x = rs * sθ * cos(ϕs); y = rs * sθ * sin(ϕs); z = rs * cos(θs)
            d2 = ((x - p[1])^2 + (y - p[2])^2 + (z - p[3])^2) / (2σ^2)
            acc += ifelse(ok & (rs > rh * (1 + T(1e-3))) & (rs < T(1e3)), A * exp(-d2) * dτ, zero(T))
        end
        outc[i] = acc
    end
    f(); t0 = time(); f(); tc = time() - t0
    og = Array(out)[1:length(sub)]
    @printf("       CPU %d threads, %d px x %d samples: %.3f s -> %.1f ns/sample; GPU speedup vs CPU(%d thr) = %.0fx; max rel diff GPU vs CPU image = %.1e\n",
            Threads.nthreads(), length(sub), N, tc, tc / (length(sub) * N) * 1e9, Threads.nthreads(), (tc / (length(sub) * N)) / (tg / (npix * N)),
            maximum(abs.(og .- outc) ./ (abs.(outc) .+ 1e-30))); flush(stdout)
    return out
end

# ---------------------------------------------------------------- run
println("\n===== Float64 ====="); flush(stdout)
cpu64, gpu64 = bench_screen(Float64, 256)
rays64 = bench_rays(cpu64, gpu64, 200); rays64 = nothing; GC.gc(); CUDA.reclaim()
out64 = bench_fused(cpu64, gpu64, Val(1000))

println("\n===== Float32 ====="); flush(stdout)
cpu32, gpu32 = bench_screen(Float32, 256)
rays32 = bench_rays(cpu32, gpu32, 200); rays32 = nothing; GC.gc(); CUDA.reclaim()
out32 = bench_fused(cpu32, gpu32, Val(1000))
# Float32 vs Float64 image error
o64 = Array(out64); o32 = Float64.(Array(out32))
rel = abs.(o32 .- o64) ./ (abs.(o64) .+ 1e-12 * maximum(abs.(o64)))
@printf("Float32 vs Float64 fused image (256^2 x 1000): median rel err %.1e, 99th pct %.1e, max %.1e, frac > 1e-3: %.4f\n",
        sort(rel)[end ÷ 2], sort(rel)[round(Int, 0.99 * end)], maximum(rel), count(rel .> 1e-3) / length(rel)); flush(stdout)

println("\n===== spacetime derivatives (a, θo) via ForwardDiff Duals inside the GPU kernel ====="); flush(stdout)
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
try
    res = 64; N = 200
    αs = [x for x in range(-10.0, 10.0, res), y in range(-10.0, 10.0, res)]; βs = [y for x in range(-10.0, 10.0, res), y in range(-10.0, 10.0, res)]
    p = [6.0, 4.0, 0.5, log(1.5), log(2.0)]
    D = ForwardDiff.Dual{Nothing,Float64,2}
    a = 0.94; θo = 60 * pi / 180
    metD = Krang.Kerr(D(a, 1.0, 0.0)); θoD = D(θo, 0.0, 1.0)
    αD = CuArray(D.(αs)); βD = CuArray(D.(βs)); pD = CuArray(D.(p)); outD = CUDA.zeros(D, res * res)
    backend = KA.get_backend(outD)
    t0 = time(); dual_kernel!(backend, 128)(outD, metD, αD, βD, θoD, pD, Val(N); ndrange = res * res); KA.synchronize(backend); t1 = time() - t0
    t0 = time(); dual_kernel!(backend, 128)(outD, metD, αD, βD, θoD, pD, Val(N); ndrange = res * res); KA.synchronize(backend); t2 = time() - t0
    h = Array(outD); Isum = sum(ForwardDiff.value.(h)); dIda = sum(ForwardDiff.partials.(h, 1)); dIdθ = sum(ForwardDiff.partials.(h, 2))
    @printf("GPU Dual kernel %d px x %d samples: %.3f s (first %.1f s); I = %.8e  dI/da = %.8e  dI/dθo = %.8e\n", res^2, N, t2, t1, Isum, dIda, dIdθ); flush(stdout)
    # finite differences on CPU (Float64) for the same sum
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
    @printf("CPU FD:  I = %.8e  dI/da = %.8e (rel err %.1e)  dI/dθo = %.8e (rel err %.1e)\n", Icpu(a, θo), dIda_fd, abs(dIda - dIda_fd) / abs(dIda_fd), dIdθ_fd, abs(dIdθ - dIdθ_fd) / abs(dIdθ_fd)); flush(stdout)
catch err
    println("DUAL KERNEL FAILED: "); showerror(stdout, err); println()
end
println("GPUGEO_DONE")
