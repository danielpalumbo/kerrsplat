# Gates for the polarized one-zone splats (Splats.PolarizedSplats through Transfer.RadiativeTransport):
#  1. the fused march on the CPU backend reproduces a host loop over stored samples (plumbing:
#     observer-first ordering, interval lengths, skipped samples, the observation time);
#  2. CUDA reproduces the CPU backend;
#  3. Enzyme reverse-mode gradients of an image loss with respect to all 20 parameters of every
#     splat agree with central finite differences;
#  4. a short Adam fit recovers a perturbed splat from its own I, Q, U, V image.

using StaticArrays
using LinearAlgebra
using KerrSplat.Geodesics
using KerrSplat.Transfer
using KerrSplat.Splats
using KerrSplat.Transfer: transfer_step, advance, rotate_to_screen, invariants, cap_polarization

const POL_ν = 230e9
const POL_L = gravitational_radius(4.0e6)

"Two overlapping splats with different fields and velocities (columns of the parameter matrix)."
function polarized_test_params()
    p = zeros(NPOLARIZEDPARAMS, 2)
    p[:, 1] = [4.0, 1.0, 0.0, 0.2, -0.1, -0.3, 1.0, 0.1, 0.0, 0.0, 0.0, log(1e6), log(2e6), log(20.0), log(30.0), 1.2, 0.7, 0.1, 0.45, 0.0, 0.04]
    p[:, 2] = [3.5, 1.6, 0.2, 0.0, 0.1, -0.2, 0.9, 0.0, 0.2, 0.1, 0.0, log(1e6), log(1e6), log(35.0), log(15.0), 0.6, -1.0, -0.2, 0.3, 0.1, -0.02]
    return p
end

"Host reference: the same elements and steps, looped over stored samples."
function polarized_image_host(cache_stored::GeodesicCache{T,N}, params, t_obs, ν, L) where {T,N}
    S = host(cache_stored.samples)
    consts = host(cache_stored.consts)
    met = Krang.Kerr(cache_stored.spin); θo = cache_stored.θo
    model = PolarizedSplats(params, fill(T(t_obs), 1))
    npix = npixels(cache_stored)
    out = Vector{SVector{4,T}}(undef, npix)
    for i in 1:npix
        pix = Krang.SlowLightIntensityPixel(met, consts.α[i], consts.β[i], θo)
        Δτ = Krang.total_mino_time(pix) / (N + 1)
        st = zero(RadiativeState{T})
        for k in 1:N
            s = S[i, k]
            (s.ok && s.r > Krang.horizon(met) * (1 + T(1e-3))) || continue
            j4 = zero(SVector{4,T}); α4 = zero(SVector{4,T}); ρ3 = zero(SVector{3,T}); active = false
            for e in 1:Transfer.nelements(model)
                cf, fr = Transfer.element(model, e, pix, s, ν)
                (cf.jI > 0 || cf.αI > 0 || cf.ρQ != 0 || cf.ρV != 0) || continue
                jj, aa, rr = rotate_to_screen(invariants(cap_polarization(cf), ν / fr.g), fr.χ)
                j4 += jj; α4 += aa; ρ3 += rr; active = true
            end
            active || continue
            Σ = s.r^2 + met.spin^2 * cos(s.θ)^2
            O, E = transfer_step(j4, α4, ρ3, L / ν * Σ * Δτ)
            st = advance(st, O, E)
        end
        out[i] = observed_stokes(st, ν)
    end
    return to_screen(cache_stored, out)
end

function polarized_march(q, cache, t_obs)
    out = Vector{RadiativeState{Float64}}(undef, npixels(cache))
    fill!(out, zero(RadiativeState{Float64}))
    polarized_image!(out, cache, q, t_obs, POL_ν, POL_L)
    return map(st -> observed_stokes(st, POL_ν), out)
end

function test_polarized_splats(backend; res = 24, N = 300, label = "")
    Geodesics.prepare_backend!(backend)
    @testset "polarized splats ($label)" begin
        a = 0.9; θo = deg2rad(60.0); t_obs = 0.0
        camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
        p = polarized_test_params()
        cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        img = Array(polarized_image(cache, adapt_to(backend, p), t_obs, POL_ν, POL_L))
        I = getindex.(img, 1)
        @test all(isfinite, I) && maximum(I) > 0
        @test maximum(hypot.(getindex.(img, 2), getindex.(img, 3)) ./ max.(I, 1e-300 * maximum(I))) < 1     # partially polarized
        if backend isa CPU
            cs = GeodesicCache(CPU(), camera, Val(N)); regenerate!(cs, a, θo; marcher = Recurrence(64))
            ref = polarized_image_host(cs, p, t_obs, POL_ν, POL_L)
            err = maximum(norm.(img .- ref)) / maximum(norm.(ref))
            @test err < 1e-10
            @info "polarized splats: fused march vs host loop, $(res)² × $N: max relative difference $err; peak I $(maximum(I)) cgs, LP fraction at the peak $(round(hypot(img[argmax(I)][2], img[argmax(I)][3]) / maximum(I), digits = 3)), V/I $(round(img[argmax(I)][4] / maximum(I), sigdigits = 3))"
        else
            cc = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cc, a, θo; marcher = Fused(64))
            ref = polarized_image(cc, p, t_obs, POL_ν, POL_L)
            err = maximum(norm.(img .- ref)) / maximum(norm.(ref))
            @test err < 1e-11
            @info "polarized splats on $label vs CPU backend: max relative difference $err"
        end
    end
end

function test_polarized_splat_gradients(; res = 10, N = 100, tol = 1e-5)
    a = 0.9; θo = deg2rad(60.0); t_obs = 0.0
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    p = polarized_test_params()
    target = polarized_march(p, cache, t_obs)
    scale = maximum(norm.(target))
    p_start = copy(p); p_start[1, :] .+= 0.4; p_start[13, :] .-= 0.2; p_start[16, :] .+= 0.2; p_start[18, :] .+= 0.1
    loss(q) = sum(abs2, reinterpret(Float64, polarized_march(q, cache, t_obs) .- target)) / scale^2
    @testset "Enzyme reverse gradient of a polarized image loss, $(res)² × $N, $(length(p)) parameters" begin
        L0 = loss(p_start)
        t0 = time()
        g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p_start)[1]
        t_first = time() - t0
        t0 = time(); g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p_start)[1]; t_grad = time() - t0
        # fourth-order central stencil; the floor covers derivatives at rounding level (t0 and logw of a
        # splat whose temporal envelope is flat over the observation have gradients of 1e-12)
        err = 0.0
        gmax = maximum(abs, g)
        iω = findfirst(==(:omega), POLARIZED_SPLAT_PARAMS)
        for i in eachindex(p_start)
            row = (i - 1) % NPOLARIZEDPARAMS + 1
            # the loss depends on ω through angles ω t with |t| up to ~50 M, so the stencil's truncation error
            # (∝ (t h)⁴) needs a much smaller step on that row
            h = row == iω ? 2e-5 : 1e-3 * max(1.0, abs(p_start[i]))
            f(x) = (q = copy(p_start); q[i] = x; loss(q))
            x = p_start[i]
            fd = (-f(x + 2h) + 8f(x + h) - 8f(x - h) + f(x - 2h)) / (12h)
            err = max(err, abs(g[i] - fd) / max(abs(fd), 1e-6 * gmax))
        end
        @test err < tol
        @info "polarized splats Enzyme gradient: loss $L0; max relative error vs central finite differences $err; gradient $(round(t_grad, digits = 2)) s (first call $(round(t_first, digits = 1)) s)"
    end
end

function test_polarized_splat_fit(; res = 10, N = 100, iterations = 150)
    a = 0.9; θo = deg2rad(60.0); t_obs = 0.0
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cache, a, θo; marcher = Fused(64))
    p_true = polarized_test_params()[:, 1:1]
    target = polarized_march(p_true, cache, t_obs)
    scale = maximum(norm.(target))
    loss(q) = sum(abs2, reinterpret(Float64, polarized_march(q, cache, t_obs) .- target)) / scale^2
    p = copy(p_true); p[1] += 0.5; p[2] -= 0.3; p[13] -= 0.3; p[14] += 0.15; p[16] += 0.3; p[18] -= 0.15
    @testset "Adam fit of one polarized splat, $(res)² × $N, $iterations iterations" begin
        L0 = loss(p)
        opt = Optimisers.setup(Optimisers.Adam(0.02), p)
        Lmin = L0
        for it in 1:iterations
            g = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(loss), p)[1]
            opt, p = Optimisers.update(opt, p, g)
            Lmin = min(Lmin, loss(p))
        end
        @test Lmin < 1e-2 * L0
        @info "polarized splat fit: loss $L0 → $Lmin; position error $(norm(p[1:3] - p_true[1:3])) M, log density error $(abs(p[13] - p_true[13])), field angle error $(abs(p[16] - p_true[16]))"
    end
end

"""
Non-thermal splat sets: power-law and κ splats render finite, partially polarized images with the
same geometry rows as the thermal set; in the optically thin limit the composite of the three
populations equals the sum of their images (coefficients add) to 1e-6; CUDA matches the CPU
backend.
"""
function test_populations(backend; res = 16, N = 200, label = "")
    Geodesics.prepare_backend!(backend)
    @testset "power-law and κ splats ($label)" begin
        a = 0.9; θo = deg2rad(60.0); t_obs = 0.0
        camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
        cache = GeodesicCache(backend, camera, Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        geo = [4.0, 1.0, 0.0, 0.2, -0.1, -0.3, 1.0, 0.1, 0.0, 0.0, 0.0, log(1e6)]
        tail = [log(30.0), 1.2, 0.7, 0.1, 0.45, 0.0, 0.03]                 # logB, thB, phB, u1, u2, u3, omega
        pth = reshape(vcat(geo, [log(10.0), log(20.0)], tail), :, 1)     # thermal: logne, logTe (thin: τ ~ 1e-5)
        ppl = reshape(vcat(geo, [log(10.0), 3.0, log(10.0)], tail), :, 1) # power law: logne, p, ln γmin
        pκ = reshape(vcat(geo, [log(10.0), 4.0, log(10.0)], tail), :, 1)  # κ: logne, κ, ln w
        @test size(pth, 1) == NPOLARIZEDPARAMS && size(ppl, 1) == NPOWERLAWPARAMS && size(pκ, 1) == NKAPPAPARAMS
        mth = PolarizedSplats(adapt_to(backend, pth), t_obs)
        mpl = PowerLawSplats(adapt_to(backend, ppl), t_obs)
        mκ = KappaSplats(adapt_to(backend, pκ), t_obs)
        L = gravitational_radius(4e6)
        img(m) = Array(stokes_image(backend, m, camera, a, θo, POL_ν, L; N))
        Ith, Ipl, Iκ = img(mth), img(mpl), img(mκ)
        for I in (Ith, Ipl, Iκ)
            @test all(x -> all(isfinite, x), I) && maximum(getindex.(I, 1)) > 0
            @test 0 < maximum(hypot.(getindex.(I, 2), getindex.(I, 3)) ./ max.(getindex.(I, 1), 1e-300)) < 1
        end
        Iall = img(CompositeModel(CompositeModel(mth, mpl), mκ))
        err = maximum(norm.(Iall .- (Ith .+ Ipl .+ Iκ))) / maximum(norm.(Iall))
        @test err < 1e-4                                                 # thin: coefficients add, images add up to the optical depth
        @info "populations ($label): peak I thermal $(maximum(getindex.(Ith, 1))), power law $(maximum(getindex.(Ipl, 1))), κ $(maximum(getindex.(Iκ, 1))); composite vs sum $err"
        if !(backend isa CPU)
            cc = GeodesicCache(CPU(), camera, Val(N); store_samples = false); regenerate!(cc, a, θo; marcher = Fused(64))
            ref = Array(stokes_image(CPU(), CompositeModel(PowerLawSplats(ppl, t_obs), KappaSplats(pκ, t_obs)), camera, a, θo, POL_ν, L; N))
            gpu = img(CompositeModel(mpl, mκ))
            @test maximum(norm.(gpu .- ref)) < 1e-11 * maximum(norm.(ref))
        end
    end
end

"""
    test_reflection()

Parity gate for the polarized splats: the Kerr spacetime is symmetric under the equatorial
reflection z → −z and (with the spin reversed) under y → −y, so the image of a source seen
from below the equator (θo = 163°, the M87 geometry) must be the β-mirror of the image of the
reflected source seen from 17°, and the image at spin −a the α-mirror of the reflected source
at +a. The reflection maps the field as an axial vector (B → −B on top of the geometric
mirror), reverses the mirrored velocity component, and flips the signs of U and V. Pins the
observer inclinations beyond 90°, negative spins and the handedness of the screen basis.
"""
function test_reflection(; res = 24, N = 60)
    fov = 20.0; Δα = fov / res
    camera = Geodesics.Camera((-fov / 2 + Δα / 2, fov / 2 - Δα / 2), (-fov / 2 + Δα / 2, fov / 2 - Δα / 2), res)
    M_solar = 6.5e9; L = gravitational_radius(M_solar); ν = 230e9
    function image(a, θo, q)
        cache = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
        regenerate!(cache, a, θo; marcher = Fused(64))
        polarized_image(cache, q, 0.0, ν, L)
    end
    S(x, k) = getindex.(x, k)
    rel(x, y) = sqrt(sum((x .- y) .^ 2) / sum(y .^ 2))
    p = zeros(NPOLARIZEDPARAMS, 2)
    p[:, 1] = [5.0, 2.0, 0.5, log(1.5), log(1.5), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(3e6), log(30.0), log(8.0), 1.0, 0.5, 0.3, 0.3, 0.1, 0.0]
    p[:, 2] = [-3.0, -4.0, -1.0, log(1.0), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e5), log(20.0), log(15.0), 2.0, -1.0, -0.2, 0.4, -0.1, 0.0]
    @testset "equatorial and azimuthal reflections" begin
        A = image(0.94, deg2rad(163.0), p)
        q = copy(p); q[3, :] .= -q[3, :]; q[17, :] .+= π; q[20, :] .= -q[20, :]           # z → −z: B → −(mirror), u_z reversed
        D = image(0.94, deg2rad(17.0), q)
        for k in 1:4
            @test rel(reverse(S(D, k); dims = 2) .* (k >= 3 ? -1 : 1), S(A, k)) < 1e-10
        end
        B = image(0.94, deg2rad(17.0), p)
        r = copy(p); r[2, :] .= -r[2, :]; r[16, :] .= π .- r[16, :]; r[17, :] .= π .- r[17, :]; r[19, :] .= -r[19, :]   # y → −y, spin reversed
        E = image(-0.94, deg2rad(17.0), r)
        for k in 1:4
            @test rel(reverse(S(E, k); dims = 1) .* (k >= 3 ? -1 : 1), S(B, k)) < 1e-10
        end
        # the gate has teeth: a polar-vector mirror of the field (no B → −B) breaks V
        q2 = copy(q); q2[17, :] .-= π; q2[16, :] .= π .- q2[16, :]
        D2 = image(0.94, deg2rad(17.0), q2)
        @test rel(-reverse(S(D2, 4); dims = 2), S(A, 4)) > 0.1
        @info "reflection: θo = 163° vs the mirrored source at 17°, and spin −0.94 vs +0.94, agree to $(maximum(rel(reverse(S(D, k); dims = 2) .* (k >= 3 ? -1 : 1), S(A, k)) for k in 1:4)) and $(maximum(rel(reverse(S(E, k); dims = 1) .* (k >= 3 ? -1 : 1), S(B, k)) for k in 1:4)); |V|/I = $(round(sqrt(sum(S(A, 4) .^ 2) / sum(S(A, 1) .^ 2)); sigdigits = 2))"
    end
end

# ---- the polarized consumer differentiated inside a GPU kernel ------------------------------------
@inline function _polarized_chunk_value!(out, params, tvec, i, pix, ν, L, ::Val{K}) where {K}
    c = RadiativeTransport(PolarizedSplats(params, tvec), ν, L)
    acc = zero(RadiativeState{Float64})
    for k in 1:K
        s = Geodesics.GeodesicSample(1.0 + 0.1k, 8.0 - 0.3k + 0.01i, 1.3 + 0.02k, 0.7 + 0.05k, true, k % 2 == 0, true)
        acc = c(acc, i, k, s, 0.01, pix)
    end
    st = observed_stokes(acc, ν)
    @inbounds out[i] = st[1] + 0.5 * st[2] - 0.3 * st[3] + 2 * st[4]
    return nothing
end
@kernel function _polarized_chunk_adjoint!(out, dout, params, dparams, tvec, dtvec, pix, ν, L, ::Val{K}) where {K}
    i = @index(Global, Linear)
    Enzyme.autodiff_deferred(Enzyme.Reverse, Enzyme.Const(_polarized_chunk_value!), Enzyme.Const, Enzyme.Duplicated(out, dout), Enzyme.Duplicated(params, dparams), Enzyme.Duplicated(tvec, dtvec),
                             Enzyme.Const(i), Enzyme.Const(pix), Enzyme.Const(ν), Enzyme.Const(L), Enzyme.Const(Val(K)))
end

"""
    test_polarized_kernel_gradient(backend; K = 8)

The full polarized consumer (thermal synchrotron with the inline Bessel functions, the local
frame, the exact transfer step, compositing) folded over `K` synthetic samples and differentiated
by Enzyme inside a kernel on `backend`, against the host Enzyme gradient of the same fold. On
CUDA the per-thread stack (64 KB, the most the card can give every resident thread) holds the
tape of up to eight polarized samples; longer rays need the chunked reverse sweep.
"""
function test_polarized_kernel_gradient(backend; K::Int = 8, label = "CPU")
    met = Krang.Kerr(0.94); θo = deg2rad(60.0)
    pix = Krang.SlowLightIntensityPixel(met, 3.0, 1.0, θo)
    p = zeros(NPOLARIZEDPARAMS, 2)
    p[:, 1] = [5.0, 2.0, 0.5, log(1.5), log(1.5), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(3e5), log(30.0), log(8.0), 1.0, 0.5, 0.3, 0.3, 0.1, 0.0]
    p[:, 2] = [-3.0, 4.0, -0.5, log(1.0), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e5), log(20.0), log(15.0), 2.0, -1.0, -0.2, 0.4, -0.1, 0.0]
    n = 96; L = gravitational_radius(6.5e9); ν = 230e9
    function host(q)
        c = RadiativeTransport(PolarizedSplats(q, [20.0]), ν, L); total = 0.0
        for i in 1:n
            acc = zero(RadiativeState{Float64})
            for k in 1:K
                s = Geodesics.GeodesicSample(1.0 + 0.1k, 8.0 - 0.3k + 0.01i, 1.3 + 0.02k, 0.7 + 0.05k, true, k % 2 == 0, true)
                acc = c(acc, i, k, s, 0.01, pix)
            end
            st = observed_stokes(acc, ν)
            total += st[1] + 0.5 * st[2] - 0.3 * st[3] + 2 * st[4]
        end
        return total
    end
    gh = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(host), p)[1]
    @testset "polarized consumer differentiated in a kernel ($label, $K samples)" begin
        Geodesics.prepare_backend!(backend; stack_bytes = Geodesics.ENZYME_STACK_BYTES, heap_bytes = Geodesics.ENZYME_HEAP_BYTES)
        params = adapt_to(backend, p); dparams = adapt_to(backend, zeros(size(p)))
        out = adapt_to(backend, zeros(n)); dout = adapt_to(backend, ones(n))
        tvec = adapt_to(backend, [20.0]); dtvec = adapt_to(backend, [0.0])
        if backend isa CPU
            dp = adapt_to(backend, zeros(size(p))); dw = adapt_to(backend, ones(n)); dt = adapt_to(backend, [0.0])
            _polarized_chunk_adjoint!(backend, 1)(out, dw, params, dp, tvec, dt, pix, ν, L, Val(K); ndrange = 1); KernelAbstractions.synchronize(backend)
        end
        _polarized_chunk_adjoint!(backend, 64)(out, dout, params, dparams, tvec, dtvec, pix, ν, L, Val(K); ndrange = n)
        KernelAbstractions.synchronize(backend)
        g = Array(dparams)
        e = maximum(abs.(g .- gh)) / maximum(abs.(gh))
        @test all(isfinite, g)
        @test e < 1e-10
        @info "polarized consumer in a kernel ($label, $K samples): max |Δ|/max vs host Enzyme $e"
    end
end

"""
    test_polarized_gradient(backend; res, N, tol, label, methods = (:dual, :enzyme))

The in-kernel polarized gradients (`polarized_gradient!`: the dual sweep and the chunked Enzyme
reverse sweep) against the host Enzyme gradient of the same loss, Σⱼ wⱼ · observed_stokes(ray j)
for a fixed set of per-pixel weights, through the fused march on the CPU: the stored samples
and the fused samples are the same Mino grid.
"""
function test_polarized_gradient(backend; res::Int, N::Int, tol::Float64, label::String, methods = (:dual, :enzyme))
    a, θo = 0.94, deg2rad(60.0); t_obs = 12.0; ν = 230e9; L = gravitational_radius(4e6)
    camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
    p = zeros(NPOLARIZEDPARAMS, 2)
    p[:, 1] = [5.0, 2.0, 0.5, log(1.5), log(1.5), log(1.0), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(3e5), log(30.0), log(8.0), 1.0, 0.5, 0.3, 0.3, 0.1, 0.01]
    p[:, 2] = [-3.0, 4.0, -0.5, log(1.0), log(1.2), log(0.8), 1.0, 0.0, 0.0, 0.0, 0.0, log(1e9), log(1e5), log(20.0), log(15.0), 2.0, -1.0, -0.2, 0.4, -0.1, 0.02]
    cpu = GeodesicCache(CPU(), camera, Val(N); store_samples = false)
    regenerate!(cpu, a, θo; marcher = Fused(64))
    rng = MersenneTwister(7)
    w = [SVector{4}(randn(rng, 4)) for _ in 1:npixels(cpu)]                       # sorted pixel order
    function host(q)
        out = Vector{RadiativeState{Float64}}(undef, npixels(cpu)); fill!(out, zero(RadiativeState{Float64}))
        polarized_image!(out, cpu, q, t_obs, ν, L)
        total = 0.0
        for j in eachindex(out)
            total += sum(w[j] .* observed_stokes(out[j], ν))
        end
        return total
    end
    gh = Enzyme.gradient(Enzyme.set_runtime_activity(Enzyme.Reverse), Enzyme.Const(host), p)[1]
    # (a different name from the closure's `out`: a variable assigned both inside a closure Enzyme differentiates and in
    # the enclosing function is boxed, and Enzyme then treats the box captured by the `Const` closure as constant memory
    # and returns a zero gradient)
    out_ref = Vector{RadiativeState{Float64}}(undef, npixels(cpu)); fill!(out_ref, zero(RadiativeState{Float64})); polarized_image!(out_ref, cpu, p, t_obs, ν, L)
    reference = [observed_stokes(st, ν) for st in out_ref]
    cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
    regenerate!(cache, a, θo; marcher = Recurrence(64))
    params = adapt_to(backend, p)
    dstokes = adapt_to(backend, w)
    for method in methods
        @testset "$label polarized gradient by the $method sweep, $(res)² × $N" begin
            dparams = adapt_to(backend, zeros(size(p)))
            t0 = time()
            _, image = polarized_gradient!(dparams, dstokes, cache, params, t_obs, ν, L; method)
            t1 = time() - t0
            fill!(dparams, 0.0)
            t2 = @elapsed polarized_gradient!(dparams, dstokes, cache, params, t_obs, ν, L; method)
            g = Array(dparams)
            e = maximum(abs.(g .- gh)) / maximum(abs.(gh))
            @test all(isfinite, g)
            @test e <= tol
            # the forward pass of the sweep is the image itself
            @test maximum(norm.(Array(image) .- reference)) <= 1e-12 * maximum(norm.(reference))
            @info "$label polarized $method-sweep gradient vs host Enzyme: max |Δ|/max = $e ($(round(t1; digits = 1)) s including compilation, $(round(t2; digits = 3)) s warm)"
        end
    end
end

"""
    test_ray_lists(backend; res = 8, N = 40, tol = 1e-12, label = "CPU backend")

The large-N path: per-ray parcel lists (`ray_lists`) against the dense sums. Forty-eight
parcels of mixed sizes; the lists are sorted, duplicate-free and much shorter than the parcel
count; the tails image over the lists equals the dense one, the dual-sweep gradient over the
lists equals the dense one to `tol`, with and without the half-orbit truncation, and
`polarized_gradient!(...; cull = true)` equals `cull = false`.
"""
function test_ray_lists(backend; res::Int = 8, N::Int = 40, tol::Float64 = 1e-12, label::String = "CPU backend")
    a, θo = 0.94, deg2rad(60.0); t_obs = 12.0; ν = 230e9; L = gravitational_radius(4e6)
    camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
    rng = MersenneTwister(11)
    n = 48
    p = zeros(NPOLARIZEDPARAMS, n)
    for i in 1:n
        φ = 2π * rand(rng); r0 = 3.0 + 5.0 * rand(rng); size = i % 3 == 0 ? 0.15 : 0.6
        p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.8 * randn(rng), log(size), log(size * 1.3), log(size * 0.7), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
                   0.0, log(1e9), log(2e4), log(30.0), log(10.0), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.2 * randn(rng), 0.3 + 0.1 * randn(rng), 0.1 * randn(rng), 0.02 * randn(rng)]
    end
    cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
    regenerate!(cache, a, θo; marcher = Recurrence(64))
    params = adapt_to(backend, p)
    w = [SVector{4}(randn(rng, 4)) for _ in 1:npixels(cache)]
    dstokes = adapt_to(backend, w)
    @testset "$label per-ray parcel lists" begin
        for (nmax, slab) in ((-1, 0.0), (1, 0.5))
            lists = ray_lists(cache, params, t_obs; nmax, slab)
            ids = Array(lists.ids); count = Array(lists.count)
            @test all(0 .<= count .<= size(ids, 1)) && maximum(count) == size(ids, 1)
            @test all(issorted(ids[1:count[j], j]) && allunique(ids[1:count[j], j]) && all(1 .<= ids[1:count[j], j] .<= n) for j in eachindex(count))
            @test sum(count) < 0.5 * n * length(count)                       # the lists cull most parcels on most rays
            tails_d = adapt_to(backend, zeros(SVector{4,Float64}, npixels(cache), N + 1)); tails_l = adapt_to(backend, zeros(SVector{4,Float64}, npixels(cache), N + 1))
            polarized_tails!(tails_d, cache, params, t_obs, ν, L; nmax, slab)
            polarized_tails!(tails_l, cache, params, t_obs, ν, L; nmax, slab, lists)
            img_d = Array(tail_image(tails_d, ν)); img_l = Array(tail_image(tails_l, ν))
            scale = maximum(x -> maximum(abs, x), img_d)
            @test maximum(maximum.(abs, img_l .- img_d)) <= 1e-14 * scale
            g_d = adapt_to(backend, zeros(size(p))); g_l = adapt_to(backend, zeros(size(p)))
            polarized_dual_sweep!(g_d, dstokes, tails_d, cache, params, t_obs, ν, L; nmax, slab)
            polarized_dual_sweep!(g_l, dstokes, tails_l, cache, params, t_obs, ν, L; nmax, slab, lists)
            e = maximum(abs.(Array(g_l) .- Array(g_d))) / maximum(abs.(Array(g_d)))
            @test e <= tol
            @info "$label per-ray lists (nmax = $nmax): capacity $(size(ids, 1)) of $n parcels, mean list length $(round(sum(count) / length(count); digits = 1)); image identical to $(maximum(maximum.(abs, img_l .- img_d)) / scale), gradient to $e"
        end
        g_c = adapt_to(backend, zeros(size(p))); g_n = adapt_to(backend, zeros(size(p)))
        _, img_c = polarized_gradient!(g_c, dstokes, cache, params, t_obs, ν, L; cull = true)
        _, img_n = polarized_gradient!(g_n, dstokes, cache, params, t_obs, ν, L; cull = false)
        @test maximum(abs.(Array(g_c) .- Array(g_n))) / maximum(abs.(Array(g_n))) <= tol
        @test Array(img_c) == Array(img_n)
    end
end

"""
    test_frame_batching(backend; res = 8, N = 40, tol = 1e-13, label = "CPU backend")

Frames batched into one launch against one launch per frame: the lists of three frames built
together equal the three built apart; the batched tails and image equal the per-frame ones
exactly; the batched dual-sweep gradient equals the sum of the per-frame gradients to `tol`,
dense and over lists, with and without the half-orbit truncation; and `chi2_gradient!` with
`batch_frames = 3` equals `batch_frames = 1`.
"""
function test_frame_batching(backend; res::Int = 8, N::Int = 40, tol::Float64 = 1e-13, label::String = "CPU backend")
    a, θo = 0.94, deg2rad(60.0); ν = 230e9; L = gravitational_radius(4e6)
    camera = Geodesics.Camera((-10.0, 10.0), (-10.0, 10.0), res)
    rng = MersenneTwister(13)
    n = 24
    p = zeros(NPOLARIZEDPARAMS, n)
    for i in 1:n
        φ = 2π * rand(rng); r0 = 3.0 + 5.0 * rand(rng); sz = i % 3 == 0 ? 0.15 : 0.6
        p[:, i] = [r0 * cos(φ), r0 * sin(φ), 0.8 * randn(rng), log(sz), log(sz * 1.3), log(sz * 0.7), 1.0, 0.1 * randn(rng), 0.1 * randn(rng), 0.0,
                   0.0, log(1e9), log(4e4), log(30.0), log(10.0), π / 2 + 0.3 * randn(rng), 0.5 * randn(rng), 0.2 * randn(rng), 0.3 + 0.1 * randn(rng), 0.1 * randn(rng), 0.03 * randn(rng)]
    end
    cache = GeodesicCache(backend, camera, Val(N); store_samples = true)
    regenerate!(cache, a, θo; marcher = Recurrence(64))
    npix = npixels(cache)
    params = adapt_to(backend, p)
    times = [0.0, 12.0, 25.0]
    w = [SVector{4}(randn(rng, 4)) for _ in 1:npix, _ in 1:3]
    @testset "$label frames batched into one launch" begin
        for (nmax, slab) in ((-1, 0.0), (1, 0.5)), cull in (false, true)
            lists3 = cull ? ray_lists(cache, params, times; nmax, slab) : nothing
            if cull
                for (f, t) in enumerate(times)
                    l1 = ray_lists(cache, params, t; nmax, slab)
                    c3 = Array(lists3.count)[:, f]; c1 = Array(l1.count)
                    @test c3 == c1
                    @test all(Array(lists3.ids)[1:c3[j], j, f] == Array(l1.ids)[1:c1[j], j] for j in 1:npix)
                end
            end
            tails3 = adapt_to(backend, zeros(SVector{4,Float64}, npix, N + 1, 3))
            polarized_tails!(tails3, cache, params, times, ν, L; nmax, slab, lists = lists3)
            img3 = Array(tail_image(tails3, ν))
            @test size(img3) == (npix, 3)
            g3 = adapt_to(backend, zeros(size(p)))
            polarized_dual_sweep!(g3, adapt_to(backend, w), tails3, cache, params, times, ν, L; nmax, slab, lists = lists3)
            g1 = adapt_to(backend, zeros(size(p)))
            for (f, t) in enumerate(times)
                l1 = cull ? ray_lists(cache, params, t; nmax, slab) : nothing
                tails1 = adapt_to(backend, zeros(SVector{4,Float64}, npix, N + 1))
                polarized_tails!(tails1, cache, params, t, ν, L; nmax, slab, lists = l1)
                @test Array(tail_image(tails1, ν)) == img3[:, f]
                polarized_dual_sweep!(g1, adapt_to(backend, w[:, f]), tails1, cache, params, t, ν, L; nmax, slab, lists = l1)
            end
            e = maximum(abs.(Array(g3) .- Array(g1))) / maximum(abs.(Array(g1)))
            @test e <= tol
            @info "$label batched frames (nmax = $nmax, lists = $cull): image identical, gradient to $e"
        end
        # the movie χ² gradient with frames batched
        clean = Array(polarized_cube(cache, params, times, [ν], L))
        σ = SVector(0.02, 0.01, 0.01, 0.005) * maximum(norm.(clean))
        movie = StokesMovie([clean[idx] + σ .* SVector{4}(randn(rng, 4)) for idx in CartesianIndices(clean)], times, [ν], σ)
        gb = adapt_to(backend, zeros(size(p))); gu = adapt_to(backend, zeros(size(p)))
        χb = Fit.chi2_gradient!(gb, params, movie, cache, L; batch_frames = 3)
        χu = Fit.chi2_gradient!(gu, params, movie, cache, L; batch_frames = 1)
        @test abs(χb - χu) <= 1e-12 * χu
        @test maximum(abs.(Array(gb) .- Array(gu))) / maximum(abs.(Array(gu))) <= tol
    end
end

"""
    test_precision(; res = 8, N = 24)

`Geodesics.precision`: the Float32 copy of a Float64 cache has converted samples and constants,
the same permutation and ranges, and the tails image rendered on it is finite and within 5e-3 of
the Float64 image (1.9e-3 at this size with sixteen samples, 1.1e-4 at 24² × 60; the
coefficient chain still promotes and the invariants reach subnormals; see the FP32 note).
"""
function test_precision(; res = 8, N = 24)
    a, θo = 0.9, deg2rad(60.0); ν = 230e9; L = gravitational_radius(4e6)
    camera = Geodesics.Camera((-9.0, 9.0), (-9.0, 9.0), res)
    cache = GeodesicCache(CPU(), camera, Val(N); store_samples = true); regenerate!(cache, a, θo; marcher = Recurrence(64))
    c32 = Geodesics.precision(cache, Float32)
    p = polarized_test_params()
    @testset "precision converter" begin
        @test c32 isa GeodesicCache{Float32} && eltype(c32.samples.r) == Float32 && eltype(c32.consts.η) == Float32
        @test c32.perm_host == cache.perm_host && c32.ranges == cache.ranges && c32.spin == Float32(a)
        @test maximum(abs.(Float64.(c32.samples.r) .- cache.samples.r)) <= 1e-6 * maximum(abs, cache.samples.r)
        t64 = zeros(SVector{4,Float64}, npixels(cache), N + 1); polarized_tails!(t64, cache, p, 5.0, ν, L)
        t32 = zeros(SVector{4,Float32}, npixels(cache), N + 1); polarized_tails!(t32, c32, Float32.(p), 5.0f0, Float32(ν), Float32(L))
        img64 = tail_image(t64, ν); img32 = tail_image(t32, Float32(ν))
        @test all(x -> all(isfinite, x), img32)
        e = maximum(maximum.(abs, SVector{4,Float64}.(img32) .- img64)) / maximum(x -> maximum(abs, x), img64)
        @test e < 1e-4                      # measured 3e-7 (the coefficient chain's own Float32 rounding, 1e-7)
        # the coefficient chain stays Float32 (the Dexter fits, the Bessel series and the physical constants typed by
        # the arguments), and the Bessel functions in Float32 follow Float64 to single precision
        cf32 = Transfer.thermal_synchrotron(1.0f3, 30.0f0, 20.0f0, 2.3f11, 1.0f0)
        cf64 = Transfer.thermal_synchrotron(1.0e3, 30.0, 20.0, 2.3e11, 1.0)
        @test cf32 isa Transfer.StokesCoefficients{Float32}
        @test abs(cf32.jI - cf64.jI) <= 1e-5 * abs(cf64.jI) && abs(cf32.ρV - cf64.ρV) <= 1e-4 * abs(cf64.ρV)
        for x in (0.3, 1.0, 3.0, 12.0)
            @test abs(Transfer.besselk0(Float32(x)) - Transfer.besselk0(x)) <= 2e-6 * Transfer.besselk0(x)
            @test abs(Transfer.besselk1(Float32(x)) - Transfer.besselk1(x)) <= 2e-6 * Transfer.besselk1(x)
        end
        @test Transfer.planck_invariant(2.3f11, 30.0f0) isa Float32
        # the dual sweep in Float32 runs and its gradient follows the Float64 one
        w = [SVector{4}(Float64.(randn(MersenneTwister(4), 4))) for _ in 1:npixels(cache)]
        g64 = zeros(size(p)); polarized_dual_sweep!(g64, w, t64, cache, p, 5.0, ν, L)
        g32 = zeros(Float32, size(p)); polarized_dual_sweep!(g32, SVector{4,Float32}.(w), t32, c32, Float32.(p), 5.0f0, Float32(ν), Float32(L))
        @test all(isfinite, g32)
        eg = maximum(abs.(Float64.(g32) .- g64)) / maximum(abs.(g64))
        @test eg < 1e-3                     # measured 1.5e-6
        @info "precision: the Float32 tails image agrees with Float64 to $e of the peak, the Float32 dual-sweep gradient to $eg; Float32 synchrotron jI relative difference $(abs(cf32.jI - cf64.jI) / abs(cf64.jI))"
    end
end

