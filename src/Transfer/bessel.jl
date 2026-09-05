# K₀ and K₁ as inline polynomials, for GPU kernels differentiated by Enzyme.
#
# The algorithms and coefficients are those of Bessels.jl (MIT License, Copyright (c) 2021-2022
# Michael Helton, Oscar Smith, and contributors; `besselk0`, `besselk1` and `constants.jl`),
# rewritten with `Base.Math.@horner` so that the coefficient tables become inline multiply-adds:
# Bessels.jl's `evalpoly` calls take the tables as constant global arrays, and Enzyme's reverse
# pass inside a CUDA kernel cannot cache such calls ("caching call: julia_evalpoly"). Accuracy is
# that of Bessels.jl (a few ulp); the gate in test_coefficients compares the two on a grid.

"""
    @muladd_chain(x, c0, c1, …)

Horner's rule as an explicit nest of `muladd`s: `Base.Math.@horner` expands to `evalpoly`, which
the compiler outlines into a call taking the coefficient tuple by reference once the tuple has
nine or more entries, and Enzyme's reverse pass on the device cannot cache that call.
"""
macro muladd_chain(x, cs...)
    ex = esc(cs[end])
    for c in reverse(cs[1:end-1])
        ex = :(muladd($(esc(x)), $ex, $(esc(c))))
    end
    return ex
end

"Modified Bessel function of the second kind of order zero, K₀(x), x > 0."
@inline function besselk0_inline(x::T) where {T<:Real}
    if x <= one(T)
        a = x * x / 4
        s = muladd(@muladd_chain(a, -1.372509002685546267e-1, 2.574916117833312855e-1, 1.395474602146869316e-2, 5.445476986653926759e-4, 7.125159422136622118e-6),
                   inv(@muladd_chain(a, 1.000000000000000000e+00, -5.458333438017788530e-02, 1.291052816975251298e-03, -1.367653946978586591e-05)),
                   T(1.137250900268554688))
        a = muladd(s, a, one(T))
        return muladd(-a, log(x), @muladd_chain(x * x, 1.159315156584124484e-01, 2.789828789146031732e-01, 2.524892993216121934e-02, 8.460350907213637784e-04,
                                                     1.491471924309617534e-05, 1.627106892422088488e-07, 1.208266102392756055e-09, 6.611686391749704310e-12))
    else
        s = exp(-x / 2)
        y = inv(x)
        a = muladd(@muladd_chain(y, 2.533141373155002416e-1, 3.628342133984595192e0, 1.868441889406606057e1, 4.306243981063412784e1, 4.424116209627428189e1,
                                     1.562095339356220468e1, -1.810138978229410898e0, -1.414237994269995877e0, -9.369168119754924625e-2),
                   inv(@muladd_chain(y, 1.000000000000000000e0, 1.494194694879908328e1, 8.265296455388554217e1, 2.162779506621866970e2, 2.845145155184222157e2,
                                         1.851714491916334995e2, 5.486540717439723515e1, 6.118075837628957015e0, 1.586261269326235053e-1)),
                   one(T)) * s / sqrt(x)
        return a * s
    end
end

"Modified Bessel function of the second kind of order one, K₁(x), x > 0."
@inline function besselk1_inline(x::T) where {T<:Real}
    if x <= one(T)
        z = x * x
        a = z / 4
        pq = muladd(@muladd_chain(a, -3.62137953440350228e-3, 7.11842087490330300e-3, 1.00302560256614306e-5, 1.77231085381040811e-6),
                    inv(@muladd_chain(a, 1.00000000000000000e0, -4.80414794429043831e-2, 9.85972641934416525e-4, -8.91196859397070326e-6)),
                    T(8.69547128677368164e-2))
        pq = muladd(pq * a, a, (a / 2 + one(T)))
        a = pq * x / 2
        pq = muladd(@muladd_chain(z, -3.07965757829206184e-1, -7.80929703673074907e-02, -2.70619343754051620e-3, -2.49549522229072008e-5) /
                    @muladd_chain(z, 1.00000000000000000e0, -2.36316836412163098e-2, 2.64524577525962719e-4, -1.49749618004162787e-6), x, inv(x))
        return muladd(a, log(x), pq)
    else
        s = exp(-x / 2)
        y = inv(x)
        a = muladd(@muladd_chain(y, -1.97028041029226295e-1, -2.32408961548087617e0, -7.98269784507699938e0, -2.39968410774221632e0, 3.28314043780858713e1,
                                     5.67713761158496058e1, 3.30907788466509823e1, 6.62582288933739787e0, 3.08851840645286691e-1),
                   inv(@muladd_chain(y, 1.00000000000000000e0, 1.41811409298826118e1, 7.35979466317556420e1, 1.77821793937080859e2, 2.11014501598705982e2,
                                         1.19425262951064454e2, 2.88448064302447607e1, 2.27912927104139732e0, 2.50358186953478678e-2)),
                   T(1.45034217834472656)) * s / sqrt(x)
        return a * s
    end
end
