"""ehtim's Sgr A* diffractive scattering kernel in the visibility domain (`sgra_kernel_uv`) at 230 GHz on a set of
(u, v) points, for two parameter sets: ehtim's constants (Bower et al. 2006: 1.309 mas, 0.640 mas, 78°) and the
Johnson et al. 2018 values the EHT's 2017 Sgr A* analysis used (1.380 mas, 0.703 mas, 81.9°). Fixture for
`test_scattering` (test/data/sgra_kernel_ehtim.csv: u, v, g_bower, g_johnson).

    python dump_scattering.py <outfile.csv>
"""
import sys, numpy as np
import ehtim.observing.obs_helpers as oh, ehtim.const_def as ehc
rf = 230e9
rng = np.random.default_rng(1)
u = np.concatenate([[0.0, 1e9, 0.0, -3e9, 5e9], rng.uniform(-9e9, 9e9, 40)])
v = np.concatenate([[0.0, 0.0, 1e9, 4e9, -5e9], rng.uniform(-9e9, 9e9, 40)])
g_bower = oh.sgra_kernel_uv(rf, u, v)
ehc.FWHM_MAJ, ehc.FWHM_MIN, ehc.POS_ANG = 1380.0, 703.0, 81.9
g_johnson = oh.sgra_kernel_uv(rf, u, v)
with open(sys.argv[1], "w") as f:
    f.write("# ehtim sgra_kernel_uv at 230 GHz: u, v (wavelengths), kernel with (1.309 mas, 0.640 mas, 78 deg), kernel with (1.380 mas, 0.703 mas, 81.9 deg)\n")
    for k in range(len(u)):
        f.write(f"{u[k]!r},{v[k]!r},{g_bower[k]!r},{g_johnson[k]!r}\n")
print(len(u), "points; kernel range", g_johnson.min(), g_johnson.max())
