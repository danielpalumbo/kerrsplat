# Fit to a GRMHD snapshot image (2026-09-04)

Plan §7.5 item 7 ii asks for recovery tests against GRMHD movies. No fluid dumps exist on this
machine, but ipole images of KHARMA snapshots do (`~/Dropbox/for_angelo/sgra_figs`, ipole
`v1.4-45`, full Stokes), so `validation/grmhd_fit` fits polarized splats to one of them:
the MAD a = 0.9375, R_high = 10 snapshot of Sgr A* seen from 130° (below the equator) at
230 GHz, 200 μas across at 8.13 kpc for 4.14e6 M⊙ (the units of the ipole header), 400²
pixels of 0.5 μas, total flux 2.28 Jy with 5.9% net linear polarization.

Setup (`grmhd_fit.jl --res 40 --samples 200 --iterations 300`): the image block-averaged to
40² blocks of 5 μas (1.3 M), noise 2/1/1/0.5% of the peak Stokes vector norm in I/Q/U/V, the
staged fit of the RIAF experiment (plasma rows first, then everything, Adam with cosine
annealing, densification every 40 iterations up to 16 → the two splittings gave 6 → 12 → 24
splats), from six splats on a Keplerian ring at 4 M with a toroidal field. 26 minutes on the
CPU backend with eight threads.

Result (`figures/kharma_sgra_snapshot_fit.png`): χ²/N = 0.039 at the assumed noise, i.e. the
24-splat model reproduces the blocked image far inside the 2% level:

| Stokes | relative L2 error (of the I norm) | data total (Jy) | model total (Jy) |
|---|---|---|---|
| I | 5.3% | 2.284 | 2.087 |
| Q | 2.1% | 0.134 | 0.153 |
| U | 2.4% | 0.108 | 0.107 |
| V | 0.46% | −0.0023 | −0.0006 |

The model's total flux is 8.6% low: the splats reproduce the bright ring and its polarization
pattern but not the faint extended emission of the simulation outside the ring (the χ² is
dominated by the bright pixels at a noise proportional to the peak), which is where a
composite with a smooth background, or a per-pixel noise model, would matter. This is a
single snapshot, static and at one frequency; the movie version needs dumps (or an ipole movie)
and remains open, as does the comparison of the fitted fields with the simulation's, which the
image files do not carry.

The same script takes the time-averaged M87 library images in `~/Dropbox/aditya_projects`
(ehtim-style FITS, 480² pixels of 0.33 μas) with `--spin`, `--inc`, `--msolar`, `--dpc`.
