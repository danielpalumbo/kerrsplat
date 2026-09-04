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

## A time-averaged M87 library image

The same script on `ma+0.94_r40_nall_tavg.fits` from the M87 GRMHD library in
`~/Dropbox/aditya_projects` (MAD, a = +0.94, R_high = 40, time-averaged; ehtim-style FITS,
480² pixels of 0.33 μas, 0.41 Jy), with the observer at 163° and the library's scaling assumed
to be M = 6.2e9 M⊙ at 16.9 Mpc (`--res 40 --samples 200 --iterations 300`, 27 minutes):
χ²/N = 1.3 at the same 2/1/1/0.5% noise, but the image is reproduced much less well than the
Sgr A* snapshot (`figures/m87_library_tavg_fit.png`):

| Stokes | relative L2 error (of the I norm) | data total (Jy) | model total (Jy) |
|---|---|---|---|
| I | 18% | 0.408 | 0.333 |
| Q | 14% | −0.0001 | 0.0043 |
| U | 13% | 0.0038 | −0.0004 |
| V | 2.6% | 0.0007 | −0.0062 |

The time average of a turbulent MAD flow has a thin, nearly uniform ring with a broad faint
halo and a polarization pattern with almost no net Q, U (the average of a rotating EVPA
pattern); 24 parcels starting from a 4 M ring capture the ring's shape and brightness
asymmetry but not the halo (18% of the flux missing) nor the fine Q, U structure at 4 μas
blocks, and the run started very far from the data (a 17 Jy ring against 0.41 Jy). This is the
harder of the two targets and the one closer to the real M87; the remedies are the same as
above (a smooth background component, a noise model with a floor, more iterations from a
scaled start), plus the correct library scaling, which the FITS header does not carry.
