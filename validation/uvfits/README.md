# uvfits reader against ehtim on real EHT data

`Fit.read_uvfits` is gated in the test suite on an ehtim-written synthetic observation
(`test/data/synth_eht2017*.uvfits`). This directory checks it on a real file, the public
EHT 2017 M87 Stokes-I data (`SR1_M87_2017_101_lo_hops_netcal_StokesI.uvfits`, HOPS pipeline,
band low, day 101, not part of the repository), against ehtim 1.2.10's parse of the same file.

    python dump_ehtim.py <file.uvfits> <out.csv>          # ehtim's parse (needs eht-imaging)
    julia --project=../.. compare.jl <file.uvfits> <out.csv> [--scan-average]

With `--scan-average`, `Fit.average_scans` is compared with ehtim's `add_scans()` followed by
`avg_coherent(0., scan_avg=True)` (dump the latter with `dump_ehtim.py --scan-average`).

Result (2026-09-03): 7447 rows; stations, times (5e-9 h) and baselines matched; u, v to 5e-9
relative, Stokes I to 5e-9 Jy and V to the bit, σ to 1.5e-8 Jy (float32 weights); the file
carries only parallel hands, with infinite cross-hand weights, which the reader reports as
infinite Q, U noise (ehtim: NaN). Scan-averaged: 216 rows over 22 scans; times, stations,
integration times and u, v matched, Stokes I to 7e-9 Jy and σ to 1e-9 Jy.
