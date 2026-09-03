# ipole reference images of its RIAF model

Full Stokes images produced by ipole (AFD-Illinois/ipole, `v1.5-5-g198adfc`, GPL; not copied here)
from its `model/riaf` with the parameters of `model/riaf/example.par` (a = 0.9375, thermal
synchrotron with `emission_type 1`: Pandya+ 2016 emissivities, Kirchhoff absorptivities, Dexter
2016 ρ_Q and Shcherbakov 2008 ρ_V; θ = 85°, 200 μas across at 8.3 kpc for M = 4.3e6 M☉, 100²
pixels). `test/riaf_model.jl` reproduces the model and `test/test_polarized.jl` compares.

Build (in a copy of the ipole checkout; HDF5 and GSL from conda, see the Phase 2 design note):

    mamba create -n ipole-build -c conda-forge hdf5 gsl
    rm -rf build_archive && make MODEL=riaf CC=gcc HDF5_DIR=$HOME/miniforge-pypy3/envs/ipole-build GSL_DIR=/usr

Runs (`LD_LIBRARY_PATH=$HOME/miniforge-pypy3/envs/ipole-build/lib`):

- `riaf_fine_*.csv`: `ipole -par riaf_fine.par --rcam=10000 --eps=0.002` at 230 GHz (camera at
  10⁴ M and ten times finer geodesic steps than the defaults, closer to KerrSplat's camera at
  infinity and its converged sampling; a run with ipole's defaults differs from it by up to 0.3%
  in the totals).
- `riaf_2thz_*.csv`: the same at 2 THz, where Faraday rotation and conversion are negligible
  (τ_F < 0.12), isolating the intrinsic polarization and its parallel transport.

The CSVs hold the `pol` dataset of ipole's HDF5 output (I, Q, U, V in cgs specific intensity),
row = image x index, column = image y index. The flux per pixel is intensity × `scale` with
`scale` = (pixel solid angle)/Jy = 9.4018 for this camera.
