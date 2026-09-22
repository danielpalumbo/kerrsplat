# Running KerrSplat on a cluster

A user-level installation (no root), one GPU per job, the long fits with checkpoints so that a
job cut short resumes. Written for a SLURM cluster with H200 nodes; the workstation numbers are
from the RTX 2080 SUPER that ran everything so far (`docs/HANDOFF.md`).

## Install (once, on a login node with internet)

```
git clone https://github.com/danielpalumbo/kerrsplat.git && cd kerrsplat
bash cluster/setup.sh 12.8          # the CUDA runtime version the nodes' driver supports (nvidia-smi on a GPU node)
```

`setup.sh` installs juliaup in `$HOME/.juliaup`, Julia 1.10 (the `lts` channel), instantiates
the project (the Manifest pins Krang.jl to the git commit `f36f43a` and Enzyme to 0.13.199;
nothing is added by hand), pins the CUDA runtime with `CUDA.set_runtime_version!`, precompiles,
and runs `cluster/check.jl`. Set `JULIA_DEPOT_PATH` to a filesystem the compute nodes share
and that is not purged (the depot holds the packages, the artifacts and the compiled caches;
several gigabytes). If the login node has no GPU, run `check.jl` from a GPU job afterwards:

```
srun --gres=gpu:1 --time=00:20:00 julia -t 8 --project=. cluster/check.jl
```

The CUDA runtime version must match the nodes' driver (the committed `LocalPreferences.toml`
says 12.8 for the workstation's driver 570; `setup.sh` overwrites it). CUDA.jl downloads its
runtime artifact for that version on first use, so the first GPU run needs internet or a
pre-warmed depot.

## Data

Not in the repository (`docs/HANDOFF.md`, "Data that is not in the repository"). For the triband
ngEHT fit, copy `validation/ngeht/output/ngeht_M87_*.uvfits` (fifteen files, 3 MB; or regenerate
them with `validation/ngeht/make_campaign.py`, which needs `ngehtsim` in a Python environment)
and any state to resume from (`<tag>_params.csv` or `<tag>_checkpoint_params.csv`).

## Jobs

`cluster/triband.sbatch` is the template: one GPU, the driver run through `tee` so its log
is readable while it runs, and `--resume` of the tag's checkpoint whenever one exists, so a
requeued or restarted job continues where the last one stopped (the driver writes
`<tag>_checkpoint_params.csv` every twenty Adam iterations and every polish step, atomically).
Adjust the partition, account and time limit; keep `-t` at the node's core count for the host
seeds and the Jacobian's host side.

```
sbatch cluster/triband.sbatch                       # the defaults in the file
TAG=triband_h200 EXTRA="--free-spacetime 1" sbatch cluster/triband.sbatch
```

## What to expect

- Float64 is the precision of the geodesics; the transport runs in Float32 on the workstation
  (`--precision Float32`, nine mechanisms in `docs/notes/2026-09-12_fp32_transport.md`) because
  the consumer card's FP64 rate is 1/32. On an H200 Float64 throughout is the right choice
  (`--precision Float64`): no accuracy questions, and the FP64 rate is half of FP32.
- The workstation's timings for the full triband fit (64² × 160 samples, three bands, 241,264
  values, 70–300 parcels): an Adam iteration 57 s, an LSQR polish step of forty solves 20 min,
  an explicit Jacobian 55 min, a chord step on it 3 s. An H200 at Float64 should be several
  times faster on the kernels; the host side (the Jacobian's Float64 normal matrix, the
  per-scan calibrations) scales with the cores.
- The joint fit with the spacetime free (`--free-spacetime 1`) regenerates the geodesics every
  iteration and runs in Float64: ten hours or more per three hundred iterations on the
  workstation, the first job for a cluster node (`docs/HANDOFF.md`, the wishlist).
- No new kernels are needed: KernelAbstractions compiles the same code for any CUDA card; the
  Enzyme device gates need a free card (they reserve large per-thread stacks), the dual sweep
  does not.

## Checks

`julia -t 8 --project=. test/ci.jl` runs the CPU gates (about fifteen minutes with compiles);
the full suite (`test/runtests.jl`, over an hour, the CUDA gates included) goes through a pipe
(`2>&1 | tee suite.log`), since output redirected to a file is buffered until exit.
