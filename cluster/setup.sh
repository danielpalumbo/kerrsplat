#!/bin/bash
# User-level installation of KerrSplat on a cluster login node: juliaup, Julia 1.10, the project's pinned environment,
# the CUDA runtime matching the nodes' driver, precompilation, a check. Usage: bash cluster/setup.sh <cuda runtime, e.g. 12.8>
set -euo pipefail
RUNTIME=${1:?usage: bash cluster/setup.sh <cuda runtime version, e.g. 12.8 (nvidia-smi on a GPU node shows the driver version)>}
cd "$(dirname "$0")/.."
export JULIA_DEPOT_PATH=${JULIA_DEPOT_PATH:-$HOME/.julia}
echo "depot: $JULIA_DEPOT_PATH (set JULIA_DEPOT_PATH to a shared, unpurged filesystem before running this)"
if ! command -v juliaup >/dev/null 2>&1 && [ ! -x "$HOME/.juliaup/bin/juliaup" ]; then
    curl -fsSL https://install.julialang.org | sh -s -- --yes --default-channel lts
fi
export PATH="$HOME/.juliaup/bin:$PATH"
juliaup add lts >/dev/null 2>&1 || true
juliaup default lts
julia --version
julia --project=. -e 'using Pkg; Pkg.instantiate(); Pkg.status()'
julia --project=. -e "using CUDA; CUDA.set_runtime_version!(v\"$RUNTIME\")"
julia --project=. -e 'using Pkg; Pkg.precompile()'
julia -t 8 --project=. cluster/check.jl || echo "check.jl did not pass on this node (no GPU on a login node is the usual reason): run it from a GPU job"
echo "done: run jobs with sbatch cluster/triband.sbatch"
