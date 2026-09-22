#!/bin/bash
#SBATCH --job-name=kerrsplat-triband
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=runs/%x-%j.log
# The triband ngEHT self-fit on one GPU, resumable: if the tag's checkpoint exists the job continues from it, otherwise it
# starts as configured. Set TAG and EXTRA in the environment (sbatch --export=ALL,TAG=...,EXTRA="...") or edit below.
# Adjust --partition/--account for the cluster; the driver's options are documented at the top of triband_selffit.jl.
set -euo pipefail
export PATH="$HOME/.juliaup/bin:$PATH"
cd "$SLURM_SUBMIT_DIR"
mkdir -p runs
TAG=${TAG:-triband_cluster}
EXTRA=${EXTRA:-}
CKPT=validation/ngeht/output/${TAG}_checkpoint_params.csv
RESUME=""
if [ -f "$CKPT" ]; then
    RESUME="--resume output/${TAG}_checkpoint_params.csv --single-stage 1"
    echo "resuming $TAG from its checkpoint"
fi
cd validation/ngeht
julia -t "$SLURM_CPUS_PER_TASK" --project=../.. triband_selffit.jl --backend cuda --precision Float64 --days 5 --res 64 --samples 160 \
    --shell 300 --iterations 300 --every 25 --frame-hours 4 --batch 8 --closures 0 --tag "$TAG" $RESUME $EXTRA 2>&1 | tee "../../runs/${TAG}.log"
