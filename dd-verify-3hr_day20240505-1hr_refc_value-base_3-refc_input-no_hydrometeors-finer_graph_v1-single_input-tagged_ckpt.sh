#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-verify-3h-base3d
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o anemoi-gpu-day20240505-3hrleadtime-1hr_refc_value-base_3-refc_input-no_hydrometeors-finer_graph_v1-single_input-tagged_ckpt-verify.%j.out
#SBATCH -e anemoi-gpu-day20240505-3hrleadtime-1hr_refc_value-base_3-refc_input-no_hydrometeors-finer_graph_v1-single_input-tagged_ckpt-verify.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

# Set this to the exact checkpoint you want to verify, for example:
# /scratch3/.../checkpoint/<run_id>/inference-last.ckpt
CHECKPOINT_PATH="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_3_graphtransformer_finer_graph_v1_single_input_refc_input_no_hydrometeors/checkpoint/51951d18-0807-4125-819e-4d81741c716a/inference-anemoi-by_epoch-epoch_030-step_000713.ckpt"
test -f "$CHECKPOINT_PATH"

training/docs/user-guide/examples/run_rrfs_verify_export_3h_refc_value_base_3_refc_input_no_hydrometeors_finer_graph_v1_single_input_tagged_ckpt.sh \
  "$CHECKPOINT_PATH" \
  2024-05-05T00:00:00 2024-05-05T23:00:00 1h
