#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-verify-12h-b3v3d
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o anemoi-gpu-day20240505-12hrleadtime-1hr_refc_value-base_3_v3-refc_input-no_hydrometeors-finer_graph_v1-single_input-verify.%j.out
#SBATCH -e anemoi-gpu-day20240505-12hrleadtime-1hr_refc_value-base_3_v3-refc_input-no_hydrometeors-finer_graph_v1-single_input-verify.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

checkpoint_root="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_3_v3_graphtransformer_finer_graph_v1_single_input_refc_input_no_hydrometeors/checkpoint"
if [[ ! -d "$checkpoint_root" ]]; then
  echo "ERROR: checkpoint directory not found: $checkpoint_root" >&2
  exit 1
fi
checkpoint_path="$(find "$checkpoint_root" -mindepth 2 -maxdepth 2 -name last.ckpt -type f -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "$checkpoint_path" ]]; then
  echo "ERROR: no last.ckpt found under $checkpoint_root" >&2
  exit 1
fi
echo "Using checkpoint: $checkpoint_path"

training/docs/user-guide/examples/run_rrfs_verify_export_12h_refc_value_base_3_v3_refc_input_no_hydrometeors_finer_graph_v1_single_input_day20240505.sh \
  "$checkpoint_path" \
  2024-05-05T00:00:00 2024-05-05T23:00:00 1h
