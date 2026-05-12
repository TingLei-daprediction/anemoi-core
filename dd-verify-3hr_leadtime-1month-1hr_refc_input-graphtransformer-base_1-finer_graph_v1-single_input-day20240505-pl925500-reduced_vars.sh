#!/bin/bash
#SBATCH -A fv3-cam
#SBATCH -J anemoi-verify-3h-bg1fgv1
#SBATCH -p u1-h100
#SBATCH -q gpuwf
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=0
#SBATCH -t 8:00:00
#SBATCH -o anemoi-gpu-3hrleadtime-1hr_refc_input-graphtransformer-base_1-finer_graph_v1-single_input-day20240505-pl925500-reduced_vars-verify.%j.out
#SBATCH -e anemoi-gpu-3hrleadtime-1hr_refc_input-graphtransformer-base_1-finer_graph_v1-single_input-day20240505-pl925500-reduced_vars-verify.%j.err

set -euo pipefail

source /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-miniconda3/bin/activate anemoi-training-env-python3.12
cd /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core

run_id="f25324d4-e3cc-4907-a195-90a012bf4eef"
training/docs/user-guide/examples/run_rrfs_verify_export_3h_refc_input_graphtransformer_base_1_finer_graph_v1_single_input_day20240505_pl925500_reduced_vars.sh \
  /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_1_graphtransformer_finer_graph_v1_single_input_day20240505_refc_input_pl925500_reduced_vars/checkpoint/${run_id}/inference-last.ckpt \
  2024-05-05T15:00:00 2024-05-05T20:00:00 1h
