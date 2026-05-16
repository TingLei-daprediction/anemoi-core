#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <checkpoint_path> <start_iso> <end_iso> <frequency>" >&2
  exit 1
fi

CHECKPOINT_PATH="$1"
START_TIME="$2"
END_TIME="$3"
FREQUENCY="$4"

CONFIG_PATH="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples"
CONFIG_NAME="anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-base_3-refc-input-no-hydrometeors-finer-graph-v1-single-input"

echo "DEBUG_CONFIG_NAME: $CONFIG_NAME"
echo "DEBUG_VERIFY_ROOT: /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_3_graphtransformer_finer_graph_v1_single_input_refc_input_no_hydrometeors/verify/"

export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-12345}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ANEMOI_LOG_LEVEL="${ANEMOI_LOG_LEVEL:-DEBUG}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

echo "Running anemoi training command to dump the composed config..."
anemoi-training train \
  --config-path "$CONFIG_PATH" \
  --config-name "$CONFIG_NAME" \
  --cfg job

echo "Running anemoi training command..."
anemoi-training train \
  --config-path "$CONFIG_PATH" \
  --config-name "$CONFIG_NAME" \
  system.input.warm_start="$CHECKPOINT_PATH" \
  dataloader.training.datasets.data.start="$START_TIME" \
  dataloader.training.datasets.data.end="$END_TIME" \
  dataloader.validation.datasets.data.start="$START_TIME" \
  dataloader.validation.datasets.data.end="$END_TIME" \
  dataloader.test.datasets.data.start="$START_TIME" \
  dataloader.test.datasets.data.end="$END_TIME" \
  data.frequency="$FREQUENCY" \
  data.timestep="$FREQUENCY"
