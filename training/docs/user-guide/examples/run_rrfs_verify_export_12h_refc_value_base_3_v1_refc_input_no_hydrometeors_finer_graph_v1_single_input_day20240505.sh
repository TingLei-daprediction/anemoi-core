#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: $0 <checkpoint_path> <start> <end> <frequency>" >&2
  exit 1
fi

CHECKPOINT_PATH="$1"
START="$2"
END="$3"
FREQ="$4"
CONFIG_NAME="anemoi-training-rrfs-lam-neural-lam-verify-202405-12h-refc-value-base_3_v1-refc-input-no-hydrometeors-finer-graph-v1-single-input-day20240505"
RESOLVED_CONFIG="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tmp-verify-resolved-day20240505-12h-refc-value-base_3_v1-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml"

echo "DEBUG_CONFIG_NAME: $CONFIG_NAME"
echo "DEBUG_VERIFY_ROOT: /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_3_v1_graphtransformer_finer_graph_v1_single_input_refc_input_no_hydrometeors/verify_12h_day20240505/"

export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-12345}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ANEMOI_LOG_LEVEL="${ANEMOI_LOG_LEVEL:-DEBUG}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

anemoi-training train \
  --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
  --config-name "$CONFIG_NAME" \
  system.input.warm_start="$CHECKPOINT_PATH" \
  dataloader.training.datasets.data.start="$START" \
  dataloader.training.datasets.data.end="$END" \
  dataloader.validation.datasets.data.start="$START" \
  dataloader.validation.datasets.data.end="$END" \
  dataloader.test.datasets.data.start="$START" \
  dataloader.test.datasets.data.end="$END" \
  data.frequency="$FREQ" \
  dataloader.training.datasets.data.frequency="$FREQ" \
  dataloader.validation.datasets.data.frequency="$FREQ" \
  dataloader.test.datasets.data.frequency="$FREQ" \
  training.num_sanity_val_steps=0 \
  --cfg job > "$RESOLVED_CONFIG"

echo "DEBUG_CONFIG_SAVED: $RESOLVED_CONFIG"

anemoi-training train \
  --config-path /scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples \
  --config-name "$CONFIG_NAME" \
  system.input.warm_start="$CHECKPOINT_PATH" \
  dataloader.training.datasets.data.start="$START" \
  dataloader.training.datasets.data.end="$END" \
  dataloader.validation.datasets.data.start="$START" \
  dataloader.validation.datasets.data.end="$END" \
  dataloader.test.datasets.data.start="$START" \
  dataloader.test.datasets.data.end="$END" \
  data.frequency="$FREQ" \
  dataloader.training.datasets.data.frequency="$FREQ" \
  dataloader.validation.datasets.data.frequency="$FREQ" \
  dataloader.test.datasets.data.frequency="$FREQ" \
  training.num_sanity_val_steps=0
