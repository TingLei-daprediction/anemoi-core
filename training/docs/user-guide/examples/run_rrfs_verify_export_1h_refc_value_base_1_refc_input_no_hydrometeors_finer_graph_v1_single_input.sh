#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 || $# -gt 5 ]]; then
  echo "Usage: $0 <checkpoint_path> <start_iso> <end_iso> <frequency> [warm_start_iso]" >&2
  exit 2
fi

CHECKPOINT_PATH="$1"
START_TIME="$2"
END_TIME="$3"
FREQUENCY="$4"
WARM_START="${5:-$START_TIME}"

CONFIG_PATH="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/training/docs/user-guide/examples"
CONFIG_NAME="anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-base_1-refc-input-no-hydrometeors-finer-graph-v1-single-input"

echo "Running anemoi training command to dump the composed config..."
anemoi-training train \
  --config-path "$CONFIG_PATH" \
  --config-name "$CONFIG_NAME" \
  --cfg job

echo "Running anemoi training command..."
anemoi-training train \
  --config-path "$CONFIG_PATH" \
  --config-name "$CONFIG_NAME" \
  system.input.warm_start="$WARM_START" \
  system.input.checkpoint="$CHECKPOINT_PATH" \
  dataloader.training.datasets.data.start="$START_TIME" \
  dataloader.training.datasets.data.end="$END_TIME" \
  dataloader.validation.datasets.data.start="$START_TIME" \
  dataloader.validation.datasets.data.end="$END_TIME" \
  dataloader.test.datasets.data.start="$START_TIME" \
  dataloader.test.datasets.data.end="$END_TIME" \
  data.frequency="$FREQUENCY" \
  data.timestep="$FREQUENCY"
