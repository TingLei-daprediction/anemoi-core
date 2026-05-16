#!/usr/bin/env bash
# Run 1-hour verification and export configured loss tables for every
# validation sample/lead/variable for the base_1 refc-input no-hydrometeors
# finer_graph_v1 single-input setup.
#
# Loss weights match base_1 training exactly:
#   general_variable.weights.refc = 20
#   range_weight_factors = [1, 2, 2, 4]
#   thresholds = [0.0, 20.0, 40.0] dBZ
#
# Usage:
#   run_rrfs_verify_loss_table_1h_refc_value_base_1_refc_input_no_hydrometeors_finer_graph_v1_single_input.sh \
#     <checkpoint_path> <start> <end> <frequency>
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "Usage: run_rrfs_verify_loss_table_1h_refc_value_base_1_refc_input_no_hydrometeors_finer_graph_v1_single_input.sh <checkpoint_path> <start> <end> <frequency>"
  exit 1
fi

CHECKPOINT_PATH="$1"
START="$2"
END="$3"
FREQ="$4"
CONFIG_NAME="anemoi-training-rrfs-lam-neural-lam-verify-202405-1h-refc-value-base_1-refc-input-no-hydrometeors-finer-graph-v1-single-input-loss-table"
RESOLVED_CONFIG="/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tmp-verify-resolved-1h-refc-value-base_1-refc-input-no-hydrometeors-finer-graph-v1-single-input-loss-table.yaml"

echo "DEBUG_CONFIG_NAME: $CONFIG_NAME"
echo "DEBUG_LOSS_TABLE_DIR: /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/base_1_graphtransformer_finer_graph_v1_single_input_refc_input_no_hydrometeors/verify/loss_tables"

export ANEMOI_BASE_SEED="${ANEMOI_BASE_SEED:-12345}"
export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export ANEMOI_LOG_LEVEL="${ANEMOI_LOG_LEVEL:-DEBUG}"
export MPLBACKEND="${MPLBACKEND:-Agg}"

python - <<PY
from anemoi.datasets import open_dataset
path = "/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/tmp/rrfs-monthly/rrfs-conus-3km-202405-bcmask-time-s.zarr"
ds = open_dataset(path, start="${START}", end="${END}", frequency="${FREQ}")
print("DEBUG_DATASET_PATH:", path)
print("DEBUG_DATASET_LEN:", len(ds.dates))
print("DEBUG_DATASET_START:", ds.dates[0])
print("DEBUG_DATASET_END:", ds.dates[-1])
PY

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
  dataloader.num_workers.validation=1 \
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
  dataloader.num_workers.validation=1 \
  training.num_sanity_val_steps=0
