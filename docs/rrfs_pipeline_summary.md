# RRFS Pipeline Summary

This note summarizes the current RRFS/Anemoi workflow as it is used in this repo, from source GRIB handling through graph creation, masked Zarr use, training, and verification.

It is intended as a review checklist for collaborators.

## 1. Prepare valid-time RRFS GRIB files

Goal:
- collapse RRFS cycles/leads into one GRIB per valid time

Expected input layout:
- `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/rrfs-valid/rrfs.vYYYYMMDDHH.grib2`

Referenced by:
- [anemoi-data-lam-rrfs-hres-example.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-data-lam-rrfs-hres-example.yaml)

Important note:
- this repo assumes those valid-time GRIB links/files already exist
- the script that generates `rrfs-valid/` is not clearly present in this repo

## 2. Build the regridding matrix

Goal:
- map RRFS native grid to the lat-lon target grid used in the dataset

Output used by dataset recipes:
- `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/regrid/rrfs-to-latlon-3km.npz`

Command family:
- `anemoi-transform make-regrid-file mir-matrix ...`

Recorded in:
- [session-context.md](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/docs/session-context.md)

Important note:
- this matrix is consumed by both the main RRFS dataset recipe and the additions recipe

## 3. Create the main RRFS Zarr dataset

Goal:
- read RRFS GRIB fields, rename them, regrid them, and write a Zarr on a 1-hour valid-time axis

Main recipe:
- [anemoi-data-lam-rrfs-hres-example.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-data-lam-rrfs-hres-example.yaml)

What it reads:
- pressure-level fields:
  - `u, v, t, q, gh, clwmr, icmr, rwmr, snmr, grle`
- surface fields:
  - `sp, tmp`
- reflectivity:
  - composite reflectivity only, renamed to `refc`
- static/forcing:
  - `sdswrf -> swdown`
  - `lsm -> landcover`
  - `orog -> orography`

What it does:
- rename variables to Anemoi names
- regrid using `rrfs-to-latlon-3km.npz`
- write a monthly Zarr

Typical output concept:
- `rrfs-conus-3km-202405.zarr`

## 4. Add static/forcing fields to an existing Zarr

Goal:
- append `landcover`, `orography`, and `swdown` if needed through the additions workflow

Recipe:
- [anemoi-data-rrfs-additions.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-data-rrfs-additions.yaml)

Commands:
- `anemoi-datasets init-additions ...`
- `anemoi-datasets load-additions ...`
- `anemoi-datasets finalise-additions ...`

Important note:
- this is an auxiliary update path, not the main monthly create recipe

## 5. Boundary-mask / bcmask dataset stage

Goal:
- provide `boundary_mask` and the masked-domain dataset used by graph building and training

Observed dataset names in this workflow:
- `test-20km-bcmask.zarr`
- `test-20km-bcmask-time-s.zarr`
- `rrfs-conus-3km-202405-bcmask-time-s.zarr`

Used by:
- graph YAMLs
- RRFS training YAMLs

Important note:
- in this repo these datasets are clearly being used
- the exact script/config that originally creates the `*-bcmask*.zarr` products is not clearly identified here
- this is an important review point: confirm where `boundary_mask` is created and how the `*-time-s.zarr` variants are produced

## 6. Build the LAM graph

Goal:
- create the data/hidden graph used by Anemoi

Baseline graph config:
- [rrfs-lam-graph.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/graphs/docs/usage/yaml/rrfs-lam-graph.yaml)

Finer graph config:
- [rrfs-lam-graph-finer_graph_v1.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/graphs/docs/usage/yaml/rrfs-lam-graph-finer_graph_v1.yaml)

Key graph details:
- data nodes come from the masked dataset Zarr
- node attributes include:
  - `boundary_mask`
  - `cutout_mask`
  - `area_weight`
- graph postprocessor removes unconnected data nodes and stores:
  - `indices_connected_nodes`

That last attribute is important because training uses it for correct masked-grid indexing.

Documented in:
- [graph-finer_graph_v1.md](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/docs/graph-finer_graph_v1.md)

## 7. Train using the bcmask-time-s Zarr plus graph

Goal:
- run Anemoi training on the masked regional dataset

Main RRFS training base:
- [anemoi-training-rrfs-lam-neural-lam-static-forcing-202405.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405.yaml)

Important training mechanics:
- dataset:
  - `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/dr-anemoi-core/anemoi-core/tmp/rrfs-monthly/rrfs-conus-3km-202405-bcmask-time-s.zarr`
- graph:
  - `/scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/graphs/rrfs-3km-lam-graph-20km.pt`
  - or a finer-graph override
- `boundary_mask` is dropped from train/validation/test variable lists
- dataloader uses:
  - `MaskedGrid`
  - `node_attribute_name: indices_connected_nodes`

That `MaskedGrid + indices_connected_nodes` combination is critical. It prevents graph/data spatial mismatch.

## 8. Specialize experiments by overriding the training YAML

Goal:
- define scientific variants without rewriting the whole base config

Examples:
- no-refc-input
- refc-input
- finer graph
- reduced vars
- one-day vs one-month
- base / base_1 / base_2
- diffusion vs GraphTransformer

Examples currently used:
- [anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml)
- [anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base_1-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base_1-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml)
- [anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base_2-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base_2-refc-input-no-hydrometeors-finer-graph-v1-single-input.yaml)

## 9. Launch training with sbatch wrappers

Goal:
- provide the exact HPC launch command and graph symlink handling

Examples:
- [d-2GPU-1hr-refc_value-base-refc_input-no_hydrometeors-finer_graph_v1-single_input-202405.sh](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/d-2GPU-1hr-refc_value-base-refc_input-no_hydrometeors-finer_graph_v1-single_input-202405.sh)
- [d-2GPU-1hr-refc_value-base_1-refc_input-no_hydrometeors-finer_graph_v1-single_input-202405.sh](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/d-2GPU-1hr-refc_value-base_1-refc_input-no_hydrometeors-finer_graph_v1-single_input-202405.sh)
- [d-2GPU-1hr-refc_value-base_2-refc_input-no_hydrometeors-finer_graph_v1-single_input-202405.sh](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/d-2GPU-1hr-refc_value-base_2-refc_input-no_hydrometeors-finer_graph_v1-single_input-202405.sh)

These wrappers typically:
- activate the conda env
- `cd` into the scratch repo
- create the expected `_data.pt` graph symlink if needed
- run `anemoi-training train ...`

## 10. Verify / export diagnostics

Goal:
- load checkpoint, run validation rollout, save plots and exported prediction files

Generic verify YAMLs:
- [anemoi-training-rrfs-lam-neural-lam-verify.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify.yaml)
- [anemoi-training-rrfs-lam-neural-lam-verify-202405.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify-202405.yaml)
- [anemoi-training-rrfs-lam-neural-lam-verify-202405-1to3h.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-training-rrfs-lam-neural-lam-verify-202405-1to3h.yaml)

Per-experiment verify wrappers and sbatch scripts exist for each run family.

## Current Known Review Points

1. `rrfs-valid` creation is external
- the repo assumes valid-time GRIBs already exist

2. `boundary_mask` / `*-bcmask*.zarr` creation is not clearly fully scripted here
- the repo uses those products heavily
- this is an important place for review

3. The graph/training interface depends critically on:
- `boundary_mask`
- `cutout_mask`
- `indices_connected_nodes`
- `MaskedGrid`

4. Verification/export had real bugs recently
- plotting and export both had data-output vs model-output index mismatches
- verify workflow also had to be changed so verification is inference-only

## Short Pipeline Version

1. Build or obtain valid-time RRFS GRIB files in `rrfs-valid/`
2. Build `rrfs-to-latlon-3km.npz`
3. Create monthly RRFS Zarr with [anemoi-data-lam-rrfs-hres-example.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-data-lam-rrfs-hres-example.yaml)
4. Add static/forcing fields with [anemoi-data-rrfs-additions.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/training/docs/user-guide/examples/anemoi-data-rrfs-additions.yaml) if needed
5. Produce or use `*-bcmask*.zarr` datasets containing `boundary_mask`
6. Build graph with [rrfs-lam-graph.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/graphs/docs/usage/yaml/rrfs-lam-graph.yaml) or [rrfs-lam-graph-finer_graph_v1.yaml](/c:/Users/Ting.Lei/Documents/GitHub/dr-anemoi/anemoi-core/graphs/docs/usage/yaml/rrfs-lam-graph-finer_graph_v1.yaml)
7. Train with `anemoi-training-rrfs-lam-neural-lam-static-forcing-202405*.yaml` variants and `d-*.sh` sbatch wrappers
8. Verify/export with `anemoi-training-rrfs-lam-neural-lam-verify*.yaml` plus `run_rrfs_verify_export_*.sh` and `dd-verify-*.sh`
