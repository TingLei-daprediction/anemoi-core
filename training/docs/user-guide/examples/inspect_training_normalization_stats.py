"""Inspect the normalization statistics actually used by training.

This script composes a Hydra training config, opens the configured dataset
with the same drop/start/end/frequency settings as the training dataloader,
builds the same IndexCollection/InputNormalizer objects used by training, and
prints the stored dataset statistics that drive normalization.

The key point: these are the statistics used by training-time normalization,
not recomputed from each batch.

Example
-------
python training/docs/user-guide/examples/inspect_training_normalization_stats.py ^
  --config-path training/docs/user-guide/examples ^
  --config-name anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base_3-refc-input-no-hydrometeors-finer-graph-v1-single-input
"""

from __future__ import annotations

import argparse
from pathlib import Path

from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.training.data.dataset import NativeGridDataset


def _compose_config(config_dir: Path, config_name: str):
    with initialize_config_dir(version_base=None, config_dir=str(config_dir.resolve())):
        cfg = compose(config_name=config_name)
    OmegaConf.resolve(cfg)
    return cfg


def _get_train_dataset_cfg(cfg, dataset_name: str):
    training_cfg = cfg.dataloader.training
    if training_cfg.get("datasets") is not None:
        return training_cfg.datasets[dataset_name]
    return training_cfg


def _get_dataset_path(cfg, dataset_name: str) -> str:
    train_ds_cfg = _get_train_dataset_cfg(cfg, dataset_name)
    dataset_cfg = train_ds_cfg.get("dataset_config")
    if dataset_cfg is not None and dataset_cfg.get("dataset") is not None:
        return dataset_cfg.dataset
    if train_ds_cfg.get("dataset") is not None:
        return train_ds_cfg.dataset
    if cfg.dataloader.get("dataset") is not None:
        return cfg.dataloader.dataset
    return cfg.system.input.dataset


def _get_role(var: str, data_cfg) -> str:
    forcing = set(OmegaConf.to_container(data_cfg.forcing, resolve=True) or [])
    diagnostic = set(OmegaConf.to_container(data_cfg.diagnostic, resolve=True) or [])
    target = set(OmegaConf.to_container(data_cfg.get("target"), resolve=True) or [])
    if var in forcing:
        return "forcing"
    if var in diagnostic:
        return "diagnostic"
    if var in target:
        return "target-only"
    return "prognostic"


def _normalization_method(normalizer: InputNormalizer, var: str) -> str:
    return normalizer.methods.get(var, normalizer.default)


def _format(value) -> str:
    try:
        return f"{float(value):.6g}"
    except Exception:
        return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name without .yaml")
    parser.add_argument("--dataset-name", default="data", help="Dataset key to inspect")
    parser.add_argument(
        "--variables",
        nargs="*",
        default=None,
        help="Optional subset of variables. Default: print all loaded dataset variables.",
    )
    args = parser.parse_args()

    cfg = _compose_config(Path(args.config_path), args.config_name)
    dataset_name = args.dataset_name
    data_cfg = cfg.data.datasets[dataset_name]
    train_ds_cfg = _get_train_dataset_cfg(cfg, dataset_name)

    dataset_path = _get_dataset_path(cfg, dataset_name)
    start = train_ds_cfg.start
    end = train_ds_cfg.end
    frequency = train_ds_cfg.get("frequency") or cfg.data.frequency
    drop = OmegaConf.to_container(train_ds_cfg.drop, resolve=True) if train_ds_cfg.get("drop") is not None else None

    reader = NativeGridDataset(dataset=dataset_path, start=start, end=end, frequency=frequency, drop=drop)
    data_indices = IndexCollection(data_config=data_cfg, name_to_index=reader.name_to_index)
    normalizer_cfg = data_cfg.processors.normalizer.config
    normalizer = InputNormalizer(config=normalizer_cfg, data_indices=data_indices, statistics=reader.statistics)

    variables = list(reader.name_to_index.keys())
    if args.variables:
        requested = set(args.variables)
        variables = [v for v in variables if v in requested]
        missing = [v for v in args.variables if v not in reader.name_to_index]
        if missing:
            print(f"MISSING_VARIABLES: {missing}")

    print(f"CONFIG_NAME: {args.config_name}")
    print(f"DATASET_NAME: {dataset_name}")
    print(f"DATASET_PATH: {dataset_path}")
    print(f"TRAIN_WINDOW: start={start} end={end} frequency={frequency}")
    print(f"DROP: {drop}")
    print("NOTE: mean/stdev/min/max below come from stored dataset statistics metadata used by InputNormalizer.")
    print("")
    print(
        "variable,role,normalization,mean,stdev,minimum,maximum,"
        "model_input_idx,data_input_idx,data_output_idx,model_output_idx"
    )

    stats = reader.statistics
    model_input_name_to_index = data_indices.model.input.name_to_index
    model_output_name_to_index = data_indices.model.output.name_to_index
    data_input_name_to_index = data_indices.data.input.name_to_index
    data_output_name_to_index = data_indices.data.output.name_to_index

    for var in variables:
        full_idx = reader.name_to_index[var]
        role = _get_role(var, data_cfg)
        method = _normalization_method(normalizer, var)
        model_input_idx = model_input_name_to_index.get(var, "")
        data_input_idx = data_input_name_to_index.get(var, "")
        data_output_idx = data_output_name_to_index.get(var, "")
        model_output_idx = model_output_name_to_index.get(var, "")
        print(
            ",".join(
                [
                    var,
                    role,
                    method,
                    _format(stats["mean"][full_idx]),
                    _format(stats["stdev"][full_idx]),
                    _format(stats["minimum"][full_idx]),
                    _format(stats["maximum"][full_idx]),
                    str(model_input_idx),
                    str(data_input_idx),
                    str(data_output_idx),
                    str(model_output_idx),
                ]
            )
        )


if __name__ == "__main__":
    main()
