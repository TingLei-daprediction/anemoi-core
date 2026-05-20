#!/usr/bin/env python3
"""
Numerically inspect exported Anemoi prediction files.

This compares:
  - prediction - last input
  - target - last input
  - target - prediction

so it is easy to tell whether the model is behaving like persistence.

If a Hydra training config is provided, it also reports the same statistics in
the training normalization space, which is useful for checking whether a large
raw-space increment is only a modest sigma-scale bias.

Example:
  python training/docs/user-guide/examples/check_exported_prediction_increments.py \
    /scratch3/NCEPDEV/fv3-cam/Ting.Lei/tlei-anemoi-training/verify/predictions/pred_target_epoch000_batch0000.nc \
    --variable temp_850

  python training/docs/user-guide/examples/check_exported_prediction_increments.py \
    /scratch3/.../pred_target_epoch000_batch0000.nc \
    --variable refc \
    --config-path training/docs/user-guide/examples \
    --config-name anemoi-training-rrfs-lam-neural-lam-static-forcing-202405-1h-refc-value-base_3-refc-input-no-hydrometeors-finer-graph-v1-single-input
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr
from hydra import compose
from hydra import initialize_config_dir
from omegaconf import OmegaConf

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.training.data.dataset import NativeGridDataset


def _open_export(path: Path) -> xr.Dataset:
    if path.suffix == ".nc":
        return xr.open_dataset(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path, consolidated=False)
    raise ValueError(f"Unsupported export file: {path}")


def _select_var(ds: xr.Dataset, var: str) -> int:
    if "variable" not in ds.coords:
        raise ValueError("Export file is missing 'variable' coordinate.")
    names = list(ds.coords["variable"].values)
    if var not in names:
        raise ValueError(f"Variable '{var}' not found. Available: {names}")
    return names.index(var)


def _stats(name: str, arr: np.ndarray) -> str:
    finite = np.isfinite(arr)
    if not finite.any():
        return f"{name}: all-nonfinite"
    vals = arr[finite]
    rms = float(np.sqrt(np.mean(vals**2)))
    mean_abs = float(np.mean(np.abs(vals)))
    return (
        f"{name}: mean={float(np.mean(vals)):.6g} "
        f"mean_abs={mean_abs:.6g} rms={rms:.6g} "
        f"min={float(np.min(vals)):.6g} max={float(np.max(vals)):.6g}"
    )


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


def _load_normalization_context(config_path: Path, config_name: str, dataset_name: str, variable: str) -> dict:
    cfg = _compose_config(config_path, config_name)
    data_cfg = cfg.data.datasets[dataset_name]
    train_ds_cfg = _get_train_dataset_cfg(cfg, dataset_name)

    dataset_path = _get_dataset_path(cfg, dataset_name)
    start = train_ds_cfg.start
    end = train_ds_cfg.end
    frequency = train_ds_cfg.get("frequency") or cfg.data.frequency
    drop = OmegaConf.to_container(train_ds_cfg.drop, resolve=True) if train_ds_cfg.get("drop") is not None else None

    reader = NativeGridDataset(dataset=dataset_path, start=start, end=end, frequency=frequency, drop=drop)
    if variable not in reader.name_to_index:
        raise ValueError(f"Variable '{variable}' not found in dataset. Available: {list(reader.name_to_index.keys())}")

    data_indices = IndexCollection(data_config=data_cfg, name_to_index=reader.name_to_index)
    normalizer = InputNormalizer(
        config=data_cfg.processors.normalizer.config,
        data_indices=data_indices,
        statistics=reader.statistics,
    )

    full_idx = reader.name_to_index[variable]
    method = normalizer.methods.get(variable, normalizer.default)
    stats = reader.statistics
    minimum = float(stats["minimum"][full_idx])
    maximum = float(stats["maximum"][full_idx])
    mean = float(stats["mean"][full_idx])
    stdev = float(stats["stdev"][full_idx])

    if method in {"mean-std", "std"}:
        scale = stdev
    elif method == "min-max":
        scale = maximum - minimum
    elif method == "max":
        scale = maximum
    elif method == "none":
        scale = 1.0
    else:
        raise ValueError(f"Unsupported normalization method for {variable}: {method}")

    if not np.isfinite(scale) or scale == 0.0:
        raise ValueError(f"Invalid normalization scale for {variable}: method={method}, scale={scale}")

    return {
        "dataset_path": dataset_path,
        "train_start": start,
        "train_end": end,
        "frequency": frequency,
        "drop": drop,
        "method": method,
        "mean": mean,
        "stdev": stdev,
        "minimum": minimum,
        "maximum": maximum,
        "scale": float(scale),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("export", type=Path, help="Exported NetCDF/Zarr from ExportPredictions.")
    p.add_argument("--variable", required=True, help="Variable name to inspect.")
    p.add_argument("--config-path", help="Optional Hydra config directory for normalized-space stats.")
    p.add_argument("--config-name", help="Optional Hydra config name for normalized-space stats.")
    p.add_argument("--dataset-name", default="data", help="Dataset key to inspect in the training config.")
    p.add_argument(
        "--time-index",
        type=int,
        default=None,
        help="Optional single target/pred time index. Default: inspect all available times.",
    )
    args = p.parse_args()

    ds = _open_export(args.export)
    var_idx = _select_var(ds, args.variable)
    norm_ctx = None
    if args.config_path or args.config_name:
        if not (args.config_path and args.config_name):
            raise ValueError("Both --config-path and --config-name are required for normalized-space stats.")
        norm_ctx = _load_normalization_context(
            config_path=Path(args.config_path),
            config_name=args.config_name,
            dataset_name=args.dataset_name,
            variable=args.variable,
        )

    inp = ds["input"].values
    targ = ds["target"].values
    pred = ds["prediction"].values

    # ExportPredictions writes:
    # input: (input_time, node, variable)
    # target/prediction: (target_time|pred_time, node, variable)
    inp = inp[:, :, var_idx]
    targ = targ[:, :, var_idx]
    pred = pred[:, :, var_idx]

    last_input = inp[-1]

    if args.time_index is None:
        indices = range(min(targ.shape[0], pred.shape[0]))
    else:
        indices = [args.time_index]

    print(f"file: {args.export}")
    print(f"variable: {args.variable}")
    print(f"input_time[-1]: {str(ds['input_time'].values[-1])}")
    if norm_ctx is not None:
        print(
            "normalization: "
            f"method={norm_ctx['method']} mean={norm_ctx['mean']:.6g} stdev={norm_ctx['stdev']:.6g} "
            f"minimum={norm_ctx['minimum']:.6g} maximum={norm_ctx['maximum']:.6g} scale={norm_ctx['scale']:.6g}"
        )
        print(
            "training_window: "
            f"start={norm_ctx['train_start']} end={norm_ctx['train_end']} frequency={norm_ctx['frequency']}"
        )

    for i in indices:
        target_time = str(ds["target_time"].values[i])
        pred_time = str(ds["pred_time"].values[i])
        targ_i = targ[i]
        pred_i = pred[i]

        pred_minus_input = pred_i - last_input
        target_minus_input = targ_i - last_input
        target_minus_pred = targ_i - pred_i

        print()
        print(f"time_index={i} target_time={target_time} pred_time={pred_time}")
        print(_stats("pred-input", pred_minus_input))
        print(_stats("target-input", target_minus_input))
        print(_stats("target-pred", target_minus_pred))
        print(_stats("persist-err[target-input]", target_minus_input))
        print(_stats("pred-err[target-pred]", target_minus_pred))

        if norm_ctx is not None:
            scale = norm_ctx["scale"]
            print(_stats("pred-input[norm]", pred_minus_input / scale))
            print(_stats("target-input[norm]", target_minus_input / scale))
            print(_stats("target-pred[norm]", target_minus_pred / scale))
            print(_stats("persist-err[norm]", target_minus_input / scale))
            print(_stats("pred-err[norm]", target_minus_pred / scale))

        denom = np.mean(np.abs(target_minus_input[np.isfinite(target_minus_input)]))
        numer = np.mean(np.abs(pred_minus_input[np.isfinite(pred_minus_input)]))
        if np.isfinite(denom) and denom > 0:
            print(f"ratio mean_abs(pred-input) / mean_abs(target-input) = {numer / denom:.6g}")
        else:
            print("ratio mean_abs(pred-input) / mean_abs(target-input) = n/a")


if __name__ == "__main__":
    main()
