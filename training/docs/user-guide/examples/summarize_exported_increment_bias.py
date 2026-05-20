#!/usr/bin/env python3
"""Summarize domain-mean increment bias from exported prediction files.

This scans ExportPredictions NetCDF/Zarr outputs and reports, for one variable:

- mean domain-mean model increment: mean(pred - last_input)
- mean domain-mean true increment: mean(target - last_input)
- mean domain-mean prediction error: mean(target - pred)
- average per-case RMSE for model and persistence
- the same quantities in normalized space using the training config

Examples
--------
python training/docs/user-guide/examples/summarize_exported_increment_bias.py \
  /scratch3/.../verify_3h/predictions \
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


def _iter_exports(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(path.glob("pred_target_epoch*_batch*.nc"))
    files.extend(sorted(path.glob("pred_target_epoch*_batch*.zarr")))
    if not files:
        raise FileNotFoundError(f"No pred_target_epoch*_batch*.nc/.zarr files found under {path}")
    return files


def _open_export(path: Path) -> xr.Dataset:
    if path.suffix == ".nc":
        return xr.open_dataset(path)
    if path.suffix == ".zarr":
        return xr.open_zarr(path, consolidated=False)
    raise ValueError(f"Unsupported export file: {path}")


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
        "method": method,
        "mean": mean,
        "stdev": stdev,
        "minimum": minimum,
        "maximum": maximum,
        "scale": float(scale),
    }


def _rmse(arr: np.ndarray) -> float:
    vals = np.asarray(arr)
    finite = np.isfinite(vals)
    if not finite.any():
        return np.nan
    vals = vals[finite]
    return float(np.sqrt(np.mean(vals**2)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_path", type=Path, help="Export file or directory containing pred_target_epoch*.nc/.zarr.")
    parser.add_argument("--variable", required=True, help="Variable to summarize.")
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name without .yaml")
    parser.add_argument("--dataset-name", default="data", help="Dataset key to inspect")
    parser.add_argument("--lead-index", type=int, default=None, help="Optional single lead index. Default: all leads.")
    args = parser.parse_args()

    norm_ctx = _load_normalization_context(
        config_path=Path(args.config_path),
        config_name=args.config_name,
        dataset_name=args.dataset_name,
        variable=args.variable,
    )
    scale = norm_ctx["scale"]
    files = _iter_exports(args.export_path)

    mean_pred_input = []
    mean_target_input = []
    mean_target_pred = []
    rmse_pred_err = []
    rmse_persist_err = []
    times = []

    for path in files:
        ds = _open_export(path)
        names = [str(v) for v in ds.coords["variable"].values]
        if args.variable not in names:
            continue
        var_idx = names.index(args.variable)

        inp = ds["input"].values[:, :, var_idx]
        targ = ds["target"].values[:, :, var_idx]
        pred = ds["prediction"].values[:, :, var_idx]
        last_input = inp[-1]

        if args.lead_index is None:
            lead_indices = range(min(targ.shape[0], pred.shape[0]))
        else:
            lead_indices = [args.lead_index]

        for lead_idx in lead_indices:
            targ_i = targ[lead_idx]
            pred_i = pred[lead_idx]
            pred_minus_input = pred_i - last_input
            target_minus_input = targ_i - last_input
            target_minus_pred = targ_i - pred_i

            mean_pred_input.append(float(np.nanmean(pred_minus_input)))
            mean_target_input.append(float(np.nanmean(target_minus_input)))
            mean_target_pred.append(float(np.nanmean(target_minus_pred)))
            rmse_pred_err.append(_rmse(target_minus_pred))
            rmse_persist_err.append(_rmse(target_minus_input))
            times.append(str(ds["target_time"].values[lead_idx]))

    if not mean_pred_input:
        raise ValueError(f"No usable samples found for variable '{args.variable}' under {args.export_path}")

    mean_pred_input = np.asarray(mean_pred_input, dtype=float)
    mean_target_input = np.asarray(mean_target_input, dtype=float)
    mean_target_pred = np.asarray(mean_target_pred, dtype=float)
    rmse_pred_err = np.asarray(rmse_pred_err, dtype=float)
    rmse_persist_err = np.asarray(rmse_persist_err, dtype=float)

    sign_mismatch = np.sign(mean_pred_input) != np.sign(mean_target_input)
    valid_sign = np.isfinite(mean_pred_input) & np.isfinite(mean_target_input) & (mean_target_input != 0.0)
    mismatch_fraction = float(np.mean(sign_mismatch[valid_sign])) if np.any(valid_sign) else np.nan

    print(f"variable: {args.variable}")
    print(f"files: {len(files)}")
    print(f"cases: {len(mean_pred_input)}")
    print(
        "normalization: "
        f"method={norm_ctx['method']} mean={norm_ctx['mean']:.6g} stdev={norm_ctx['stdev']:.6g} "
        f"minimum={norm_ctx['minimum']:.6g} maximum={norm_ctx['maximum']:.6g} scale={scale:.6g}"
    )
    print(f"training_window: start={norm_ctx['train_start']} end={norm_ctx['train_end']} frequency={norm_ctx['frequency']}")
    print("")
    print(
        f"mean domain-mean(pred-input): raw={np.nanmean(mean_pred_input):.6g} "
        f"norm={np.nanmean(mean_pred_input / scale):.6g}"
    )
    print(
        f"mean domain-mean(target-input): raw={np.nanmean(mean_target_input):.6g} "
        f"norm={np.nanmean(mean_target_input / scale):.6g}"
    )
    print(
        f"mean domain-mean(target-pred): raw={np.nanmean(mean_target_pred):.6g} "
        f"norm={np.nanmean(mean_target_pred / scale):.6g}"
    )
    print(
        f"mean RMSE(pred err): raw={np.nanmean(rmse_pred_err):.6g} "
        f"norm={np.nanmean(rmse_pred_err / scale):.6g}"
    )
    print(
        f"mean RMSE(persist err): raw={np.nanmean(rmse_persist_err):.6g} "
        f"norm={np.nanmean(rmse_persist_err / scale):.6g}"
    )
    print(
        f"ratio mean_RMSE(pred err)/mean_RMSE(persist err) = "
        f"{np.nanmean(rmse_pred_err) / np.nanmean(rmse_persist_err):.6g}"
    )
    print(f"sign_mismatch_fraction(pred-input vs target-input) = {mismatch_fraction:.6g}")


if __name__ == "__main__":
    main()
