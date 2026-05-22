#!/usr/bin/env python3
"""Summarize persistence-like behavior for every exported variable.

This scans ExportPredictions NetCDF/Zarr outputs and emits one CSV row per
variable so it is easy to see whether a run collapsed to persistence only for a
few variables or for nearly all of them.

Reported metrics are aggregated over all selected exported cases:

- mean_abs(pred-input)
- mean_abs(target-input)
- mean_abs(pred-err)
- mean_abs(persist-err)
- mean RMSE(pred err)
- mean RMSE(persist err)
- ratios against the true increment / persistence baseline
- mean domain-mean increments and errors

All of the above are reported in both raw space and normalization space using
the stored dataset statistics from the training config.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from anemoi.models.data_indices.collection import IndexCollection
from anemoi.models.preprocessing.normalizer import InputNormalizer
from anemoi.training.data.dataset import NativeGridDataset
from summarize_exported_increment_bias import _compose_config
from summarize_exported_increment_bias import _get_dataset_path
from summarize_exported_increment_bias import _get_train_dataset_cfg
from summarize_exported_increment_bias import _iter_exports
from summarize_exported_increment_bias import _open_export
from summarize_exported_increment_bias import _rmse
from omegaconf import OmegaConf


def _mean_abs(arr: np.ndarray) -> float:
    vals = np.asarray(arr)
    finite = np.isfinite(vals)
    if not finite.any():
        return np.nan
    return float(np.mean(np.abs(vals[finite])))


def _safe_ratio(num: float, den: float) -> float:
    if not np.isfinite(num) or not np.isfinite(den) or den == 0.0:
        return np.nan
    return float(num / den)


def _load_all_normalization_contexts(config_path: Path, config_name: str, dataset_name: str) -> dict[str, dict]:
    cfg = _compose_config(config_path, config_name)
    data_cfg = cfg.data.datasets[dataset_name]
    train_ds_cfg = _get_train_dataset_cfg(cfg, dataset_name)

    dataset_path = _get_dataset_path(cfg, dataset_name)
    start = train_ds_cfg.start
    end = train_ds_cfg.end
    frequency = train_ds_cfg.get("frequency") or cfg.data.frequency
    drop = OmegaConf.to_container(train_ds_cfg.drop, resolve=True) if train_ds_cfg.get("drop") is not None else None

    reader = NativeGridDataset(dataset=dataset_path, start=start, end=end, frequency=frequency, drop=drop)
    data_indices = IndexCollection(data_config=data_cfg, name_to_index=reader.name_to_index)
    normalizer = InputNormalizer(
        config=data_cfg.processors.normalizer.config,
        data_indices=data_indices,
        statistics=reader.statistics,
    )

    stats = reader.statistics
    contexts: dict[str, dict] = {}
    for variable, full_idx in reader.name_to_index.items():
        method = normalizer.methods.get(variable, normalizer.default)
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

        contexts[variable] = {
            "method": method,
            "mean": mean,
            "stdev": stdev,
            "minimum": minimum,
            "maximum": maximum,
            "scale": float(scale),
            "dataset_path": dataset_path,
            "train_start": start,
            "train_end": end,
            "frequency": frequency,
        }

    return contexts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_path", type=Path, help="Export file or directory containing pred_target_epoch*.nc/.zarr.")
    parser.add_argument("--config-path", required=True, help="Hydra config directory")
    parser.add_argument("--config-name", required=True, help="Hydra config name without .yaml")
    parser.add_argument("--dataset-name", default="data", help="Dataset key to inspect")
    parser.add_argument("--lead-index", type=int, default=None, help="Optional single lead index. Default: all leads.")
    parser.add_argument("--variables", nargs="*", default=None, help="Optional subset of variables to summarize.")
    args = parser.parse_args()

    files = _iter_exports(args.export_path)
    norm_ctx = _load_all_normalization_contexts(
        config_path=Path(args.config_path),
        config_name=args.config_name,
        dataset_name=args.dataset_name,
    )

    first = _open_export(files[0])
    exported_variables = [str(v) for v in first.coords["variable"].values]
    if args.variables:
        missing = [v for v in args.variables if v not in exported_variables]
        if missing:
            raise ValueError(f"Requested variables not found in export: {missing}")
        variables = list(args.variables)
    else:
        variables = exported_variables

    rows = []
    for variable in variables:
        if variable not in norm_ctx:
            continue

        scale = norm_ctx[variable]["scale"]
        inc_mean_abs = []
        true_inc_mean_abs = []
        err_mean_abs = []
        persist_mean_abs = []
        err_rmse = []
        persist_rmse = []
        inc_domain_mean = []
        true_inc_domain_mean = []
        err_domain_mean = []

        for path in files:
            ds = _open_export(path)
            names = [str(v) for v in ds.coords["variable"].values]
            if variable not in names:
                continue
            var_idx = names.index(variable)

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

                inc_mean_abs.append(_mean_abs(pred_minus_input))
                true_inc_mean_abs.append(_mean_abs(target_minus_input))
                err_mean_abs.append(_mean_abs(target_minus_pred))
                persist_mean_abs.append(_mean_abs(target_minus_input))
                err_rmse.append(_rmse(target_minus_pred))
                persist_rmse.append(_rmse(target_minus_input))
                inc_domain_mean.append(float(np.nanmean(pred_minus_input)))
                true_inc_domain_mean.append(float(np.nanmean(target_minus_input)))
                err_domain_mean.append(float(np.nanmean(target_minus_pred)))

        if not inc_mean_abs:
            continue

        inc_mean_abs_a = np.asarray(inc_mean_abs, dtype=float)
        true_inc_mean_abs_a = np.asarray(true_inc_mean_abs, dtype=float)
        err_mean_abs_a = np.asarray(err_mean_abs, dtype=float)
        persist_mean_abs_a = np.asarray(persist_mean_abs, dtype=float)
        err_rmse_a = np.asarray(err_rmse, dtype=float)
        persist_rmse_a = np.asarray(persist_rmse, dtype=float)
        inc_domain_mean_a = np.asarray(inc_domain_mean, dtype=float)
        true_inc_domain_mean_a = np.asarray(true_inc_domain_mean, dtype=float)
        err_domain_mean_a = np.asarray(err_domain_mean, dtype=float)

        rows.append(
            {
                "variable": variable,
                "cases": len(inc_mean_abs_a),
                "normalization": norm_ctx[variable]["method"],
                "scale": scale,
                "mean_abs_pred_input_raw": float(np.nanmean(inc_mean_abs_a)),
                "mean_abs_pred_input_norm": float(np.nanmean(inc_mean_abs_a / scale)),
                "mean_abs_target_input_raw": float(np.nanmean(true_inc_mean_abs_a)),
                "mean_abs_target_input_norm": float(np.nanmean(true_inc_mean_abs_a / scale)),
                "mean_abs_pred_err_raw": float(np.nanmean(err_mean_abs_a)),
                "mean_abs_pred_err_norm": float(np.nanmean(err_mean_abs_a / scale)),
                "mean_abs_persist_err_raw": float(np.nanmean(persist_mean_abs_a)),
                "mean_abs_persist_err_norm": float(np.nanmean(persist_mean_abs_a / scale)),
                "mean_rmse_pred_err_raw": float(np.nanmean(err_rmse_a)),
                "mean_rmse_pred_err_norm": float(np.nanmean(err_rmse_a / scale)),
                "mean_rmse_persist_err_raw": float(np.nanmean(persist_rmse_a)),
                "mean_rmse_persist_err_norm": float(np.nanmean(persist_rmse_a / scale)),
                "ratio_mean_abs_pred_input_to_target_input": _safe_ratio(
                    float(np.nanmean(inc_mean_abs_a)),
                    float(np.nanmean(true_inc_mean_abs_a)),
                ),
                "ratio_mean_rmse_pred_to_persist": _safe_ratio(
                    float(np.nanmean(err_rmse_a)),
                    float(np.nanmean(persist_rmse_a)),
                ),
                "mean_domain_mean_pred_input_raw": float(np.nanmean(inc_domain_mean_a)),
                "mean_domain_mean_pred_input_norm": float(np.nanmean(inc_domain_mean_a / scale)),
                "mean_domain_mean_target_input_raw": float(np.nanmean(true_inc_domain_mean_a)),
                "mean_domain_mean_target_input_norm": float(np.nanmean(true_inc_domain_mean_a / scale)),
                "mean_domain_mean_target_pred_raw": float(np.nanmean(err_domain_mean_a)),
                "mean_domain_mean_target_pred_norm": float(np.nanmean(err_domain_mean_a / scale)),
            }
        )

    rows.sort(key=lambda row: row["variable"])
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "variable",
            "cases",
            "normalization",
            "scale",
            "mean_abs_pred_input_raw",
            "mean_abs_pred_input_norm",
            "mean_abs_target_input_raw",
            "mean_abs_target_input_norm",
            "mean_abs_pred_err_raw",
            "mean_abs_pred_err_norm",
            "mean_abs_persist_err_raw",
            "mean_abs_persist_err_norm",
            "mean_rmse_pred_err_raw",
            "mean_rmse_pred_err_norm",
            "mean_rmse_persist_err_raw",
            "mean_rmse_persist_err_norm",
            "ratio_mean_abs_pred_input_to_target_input",
            "ratio_mean_rmse_pred_to_persist",
            "mean_domain_mean_pred_input_raw",
            "mean_domain_mean_pred_input_norm",
            "mean_domain_mean_target_input_raw",
            "mean_domain_mean_target_input_norm",
            "mean_domain_mean_target_pred_raw",
            "mean_domain_mean_target_pred_norm",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)


if __name__ == "__main__":
    main()
