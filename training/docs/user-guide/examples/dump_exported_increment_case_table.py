#!/usr/bin/env python3
"""Dump per-sample increment/error statistics for every exported variable.

This emits one CSV row per exported file, lead index, and variable, so you can
inspect the exact increment/error values behind the aggregated summaries.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

from summarize_exported_increment_bias import _iter_exports
from summarize_exported_increment_bias import _open_export


def _stats(arr: np.ndarray) -> dict[str, float]:
    vals = np.asarray(arr)
    finite = np.isfinite(vals)
    if not finite.any():
        return {
            "mean": np.nan,
            "mean_abs": np.nan,
            "rms": np.nan,
            "min": np.nan,
            "max": np.nan,
        }

    vals = vals[finite]
    return {
        "mean": float(np.mean(vals)),
        "mean_abs": float(np.mean(np.abs(vals))),
        "rms": float(np.sqrt(np.mean(vals**2))),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export_path", type=Path, help="Export file or directory containing pred_target_epoch*.nc/.zarr.")
    parser.add_argument("--lead-index", type=int, default=None, help="Optional single lead index. Default: all leads.")
    parser.add_argument("--variables", nargs="*", default=None, help="Optional subset of variables to dump.")
    args = parser.parse_args()

    files = _iter_exports(args.export_path)
    first = _open_export(files[0])
    exported_variables = [str(v) for v in first.coords["variable"].values]

    if args.variables:
        missing = [v for v in args.variables if v not in exported_variables]
        if missing:
            raise ValueError(f"Requested variables not found in export: {missing}")
        variables = list(args.variables)
    else:
        variables = exported_variables

    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=[
            "file",
            "lead_index",
            "input_time_last",
            "target_time",
            "pred_time",
            "variable",
            "pred_input_mean",
            "pred_input_mean_abs",
            "pred_input_rms",
            "pred_input_min",
            "pred_input_max",
            "target_input_mean",
            "target_input_mean_abs",
            "target_input_rms",
            "target_input_min",
            "target_input_max",
            "target_pred_mean",
            "target_pred_mean_abs",
            "target_pred_rms",
            "target_pred_min",
            "target_pred_max",
        ],
    )
    writer.writeheader()

    for path in files:
        ds = _open_export(path)
        names = [str(v) for v in ds.coords["variable"].values]
        input_time_last = str(ds["input_time"].values[-1])

        if args.lead_index is None:
            lead_indices = range(min(ds["target"].shape[0], ds["prediction"].shape[0]))
        else:
            lead_indices = [args.lead_index]

        for lead_idx in lead_indices:
            target_time = str(ds["target_time"].values[lead_idx])
            pred_time = str(ds["pred_time"].values[lead_idx])

            for variable in variables:
                if variable not in names:
                    continue
                var_idx = names.index(variable)
                inp = ds["input"].values[:, :, var_idx]
                targ = ds["target"].values[lead_idx, :, var_idx]
                pred = ds["prediction"].values[lead_idx, :, var_idx]
                last_input = inp[-1]

                pred_minus_input = pred - last_input
                target_minus_input = targ - last_input
                target_minus_pred = targ - pred

                pred_input = _stats(pred_minus_input)
                target_input = _stats(target_minus_input)
                target_pred = _stats(target_minus_pred)

                writer.writerow(
                    {
                        "file": str(path),
                        "lead_index": lead_idx,
                        "input_time_last": input_time_last,
                        "target_time": target_time,
                        "pred_time": pred_time,
                        "variable": variable,
                        "pred_input_mean": pred_input["mean"],
                        "pred_input_mean_abs": pred_input["mean_abs"],
                        "pred_input_rms": pred_input["rms"],
                        "pred_input_min": pred_input["min"],
                        "pred_input_max": pred_input["max"],
                        "target_input_mean": target_input["mean"],
                        "target_input_mean_abs": target_input["mean_abs"],
                        "target_input_rms": target_input["rms"],
                        "target_input_min": target_input["min"],
                        "target_input_max": target_input["max"],
                        "target_pred_mean": target_pred["mean"],
                        "target_pred_mean_abs": target_pred["mean_abs"],
                        "target_pred_rms": target_pred["rms"],
                        "target_pred_min": target_pred["min"],
                        "target_pred_max": target_pred["max"],
                    }
                )


if __name__ == "__main__":
    main()
