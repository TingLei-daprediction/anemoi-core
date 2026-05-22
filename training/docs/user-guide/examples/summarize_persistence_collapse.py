#!/usr/bin/env python3
"""Summarize how close exported predictions stay to persistence.

This is similar to summarize_exported_increment_bias.py, but focuses on:

- how large the model increment |pred - last_input| is
- how large the true increment |target - last_input| is
- how model error compares with persistence error
- the same quantities in normalized space

Examples
--------
python training/docs/user-guide/examples/summarize_persistence_collapse.py \
  /scratch3/.../verify/predictions \
  --variable refc \
  --config-path training/docs/user-guide/examples \
  --config-name anemoi-training-...-base_3_v1-...
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr

from summarize_exported_increment_bias import _compose_config
from summarize_exported_increment_bias import _get_dataset_path
from summarize_exported_increment_bias import _get_train_dataset_cfg
from summarize_exported_increment_bias import _iter_exports
from summarize_exported_increment_bias import _load_normalization_context
from summarize_exported_increment_bias import _open_export
from summarize_exported_increment_bias import _rmse


def _mean_abs(arr: np.ndarray) -> float:
    vals = np.asarray(arr)
    finite = np.isfinite(vals)
    if not finite.any():
        return np.nan
    return float(np.mean(np.abs(vals[finite])))


def _q(arr: np.ndarray, q: float) -> float:
    vals = np.asarray(arr)
    finite = np.isfinite(vals)
    if not finite.any():
        return np.nan
    return float(np.quantile(vals[finite], q))


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
        raise ValueError(f"No usable samples found for variable '{args.variable}' under {args.export_path}")

    inc_mean_abs = np.asarray(inc_mean_abs, dtype=float)
    true_inc_mean_abs = np.asarray(true_inc_mean_abs, dtype=float)
    err_mean_abs = np.asarray(err_mean_abs, dtype=float)
    persist_mean_abs = np.asarray(persist_mean_abs, dtype=float)
    err_rmse = np.asarray(err_rmse, dtype=float)
    persist_rmse = np.asarray(persist_rmse, dtype=float)
    inc_domain_mean = np.asarray(inc_domain_mean, dtype=float)
    true_inc_domain_mean = np.asarray(true_inc_domain_mean, dtype=float)
    err_domain_mean = np.asarray(err_domain_mean, dtype=float)

    print(f"variable: {args.variable}")
    print(f"files: {len(files)}")
    print(f"cases: {len(inc_mean_abs)}")
    print(
        "normalization: "
        f"method={norm_ctx['method']} mean={norm_ctx['mean']:.6g} stdev={norm_ctx['stdev']:.6g} "
        f"minimum={norm_ctx['minimum']:.6g} maximum={norm_ctx['maximum']:.6g} scale={scale:.6g}"
    )
    print(f"training_window: start={norm_ctx['train_start']} end={norm_ctx['train_end']} frequency={norm_ctx['frequency']}")
    print("")
    print(
        f"mean_abs(pred-input): raw={np.nanmean(inc_mean_abs):.6g} "
        f"norm={np.nanmean(inc_mean_abs / scale):.6g}"
    )
    print(
        f"mean_abs(target-input): raw={np.nanmean(true_inc_mean_abs):.6g} "
        f"norm={np.nanmean(true_inc_mean_abs / scale):.6g}"
    )
    print(
        f"mean_abs(pred-err): raw={np.nanmean(err_mean_abs):.6g} "
        f"norm={np.nanmean(err_mean_abs / scale):.6g}"
    )
    print(
        f"mean_abs(persist-err): raw={np.nanmean(persist_mean_abs):.6g} "
        f"norm={np.nanmean(persist_mean_abs / scale):.6g}"
    )
    print(
        f"mean RMSE(pred err): raw={np.nanmean(err_rmse):.6g} "
        f"norm={np.nanmean(err_rmse / scale):.6g}"
    )
    print(
        f"mean RMSE(persist err): raw={np.nanmean(persist_rmse):.6g} "
        f"norm={np.nanmean(persist_rmse / scale):.6g}"
    )
    print(
        f"ratio mean_abs(pred-input) / mean_abs(target-input) = "
        f"{np.nanmean(inc_mean_abs) / np.nanmean(true_inc_mean_abs):.6g}"
    )
    print(
        f"ratio mean_RMSE(pred err) / mean_RMSE(persist err) = "
        f"{np.nanmean(err_rmse) / np.nanmean(persist_rmse):.6g}"
    )
    print(
        f"mean domain-mean(pred-input): raw={np.nanmean(inc_domain_mean):.6g} "
        f"norm={np.nanmean(inc_domain_mean / scale):.6g}"
    )
    print(
        f"mean domain-mean(target-input): raw={np.nanmean(true_inc_domain_mean):.6g} "
        f"norm={np.nanmean(true_inc_domain_mean / scale):.6g}"
    )
    print(
        f"mean domain-mean(target-pred): raw={np.nanmean(err_domain_mean):.6g} "
        f"norm={np.nanmean(err_domain_mean / scale):.6g}"
    )
    print(
        f"quantiles mean_abs(pred-input): "
        f"q10={_q(inc_mean_abs, 0.10):.6g} q50={_q(inc_mean_abs, 0.50):.6g} q90={_q(inc_mean_abs, 0.90):.6g}"
    )


if __name__ == "__main__":
    main()
