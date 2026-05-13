#!/usr/bin/env python3
"""Plot diffusion training loss from Anemoi logs.

This is intended for the current diffusion training logs, which commonly expose
`train_multi_dataset_loss_step=...` in the progress-bar output rather than only
epoch-level metrics.

The script handles two quirks in those logs:
- progress-bar redraws can repeat the exact same `*_step` values
- `*_epoch` metrics are often reprinted on many later lines within the same
  epoch, which otherwise produces a badly overplotted figure
"""

from __future__ import annotations

import argparse
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ANSI_RE = re.compile(r"\x1B\[[0-9;]*[A-Za-z]")
STEP_LOSS_RE = re.compile(
    r"(?P<key>(?:train|val)_[A-Za-z0-9_]*(?:loss|mse)_(?:step|epoch))="
    r"(?P<val>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)"
)
EPOCH_RE = re.compile(r"Epoch\s+(?P<epoch>\d+):")


def parse_log(path: str) -> dict[str, list[tuple[int, float]]]:
    """Parse diffusion-relevant loss metrics from a log file.

    Returns a dict keyed by metric name, where each value is a list of
    `(sample_index, value)` pairs in log order.
    """
    step_metrics: dict[str, list[tuple[int, float]]] = defaultdict(list)
    epoch_metrics: dict[str, dict[int, float]] = defaultdict(dict)
    counters: dict[str, int] = defaultdict(int)
    last_step_value: dict[str, float] = {}

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw)

            epoch = None
            m_epoch = EPOCH_RE.search(line)
            if m_epoch:
                epoch = int(m_epoch.group("epoch"))

            for m_loss in STEP_LOSS_RE.finditer(line):
                key = m_loss.group("key")
                val = float(m_loss.group("val"))

                if key.endswith("_epoch"):
                    if epoch is not None:
                        # Keep the latest value seen for a given epoch.
                        epoch_metrics[key][epoch] = val
                    continue

                # Suppress exact progress-bar redraw duplicates for step metrics.
                if key in last_step_value and last_step_value[key] == val:
                    continue
                step_metrics[key].append((counters[key], val))
                counters[key] += 1
                last_step_value[key] = val

    by_metric: dict[str, list[tuple[int, float]]] = {}
    for key, series in step_metrics.items():
        by_metric[key] = series
    for key, by_epoch in epoch_metrics.items():
        by_metric[key] = sorted(by_epoch.items())
    return by_metric


def choose_default_keys(all_keys: list[str]) -> list[str]:
    preferred = [
        "train_multi_dataset_loss_step",
        "train_multi_dataset_loss_epoch",
        "val_multi_dataset_loss_step",
        "val_multi_dataset_loss_epoch",
        "val_mse_epoch",
        "train_mse_loss_epoch",
        "val_mse_loss_epoch",
    ]
    selected = [k for k in preferred if k in all_keys]
    if selected:
        return selected
    return sorted(all_keys)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot diffusion training loss from an Anemoi log.")
    p.add_argument("log_file", help="Path to the log file (for example dd.output)")
    p.add_argument("output_png", nargs="?", default="diffusion_training_loss.png", help="Output figure path")
    p.add_argument("--keys", default="", help="Comma-separated metric keys to plot. Default: auto-detect.")
    p.add_argument("--title", default="Diffusion Training Loss", help="Plot title")
    args = p.parse_args()

    metrics = parse_log(args.log_file)
    if not metrics:
        raise SystemExit("No diffusion-style '*loss_step', '*loss_epoch', or '*mse_epoch' metrics found in log.")

    all_keys = sorted(metrics.keys())
    if args.keys.strip():
        keys = [k.strip() for k in args.keys.split(",") if k.strip()]
        missing = [k for k in keys if k not in metrics]
        if missing:
            raise SystemExit(f"Requested keys not found: {missing}. Available: {all_keys}")
    else:
        keys = choose_default_keys(all_keys)

    step_keys = [k for k in keys if k.endswith("_step")]
    epoch_keys = [k for k in keys if k.endswith("_epoch")]
    nrows = int(bool(step_keys)) + int(bool(epoch_keys))
    fig, axes = plt.subplots(nrows=nrows, ncols=1, figsize=(9.0, 4.5 * nrows))
    if nrows == 1:
        axes = [axes]

    ax_idx = 0
    if step_keys:
        ax = axes[ax_idx]
        ax_idx += 1
        for key in step_keys:
            series = metrics[key]
            if not series:
                continue
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            ax.plot(xs, ys, linewidth=1.0, label=key)
        ax.set_xlabel("Step index")
        ax.set_ylabel("Loss")
        ax.set_title(f"{args.title} - Step Metrics")
        ax.grid(True, alpha=0.3)
        ax.legend()

    if epoch_keys:
        ax = axes[ax_idx]
        for key in epoch_keys:
            series = metrics[key]
            if not series:
                continue
            xs = [x for x, _ in series]
            ys = [y for _, y in series]
            ax.plot(xs, ys, linewidth=1.2, marker="o", label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title(f"{args.title} - Epoch Metrics")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(args.output_png, dpi=150)

    print("Available keys:", ", ".join(all_keys))
    print("Plotted keys:", ", ".join(keys))
    print(f"Wrote {args.output_png}")


if __name__ == "__main__":
    main()
