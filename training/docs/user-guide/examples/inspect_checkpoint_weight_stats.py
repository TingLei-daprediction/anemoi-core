#!/usr/bin/env python3
"""Summarize parameter statistics from an Anemoi checkpoint.

Examples
--------
python training/docs/user-guide/examples/inspect_checkpoint_weight_stats.py \
  /scratch3/.../last.ckpt

python training/docs/user-guide/examples/inspect_checkpoint_weight_stats.py \
  /scratch3/.../last.ckpt \
  --pattern node_data_extractor
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    raw = torch.load(path, map_location="cpu", weights_only=False)

    if hasattr(raw, "state_dict"):
        return raw.state_dict()
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw["state_dict"]
    if isinstance(raw, dict):
        tensor_items = {k: v for k, v in raw.items() if torch.is_tensor(v)}
        if tensor_items:
            return tensor_items
    raise ValueError(f"Could not extract a state_dict-like mapping from {path}")


def _tensor_stats(t: torch.Tensor) -> dict[str, float]:
    x = t.detach().float().cpu()
    return {
        "numel": int(x.numel()),
        "mean": float(x.mean()),
        "mean_abs": float(x.abs().mean()),
        "rms": float(torch.sqrt(torch.mean(x * x))),
        "std": float(x.std(unbiased=False)),
        "min": float(x.min()),
        "max": float(x.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Path to last.ckpt or inference checkpoint")
    parser.add_argument("--pattern", default=None, help="Only print parameters whose name contains this substring")
    parser.add_argument("--topk", type=int, default=200, help="Maximum number of matching parameters to print")
    args = parser.parse_args()

    state_dict = _load_state_dict(args.checkpoint)
    names = sorted(state_dict.keys())
    if args.pattern:
        names = [name for name in names if args.pattern in name]

    if not names:
        raise ValueError(f"No parameters matched pattern={args.pattern!r}")

    print(f"checkpoint: {args.checkpoint}")
    print(f"pattern: {args.pattern or '<all>'}")
    print(f"matched_parameters: {len(names)}")
    print("")
    print("name,numel,mean,mean_abs,rms,std,min,max")

    for name in names[: args.topk]:
        tensor = state_dict[name]
        if not torch.is_tensor(tensor):
            continue
        stats = _tensor_stats(tensor)
        print(
            f"{name},{stats['numel']},{stats['mean']:.6g},{stats['mean_abs']:.6g},"
            f"{stats['rms']:.6g},{stats['std']:.6g},{stats['min']:.6g},{stats['max']:.6g}"
        )

    # Helpful grouped summary for common model components.
    print("")
    print("group,parameters,total_numel,mean_abs,rms")
    groups = {
        "decoder": [n for n in names if ".decoder." in n],
        "decoder_node_data_extractor": [n for n in names if "node_data_extractor" in n],
        "encoder": [n for n in names if ".encoder." in n],
        "processor": [n for n in names if ".processor." in n],
        "bias_terms": [n for n in names if n.endswith(".bias")],
    }
    for group_name, group_names in groups.items():
        if not group_names:
            continue
        flat = torch.cat([state_dict[n].detach().float().reshape(-1).cpu() for n in group_names if torch.is_tensor(state_dict[n])])
        print(
            f"{group_name},{len(group_names)},{flat.numel()},"
            f"{float(flat.abs().mean()):.6g},{float(torch.sqrt(torch.mean(flat * flat))):.6g}"
        )


if __name__ == "__main__":
    main()
