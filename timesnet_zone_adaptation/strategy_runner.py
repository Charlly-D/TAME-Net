"""Shared launcher for TimesNet zone adaptation strategy scripts."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.timesnet_zone_adaptation.adaptation_utils import (
    parse_key_value_pairs,
    resolve_zone_configs,
    write_json,
)


def add_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--zones", nargs="+", required=True)
    parser.add_argument("--zone-config-template", default=None)
    parser.add_argument("--zone-configs", nargs="*", default=None, help="Explicit mappings, e.g. CT=configs/ct.json")
    parser.add_argument("--global-checkpoint", default=None)
    parser.add_argument("--baseline-results", nargs="*", default=None, help="Optional zone=path mappings.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--router-lr", type=float, default=1e-4)
    parser.add_argument("--full-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--full-finetune-weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resize-direct-reuse", action="store_true")
    parser.add_argument("--resize-full-finetune", action="store_true")


def run_strategy(args: argparse.Namespace, strategy: str):
    if strategy in {"direct_reuse", "encoder_transfer"} and not args.global_checkpoint:
        raise ValueError(f"--global-checkpoint is required for {strategy}.")

    zone_configs = resolve_zone_configs(args.zones, args.zone_config_template, args.zone_configs)
    baseline_map = parse_key_value_pairs(args.baseline_results)
    script = Path(__file__).resolve().parent / "train_zone_strategy.py"
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    results = {}

    for zone in args.zones:
        zone_output = output_root / zone
        zone_output.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(script),
            "--zone",
            zone,
            "--strategy",
            strategy,
            "--config",
            zone_configs[zone],
            "--output-dir",
            str(zone_output),
            "--device",
            args.device,
            "--epochs",
            str(args.epochs),
            "--router-epochs",
            str(args.router_epochs),
            "--batch-size",
            str(args.batch_size),
            "--num-workers",
            str(args.num_workers),
            "--backbone-lr",
            str(args.backbone_lr),
            "--encoder-lr",
            str(args.encoder_lr),
            "--router-lr",
            str(args.router_lr),
            "--full-finetune-lr",
            str(args.full_finetune_lr),
            "--full-finetune-weight-decay",
            str(args.full_finetune_weight_decay),
            "--patience",
            str(args.patience),
            "--seed",
            str(args.seed),
        ]
        if args.global_checkpoint:
            cmd.extend(["--global-checkpoint", args.global_checkpoint])
        if zone in baseline_map:
            cmd.extend(["--baseline-results", baseline_map[zone]])
        if args.resize_direct_reuse:
            cmd.append("--resize-direct-reuse")
        if args.resize_full_finetune:
            cmd.append("--resize-full-finetune")

        print("=" * 88)
        print(f"Running zone={zone}, strategy={strategy}")
        print(" ".join(cmd))
        print("=" * 88)
        subprocess.run(cmd, check=True)

        result_path = zone_output / f"{zone}_{strategy}_results.json"
        with result_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        results[zone] = {
            "result_path": str(result_path),
            "test_metrics": payload.get("test_metrics"),
            "baseline_metrics": payload.get("baseline_metrics"),
            "improvement_vs_baseline": payload.get("improvement_vs_baseline"),
            "transfer_report": {
                key: payload.get("transfer_report", {}).get(key)
                for key in ("loaded_count", "resized_count", "skipped_count")
            } if payload.get("transfer_report") else None,
        }

    summary_path = output_root / f"{strategy}_summary.json"
    write_json({
        "strategy": strategy,
        "zones": args.zones,
        "global_checkpoint": args.global_checkpoint,
        "settings": {
            "epochs": args.epochs,
            "router_epochs": args.router_epochs,
            "batch_size": args.batch_size,
            "backbone_lr": args.backbone_lr,
            "encoder_lr": args.encoder_lr,
            "router_lr": args.router_lr,
            "full_finetune_lr": args.full_finetune_lr,
            "full_finetune_weight_decay": args.full_finetune_weight_decay,
            "patience": args.patience,
            "seed": args.seed,
        },
        "results": results,
    }, summary_path)
    print(f"Saved summary to {summary_path}")
