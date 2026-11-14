"""Run zone-adaptation experiments across multiple seeds and strategies."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SCRIPT_DIR / "train_zone_strategy.py"
TRAINABLE_STRATEGIES = (
    "local_scratch",
    "encoder_transfer",
    "full_model_finetune",
)


def parse_zone_configs(items: list[str]) -> dict[str, str]:
    mappings: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Invalid zone config mapping {item!r}; expected ZONE=PATH."
            )
        zone, path = item.split("=", 1)
        zone = zone.strip()
        path = path.strip()
        if not zone or not path:
            raise ValueError(
                f"Invalid zone config mapping {item!r}; expected ZONE=PATH."
            )
        mappings[zone] = path
    return mappings


def load_completed_result(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None

    metrics = payload.get("test_metrics")
    if not isinstance(metrics, dict) or not metrics:
        return None
    return payload


def write_summary(path: Path, args: argparse.Namespace, rows: list[dict]) -> None:
    payload = {
        "seeds": args.seeds,
        "strategies": args.strategies,
        "zones": args.zones,
        "global_checkpoint": args.global_checkpoint,
        "settings": {
            "device": args.device,
            "epochs": args.epochs,
            "router_epochs": args.router_epochs,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "backbone_lr": args.backbone_lr,
            "encoder_lr": args.encoder_lr,
            "router_lr": args.router_lr,
            "router_init_bias": args.router_init_bias,
            "full_finetune_lr": args.full_finetune_lr,
            "full_finetune_weight_decay": args.full_finetune_weight_decay,
            "resize_full_finetune": args.resize_full_finetune,
            "patience": args.patience,
        },
        "runs": rows,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run local adaptation for multiple random seeds."
    )
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument(
        "--strategies",
        nargs="+",
        choices=TRAINABLE_STRATEGIES,
        default=list(TRAINABLE_STRATEGIES),
    )
    parser.add_argument("--zones", nargs="+", required=True)
    parser.add_argument(
        "--zone-configs",
        nargs="+",
        required=True,
        metavar="ZONE=PATH",
    )
    parser.add_argument("--global-checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--router-lr", type=float, default=1e-4)
    parser.add_argument("--router-init-bias", type=float, default=None)
    parser.add_argument("--full-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--full-finetune-weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resize-full-finetune", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun experiments even when a complete result JSON already exists.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    zone_configs = parse_zone_configs(args.zone_configs)
    missing = [zone for zone in args.zones if zone not in zone_configs]
    if missing:
        raise ValueError(f"Missing --zone-configs entries for: {', '.join(missing)}")

    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / "multiseed_adaptation_summary.json"
    rows: list[dict] = []

    for seed in args.seeds:
        for strategy in args.strategies:
            for zone in args.zones:
                run_dir = output_root / f"seed_{seed}" / strategy / zone
                result_path = run_dir / f"{zone}_{strategy}_results.json"
                payload = None if args.overwrite else load_completed_result(result_path)

                if payload is not None:
                    print(
                        f"[skip] seed={seed} strategy={strategy} zone={zone}: "
                        f"{result_path}"
                    )
                    status = "skipped_existing"
                else:
                    run_dir.mkdir(parents=True, exist_ok=True)
                    cmd = [
                        sys.executable,
                        "-u",
                        str(TRAIN_SCRIPT),
                        "--zone",
                        zone,
                        "--strategy",
                        strategy,
                        "--config",
                        zone_configs[zone],
                        "--global-checkpoint",
                        args.global_checkpoint,
                        "--output-dir",
                        str(run_dir),
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
                        str(seed),
                    ]
                    if args.router_init_bias is not None:
                        cmd.extend(["--router-init-bias", str(args.router_init_bias)])
                    if args.resize_full_finetune:
                        cmd.append("--resize-full-finetune")
                    print("=" * 88)
                    print(f"Running seed={seed}, strategy={strategy}, zone={zone}")
                    print(" ".join(cmd))
                    print("=" * 88)
                    subprocess.run(cmd, check=True)
                    payload = load_completed_result(result_path)
                    if payload is None:
                        raise RuntimeError(
                            f"Training finished without a valid result: {result_path}"
                        )
                    status = "completed"

                rows.append(
                    {
                        "seed": seed,
                        "strategy": strategy,
                        "zone": zone,
                        "status": status,
                        "result_path": str(result_path),
                        "test_metrics": payload.get("test_metrics"),
                        "transfer_report": payload.get("transfer_report"),
                        "timing": payload.get("timing"),
                    }
                )
                write_summary(summary_path, args, rows)

    print(f"Saved combined summary to {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
