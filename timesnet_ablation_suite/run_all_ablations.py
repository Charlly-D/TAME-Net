from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


DEFAULT_VARIANTS = ["full", "no_image", "no_weather", "no_route", "cross_attention", "end_to_end"]

def main():
    parser = argparse.ArgumentParser(description="Run multimodal ablation variants sequentially.")
    parser.add_argument("--variants", nargs="*", default=DEFAULT_VARIANTS)
    parser.add_argument("--config", default="zone_config.json")
    parser.add_argument("--save-root", default=None)
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=None)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--router-epochs", type=int, default=None)
    parser.add_argument("--end-to-end-epochs", type=int, default=None)
    parser.add_argument("--end-to-end-lr", type=float, default=None)
    parser.add_argument("--router-init-bias", type=float, default=None)
    parser.add_argument("--aux-delta-scale", type=float, default=None)
    parser.add_argument("--no-fallback", action="store_true")
    args = parser.parse_args()

    script = Path(__file__).resolve().parent / "train_ablation.py"
    seeds = args.seeds if args.seeds is not None else [args.seed]
    results = []
    for seed in seeds:
        for variant in args.variants:
            cmd = [
            sys.executable,
            str(script),
            "--variant",
            variant,
            "--config",
            args.config,
            "--seed",
            str(seed),
            "--gpu",
            str(args.gpu),
            ]
            save_dir = None
            if args.save_root:
                seed_root = Path(args.save_root) / f"seed_{seed}" if args.seeds is not None else Path(args.save_root)
                save_dir = str(seed_root / variant)
                cmd.extend(["--save-dir", save_dir])
            if args.backbone_checkpoint:
                cmd.extend(["--backbone-checkpoint", args.backbone_checkpoint])
            if args.cpu:
                cmd.append("--cpu")
            if args.batch_size is not None:
                cmd.extend(["--batch-size", str(args.batch_size)])
            if args.num_workers is not None:
                cmd.extend(["--num-workers", str(args.num_workers)])
            if args.pretrain_epochs is not None:
                cmd.extend(["--pretrain-epochs", str(args.pretrain_epochs)])
            if args.router_epochs is not None:
                cmd.extend(["--router-epochs", str(args.router_epochs)])
            if args.end_to_end_epochs is not None:
                cmd.extend(["--end-to-end-epochs", str(args.end_to_end_epochs)])
            if args.end_to_end_lr is not None:
                cmd.extend(["--end-to-end-lr", str(args.end_to_end_lr)])
            if args.router_init_bias is not None:
                cmd.extend(["--router-init-bias", str(args.router_init_bias)])
            if args.aux_delta_scale is not None:
                cmd.extend(["--aux-delta-scale", str(args.aux_delta_scale)])
            if args.no_fallback:
                cmd.append("--no-fallback")

            print("=" * 80)
            print("Running:", " ".join(cmd))
            print("=" * 80)
            subprocess.run(cmd, check=True)

            if save_dir:
                result_path = Path(save_dir) / f"{variant}_results.json"
                if result_path.exists():
                    with result_path.open("r", encoding="utf-8") as f:
                        payload = json.load(f)
                    results.append({
                        "seed": seed,
                        "variant": variant,
                        "metrics": payload.get("final_metrics", {}),
                        "result_path": str(result_path),
                    })

    if args.save_root and results:
        summary_path = Path(args.save_root) / "ablation_summary.json"
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump({"results": results}, f, indent=2, ensure_ascii=False)
        print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
