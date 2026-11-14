"""Run validation-only sensitivity analysis for the interpolation TAME-Net model."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SUITE_DIR = Path(__file__).resolve().parent
TRAIN_SCRIPT = SUITE_DIR / "train_ablation.py"


def value_tag(value: float) -> str:
    text = format(value, "g")
    return text.replace("-", "neg").replace(".", "p")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def result_is_reusable(path: Path, param: str, value: float) -> bool:
    if not path.is_file():
        return False
    try:
        payload = load_json(path)
        return (
            payload.get("reported_split") == "validation"
            and float(payload.get("config", {}).get(param)) == float(value)
            and payload.get("final_metrics", {}).get("MAPE(%)") is not None
        )
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validation-only sensitivity analysis for the revised interpolation model. "
            "The fixed reference configuration is reported but is not replaced by the sweep."
        )
    )
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--param",
        required=True,
        choices=("aux_delta_scale", "router_init_bias"),
    )
    parser.add_argument("--values", nargs="+", required=True, type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--fixed-aux-delta-scale", type=float, default=0.1)
    parser.add_argument("--fixed-router-init-bias", type=float, default=-5.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = []

    for value in args.values:
        aux_delta_scale = (
            value if args.param == "aux_delta_scale" else args.fixed_aux_delta_scale
        )
        router_init_bias = (
            value if args.param == "router_init_bias" else args.fixed_router_init_bias
        )
        run_dir = output_dir / f"{args.param}_{value_tag(value)}"
        result_path = run_dir / "full_results.json"

        if not result_is_reusable(result_path, args.param, value):
            command = [
                sys.executable,
                "-u",
                str(TRAIN_SCRIPT),
                "--variant",
                "full",
                "--config",
                args.config,
                "--save-dir",
                str(run_dir),
                "--seed",
                str(args.seed),
                "--gpu",
                str(args.gpu),
                "--aux-delta-scale",
                str(aux_delta_scale),
                "--router-init-bias",
                str(router_init_bias),
                "--report-split",
                "validation",
            ]
            if args.batch_size is not None:
                command.extend(["--batch-size", str(args.batch_size)])
            if args.num_workers is not None:
                command.extend(["--num-workers", str(args.num_workers)])
            if args.backbone_checkpoint:
                command.extend(["--backbone-checkpoint", args.backbone_checkpoint])
            subprocess.run(command, check=True)

        payload = load_json(result_path)
        runs.append(
            {
                "value": value,
                "validation_metrics": payload["final_metrics"],
                "result_path": str(result_path),
            }
        )

    summary = {
        "analysis_type": "sensitivity_only",
        "selection_split": "validation",
        "test_split_evaluated": False,
        "model_variant": "full_interpolation",
        "seed": args.seed,
        "parameter": args.param,
        "values": args.values,
        "fixed_reference": {
            "aux_delta_scale": args.fixed_aux_delta_scale,
            "router_init_bias": args.fixed_router_init_bias,
        },
        "note": (
            "The reference configuration was fixed before this sensitivity analysis. "
            "The sweep characterizes robustness and does not replace the reported main configuration."
        ),
        "runs": runs,
    }
    summary_path = output_dir / "sensitivity_summary.json"
    with summary_path.open("w", encoding="utf-8") as stream:
        json.dump(summary, stream, indent=2, ensure_ascii=False)
    print(f"Sensitivity summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
