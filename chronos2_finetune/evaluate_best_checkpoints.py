"""Evaluate validation-selected Chronos-2 checkpoints from completed runs."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.chronos2_finetune import train_chronos2 as training  # noqa: E402


SEED_PATTERN = re.compile(r"seed[_-]?(\d+)", re.IGNORECASE)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def find_best_checkpoint(run_dir: Path) -> tuple[Path, float, int]:
    records = []
    for state_path in sorted((run_dir / "trainer").glob("checkpoint-*/trainer_state.json")):
        state = read_json(state_path)
        metric = state.get("best_metric")
        checkpoint_value = state.get("best_model_checkpoint")
        if metric is None or not checkpoint_value:
            continue
        checkpoint_name = Path(checkpoint_value).name
        candidates = (
            Path(checkpoint_value),
            run_dir / "trainer" / checkpoint_name,
            state_path.parent,
        )
        checkpoint = next(
            (candidate.resolve() for candidate in candidates if candidate.is_dir()),
            None,
        )
        if checkpoint is not None:
            records.append((float(metric), checkpoint, int(state.get("best_global_step", 0))))

    if not records:
        raise FileNotFoundError(
            f"No validation-selected checkpoint found under {run_dir / 'trainer'}"
        )
    metric, checkpoint, step = min(records, key=lambda item: item[0])
    return checkpoint, metric, step


def infer_seed(payload: dict[str, Any], run_dir: Path) -> int:
    value = payload.get("finetune", {}).get("seed")
    if value is not None:
        return int(value)
    match = SEED_PATTERN.search(run_dir.as_posix())
    return int(match.group(1)) if match else 0


def to_float_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    return {key: float(value) for key, value in metrics.items()}


def evaluate_run(
    run_dir: Path,
    config_path: Path,
    device: str,
    eval_batch_size: int,
    num_samples: int,
) -> dict[str, Any]:
    result_path = run_dir / "chronos2_finetune_metrics.json"
    if not result_path.is_file():
        raise FileNotFoundError(result_path)
    original = read_json(result_path)
    best_checkpoint, best_val_loss, best_step = find_best_checkpoint(run_dir)

    use_gap_weather = bool(original.get("use_gap_weather", False))
    weather_vars = original.get("weather_vars") or ["t2m", "u10", "v10", "sp", "tp", "d2m"]
    args = SimpleNamespace(
        config=str(config_path),
        model_id=str(best_checkpoint),
        context_len=int(original.get("context_len", 96)),
        pred_len=int(original.get("pred_len", 96)),
        split="test",
        stride=int(original.get("stride", 1)),
        limit_windows=None,
        eval_batch_size=eval_batch_size,
        num_samples=num_samples,
        device=device,
        torch_dtype="auto",
        trust_remote_code=False,
        local_files_only=True,
        use_gap_weather=use_gap_weather,
        era5_path=None,
        weather_vars=weather_vars,
        standardize_load=bool(original.get("standardize_load", False)),
    )

    raw_config = training.cb.load_json(config_path)
    flat_config = training.cb.flatten_config(raw_config)
    _, _, raw_load, _, covariate_stats = training.build_train_val_series(
        args, config_path, raw_config, flat_config
    )
    pipeline, resolved_path = training.load_chronos2_pipeline(str(best_checkpoint), args)
    _, truths, predictions, metrics = training.evaluate_pipeline(
        pipeline,
        args,
        config_path,
        flat_config,
        raw_load,
        covariate_stats,
    )

    predictions_path = run_dir / "best_checkpoint_predictions.npy"
    truths_path = run_dir / "best_checkpoint_truths.npy"
    output_path = run_dir / "best_checkpoint_test_metrics.json"
    np.save(predictions_path, predictions.astype("float32"))
    np.save(truths_path, truths.astype("float32"))
    output = {
        "model": "Chronos-2",
        "variant": "gap_weather" if use_gap_weather else "load_only",
        "seed": infer_seed(original, run_dir),
        "run_dir": str(run_dir),
        "selection_split": "validation",
        "evaluation_split": "test",
        "best_checkpoint": str(best_checkpoint),
        "best_global_step": best_step,
        "best_validation_loss": best_val_loss,
        "resolved_model_path": resolved_path,
        "training_protocol": original.get("finetune"),
        "test_metrics": to_float_metrics(metrics),
        "original_reported_metrics": original.get("reloaded_metrics") or original.get("metrics"),
        "predictions": str(predictions_path),
        "truths": str(truths_path),
    }
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
    print(
        f"[{output['variant']} seed={output['seed']}] "
        f"best_step={best_step}, val_loss={best_val_loss:.6f}, "
        f"MAPE={output['test_metrics'].get('MAPE(%)', float('nan')):.4f}%"
    )
    return output


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for variant in sorted({row["variant"] for row in rows}):
        selected = [row for row in rows if row["variant"] == variant]
        item: dict[str, Any] = {
            "variant": variant,
            "n": len(selected),
            "seeds": [row["seed"] for row in selected],
        }
        for metric in ("MAE", "RMSE", "MAPE(%)", "WAPE(%)", "R2", "Corr"):
            values = [row["test_metrics"][metric] for row in selected if metric in row["test_metrics"]]
            if values:
                item[metric] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                    "values": values,
                }
        output.append(item)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--run-dirs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--num-samples", type=int, default=20)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    rows = [
        evaluate_run(
            Path(run_dir).resolve(),
            config_path,
            args.device,
            args.eval_batch_size,
            args.num_samples,
        )
        for run_dir in args.run_dirs
    ]
    payload = {
        "selection_rule": "Minimum validation loss among saved checkpoints; test split evaluated once.",
        "runs": rows,
        "aggregate": aggregate(rows),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    print(f"Saved aggregate summary to {output_path}")


if __name__ == "__main__":
    main()
