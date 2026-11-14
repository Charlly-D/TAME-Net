"""Fine-tune Chronos-2 with load as target and historical GAP weather as covariates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import train_chronos2 as base


def standardize_weather(weather: np.ndarray, train_end: int):
    mean = weather[:train_end].mean(axis=0)
    std = weather[:train_end].std(axis=0) + 1e-5
    return ((weather - mean) / std).astype("float32"), mean, std


def make_item(load: np.ndarray, weather: np.ndarray, names: list[str]):
    return {
        "target": np.asarray(load, dtype="float32"),
        "past_covariates": {
            name: np.asarray(weather[:, index], dtype="float32")
            for index, name in enumerate(names)
        },
    }


def prepare_inputs(items: list[dict], prediction_length: int):
    try:
        from chronos.chronos2.preprocess import from_list_of_dicts
    except ImportError as exc:
        raise ImportError(
            "Chronos-2 covariate preprocessing is unavailable. Upgrade chronos-forecasting "
            "to a version that provides chronos.chronos2.preprocess.from_list_of_dicts."
        ) from exc
    return from_list_of_dicts(items, prediction_length=prediction_length)


def build_eval_windows(
    load: np.ndarray,
    weather: np.ndarray,
    names: list[str],
    split: str,
    base_seq_len: int,
    context_len: int,
    pred_len: int,
    stride: int,
    limit_windows: int | None,
):
    train_end, val_end, total_len = base.cb.split_bounds(len(load))
    split_start, split_end = {
        "val": (train_end, val_end),
        "test": (val_end, total_len),
    }[split]
    first_target = split_start + base_seq_len
    last_target = split_end - pred_len
    items, targets = [], []

    for target_start in range(first_target, last_target + 1, stride):
        context_start = target_start - context_len
        if context_start < 0:
            continue
        items.append(make_item(load[context_start:target_start], weather[context_start:target_start], names))
        targets.append(load[target_start:target_start + pred_len])
        if limit_windows is not None and len(items) >= limit_windows:
            break

    if not items:
        raise RuntimeError(f"No evaluation windows were generated for split={split}.")
    return items, np.stack(targets).astype("float32")


def means_to_numpy(means) -> np.ndarray:
    rows = []
    for value in means:
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value, dtype="float32")
        if value.ndim == 2:
            value = value[0]
        elif value.ndim != 1:
            raise RuntimeError(f"Unsupported Chronos-2 mean prediction shape: {value.shape}")
        rows.append(value)
    return np.stack(rows).astype("float32")


def evaluate(pipeline, items: list[dict], targets: np.ndarray, pred_len: int, batch_size: int):
    forecasts = []
    for start in range(0, len(items), batch_size):
        prepared = prepare_inputs(items[start:start + batch_size], pred_len)
        _, means = pipeline.predict_quantiles(
            inputs=prepared,
            prediction_length=pred_len,
            quantile_levels=[0.5],
        )
        forecasts.append(means_to_numpy(means))
    predictions = np.concatenate(forecasts, axis=0)
    return predictions, base.cb.compute_metrics(predictions, targets)


def run(args):
    base.set_seed(args.seed)
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    config_path = Path(args.config)
    raw_config = base.cb.load_json(config_path)
    flat_config = base.cb.flatten_config(raw_config)
    data_paths = raw_config.get("env_settings", {}).get("data_paths", {})

    load_path = base.cb.resolve_path(config_path, data_paths["load"])
    era5_value = args.era5_path or data_paths.get("era5")
    if not era5_value:
        raise ValueError("An ERA5 path is required through --era5-path or config env_settings.data_paths.era5.")
    era5_path = base.cb.resolve_path(config_path, era5_value)

    load, load_column = base.cb.read_load_series(load_path, flat_config)
    weather = base.cb.read_gap_weather(era5_path, args.weather_vars)
    aligned_len = min(len(load), len(weather))
    if len(load) != len(weather):
        print(f"[ChronosCovariates] Warning: truncating load/weather to {aligned_len} aligned positions.")
    load = np.asarray(load[:aligned_len], dtype="float32")
    weather = np.asarray(weather[:aligned_len], dtype="float32")

    train_end, val_end, _ = base.cb.split_bounds(aligned_len)
    weather, weather_mean, weather_std = standardize_weather(weather, train_end)
    pred_len = int(args.pred_len or flat_config.get("pred_len", 96))
    context_len = int(args.context_len or flat_config.get("seq_len", 96))

    train_items = prepare_inputs(
        [make_item(load[:train_end], weather[:train_end], args.weather_vars)], pred_len
    )
    val_items = prepare_inputs(
        [make_item(load[train_end:val_end], weather[train_end:val_end], args.weather_vars)], pred_len
    )

    pipeline, resolved_model = base.load_chronos2_pipeline(args.model_id, args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_kwargs = {
        "inputs": train_items,
        "prediction_length": pred_len,
        "validation_inputs": val_items,
        "finetune_mode": args.finetune_mode,
        "context_length": context_len,
        "learning_rate": args.learning_rate,
        "num_steps": args.num_steps,
        "batch_size": args.batch_size,
        "output_dir": str(output_dir / "trainer"),
        "finetuned_ckpt_name": args.finetuned_ckpt_name,
        "disable_data_parallel": args.disable_data_parallel,
        "seed": args.seed,
        "data_seed": args.seed,
    }
    if args.finetune_mode == "lora":
        fit_kwargs["lora_config"] = {
            "r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": args.lora_dropout,
        }

    print("Chronos-2 fine-tuning with historical GAP weather covariates")
    print("-" * 64)
    print(f"model_id        : {args.model_id}")
    print(f"resolved_model  : {resolved_model}")
    print(f"load_column     : {load_column}")
    print(f"weather_vars    : {', '.join(args.weather_vars)}")
    print(f"context_len     : {context_len}")
    print(f"pred_len        : {pred_len}")
    print(f"finetune_mode   : {args.finetune_mode}")
    print(f"num_steps       : {args.num_steps}")
    print(f"seed            : {args.seed}")
    print("future_weather  : disabled")
    print("-" * 64)

    finetuned = pipeline.fit(**fit_kwargs)
    eval_items, targets = build_eval_windows(
        load=load,
        weather=weather,
        names=args.weather_vars,
        split=args.split,
        base_seq_len=int(flat_config.get("seq_len", 96)),
        context_len=context_len,
        pred_len=pred_len,
        stride=args.stride,
        limit_windows=args.limit_windows,
    )
    predictions, metrics = evaluate(finetuned, eval_items, targets, pred_len, args.eval_batch_size)

    checkpoint_dir = output_dir / "evaluated-finetuned-ckpt"
    saved_checkpoint = base.save_current_pipeline(finetuned, checkpoint_dir)
    result = {
        "model": "Chronos-2",
        "input_protocol": "load target + historical GAP weather past_covariates",
        "future_covariates_used": False,
        "model_id": args.model_id,
        "saved_checkpoint": saved_checkpoint,
        "config": str(config_path),
        "load_column": load_column,
        "weather_vars": args.weather_vars,
        "weather_mean": weather_mean.tolist(),
        "weather_std": weather_std.tolist(),
        "split": args.split,
        "context_len": context_len,
        "pred_len": pred_len,
        "num_windows": len(eval_items),
        "finetune_mode": args.finetune_mode,
        "num_steps": args.num_steps,
        "learning_rate": args.learning_rate,
        "seed": args.seed,
        "metrics": metrics,
    }
    metrics_path = output_dir / "chronos2_covariate_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    np.save(output_dir / "predictions.npy", predictions)
    np.save(output_dir / "truths.npy", targets)

    print("\nChronos-2 historical-weather covariate evaluation")
    print("-" * 64)
    for name, value in metrics.items():
        print(f"{name:10s}: {value:.4f}")
    print(f"Saved metrics to: {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-2 with historical GAP weather covariates.")
    parser.add_argument("--config", default="config.json")
    parser.add_argument("--model-id", default="amazon/chronos-2")
    parser.add_argument("--output-dir", default="experiments/chronos2_finetune/checkpoints/covariates_latest")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--weather-vars", nargs="+", default=["t2m", "u10", "v10", "sp", "tp", "d2m"])
    parser.add_argument("--context-len", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--limit-windows", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--num-steps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--finetune-mode", choices=["full", "lora"], default="full")
    parser.add_argument("--finetuned-ckpt-name", default="finetuned-ckpt")
    parser.add_argument("--disable-data-parallel", action="store_true", default=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    run(parser.parse_args())
