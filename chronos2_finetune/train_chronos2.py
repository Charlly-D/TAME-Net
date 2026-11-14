"""Fine-tune Chronos-2 for ISO-NE load forecasting.

This script keeps the evaluation protocol aligned with
``experiments/chronos2_baseline/run_chronos2.py`` while adding Chronos-2
fine-tuning through ``Chronos2Pipeline.fit``.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
BASELINE_SCRIPT = ROOT / "experiments" / "chronos2_baseline" / "run_chronos2.py"


def import_chronos_baseline():
    spec = importlib.util.spec_from_file_location("chronos2_baseline_run", BASELINE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not import Chronos baseline helpers from {BASELINE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.np = np
    module.torch = torch
    return module


cb = import_chronos_baseline()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def split_series_for_fit(series: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    train_end, val_end, _ = cb.split_bounds(len(series))
    return series[:train_end], series[train_end:val_end]


def to_chronos_fit_tensor(series: np.ndarray) -> torch.Tensor:
    """Return [n_series, n_variates, history_length] float tensor."""
    arr = np.asarray(series, dtype="float32")
    if arr.ndim == 1:
        arr = arr.reshape(1, 1, -1)
    elif arr.ndim == 2:
        # Input layout is [T, V], Chronos-2 fit expects [N, V, T].
        arr = arr.T.reshape(1, arr.shape[1], arr.shape[0])
    elif arr.ndim != 3:
        raise ValueError(f"Expected 1-D, 2-D, or 3-D series, got shape {arr.shape}")
    return torch.tensor(arr, dtype=torch.float32)


def build_train_val_series(args, config_path: Path, raw_config: Dict, flat_config: Dict):
    env = raw_config.get("env_settings", {})
    data_paths = env.get("data_paths", {})
    if "load" not in data_paths:
        raise ValueError("config env_settings.data_paths.load is required")

    load_csv = cb.resolve_path(config_path, data_paths["load"])
    load_series, load_column = cb.read_load_series(load_csv, flat_config)

    if not args.use_gap_weather:
        train_series, val_series = split_series_for_fit(load_series)
        stats = None
        return train_series, val_series, load_series, load_column, stats

    era5_value = args.era5_path or data_paths.get("era5")
    if not era5_value:
        raise ValueError("--use-gap-weather requires --era5-path or config env_settings.data_paths.era5")
    era5_path = cb.resolve_path(config_path, era5_value)
    weather = cb.read_gap_weather(era5_path, args.weather_vars)
    aligned_len = min(len(load_series), len(weather))
    train_end, val_end, _ = cb.split_bounds(aligned_len)
    model_series, stats = cb.build_multivariate_series(
        load=load_series[:aligned_len],
        weather=weather[:aligned_len],
        train_end=train_end,
        standardize_load=args.standardize_load,
    )
    return (
        model_series[:train_end],
        model_series[train_end:val_end],
        load_series[:aligned_len],
        load_column,
        stats,
    )


def evaluate_pipeline(pipeline, args, config_path: Path, flat_config: Dict, raw_load: np.ndarray, covariate_stats):
    base_seq_len = int(flat_config.get("seq_len", 96))
    pred_len = int(args.pred_len or flat_config.get("pred_len", 96))
    context_len = int(args.context_len or base_seq_len)

    if args.use_gap_weather:
        env = cb.load_json(config_path).get("env_settings", {})
        era5_value = args.era5_path or env.get("data_paths", {}).get("era5")
        if not era5_value:
            raise ValueError("--use-gap-weather requires --era5-path or config env_settings.data_paths.era5")
        weather = cb.read_gap_weather(cb.resolve_path(config_path, era5_value), args.weather_vars)
        aligned_len = min(len(raw_load), len(weather))
        train_end, _, _ = cb.split_bounds(aligned_len)
        model_series, covariate_stats = cb.build_multivariate_series(
            load=raw_load[:aligned_len],
            weather=weather[:aligned_len],
            train_end=train_end,
            standardize_load=args.standardize_load,
        )
        contexts, targets = cb.make_multivariate_windows(
            series=model_series,
            raw_load=raw_load[:aligned_len],
            split=args.split,
            base_seq_len=base_seq_len,
            context_len=context_len,
            pred_len=pred_len,
            stride=args.stride,
            limit_windows=args.limit_windows,
        )
    else:
        contexts, targets = cb.make_windows(
            series=raw_load,
            split=args.split,
            base_seq_len=base_seq_len,
            context_len=context_len,
            pred_len=pred_len,
            stride=args.stride,
            limit_windows=args.limit_windows,
        )

    forecasts = []
    for batch in cb.iter_batches(contexts, args.eval_batch_size):
        forecasts.append(cb.forecast_batch(pipeline, batch, pred_len, args.num_samples, target_index=0))
    preds = np.concatenate(forecasts, axis=0)

    if args.use_gap_weather and args.standardize_load:
        load_mean = float(covariate_stats["load_mean"][0])
        load_std = float(covariate_stats["load_std"][0])
        preds = preds * load_std + load_mean

    metrics = cb.compute_metrics(preds, targets)
    return contexts, targets, preds, metrics


def load_chronos2_pipeline(model_id: str, args):
    model_path = cb.resolve_offline_model_path(model_id, args.local_files_only)
    try:
        from chronos import Chronos2Pipeline  # type: ignore
    except Exception as exc:
        raise ImportError(
            "Chronos2Pipeline is required for fine-tuning. Install/upgrade chronos-forecasting "
            "to a version that includes Chronos-2 fine-tuning support."
        ) from exc

    dtype_map = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map[args.torch_dtype]
    kwargs = {"device_map": args.device}
    if args.local_files_only:
        kwargs["local_files_only"] = True
    if dtype != "auto":
        kwargs["torch_dtype"] = dtype
    if args.trust_remote_code:
        kwargs["trust_remote_code"] = True

    try:
        pipeline = Chronos2Pipeline.from_pretrained(model_path, **kwargs)
    except TypeError:
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("local_files_only", None)
        pipeline = Chronos2Pipeline.from_pretrained(model_path, **kwargs)

    if not hasattr(pipeline, "fit"):
        raise RuntimeError(
            "The loaded Chronos2Pipeline does not expose fit(). Upgrade chronos-forecasting "
            "or use the package version that supports Chronos-2 fine-tuning."
        )
    return pipeline, str(model_path)


def save_current_pipeline(pipeline, save_dir: Path):
    """Best-effort save of the pipeline object that was actually evaluated."""
    save_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(pipeline, "save_pretrained"):
        pipeline.save_pretrained(save_dir)
        return str(save_dir)

    model = getattr(pipeline, "model", None)
    if model is not None and hasattr(model, "save_pretrained"):
        model.save_pretrained(save_dir)
        tokenizer = getattr(pipeline, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "save_pretrained"):
            tokenizer.save_pretrained(save_dir)
        return str(save_dir)

    raise RuntimeError(
        "The fine-tuned Chronos pipeline does not expose save_pretrained(), "
        "and no save_pretrained()-capable .model was found."
    )


def run(args):
    set_seed(args.seed)
    config_path = Path(args.config)
    raw_config = cb.load_json(config_path)
    flat_config = cb.flatten_config(raw_config)
    pred_len = int(args.pred_len or flat_config.get("pred_len", 96))
    context_len = int(args.context_len or flat_config.get("seq_len", 96))

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_series, val_series, raw_load, load_column, covariate_stats = build_train_val_series(
        args, config_path, raw_config, flat_config
    )
    train_inputs = to_chronos_fit_tensor(train_series)
    val_inputs = to_chronos_fit_tensor(val_series)

    pipeline, resolved_model_path = load_chronos2_pipeline(args.model_id, args)

    fit_kwargs = {
        "inputs": train_inputs,
        "prediction_length": pred_len,
        "validation_inputs": val_inputs,
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
            "target_modules": args.lora_target_modules.split(",") if args.lora_target_modules else None,
        }
        if fit_kwargs["lora_config"]["target_modules"] is None:
            fit_kwargs["lora_config"].pop("target_modules")

    print("Chronos-2 fine-tuning")
    print("-" * 40)
    print(f"model_id          : {args.model_id}")
    print(f"resolved_model    : {resolved_model_path}")
    print(f"config            : {config_path}")
    print(f"load_column       : {load_column}")
    print(f"use_gap_weather   : {args.use_gap_weather}")
    print(f"train_input_shape : {tuple(train_inputs.shape)}")
    print(f"val_input_shape   : {tuple(val_inputs.shape)}")
    print(f"context_len       : {context_len}")
    print(f"pred_len          : {pred_len}")
    print(f"finetune_mode     : {args.finetune_mode}")
    print(f"num_steps         : {args.num_steps}")
    print(f"learning_rate     : {args.learning_rate}")
    print(f"seed              : {args.seed}")
    print("-" * 40)

    finetuned_pipeline = pipeline.fit(**fit_kwargs)

    contexts, targets, preds, metrics = evaluate_pipeline(
        finetuned_pipeline, args, config_path, flat_config, raw_load, covariate_stats
    )

    evaluated_ckpt_dir = output_dir / "evaluated-finetuned-ckpt"
    reloaded_metrics = None
    reloaded_preds = None
    try:
        saved_eval_path = save_current_pipeline(finetuned_pipeline, evaluated_ckpt_dir)
        reload_args = argparse.Namespace(**vars(args))
        reload_args.model_id = saved_eval_path
        reloaded_pipeline, reloaded_model_path = load_chronos2_pipeline(saved_eval_path, reload_args)
        _, _, reloaded_preds, reloaded_metrics = evaluate_pipeline(
            reloaded_pipeline, args, config_path, flat_config, raw_load, covariate_stats
        )
    except Exception as exc:
        saved_eval_path = None
        reloaded_model_path = None
        reloaded_metrics = {"reload_error": str(exc)}

    result = {
        "model": "Chronos-2",
        "model_id": args.model_id,
        "resolved_model_path": resolved_model_path,
        "config": str(config_path),
        "load_column": load_column,
        "dataset_mode": flat_config.get("dataset_mode", "system"),
        "zone_name": flat_config.get("zone_name"),
        "split": args.split,
        "context_len": context_len,
        "pred_len": pred_len,
        "stride": args.stride,
        "num_windows": int(len(contexts)),
        "use_gap_weather": bool(args.use_gap_weather),
        "weather_vars": list(args.weather_vars) if args.use_gap_weather else [],
        "n_variates": int(contexts.shape[1]) if contexts.ndim == 3 else 1,
        "standardize_load": bool(args.standardize_load),
        "finetune": {
            "mode": args.finetune_mode,
            "num_steps": args.num_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "seed": args.seed,
            "lora_r": args.lora_r if args.finetune_mode == "lora" else None,
            "lora_alpha": args.lora_alpha if args.finetune_mode == "lora" else None,
            "lora_dropout": args.lora_dropout if args.finetune_mode == "lora" else None,
            "trainer_output_dir": str(output_dir / "trainer"),
            "evaluated_ckpt_dir": saved_eval_path,
            "reloaded_model_path": reloaded_model_path,
        },
        "metrics": metrics,
        "reloaded_metrics": reloaded_metrics,
    }

    metrics_path = output_dir / "chronos2_finetune_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    np.save(output_dir / "predictions.npy", preds.astype("float32"))
    np.save(output_dir / "truths.npy", targets.astype("float32"))
    if reloaded_preds is not None:
        np.save(output_dir / "reloaded_predictions.npy", reloaded_preds.astype("float32"))

    print("\nChronos-2 fine-tuned evaluation")
    print("-" * 40)
    for key, value in metrics.items():
        print(f"{key:10s}: {value:.4f}")
    if reloaded_metrics is not None:
        print("\nReloaded checkpoint evaluation")
        print("-" * 40)
        for key, value in reloaded_metrics.items():
            if isinstance(value, (int, float)):
                print(f"{key:10s}: {value:.4f}")
            else:
                print(f"{key:10s}: {value}")
    print("-" * 40)
    print(f"Saved metrics to: {metrics_path}")
    if saved_eval_path:
        print(f"Saved evaluated checkpoint to: {saved_eval_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune Chronos-2 for load forecasting.")
    parser.add_argument("--config", default="config.json", help="Project config JSON path.")
    parser.add_argument("--model-id", default="amazon/chronos-2", help="Hugging Face model id or local model path.")
    parser.add_argument("--output-dir", default="experiments/chronos2_finetune/checkpoints/latest_run")
    parser.add_argument("--context-len", type=int, default=None, help="Context length. Defaults to config seq_len.")
    parser.add_argument("--pred-len", type=int, default=None, help="Forecast horizon. Defaults to config pred_len.")
    parser.add_argument("--split", default="test", choices=["val", "test"], help="Evaluation split after fine-tuning.")
    parser.add_argument("--stride", type=int, default=1, help="Sliding-window evaluation stride.")
    parser.add_argument("--limit-windows", type=int, default=None, help="Optional evaluation window limit for smoke tests.")
    parser.add_argument("--batch-size", type=int, default=32, help="Chronos fine-tuning batch size.")
    parser.add_argument("--eval-batch-size", type=int, default=16, help="Chronos inference batch size.")
    parser.add_argument("--num-samples", type=int, default=20, help="Number of probabilistic samples when supported.")
    parser.add_argument("--num-steps", type=int, default=1000, help="Fine-tuning steps.")
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--finetune-mode", default="lora", choices=["lora", "full"])
    parser.add_argument("--finetuned-ckpt-name", default="finetuned-ckpt")
    parser.add_argument("--disable-data-parallel", action="store_true", default=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or a device_map accepted by Chronos.")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"])
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--use-gap-weather", action="store_true", help="Fine-tune/evaluate on load plus GAP weather variables.")
    parser.add_argument("--era5-path", default=None)
    parser.add_argument("--weather-vars", nargs="+", default=["t2m", "u10", "v10", "sp", "tp", "d2m"])
    parser.add_argument("--standardize-load", action="store_true", help="Standardize load only in multivariate GAP-weather mode.")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default=None, help="Optional comma-separated PEFT target_modules.")
    run(parser.parse_args())
