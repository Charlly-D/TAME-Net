"""Utilities for TimesNet global-to-local zone adaptation experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable


def read_json(path: str | Path) -> dict:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def write_json(payload: dict, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def parse_key_value_pairs(items: Iterable[str] | None) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"Expected NAME=PATH format, got: {item}")
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key or not value:
            raise ValueError(f"Expected NAME=PATH format, got: {item}")
        mapping[key] = value
    return mapping


def resolve_zone_configs(
    zones: Iterable[str],
    zone_config_template: str | None,
    zone_configs: Iterable[str] | None,
) -> Dict[str, str]:
    explicit = parse_key_value_pairs(zone_configs)
    resolved = {}
    for zone in zones:
        if zone in explicit:
            resolved[zone] = explicit[zone]
        elif zone_config_template:
            resolved[zone] = zone_config_template.format(zone=zone)
        else:
            raise ValueError(f"No config path provided for zone {zone}.")
    return resolved


def flatten_config(raw: dict) -> dict:
    config = {}
    config.update(raw.get("data_settings", {}))
    config.update(raw.get("model_settings", {}))
    training = raw.get("training_settings", {})
    config.update(training)
    for stage in ("stage1", "stage2", "stage3"):
        config.update(training.get(stage, {}))
    return config


def resolve_data_paths(config_path: str | Path, raw: dict) -> dict:
    base_dir = Path(config_path).resolve().parent
    resolved = {}
    for key, value in raw["env_settings"]["data_paths"].items():
        path = Path(value)
        resolved[key] = str(path if path.is_absolute() else base_dir / path)
    return resolved


def resize_tensor(source: Any, target: Any) -> Any | None:
    if source.shape == target.shape:
        return source
    if source.ndim != target.ndim:
        return None

    import torch.nn.functional as F

    if source.ndim == 3 and source.shape[0] == target.shape[0] and source.shape[-1] == target.shape[-1]:
        x = source.transpose(1, 2)
        x = F.interpolate(x, size=target.shape[1], mode="linear", align_corners=False)
        return x.transpose(1, 2).to(dtype=target.dtype)

    if source.ndim == 4 and source.shape[:2] == target.shape[:2]:
        x = F.interpolate(source, size=target.shape[-2:], mode="bilinear", align_corners=False)
        return x.to(dtype=target.dtype)

    if source.ndim == 2 and source.shape[0] == target.shape[0]:
        x = F.interpolate(source.unsqueeze(1), size=target.shape[1], mode="linear", align_corners=False)
        return x.squeeze(1).to(dtype=target.dtype)

    return None


def load_state_dict_payload(path: str | Path) -> dict:
    import torch

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)}")
    return payload


def load_global_weights(
    model: Any,
    checkpoint_path: str | Path,
    include_prefixes: tuple[str, ...] | None = None,
    allow_resize: bool = False,
) -> dict:
    """Load compatible global parameters into a target-zone model.

    For encoder-transfer experiments, include_prefixes should be set to
    ("weather_encoder.", "image_encoder."). Shape-mismatched encoder tensors are
    skipped and remain trainable as local private parameters.
    """
    source = load_state_dict_payload(checkpoint_path)
    target = model.state_dict()
    to_load = {}
    loaded = []
    resized = []
    skipped = []
    missing = []

    for key, target_value in target.items():
        if include_prefixes and not key.startswith(include_prefixes):
            continue

        source_key = key
        if source_key not in source and f"module.{key}" in source:
            source_key = f"module.{key}"
        if source_key not in source:
            missing.append(key)
            continue

        source_value = source[source_key].detach().cpu()
        if source_value.shape == target_value.shape:
            to_load[key] = source_value
            loaded.append(key)
            continue

        if allow_resize:
            resized_value = resize_tensor(source_value, target_value.detach().cpu())
            if resized_value is not None:
                to_load[key] = resized_value
                resized.append({
                    "key": key,
                    "global_shape": list(source_value.shape),
                    "local_shape": list(target_value.shape),
                })
                continue

        skipped.append({
            "key": key,
            "global_shape": list(source_value.shape),
            "local_shape": list(target_value.shape),
        })

    msg = model.load_state_dict(to_load, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "include_prefixes": list(include_prefixes) if include_prefixes else None,
        "allow_resize": allow_resize,
        "loaded_count": len(loaded),
        "resized_count": len(resized),
        "skipped_count": len(skipped),
        "missing_count": len(missing),
        "loaded_keys_preview": loaded[:30],
        "resized": resized,
        "skipped_preview": skipped[:50],
        "missing_preview": missing[:50],
        "load_state_missing_keys_preview": list(msg.missing_keys)[:50],
        "load_state_unexpected_keys_preview": list(msg.unexpected_keys)[:50],
    }


def inverse_load_scale(tensor: Any, stats: dict) -> Any:
    import torch

    mean = torch.tensor(stats["load_mean"], dtype=tensor.dtype).view(1, 1, -1)
    std = torch.tensor(stats["load_std"], dtype=tensor.dtype).view(1, 1, -1)
    return tensor.cpu() * std + mean


def jsonable_metrics(metrics: dict | None) -> dict | None:
    if metrics is None:
        return None
    return {key: float(value) for key, value in metrics.items()}


def metric_brief(metrics: dict | None) -> str:
    if not metrics:
        return "{}"
    keys = ("MAE", "RMSE", "MAPE(%)", "R2", "Corr")
    return ", ".join(f"{key}={float(metrics[key]):.4f}" for key in keys if key in metrics)


def improvement(base: dict | None, target: dict | None) -> dict:
    if not base or not target:
        return {}
    output = {}
    for key in ("MAE", "RMSE", "MAPE(%)", "WAPE(%)"):
        if key in base and key in target and float(base[key]) != 0:
            output[f"{key}_reduction(%)"] = (float(base[key]) - float(target[key])) / float(base[key]) * 100.0
    return output
