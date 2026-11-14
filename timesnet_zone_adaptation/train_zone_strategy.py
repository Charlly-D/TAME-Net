"""Run one TimesNet global-to-local adaptation strategy on one target zone."""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_provider import build_dataloaders
from engines.trainer import evaluate, train_one_epoch
# Legacy Cross-Attention implementation:
# from models.multimodal_load_model import MultimodalLoadModel
from experiments.timesnet_ablation_suite.ablation_model import AblationMultimodalLoadModel
from utils.eval_metrics import eval_load_forecasting
from experiments.timesnet_zone_adaptation.adaptation_utils import (
    flatten_config,
    improvement,
    inverse_load_scale,
    jsonable_metrics,
    load_global_weights,
    metric_brief,
    read_json,
    resolve_data_paths,
    write_json,
)


STRATEGIES = (
    "direct_reuse",
    "local_scratch",
    "encoder_transfer",
    "full_model_finetune",
)


def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_trainable(model: torch.nn.Module, predicate):
    for name, param in model.named_parameters():
        param.requires_grad = bool(predicate(name))


def synchronize_device(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def eval_metrics(model, loader, criterion, device, config, mode="finetune"):
    loss, preds, truths = evaluate(model, loader, criterion, device, mode=mode, config=config)
    stats = loader.dataset.stats
    preds_real = inverse_load_scale(preds, stats)
    truths_real = inverse_load_scale(truths, stats)
    metrics = eval_load_forecasting(preds_real, truths_real, verbose=False)
    return float(loss), jsonable_metrics(metrics), preds, truths


def fit_stage(model, train_loader, val_loader, criterion, device, config, mode, stage_name, predicate, lr, weight_decay, epochs, patience_limit):
    set_trainable(model, predicate)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[{stage_name}] trainable parameters: {sum(p.numel() for p in trainable)}")
    if not trainable:
        raise RuntimeError(f"No trainable parameters for {stage_name}")

    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=config.get("when", 20),
        factor=0.1,
    )
    best_val = float("inf")
    best_state = None
    patience = 0
    history = []
    synchronize_device(device)
    stage_start = time.perf_counter()

    for epoch in range(epochs):
        synchronize_device(device)
        epoch_start = time.perf_counter()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, epoch, mode=mode, config=config
        )
        val_loss, _, _ = evaluate(model, val_loader, criterion, device, mode=mode, config=config)
        val_metrics = None
        if mode == "finetune":
            _, val_metrics, _, _ = eval_metrics(model, val_loader, criterion, device, config, mode=mode)
        scheduler.step(val_loss)
        synchronize_device(device)
        epoch_seconds = time.perf_counter() - epoch_start
        stage_elapsed_seconds = time.perf_counter() - stage_start
        row = {
            "stage": stage_name,
            "epoch": epoch + 1,
            "mode": mode,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_metrics": val_metrics,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_seconds": float(epoch_seconds),
            "stage_elapsed_seconds": float(stage_elapsed_seconds),
        }
        history.append(row)
        print(
            f"[{stage_name}] Epoch {epoch + 1:02d}/{epochs} | "
            f"Train {train_loss:.5f} | Val {val_loss:.5f} | LR {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            patience = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= patience_limit:
                print(f"[{stage_name}] Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return history


def train_staged(model, train_loader, val_loader, criterion, device, config, args, prefix):
    history = []
    stage_summaries = []
    elapsed_before_stage = 0.0
    epochs = args.epochs
    patience = args.patience

    def append_stage(stage_history):
        nonlocal elapsed_before_stage
        if not stage_history:
            return
        for row in stage_history:
            row["training_elapsed_seconds"] = float(
                elapsed_before_stage + row["stage_elapsed_seconds"]
            )
        best_row = min(stage_history, key=lambda row: row["val_loss"])
        stage_total = float(stage_history[-1]["stage_elapsed_seconds"])
        stage_summaries.append({
            "stage": stage_history[0]["stage"],
            "mode": stage_history[0]["mode"],
            "epochs_run": len(stage_history),
            "best_epoch": int(best_row["epoch"]),
            "best_val_loss": float(best_row["val_loss"]),
            "stage_total_seconds": stage_total,
            "time_to_best_seconds": float(best_row["stage_elapsed_seconds"]),
            "cumulative_time_to_best_seconds": float(
                elapsed_before_stage + best_row["stage_elapsed_seconds"]
            ),
        })
        history.extend(stage_history)
        elapsed_before_stage += stage_total

    append_stage(fit_stage(
        model, train_loader, val_loader, criterion, device, config,
        "pretrain_backbone", f"{prefix} Backbone",
        lambda name: name.startswith("backbone"),
        args.backbone_lr, args.backbone_weight_decay, epochs, patience,
    ))
    append_stage(fit_stage(
        model, train_loader, val_loader, criterion, device, config,
        "pretrain_weather", f"{prefix} Weather Residual",
        lambda name: (
            name.startswith("weather_encoder")
            or name.startswith("weather_proj")
            or name.startswith("router.weather_align")
            or name.startswith("router.weather_norm")
        ),
        args.encoder_lr, args.encoder_weight_decay, epochs, patience,
    ))
    append_stage(fit_stage(
        model, train_loader, val_loader, criterion, device, config,
        "pretrain_img", f"{prefix} Image Residual",
        lambda name: (
            name.startswith("image_encoder")
            or name.startswith("image_proj")
            or name.startswith("router.image_align")
            or name.startswith("router.image_norm")
        ),
        args.encoder_lr, args.encoder_weight_decay, epochs, patience,
    ))
    model.fusion_mode = config.get("fusion_mode", "residual_expert")
    append_stage(fit_stage(
        model, train_loader, val_loader, criterion, device, config,
        "finetune", f"{prefix} Router",
        lambda name: name.startswith("router.router_mlp") or name.startswith("router.pool_score"),
        args.router_lr, args.router_weight_decay, args.router_epochs, patience,
    ))
    final_stage = stage_summaries[-1]
    timing = {
        "scope": "training_and_validation_only",
        "cuda_synchronized": device.type == "cuda",
        "total_training_seconds": float(elapsed_before_stage),
        "time_to_final_stage_best_seconds": float(final_stage["cumulative_time_to_best_seconds"]),
        "stages": stage_summaries,
    }
    return history, timing


def train_full_model_finetune(
    model, train_loader, val_loader, criterion, device, config, args
):
    history = fit_stage(
        model,
        train_loader,
        val_loader,
        criterion,
        device,
        config,
        "finetune",
        "Full-Model Fine-Tuning",
        lambda _name: True,
        args.full_finetune_lr,
        args.full_finetune_weight_decay,
        args.epochs,
        args.patience,
    )
    best_row = min(history, key=lambda row: row["val_loss"])
    stage_total = float(history[-1]["stage_elapsed_seconds"])
    time_to_best = float(best_row["stage_elapsed_seconds"])
    for row in history:
        row["training_elapsed_seconds"] = float(row["stage_elapsed_seconds"])
    timing = {
        "scope": "training_and_validation_only",
        "cuda_synchronized": device.type == "cuda",
        "total_training_seconds": stage_total,
        "time_to_final_stage_best_seconds": time_to_best,
        "stages": [
            {
                "stage": history[0]["stage"],
                "mode": history[0]["mode"],
                "epochs_run": len(history),
                "best_epoch": int(best_row["epoch"]),
                "best_val_loss": float(best_row["val_loss"]),
                "stage_total_seconds": stage_total,
                "time_to_best_seconds": time_to_best,
                "cumulative_time_to_best_seconds": time_to_best,
            }
        ],
    }
    return history, timing


def run(args):
    if args.strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy={args.strategy}")

    set_seed(args.seed)
    raw = read_json(args.config)
    config = flatten_config(raw)
    config.update({
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "shuffle_val_test": False,
    })
    if args.router_init_bias is not None:
        config["router_init_bias"] = args.router_init_bias
    data_paths = resolve_data_paths(args.config, raw)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_loader, val_loader, test_loader = build_dataloaders(data_paths, config, str(output_dir))
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    criterion = torch.nn.SmoothL1Loss()
    # Legacy Cross-Attention model:
    # model = MultimodalLoadModel(config).to(device)
    model = AblationMultimodalLoadModel(config, variant="full").to(device)

    transfer_report = None
    if args.strategy == "direct_reuse":
        if not args.global_checkpoint:
            raise ValueError("--global-checkpoint is required for direct_reuse")
        transfer_report = load_global_weights(
            model,
            args.global_checkpoint,
            include_prefixes=None,
            allow_resize=args.resize_direct_reuse,
        )
        model = model.to(device)
        print(f"[{args.zone}:direct_reuse] {transfer_report['loaded_count']} tensors loaded, "
              f"{transfer_report['resized_count']} resized, {transfer_report['skipped_count']} skipped.")
        test_loss, test_metrics, preds, truths = eval_metrics(model, test_loader, criterion, device, config)
        history = []
        timing = {
            "scope": "no_local_training",
            "cuda_synchronized": device.type == "cuda",
            "total_training_seconds": 0.0,
            "time_to_final_stage_best_seconds": 0.0,
            "stages": [],
        }
    elif args.strategy == "local_scratch":
        history, timing = train_staged(model, train_loader, val_loader, criterion, device, config, args, "Local Scratch")
        test_loss, test_metrics, preds, truths = eval_metrics(model, test_loader, criterion, device, config)
    elif args.strategy == "encoder_transfer":
        if not args.global_checkpoint:
            raise ValueError("--global-checkpoint is required for encoder_transfer")
        transfer_report = load_global_weights(
            model,
            args.global_checkpoint,
            include_prefixes=("weather_encoder.", "image_encoder."),
            allow_resize=False,
        )
        model = model.to(device)
        print(f"[{args.zone}:encoder_transfer] {transfer_report['loaded_count']} encoder tensors loaded, "
              f"{transfer_report['skipped_count']} incompatible tensors kept local/trainable.")
        history, timing = train_staged(model, train_loader, val_loader, criterion, device, config, args, "Encoder Transfer")
        test_loss, test_metrics, preds, truths = eval_metrics(model, test_loader, criterion, device, config)
    else:
        if not args.global_checkpoint:
            raise ValueError("--global-checkpoint is required for full_model_finetune")
        transfer_report = load_global_weights(
            model,
            args.global_checkpoint,
            include_prefixes=None,
            allow_resize=args.resize_full_finetune,
        )
        model = model.to(device)
        model.fusion_mode = config.get("fusion_mode", "residual_expert")
        print(
            f"[{args.zone}:full_model_finetune] "
            f"{transfer_report['loaded_count']} tensors loaded, "
            f"{transfer_report['resized_count']} resized, "
            f"{transfer_report['skipped_count']} skipped."
        )
        history, timing = train_full_model_finetune(
            model, train_loader, val_loader, criterion, device, config, args
        )
        test_loss, test_metrics, preds, truths = eval_metrics(
            model, test_loader, criterion, device, config
        )

    checkpoint_path = output_dir / f"{args.zone}_{args.strategy}.pth"
    pred_path = output_dir / f"{args.zone}_{args.strategy}_test_predictions.pt"
    result_path = output_dir / f"{args.zone}_{args.strategy}_results.json"
    torch.save(model.state_dict(), checkpoint_path)
    torch.save({"preds": preds, "truths": truths}, pred_path)

    result = {
        "zone": args.zone,
        "strategy": args.strategy,
        "config": args.config,
        "global_checkpoint": args.global_checkpoint,
        "checkpoint": str(checkpoint_path),
        "prediction_path": str(pred_path),
        "test_loss": test_loss,
        "test_metrics": test_metrics,
        "transfer_report": transfer_report,
        "history": history,
        "timing": timing,
        "settings": {
            "interaction_operator": "interpolation",
            "router_init_bias": config.get("router_init_bias"),
            "epochs": args.epochs,
            "router_epochs": args.router_epochs,
            "backbone_lr": args.backbone_lr,
            "encoder_lr": args.encoder_lr,
            "router_lr": args.router_lr,
            "full_finetune_lr": args.full_finetune_lr,
            "full_finetune_weight_decay": args.full_finetune_weight_decay,
            "resize_full_finetune": args.resize_full_finetune,
            "batch_size": args.batch_size,
            "patience": args.patience,
            "seed": args.seed,
        },
    }
    if args.baseline_results:
        baseline = read_json(args.baseline_results)
        baseline_metrics = baseline.get("metrics") or baseline.get("final_metrics") or baseline.get("test_metrics")
        result["baseline_metrics"] = baseline_metrics
        result["improvement_vs_baseline"] = improvement(baseline_metrics, test_metrics)

    write_json(result, result_path)
    print(f"[{args.zone}:{args.strategy}] test {metric_brief(test_metrics)}")
    print(f"Saved result to {result_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one TimesNet zone adaptation strategy.")
    parser.add_argument("--zone", required=True)
    parser.add_argument("--strategy", required=True, choices=STRATEGIES)
    parser.add_argument("--config", required=True)
    parser.add_argument("--global-checkpoint", default=None)
    parser.add_argument("--baseline-results", default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--router-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--backbone-weight-decay", type=float, default=1e-2)
    parser.add_argument("--encoder-lr", type=float, default=1e-4)
    parser.add_argument("--encoder-weight-decay", type=float, default=1e-2)
    parser.add_argument("--router-lr", type=float, default=1e-4)
    parser.add_argument("--router-init-bias", type=float, default=None)
    parser.add_argument("--router-weight-decay", type=float, default=1e-3)
    parser.add_argument("--full-finetune-lr", type=float, default=1e-4)
    parser.add_argument("--full-finetune-weight-decay", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--resize-direct-reuse", action="store_true")
    parser.add_argument("--resize-full-finetune", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    run(parser.parse_args())
