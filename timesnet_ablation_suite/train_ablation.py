from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

SUITE_DIR = Path(__file__).resolve().parent
ROOT_DIR = SUITE_DIR.parents[1]
for path in (str(SUITE_DIR), str(ROOT_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from ablation_model import AblationMultimodalLoadModel
from data_provider import build_dataloaders
from engines.trainer import evaluate, train_one_epoch
from utils.eval_metrics import eval_load_forecasting


def load_project_config(path: str) -> Tuple[Dict, Dict, str]:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8-sig") as f:
        raw = json.load(f)

    config: Dict = {}
    config.update(raw.get("data_settings", {}))
    config.update(raw.get("model_settings", {}))
    training = raw.get("training_settings", {})
    config.update(training)
    for stage_name in ("stage1", "stage2", "stage3"):
        config.update(training.get(stage_name, {}))

    env = raw.get("env_settings", {})
    return config, env.get("data_paths", {}), env.get("save_dir", "./checkpoints/ablation_suite")


def resolve_paths(config_path: str, data_paths: Dict[str, str]) -> Dict[str, str]:
    root = Path(config_path).resolve().parent
    resolved = {}
    for key, value in data_paths.items():
        path = Path(value)
        resolved[key] = str(path if path.is_absolute() else root / path)
    return resolved


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def inverse_load_scale(tensor: torch.Tensor, stats: Dict) -> torch.Tensor:
    mean = torch.tensor(stats["load_mean"], dtype=tensor.dtype).view(1, 1, -1)
    std = torch.tensor(stats["load_std"], dtype=tensor.dtype).view(1, 1, -1)
    return tensor * std + mean


class AblationTrainer:
    def __init__(
        self,
        config: Dict,
        save_dir: str,
        variant: str,
        device: torch.device,
        backbone_checkpoint=None,
        report_split: str = "test",
    ):
        self.config = config
        self.save_dir = save_dir
        self.variant = variant
        self.device = device
        self.backbone_checkpoint = backbone_checkpoint
        self.report_split = report_split
        self.model = AblationMultimodalLoadModel(config, variant=variant).to(device)
        self.stage_results = []
        self.last_fit_summary = None

    def prepare_data(self, data_paths):
        self.train_loader, self.val_loader, self.test_loader = build_dataloaders(
            data_paths, self.config, self.save_dir
        )

    def _set_trainable_by_name(self, predicate):
        for name, param in self.model.named_parameters():
            param.requires_grad = predicate(name)

    def _load_compatible_state(self, path, label, allow_partial=False, include=None):
        if not os.path.exists(path):
            return False
        state = torch.load(path, map_location=self.device)
        if include is not None:
            state = {name: value for name, value in state.items() if include(name)}
        current = self.model.state_dict()
        compatible = {k: v for k, v in state.items() if k in current and current[k].shape == v.shape}
        skipped = sorted(set(state.keys()) - set(compatible.keys()))
        self.model.load_state_dict(compatible, strict=False)
        print(f"[{label}] Loaded {len(compatible)} compatible tensors from {path}")
        if skipped:
            print(f"[{label}] Skipped {len(skipped)} incompatible or unknown tensors.")
        return allow_partial or not skipped

    def _fit_stage(self, mode, optimizer, epochs, best_path, stage_name):
        criterion = nn.SmoothL1Loss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.config.get("when", 20), factor=0.1
        )
        best_val = float("inf")
        best_epoch = 0
        patience = 0
        patience_limit = self.config.get("early_stop_patience", 5)
        history = []

        for epoch in range(epochs):
            train_loss = train_one_epoch(
                self.model,
                self.train_loader,
                optimizer,
                criterion,
                self.device,
                epoch,
                mode=mode,
                config=self.config,
            )
            val_loss, _, _ = evaluate(
                self.model, self.val_loader, criterion, self.device, mode=mode, config=self.config
            )
            scheduler.step(val_loss)
            lr = optimizer.param_groups[0]["lr"]
            history.append({
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "learning_rates": [float(group["lr"]) for group in optimizer.param_groups],
            })
            print(
                f"[{self.variant} | {stage_name}] Epoch {epoch + 1:02d}/{epochs} | "
                f"Train {train_loss:.5f} | Val {val_loss:.5f} | LR {lr:.2e}"
            )
            if val_loss < best_val:
                best_val = val_loss
                best_epoch = epoch + 1
                patience = 0
                torch.save(self.model.state_dict(), best_path)
            else:
                patience += 1
                if patience >= patience_limit:
                    print(f"[{self.variant} | {stage_name}] Early stopping.")
                    break

        history_name = stage_name.lower().replace(" ", "_").replace("-", "_")
        history_path = os.path.join(self.save_dir, f"{history_name}_history.json")
        self.last_fit_summary = {
            "stage": stage_name,
            "mode": mode,
            "maximum_epochs": int(epochs),
            "epochs_run": len(history),
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "early_stop_patience": int(patience_limit),
            "history_path": history_path,
        }
        with open(history_path, "w", encoding="utf-8") as stream:
            json.dump({**self.last_fit_summary, "history": history}, stream, indent=2, ensure_ascii=False)

        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        return best_val

    def stage1_pretrain(self):
        os.makedirs(self.save_dir, exist_ok=True)
        epochs = self.config.get("pretrain_epochs", 40)

        backbone_path = os.path.join(self.save_dir, "stage1a_backbone.pth")
        backbone_loaded = self._load_compatible_state(
            backbone_path,
            "Stage 1a",
            allow_partial=True,
            include=lambda name: name.startswith("backbone"),
        )
        if not backbone_loaded and self.backbone_checkpoint:
            backbone_loaded = self._load_compatible_state(
                self.backbone_checkpoint,
                "External Stage 1a",
                allow_partial=True,
                include=lambda name: name.startswith("backbone"),
            )
        if not backbone_loaded:
            self._set_trainable_by_name(lambda name: name.startswith("backbone"))
            optimizer = torch.optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.get("pretrain_lr", 1e-4),
                weight_decay=self.config.get("backbone_pretrain_weight_decay", 1e-2),
            )
            self._fit_stage("pretrain_backbone", optimizer, epochs, backbone_path, "Stage 1a Backbone")
        self.record_metrics("Stage 1a Backbone", mode="pretrain_backbone")

        if self.variant in {"base_only", "no_residual_expert"}:
            return

        if self.model.use_weather:
            weather_path = os.path.join(self.save_dir, "stage1b_weather_residual.pth")
            weather_parameter = lambda name: (
                ("weather_encoder" in name)
                or ("weather_proj" in name)
                or (self.model.use_cross_attention and "router.weather_align" in name)
                or (self.model.use_cross_attention and "router.weather_norm" in name)
            )
            if not self._load_compatible_state(
                weather_path,
                "Stage 1b",
                include=weather_parameter,
            ):
                self._set_trainable_by_name(weather_parameter)
                optimizer = torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.config.get("weather_pretrain_lr", self.config.get("pretrain_lr", 1e-4)),
                    weight_decay=self.config.get("modal_pretrain_weight_decay", 1e-2),
                )
                self._fit_stage("pretrain_weather", optimizer, epochs, weather_path, "Stage 1b Weather Residual")

        if self.model.use_image:
            image_path = os.path.join(self.save_dir, "stage1c_image_residual.pth")
            image_parameter = lambda name: (
                ("image_encoder" in name)
                or ("image_proj" in name)
                or (self.model.use_cross_attention and "router.image_align" in name)
                or (self.model.use_cross_attention and "router.image_norm" in name)
            )
            if not self._load_compatible_state(
                image_path,
                "Stage 1c",
                include=image_parameter,
            ):
                self._set_trainable_by_name(image_parameter)
                optimizer = torch.optim.AdamW(
                    [p for p in self.model.parameters() if p.requires_grad],
                    lr=self.config.get("image_pretrain_lr", self.config.get("pretrain_lr", 1e-4)),
                    weight_decay=self.config.get("modal_pretrain_weight_decay", 1e-2),
                )
                self._fit_stage("pretrain_img", optimizer, epochs, image_path, "Stage 1c Image Residual")

    def stage2_train_fusion(self):
        if self.variant == "base_only":
            self.record_metrics("Stage 2 Skipped", mode="finetune", extra={"skipped": True})
            return

        criterion = nn.SmoothL1Loss()
        base_val, _, _ = evaluate(
            self.model, self.val_loader, criterion, self.device, mode="pretrain_backbone", config=self.config
        )
        before_stage2 = copy.deepcopy(self.model.state_dict())
        print(f"[{self.variant} | Stage 2] Base-only Val Loss: {base_val:.5f}")

        if self.variant == "no_route":
            self.record_metrics("Stage 2 Fixed-Weight Fusion", mode="finetune", extra={
                "base_val_loss": float(base_val),
                "note": "No trainable route; residual experts use fixed modality weights.",
            })
            return

        if self.variant == "no_residual_expert":
            self._set_trainable_by_name(lambda name: not name.startswith("backbone"))
        else:
            self._set_trainable_by_name(lambda name: (
                name.startswith("router.router_mlp") or name.startswith("router.pool_score")
            ))

        trainable = [p for p in self.model.parameters() if p.requires_grad]
        print(f"[{self.variant} | Stage 2] Trainable parameters: {sum(p.numel() for p in trainable)}")
        if not trainable:
            self.record_metrics("Stage 2 Skipped", mode="finetune", extra={"skipped": True})
            return

        optimizer = torch.optim.AdamW(
            trainable,
            lr=self.config.get("router_lr", 1e-4),
            weight_decay=self.config.get("router_weight_decay", 1e-3),
        )
        best_path = os.path.join(self.save_dir, "stage2_fusion.pth")
        best_val = self._fit_stage(
            "finetune", optimizer, self.config.get("router_epochs", 20), best_path, "Stage 2 Fusion"
        )
        min_gain = self.config.get("fusion_min_gain", 0.0)
        accepted = True
        if self.config.get("fallback_to_base_if_worse", False) and best_val >= base_val - min_gain:
            self.model.load_state_dict(before_stage2)
            accepted = False
        self.record_metrics("Stage 2 Fusion", mode="finetune", extra={
            "best_val_loss": float(best_val),
            "base_val_loss": float(base_val),
            "accepted_fusion": accepted,
        })

    def train_end_to_end(self):
        """Train the full multimodal model from scratch without staged freezing."""
        os.makedirs(self.save_dir, exist_ok=True)
        self._set_trainable_by_name(lambda name: (
            self.model.use_cross_attention
            or ("router.weather_align" not in name
                and "router.weather_norm" not in name
                and "router.image_align" not in name
                and "router.image_norm" not in name)
        ))
        group_settings = {
            "backbone": {
                "lr": self.config.get("pretrain_lr", 1e-4),
                "weight_decay": self.config.get("backbone_pretrain_weight_decay", 1e-2),
            },
            "weather_branch": {
                "lr": self.config.get("weather_pretrain_lr", self.config.get("pretrain_lr", 1e-4)),
                "weight_decay": self.config.get("modal_pretrain_weight_decay", 1e-2),
            },
            "image_branch": {
                "lr": self.config.get("image_pretrain_lr", self.config.get("pretrain_lr", 1e-4)),
                "weight_decay": self.config.get("modal_pretrain_weight_decay", 1e-2),
            },
            "router_scoring": {
                "lr": self.config.get("router_lr", 1e-4),
                "weight_decay": self.config.get("router_weight_decay", 1e-3),
            },
            "other": {
                "lr": self.config.get("end_to_end_lr", self.config.get("finetune_lr", 1e-4)),
                "weight_decay": self.config.get("end_to_end_weight_decay", 1e-2),
            },
        }
        grouped_parameters = {name: [] for name in group_settings}
        for name, parameter in self.model.named_parameters():
            if not parameter.requires_grad:
                continue
            if name.startswith("backbone."):
                group_name = "backbone"
            elif (
                name.startswith("weather_encoder.")
                or name.startswith("weather_proj.")
                or name.startswith("router.weather_align.")
                or name.startswith("router.weather_norm.")
            ):
                group_name = "weather_branch"
            elif (
                name.startswith("image_encoder.")
                or name.startswith("image_proj.")
                or name.startswith("router.image_align.")
                or name.startswith("router.image_norm.")
            ):
                group_name = "image_branch"
            elif name.startswith("router.router_mlp.") or name.startswith("router.pool_score."):
                group_name = "router_scoring"
            else:
                group_name = "other"
            grouped_parameters[group_name].append(parameter)

        optimizer_groups = []
        optimizer_report = {}
        for group_name, parameters in grouped_parameters.items():
            if not parameters:
                continue
            settings = group_settings[group_name]
            optimizer_groups.append({"params": parameters, **settings})
            optimizer_report[group_name] = {
                **settings,
                "parameter_count": int(sum(parameter.numel() for parameter in parameters)),
            }

        trainable = [parameter for parameters in grouped_parameters.values() for parameter in parameters]
        print(f"[{self.variant} | End-to-End] Trainable parameters: {sum(p.numel() for p in trainable)}")
        print(f"[{self.variant} | End-to-End] Optimizer groups: {json.dumps(optimizer_report, ensure_ascii=False)}")
        optimizer = torch.optim.AdamW(optimizer_groups)
        best_path = os.path.join(self.save_dir, "end_to_end_model.pth")
        best_val = self._fit_stage(
            "finetune",
            optimizer,
            self.config.get("end_to_end_epochs", self.config.get("pretrain_epochs", 40)),
            best_path,
            "End-to-End",
        )
        self.record_metrics("End-to-End Full Model", mode="finetune", extra={
            "best_val_loss": float(best_val),
            "training_strategy": "end_to_end_from_scratch",
            "optimizer_groups": optimizer_report,
            "fit_summary": self.last_fit_summary,
            "note": "Full model trained jointly from random initialization with component-matched optimizer settings and without staged pretraining.",
        })

    def evaluate_model(self, split="test", mode="finetune"):
        if split == "validation":
            loader = self.val_loader
        elif split == "test":
            loader = self.test_loader
        else:
            raise ValueError(f"Unsupported evaluation split: {split}")
        criterion = nn.SmoothL1Loss()
        _, results, truths = evaluate(
            self.model, loader, criterion, self.device, mode=mode, config=self.config
        )
        stats = self.train_loader.dataset.stats
        preds_real = inverse_load_scale(results.cpu(), stats)
        truths_real = inverse_load_scale(truths.cpu(), stats)
        metrics = eval_load_forecasting(preds_real, truths_real, verbose=True)
        return {k: float(v) for k, v in metrics.items()}

    def test_model(self, mode="finetune"):
        return self.evaluate_model(split="test", mode=mode)

    def record_metrics(self, stage_name, mode="finetune", extra=None):
        print(f"[{self.variant} | {stage_name}] {self.report_split.title()} evaluation ({mode})")
        metrics = self.evaluate_model(split=self.report_split, mode=mode)
        record = {
            "stage": stage_name,
            "mode": mode,
            "split": self.report_split,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "metrics": metrics,
        }
        if extra:
            record.update(extra)
        self.stage_results.append(record)
        return metrics

    def save_artifacts(self, final_metrics):
        weights_path = os.path.join(self.save_dir, f"{self.variant}_final.pth")
        torch.save(self.model.state_dict(), weights_path)
        payload = {
            "variant": self.variant,
            "reported_split": self.report_split,
            "description": self.model.ablation_description(),
            "final_metrics": final_metrics,
            "stage_results": self.stage_results,
            "config": self.config,
            "artifacts": {
                "weights_path": weights_path,
                "scaler_path": os.path.join(self.save_dir, "scaler_stats.json"),
            },
        }
        results_path = os.path.join(self.save_dir, f"{self.variant}_results.json")
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return weights_path, results_path


def run(args):
    set_seed(args.seed)
    if args.backbone_checkpoint and not os.path.isfile(args.backbone_checkpoint):
        raise FileNotFoundError(f"Backbone checkpoint not found: {args.backbone_checkpoint}")
    config, data_paths, base_save_dir = load_project_config(args.config)
    config["batch_size"] = args.batch_size or config.get("batch_size", 32)
    config["num_workers"] = args.num_workers if args.num_workers is not None else config.get("num_workers", 4)
    config["pretrain_epochs"] = args.pretrain_epochs or config.get("pretrain_epochs", 40)
    config["router_epochs"] = args.router_epochs or config.get("router_epochs", 20)
    config["end_to_end_epochs"] = args.end_to_end_epochs or config.get("end_to_end_epochs", config["pretrain_epochs"])
    config["end_to_end_lr"] = args.end_to_end_lr or config.get("end_to_end_lr", config.get("finetune_lr", 1e-4))
    if args.pred_len is not None:
        config["pred_len"] = args.pred_len
    if args.router_init_bias is not None:
        config["router_init_bias"] = args.router_init_bias
    if args.aux_delta_scale is not None:
        config["aux_delta_scale"] = args.aux_delta_scale
    if args.no_fallback:
        config["fallback_to_base_if_worse"] = False

    data_paths = resolve_paths(args.config, data_paths)
    save_dir = args.save_dir or os.path.join(base_save_dir, "ablation_suite", args.variant)
    os.makedirs(save_dir, exist_ok=True)

    if torch.cuda.is_available() and not args.cpu:
        device = torch.device(f"cuda:{args.gpu}")
    else:
        device = torch.device("cpu")
    print(f"[Ablation] variant={args.variant} device={device} save_dir={save_dir}")

    trainer = AblationTrainer(
        config,
        save_dir,
        args.variant,
        device,
        backbone_checkpoint=args.backbone_checkpoint,
        report_split=args.report_split,
    )
    trainer.prepare_data(data_paths)
    if args.variant == "end_to_end":
        trainer.train_end_to_end()
        final_metrics = trainer.record_metrics("Final Model", mode="finetune")
    else:
        trainer.stage1_pretrain()
        trainer.stage2_train_fusion()
        final_metrics = trainer.record_metrics("Final Model", mode="pretrain_backbone" if args.variant == "base_only" else "finetune")
    _, results_path = trainer.save_artifacts(final_metrics)
    print(f"[Ablation] Saved results to {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Train one ablation variant of the multimodal residual-expert model.")
    parser.add_argument("--variant", required=True, choices=sorted(AblationMultimodalLoadModel.VALID_VARIANTS))
    parser.add_argument("--config", default="zone_config.json")
    parser.add_argument("--save-dir", default=None)
    parser.add_argument(
        "--backbone-checkpoint",
        default=None,
        help="Optional checkpoint from which only TimesNet backbone parameters are loaded.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--pred-len", type=int, default=None)
    parser.add_argument("--pretrain-epochs", type=int, default=None)
    parser.add_argument("--router-epochs", type=int, default=None)
    parser.add_argument("--end-to-end-epochs", type=int, default=None)
    parser.add_argument("--end-to-end-lr", type=float, default=None)
    parser.add_argument("--router-init-bias", type=float, default=None)
    parser.add_argument("--aux-delta-scale", type=float, default=None)
    parser.add_argument("--no-fallback", action="store_true", help="Do not restore base-only weights when fusion validation loss is worse.")
    parser.add_argument(
        "--report-split",
        choices=("test", "validation"),
        default="test",
        help="Split used for reported stage and final metrics. Training and early stopping always use validation loss.",
    )
    args = parser.parse_args()
    try:
        run(args)
    except Exception as exc:
        print(str(exc))
        print(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
