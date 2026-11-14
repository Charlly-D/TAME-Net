from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.multimodal_load_model import MultimodalLoadModel


class AblationMultimodalLoadModel(MultimodalLoadModel):
    """Multimodal model with switchable ablation variants.

    Variants:
        full: interpolation-based residual-expert model (revised default).
        base_only: load backbone only.
        no_route: residual experts with fixed modality weights.
        cross_attention: residual experts with learned cross-attention interaction.
        no_alignment: legacy alias for the interpolation-based full model.
        no_residual_expert: direct concatenation fusion head, no separate residual experts.
        no_image: weather residual expert only.
        no_weather: image residual expert only.
        end_to_end: same structure as full, trained without staged pretraining.
    """

    VALID_VARIANTS = {
        "full",
        "base_only",
        "no_route",
        "no_residual_expert",
        "no_image",
        "no_weather",
        "no_alignment",
        "cross_attention",
        "end_to_end",
    }

    def __init__(self, config, variant: str = "full"):
        if variant not in self.VALID_VARIANTS:
            raise ValueError(f"Unknown ablation variant={variant}. Available: {sorted(self.VALID_VARIANTS)}")
        super().__init__(config)
        cfg = SimpleNamespace(**config)
        self.variant = variant
        self.use_cross_attention = variant == "cross_attention"
        self.use_weather = variant != "no_weather"
        self.use_image = variant != "no_image"
        self.use_route = variant not in {"base_only", "no_route", "no_residual_expert"}
        self.use_residual_experts = variant not in {"base_only", "no_residual_expert"}

        self.fixed_weather_weight = float(getattr(cfg, "ablation_fixed_weather_weight", 0.5))
        self.fixed_image_weight = float(getattr(cfg, "ablation_fixed_image_weight", 0.5))

        if variant == "no_residual_expert":
            fusion_in = cfg.pred_len * cfg.d_model * 3
            hidden = max(cfg.d_model * 4, 64)
            self.direct_fusion_head = nn.Sequential(
                nn.Linear(fusion_in, hidden),
                nn.GELU(),
                nn.Dropout(getattr(cfg, "branch_dropout", 0.0)),
                nn.Linear(hidden, cfg.pred_len * cfg.c_out),
            )
            nn.init.zeros_(self.direct_fusion_head[-1].weight)
            nn.init.zeros_(self.direct_fusion_head[-1].bias)

    def ablation_description(self):
        return {
            "variant": self.variant,
            "use_weather": self.use_weather,
            "use_image": self.use_image,
            "use_route": self.use_route,
            "interaction_operator": "cross_attention" if self.use_cross_attention else "interpolation",
            "use_residual_experts": self.use_residual_experts,
            "fixed_weather_weight": self.fixed_weather_weight if self.variant == "no_route" else None,
            "fixed_image_weight": self.fixed_image_weight if self.variant == "no_route" else None,
        }

    def _zero_like_pred(self, pred_base):
        return torch.zeros_like(pred_base)

    def _zero_weight(self, pred_base):
        return pred_base.new_zeros(pred_base.size(0), 1, 1)

    def _encode_auxiliary(self, x_weather, x_img, feat_load):
        if self.use_weather:
            feat_weather = self.weather_encoder(x_weather)
        else:
            feat_weather = torch.zeros_like(feat_load)

        if self.use_image:
            feat_img = self.image_encoder(x_img)
        else:
            feat_img = torch.zeros_like(feat_load)
        return feat_weather, feat_img

    def _fixed_weights(self, pred_base):
        weather = pred_base.new_full((pred_base.size(0), 1, 1), self.fixed_weather_weight if self.use_weather else 0.0)
        image = pred_base.new_full((pred_base.size(0), 1, 1), self.fixed_image_weight if self.use_image else 0.0)
        return torch.cat([weather, image], dim=-1)

    def _direct_fusion_delta(self, feat_load, aligned_weather, aligned_image):
        if not self.use_weather:
            aligned_weather = torch.zeros_like(feat_load)
        if not self.use_image:
            aligned_image = torch.zeros_like(feat_load)
        x = torch.cat([feat_load, aligned_weather, aligned_image], dim=-1)
        raw = self.direct_fusion_head(x.reshape(x.size(0), -1))
        raw = raw.reshape(x.size(0), self.pred_len, self.c_out)
        return self._bounded_delta(raw)

    def _match_time_length(self, feat, target_len):
        if feat.size(1) == target_len:
            return feat
        return F.interpolate(
            feat.transpose(1, 2),
            size=target_len,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def _unaligned_aux_context(self, feat_load, feat_weather, feat_img):
        aligned_weather = self._match_time_length(feat_weather, feat_load.size(1))
        aligned_image = self._match_time_length(feat_img, feat_load.size(1))
        diff_weather = torch.abs(feat_load - aligned_weather)
        diff_image = torch.abs(feat_load - aligned_image)
        return aligned_weather, aligned_image, diff_weather, diff_image

    def _unaligned_router_weights(self, feat_load, aligned_weather, aligned_image, diff_weather, diff_image):
        context = torch.cat([feat_load, aligned_weather, aligned_image, diff_weather, diff_image], dim=-1)
        step_weights = torch.sigmoid(self.router.router_mlp(context)) * self.router.max_weight
        pool = torch.softmax(self.router.pool_score(context), dim=1)
        return torch.sum(step_weights * pool, dim=1, keepdim=True)

    def forward(self, x_load, x_mark, x_weather, x_img, mode="finetune"):
        pred_base, feat_load = self.backbone(x_load, x_mark, return_feat=True)
        if mode == "pretrain_backbone" or self.variant == "base_only":
            if "pretrain" in mode or self.variant == "base_only":
                if mode == "pretrain_backbone":
                    return pred_base
            if self.variant == "base_only":
                zero = self._zero_like_pred(pred_base)
                return pred_base, self._zero_weight(pred_base), self._zero_weight(pred_base), zero, zero

        feat_weather, feat_img = self._encode_auxiliary(x_weather, x_img, feat_load)
        if self.use_cross_attention:
            aligned_weather, aligned_image, diff_weather, diff_image, _ = self.router.align(
                feat_load, feat_weather, feat_img
            )
        else:
            aligned_weather, aligned_image, diff_weather, diff_image = self._unaligned_aux_context(
                feat_load, feat_weather, feat_img
            )

        if mode == "pretrain_weather":
            if not self.use_weather or not self.use_residual_experts:
                return self._zero_like_pred(pred_base)
            return self._weather_delta(feat_load, aligned_weather, diff_weather)
        if mode == "pretrain_img":
            if not self.use_image or not self.use_residual_experts:
                return self._zero_like_pred(pred_base)
            return self._image_delta(feat_load, aligned_image, diff_image)

        if self.variant == "no_residual_expert":
            delta = self._direct_fusion_delta(feat_load, aligned_weather, aligned_image)
            final_pred = pred_base + delta
            zero_weight = self._zero_weight(pred_base)
            zero_bias = self._zero_like_pred(pred_base)
            return final_pred, zero_weight, zero_weight, zero_bias, zero_bias

        delta_weather = (
            self._weather_delta(feat_load, aligned_weather, diff_weather)
            if self.use_weather else self._zero_like_pred(pred_base)
        )
        delta_img = (
            self._image_delta(feat_load, aligned_image, diff_image)
            if self.use_image else self._zero_like_pred(pred_base)
        )

        if self.variant == "no_route":
            weights = self._fixed_weights(pred_base)
        elif self.use_cross_attention:
            weights = self.router(feat_load, feat_weather, feat_img)
            if not self.use_weather:
                weights = weights.clone()
                weights[:, :, 0:1] = 0.0
            if not self.use_image:
                weights = weights.clone()
                weights[:, :, 1:2] = 0.0
        else:
            weights = self._unaligned_router_weights(
                feat_load, aligned_weather, aligned_image, diff_weather, diff_image
            )

        weighted_weather = weights[:, :, 0:1] * delta_weather
        weighted_img = weights[:, :, 1:2] * delta_img
        final_pred = pred_base + weighted_weather + weighted_img
        return final_pred, weights[:, :, 0:1], weights[:, :, 1:2], weighted_weather, weighted_img
