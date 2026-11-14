from types import SimpleNamespace

import torch
import torch.nn as nn

from models.encoders import ERA5Encoder, GOESEncoder
from models.timesnet import TimesNetStub


class LoadGuidedAlignmentRouter(nn.Module):
    """Load-guided cross-modal alignment and reliability routing."""

    def __init__(self, d_model, nhead=4, dropout=0.1, init_bias=-3.0, max_weight=1.0):
        super().__init__()
        if isinstance(max_weight, (list, tuple)):
            max_weight = torch.tensor(max_weight, dtype=torch.float32).view(1, 1, 2)
        else:
            max_weight = torch.tensor([max_weight, max_weight], dtype=torch.float32).view(1, 1, 2)
        self.register_buffer('max_weight', max_weight)

        self.weather_align = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.image_align = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.weather_norm = nn.LayerNorm(d_model)
        self.image_norm = nn.LayerNorm(d_model)

        router_in = d_model * 5
        self.router_mlp = nn.Sequential(
            nn.Linear(router_in, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, 2),
        )
        self.pool_score = nn.Sequential(
            nn.Linear(router_in, d_model),
            nn.Tanh(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.router_mlp[-1].weight)
        nn.init.constant_(self.router_mlp[-1].bias, init_bias)

    def align(self, feat_load, feat_weather, feat_img):
        # MultiheadAttention allows Q and K/V to have different sequence lengths.
        # Thus GOES image tokens [B, 24, D] are aligned to load tokens [B, 96, D]
        # directly by attention, without hard temporal repetition.
        aligned_weather, _ = self.weather_align(feat_load, feat_weather, feat_weather)
        aligned_image, _ = self.image_align(feat_load, feat_img, feat_img)
        aligned_weather = self.weather_norm(aligned_weather)
        aligned_image = self.image_norm(aligned_image)
        diff_weather = torch.abs(feat_load - aligned_weather)
        diff_image = torch.abs(feat_load - aligned_image)
        context = torch.cat(
            [feat_load, aligned_weather, aligned_image, diff_weather, diff_image],
            dim=-1,
        )
        return aligned_weather, aligned_image, diff_weather, diff_image, context

    def forward(self, feat_load, feat_weather, feat_img):
        _, _, _, _, context = self.align(feat_load, feat_weather, feat_img)
        step_weights = torch.sigmoid(self.router_mlp(context)) * self.max_weight
        # The router outputs a weight for each auxiliary modality at each time step. We then pool these weights across time steps to get a global weight for each modality.
        pool = torch.softmax(self.pool_score(context), dim=1)
        global_weights = torch.sum(step_weights * pool, dim=1, keepdim=True)
        return global_weights


class MultimodalLoadModel(nn.Module):
    """Asymmetric residual-expert network for multimodal load forecasting."""

    def __init__(self, config, pretrained_timesnet=None, pretrained_img_enc=None, pretrained_weather_enc=None):
        super().__init__()
        config = SimpleNamespace(**config)
        self.pred_len = config.pred_len
        self.c_out = config.c_out
        self.d_model = config.d_model
        self.fusion_mode = getattr(config, 'fusion_mode', 'residual_expert')
        self.aux_delta_scale = getattr(config, 'aux_delta_scale', getattr(config, 'cross_delta_scale', 0.1))
        self.branch_dropout = nn.Dropout(getattr(config, 'branch_dropout', 0.0))

        self.backbone = pretrained_timesnet if pretrained_timesnet is not None else TimesNetStub(config)
        self.weather_encoder = pretrained_weather_enc if pretrained_weather_enc is not None else ERA5Encoder(
            in_channels=getattr(config, 'weather_in_channels', 6),
            spatial_size=getattr(config, 'weather_spatial_size', 27),
            d_model=config.d_model,
            nhead=getattr(config, 'weather_nhead', 4),
            num_layers=getattr(config, 'weather_num_layers', 1),
        )
        self.image_encoder = pretrained_img_enc if pretrained_img_enc is not None else GOESEncoder(
            in_channels=getattr(config, 'img_in_channels', 1),
            img_height=getattr(config, 'img_height', 128),
            img_width=getattr(config, 'img_width', 256),
            patch_size=getattr(config, 'patch_size', 16),
            d_model=config.d_model,
            num_layer=getattr(config, 'img_num_layers', 2),
        )

        # Residual experts see the load state plus the aligned auxiliary context.
        # The expert output is a bounded correction, not a full load forecast.
        residual_in = config.pred_len * config.d_model * 3
        hidden = max(config.d_model * 4, 64)
        self.weather_proj = nn.Sequential(
            nn.Linear(residual_in, hidden),
            nn.GELU(),
            nn.Dropout(getattr(config, 'branch_dropout', 0.0)),
            nn.Linear(hidden, config.pred_len * config.c_out),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(residual_in, hidden),
            nn.GELU(),
            nn.Dropout(getattr(config, 'branch_dropout', 0.0)),
            nn.Linear(hidden, config.pred_len * config.c_out),
        )
        nn.init.zeros_(self.weather_proj[-1].weight)
        nn.init.zeros_(self.weather_proj[-1].bias)
        nn.init.zeros_(self.image_proj[-1].weight)
        nn.init.zeros_(self.image_proj[-1].bias)

        self.router = LoadGuidedAlignmentRouter(
            d_model=config.d_model,
            nhead=getattr(config, 'router_nhead', 4),
            dropout=getattr(config, 'router_dropout', 0.1),
            init_bias=getattr(config, 'router_init_bias', -3.0),
            max_weight=[
                getattr(config, 'router_weather_max_weight', getattr(config, 'router_max_weight', 1.0)),
                getattr(config, 'router_image_max_weight', getattr(config, 'router_max_weight', 1.0)),
            ],
        )

    # Bound the residual expert output to a reasonable range, e.g., [-0.1, 0.1] for normalized load.
    def _bounded_delta(self, raw_delta):
        return self.aux_delta_scale * torch.tanh(raw_delta)

    def _weather_delta(self, feat_load, aligned_weather, diff_weather):
        x = torch.cat([feat_load, aligned_weather, diff_weather], dim=-1)
        x = self.branch_dropout(x)
        raw = self.weather_proj(x.reshape(x.size(0), -1))
        raw = raw.reshape(x.size(0), self.pred_len, self.c_out)
        return self._bounded_delta(raw)

    def _image_delta(self, feat_load, aligned_image, diff_image):
        x = torch.cat([feat_load, aligned_image, diff_image], dim=-1)
        x = self.branch_dropout(x)
        raw = self.image_proj(x.reshape(x.size(0), -1))
        raw = raw.reshape(x.size(0), self.pred_len, self.c_out)
        return self._bounded_delta(raw)

    def forward(self, x_load, x_mark, x_weather, x_img, mode='finetune'):
        pred_base, feat_load = self.backbone(x_load, x_mark, return_feat=True)
        if mode == 'pretrain_backbone':
            return pred_base

        feat_weather = self.weather_encoder(x_weather)
        feat_img = self.image_encoder(x_img)
        aligned_weather, aligned_image, diff_weather, diff_image, _ = self.router.align(
            feat_load, feat_weather, feat_img
        )

        if mode == 'pretrain_weather':
            return self._weather_delta(feat_load, aligned_weather, diff_weather)
        if mode == 'pretrain_img':
            return self._image_delta(feat_load, aligned_image, diff_image)

        delta_weather = self._weather_delta(feat_load, aligned_weather, diff_weather)
        delta_img = self._image_delta(feat_load, aligned_image, diff_image)
        weights = self.router(feat_load, feat_weather, feat_img)

        if self.fusion_mode == 'base_only':
            weighted_weather = torch.zeros_like(pred_base)
            weighted_img = torch.zeros_like(pred_base)
            final_pred = pred_base
        else:
            weighted_weather = weights[:, :, 0:1] * delta_weather
            weighted_img = weights[:, :, 1:2] * delta_img
            final_pred = pred_base + weighted_weather + weighted_img

        return final_pred, weights[:, :, 0:1], weights[:, :, 1:2], weighted_weather, weighted_img
