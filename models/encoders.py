import torch
import torch.nn as nn
import math


from models.embed import PositionalEmbedding


# class PositionalEncoding(nn.Module):
#     def __init__(self, d_model, max_len=5000):
#         super().__init__()
#         # 标准的正弦位置编码
#         pe = torch.zeros(max_len, d_model)
#         position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
#         div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
#         pe[:, 0::2] = torch.sin(position * div_term)
#         pe[:, 1::2] = torch.cos(position * div_term)
#         self.register_buffer('pe', pe.unsqueeze(0)) # [1, max_len, d_model]
#
#     def forward(self, x):
#         # x: [Batch, Seq_Len, D_model]
#         # 截取对应长度的位置编码相加
#         return x + self.pe[:, :x.size(1), :]

class ERA5Encoder(nn.Module):
    def __init__(self, in_channels=6, spatial_size=27, d_model=128, nhead=4, num_layers=1):
        super().__init__()

        # 动态计算经过两次 MaxPool2d(2) 之后的特征图大小
        # 例如：27 // 2 = 13； 13 // 2 = 6。最终是 6x6
        # 如果以后 config 里是 30：30 // 2 = 15； 15 // 2 = 7。最终会自动变成 7x7
        pooled_size = (spatial_size // 2) // 2
        flattened_dim = 64 * pooled_size * pooled_size

        # 1. 空间编码 (CNN)
        # 针对 27x27 的小型 CNN，把 27x27 的图变成一个向量
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 13x13

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),  # -> 6x6

            nn.Flatten(),  # 64 * 6 * 6 = 2304
            nn.Linear(flattened_dim, d_model),
            nn.ReLU()
        )

        # 2. 位置编码
        # 让模型知道这是第几小时的天气
        self.pos_encoder = PositionalEmbedding(d_model, max_len=96)

        # 3. 时序编码
        # 理解 24 小时内的气象演变逻辑
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 2,  # 轻量化，不用标准的 d_model*4
            dropout=0.1,
            batch_first=True
        )
        self.temporal_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x: [B, T, 6, 27, 27]
        b, t, c, h, w = x.shape

        # --- A. 空间特征提取 (逐帧独立处理) ---
        x = x.reshape(b * t, c, h, w)
        x = self.cnn(x)  # -> [B*T, d_model]
        x = x.reshape(b, t, -1)  # -> [B, T, d_model]

        # --- B. 注入时间信息 ---
        x = x + self.pos_encoder(x)

        # --- C. 时序特征建模
        # 让第 t 个时刻的特征聚合上下文信息
        x = self.temporal_transformer(x)  # -> [B, T, d_model]

        return x

class GOESEncoder(nn.Module):
    def __init__(self, in_channels=1, img_height=128, img_width=256, patch_size=16, d_model=128,  num_layer=2):
        super().__init__()

        # 1. Patch Embedding using Conv2d
        # Input: [B*T, 1, 128, 256] -> Output: [B*T, d_model, 8, 16] (8*16=128 patches)
        self.patch_embed = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)

        # 计算 patches 数量: (128/16) * (256/16) = 8 * 16 = 128
        self.num_patches = (img_height // patch_size) * (img_width // patch_size)

        # 2. Spatial Positional Embedding
        # 3D 的 [1, 128, d_model]，避免之前的 broadcasting error
        self.spatial_pos_emb = nn.Parameter(torch.randn(1, self.num_patches, d_model))

        # 3. Spatial Transformer (ViT Block)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True)
        self.spatial_transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layer)

        # 4. Temporal Positional Encoding (for the T=6 sequence)
        self.temporal_pos_encoder = PositionalEmbedding(d_model, max_len=24)

    def forward(self, x):
        # x: [B, T, 1, 128, 256]
        b, t, c, h, w = x.shape

        # --- A. Fold Time ---
        x = x.reshape(b * t, c, h, w)

        # --- B. Patch Embedding ---
        # [B*T, 1, 128, 256] -> [B*T, d_model, 8, 16]
        x = self.patch_embed(x)

        # Flatten patches: [B*T, d_model, 8, 16] -> [B*T, d_model, 128]
        x = x.flatten(2)

        # Transpose for Transformer: [B*T, 128, d_model]
        x = x.transpose(1, 2)

        # --- C. Add Spatial Pos Emb ---
        # [B*T, 128, d_model] + [1, 128, d_model]
        x = x + self.spatial_pos_emb

        # --- D. Spatial Modeling ---
        x = self.spatial_transformer(x)  # -> [B*T, 128, d_model]

        # --- E. Aggregation (Pooling) ---
        # 把一张图的所有 patch 融合为一个向量
        # [B*T, 128, d_model] -> [B*T, d_model]
        x = x.mean(dim=1)

        # --- F. Unfold Time ---
        x = x.reshape(b, t, -1)  # -> [B, 6, d_model]

        # --- G. Temporal Positional Encoding ---
        # 告诉模型这是 T=1 的图还是 T=6 的图
        x = x + self.temporal_pos_encoder(x)

        return x