from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPMultiheadAttention

from custom_models import ModelSpec


class OpacusMLP(nn.Module):
    def __init__(self, num_classes: int = 10, input_shape: Tuple[int, int, int] = (1, 28, 28)):
        super().__init__()
        c, h, w = input_shape
        in_features = c * h * w
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class OpacusCNN(nn.Module):
    def __init__(self, in_channels: int = 1, num_classes: int = 10, image_size: int = 28):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)

        pooled = image_size // 4
        self.fc1 = nn.Linear(64 * pooled * pooled, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class OpacusSequenceClassifier(nn.Module):
    def __init__(
        self,
        layer_type: str,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        layer_type = layer_type.lower()
        if layer_type == "rnn":
            self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        elif layer_type == "lstm":
            self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        elif layer_type == "gru":
            self.rnn = nn.GRU(input_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")

        self.layer_type = layer_type
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, state = self.rnn(x)
        if self.layer_type == "lstm":
            h, _ = state
        else:
            h = state
        rep = h[-1]
        return self.classifier(rep)


class OpacusMHAClassifier(nn.Module):
    def __init__(self, input_dim: int, embed_dim: int, num_heads: int, num_classes: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.mha = DPMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_proj(x)
        x, _ = self.mha(x, x, x, need_weights=False)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class OpacusTransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = DPMultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class OpacusViT(nn.Module):
    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        if dropout != 0.0:
            raise ValueError("OpacusViT requires dropout=0.0 for per-sample gradients")

        self.blocks = nn.ModuleList(
            [OpacusTransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)

        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1), :]

        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])


def build_opacus_model(spec: ModelSpec) -> nn.Module:
    name = spec.model_name.lower()
    if name == "mlp":
        return OpacusMLP(num_classes=spec.num_classes, input_shape=(spec.in_channels, spec.image_size, spec.image_size))
    if name == "cnn":
        return OpacusCNN(in_channels=spec.in_channels, num_classes=spec.num_classes, image_size=spec.image_size)
    if name == "rnn":
        return OpacusSequenceClassifier(
            layer_type="rnn",
            input_dim=spec.seq_input_dim,
            hidden_dim=spec.seq_hidden_dim,
            num_classes=spec.num_classes,
        )
    if name == "lstm":
        return OpacusSequenceClassifier(
            layer_type="lstm",
            input_dim=spec.seq_input_dim,
            hidden_dim=spec.seq_hidden_dim,
            num_classes=spec.num_classes,
        )
    if name == "gru":
        return OpacusSequenceClassifier(
            layer_type="gru",
            input_dim=spec.seq_input_dim,
            hidden_dim=spec.seq_hidden_dim,
            num_classes=spec.num_classes,
        )
    if name == "mha":
        return OpacusMHAClassifier(
            input_dim=spec.seq_input_dim,
            embed_dim=spec.seq_hidden_dim,
            num_heads=4,
            num_classes=spec.num_classes,
        )
    if name == "vit":
        return OpacusViT(
            image_size=spec.image_size,
            in_channels=spec.in_channels,
            num_classes=spec.num_classes,
        )
    raise ValueError(f"Unsupported model name: {spec.model_name}")

