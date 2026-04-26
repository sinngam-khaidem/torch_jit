from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus.layers import DPGRU, DPLSTM, DPRNN, DPMultiheadAttention

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
            self.rnn = DPRNN(input_dim, hidden_dim, batch_first=True)
        elif layer_type == "lstm":
            self.rnn = DPLSTM(input_dim, hidden_dim, batch_first=True)
        elif layer_type == "gru":
            self.rnn = DPGRU(input_dim, hidden_dim, batch_first=True)
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
    raise ValueError(f"Unsupported model name: {spec.model_name}")

