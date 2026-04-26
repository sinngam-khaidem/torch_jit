from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
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


class SmallCNN(nn.Module):
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

# Recurrent Layers
# https://pytorch.org/blog/optimizing-cuda-rnn-with-torchscript/
def rnn_layer(
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
    use_tanh: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:

    W = params["W"]  # [input + hidden, hidden]
    b = params["b"]  # [hidden]

    B, T, _ = x.shape
    H = b.shape[0]

    h = torch.zeros(B, H, device=x.device, dtype=x.dtype) if h0 is None else h0

    outputs: List[torch.Tensor] = []

    for t in range(T):
        combined = torch.cat([x[:, t], h], dim=-1)

        # single fused matmul (TorchScript-friendly)
        h = combined @ W + b

        h = torch.tanh(h) if use_tanh else torch.relu(h)
        outputs.append(h)

    return torch.stack(outputs, dim=1), h

class FunctionalRNNWrapper(nn.Module):
    def __init__(self, params, fn):
        super().__init__()
        self.params = nn.ParameterDict(params)
        self.fn = fn

    def forward(self, x, state=None):
        return self.fn(self.params, x, state)


def lstm_layer(
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
    state: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
):

    W = params["W"] # [input + hidden, 4H]
    b = params["b"]  # [4H]

    B, T, _ = x.shape
    H = b.shape[0] // 4

    if state is None:
        h = torch.zeros(B, H, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, H, device=x.device, dtype=x.dtype)
    else:
        h, c = state

    outputs: List[torch.Tensor] = []

    for t in range(T):
        combined = torch.cat([x[:, t], h], dim=-1)

        # single fused matmul
        gates = combined @ W + b

        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c + i * g
        h = o * torch.tanh(c)

        outputs.append(h)

    return torch.stack(outputs, dim=1), (h, c)



def gru_layer(
    params: Dict[str, torch.Tensor],
    x: torch.Tensor,
    h0: Optional[torch.Tensor] = None,
):

    W = params["W"] # [input + hidden, 3H]
    b = params["b"]  # [3H]

    B, T, _ = x.shape
    H = b.shape[0] // 3

    h = torch.zeros(B, H, device=x.device, dtype=x.dtype) if h0 is None else h0

    outputs: List[torch.Tensor] = []

    for t in range(T):
        combined = torch.cat([x[:, t], h], dim=-1)

        # single fused matmul
        gates = combined @ W + b

        r, z, n = gates.chunk(3, dim=-1)

        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n)

        h = (1.0 - z) * n + z * h
        outputs.append(h)

    return torch.stack(outputs, dim=1), h
def init_rnn(input_size, hidden_size):
    return nn.ParameterDict({
        "W": nn.Parameter(torch.randn(input_size + hidden_size, hidden_size) * 0.02),
        "b": nn.Parameter(torch.zeros(hidden_size)),
    })


def init_lstm(input_size, hidden_size):
    return nn.ParameterDict({
        "W": nn.Parameter(torch.randn(input_size + hidden_size, 4 * hidden_size) * 0.02),
        "b": nn.Parameter(torch.zeros(4 * hidden_size)),
    })


def init_gru(input_size, hidden_size):
    return nn.ParameterDict({
        "W": nn.Parameter(torch.randn(input_size + hidden_size, 3 * hidden_size) * 0.02),
        "b": nn.Parameter(torch.zeros(3 * hidden_size)),
    })


class SequenceClassifier(nn.Module):
    """Classifier wrapper over VMAP/JIT-friendly recurrent layers."""

    def __init__(
        self,
        layer_type: str,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        vocab_size: Optional[int] = None,
        embed_dim: Optional[int] = None,
    ):
        super().__init__()
        self.uses_embedding = vocab_size is not None
        if self.uses_embedding:
            if embed_dim is None:
                raise ValueError("embed_dim must be provided when vocab_size is used")
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            recurrent_input = embed_dim
        else:
            self.embedding = None
            recurrent_input = input_dim

        layer_type = layer_type.lower()
        if layer_type == "rnn":
            # self.recurrent = RNNLayer(recurrent_input, hidden_dim)
            self.rnn_params = init_rnn(recurrent_input, hidden_dim)
            self.rnn_fn = rnn_layer
        elif layer_type == "lstm":
            # self.recurrent = LSTMLayer(recurrent_input, hidden_dim)
            self.rnn_params = init_lstm(recurrent_input, hidden_dim)
            self.rnn_fn = lstm_layer
        elif layer_type == "gru":
            # self.recurrent = GRULayer(recurrent_input, hidden_dim)
            self.rnn_params = init_gru(recurrent_input, hidden_dim)
            self.rnn_fn = gru_layer
        else:
            raise ValueError(f"Unsupported layer_type: {layer_type}")

        self.layer_type = layer_type
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_embedding:
            x = self.embedding(x)

        out, final_state = self.rnn_fn(self.rnn_params, x)
        if self.layer_type == "lstm":
            h, _ = final_state
            rep = h
        else:
            rep = final_state
        return self.classifier(rep)



@dataclass
class ModelSpec:
    model_name: str
    num_classes: int = 10
    in_channels: int = 1
    image_size: int = 28
    seq_input_dim: int = 64
    seq_hidden_dim: int = 128
    seq_len: int = 64


def build_model(spec: ModelSpec) -> nn.Module:
    name = spec.model_name.lower()
    if name == "mlp":
        return MLP(num_classes=spec.num_classes, input_shape=(spec.in_channels, spec.image_size, spec.image_size))
    if name == "cnn":
        return SmallCNN(in_channels=spec.in_channels, num_classes=spec.num_classes, image_size=spec.image_size)
    if name == "rnn":
        return SequenceClassifier(
            layer_type="rnn",
            input_dim=spec.seq_input_dim,
            hidden_dim=spec.seq_hidden_dim,
            num_classes=spec.num_classes,
        )
    if name == "lstm":
        return SequenceClassifier(
            layer_type="lstm",
            input_dim=spec.seq_input_dim,
            hidden_dim=spec.seq_hidden_dim,
            num_classes=spec.num_classes,
        )
    if name == "gru":
        return SequenceClassifier(
            layer_type="gru",
            input_dim=spec.seq_input_dim,
            hidden_dim=spec.seq_hidden_dim,
            num_classes=spec.num_classes,
        )
    raise ValueError(f"Unsupported model name: {spec.model_name}")


