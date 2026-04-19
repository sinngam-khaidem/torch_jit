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

        # 🔥 single fused matmul (TorchScript-friendly)
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

        # 🔥 single fused matmul
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

        # 🔥 single fused matmul
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

class MultiHeadAttention(nn.Module):
    """Simple MHA implementation using plain tensor ops and explicit projection weights."""

    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: [batch, seq_len, embed_dim]
        b, t, _ = x.shape

        q = self.q_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            # attn_mask expected shape [t, t] or [1, t, t]
            if attn_mask.dim() == 2:
                attn_scores = attn_scores + attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_scores = attn_scores + attn_mask.unsqueeze(1)

        if key_padding_mask is not None:
            # key_padding_mask: [batch, seq_len] (True means padding)
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(mask, float("-inf"))

        attn_probs = torch.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        context = torch.matmul(attn_probs, v)

        context = context.transpose(1, 2).contiguous().view(b, t, self.embed_dim)
        out = self.out_proj(context)

        # return mean attention over heads for easier inspection
        return out, attn_probs.mean(dim=1)


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



class AttentionClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_heads: int,
        num_classes: int,
        vocab_size: Optional[int] = None,
    ):
        super().__init__()
        self.uses_embedding = vocab_size is not None
        if self.uses_embedding:
            self.embedding = nn.Embedding(vocab_size, embed_dim)
            self.input_proj = nn.Identity()
        else:
            self.embedding = None
            self.input_proj = nn.Linear(input_dim, embed_dim)

        self.mha = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads)
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.uses_embedding:
            x = self.embedding(x)
        else:
            x = self.input_proj(x)

        x, _ = self.mha(x)
        x = self.norm(x)
        pooled = x.mean(dim=1)
        return self.classifier(pooled)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(embed_dim * mlp_ratio)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(self.norm1(x))
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    """Small ViT variant for DP-SGD runtime comparisons."""

    def __init__(
        self,
        image_size: int = 32,
        patch_size: int = 4,
        in_channels: int = 3,
        num_classes: int = 10,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 6,
    ):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))

        self.blocks = nn.ModuleList(
            [TransformerBlock(embed_dim=embed_dim, num_heads=num_heads) for _ in range(depth)]
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
    if name == "mha":
        return AttentionClassifier(
            input_dim=spec.seq_input_dim,
            embed_dim=spec.seq_hidden_dim,
            num_heads=4,
            num_classes=spec.num_classes,
        )
    if name == "vit":
        return TinyViT(
            image_size=spec.image_size,
            in_channels=spec.in_channels,
            num_classes=spec.num_classes,
        )
    raise ValueError(f"Unsupported model name: {spec.model_name}")


