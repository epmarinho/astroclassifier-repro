import math

import torch
from torch import nn
from torch.nn import functional as functional


VERSION = "phase3d-custom-family-v1"
VARIANTS = ("historical_singleton", "spatial_stem_only", "spatial_transformer")
CHANNELS = (128, 256, 512, 1024)
EMBEDDING = 128
GRID_SIZE = 8
TOKEN_COUNT = GRID_SIZE * GRID_SIZE


class DeterministicGridMaxPool(nn.Module):
    """Exact adaptive-max equivalent for the fixed 8x8/16x16 feature grids."""

    def forward(self, features):
        height, width = features.shape[-2:]
        if (height, width) == (GRID_SIZE, GRID_SIZE):
            return features
        if (height, width) != (2 * GRID_SIZE, 2 * GRID_SIZE):
            raise ValueError("Expected an 8x8 or 16x16 feature grid")
        return functional.max_pool2d(features, kernel_size=2, stride=2)


class DeterministicGlobalMaxPool(nn.Module):
    """Global maximum with CUDA-deterministic backward for this model."""

    def forward(self, features):
        return torch.amax(features, dim=(-2, -1), keepdim=True)


def fixed_2d_sinusoidal_encoding(grid_size=GRID_SIZE, embedding=EMBEDDING):
    if grid_size != 8 or embedding != 128:
        raise ValueError("Phase 3D positional encoding is fixed to an 8x8 grid and 128 dimensions")
    coordinates = torch.arange(grid_size, dtype=torch.float32)
    vertical, horizontal = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequencies = torch.pow(10000.0, -torch.arange(0, 64, 2, dtype=torch.float32) / 64.0)
    encoding = torch.empty(grid_size, grid_size, embedding, dtype=torch.float32)
    encoding[:, :, 0:32] = torch.sin(horizontal[:, :, None] * frequencies)
    encoding[:, :, 32:64] = torch.cos(horizontal[:, :, None] * frequencies)
    encoding[:, :, 64:96] = torch.sin(vertical[:, :, None] * frequencies)
    encoding[:, :, 96:128] = torch.cos(vertical[:, :, None] * frequencies)
    return encoding.reshape(1, TOKEN_COUNT, embedding)


def validate_model_configuration(configuration):
    expected = {
        "adapter_version": VERSION,
        "channels": list(CHANNELS),
        "embedding": EMBEDDING,
        "grid_size": GRID_SIZE,
        "token_count": 1 if configuration.get("variant") == "historical_singleton" else TOKEN_COUNT,
        "transformer_layers": 0 if configuration.get("variant") == "spatial_stem_only" else 2,
        "attention_heads": 8 if configuration.get("variant") != "spatial_stem_only" else 0,
        "positional_encoding": "none" if configuration.get("variant") != "spatial_transformer" else "fixed_2d_sinusoidal_xy_v1",
    }
    if configuration.get("variant") not in VARIANTS:
        raise ValueError("Unknown Phase 3D variant")
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"Phase 3D architecture field mismatch: {key}")
    if configuration.get("variant") == "historical_singleton" and configuration.get("claims_spatial_attention") is not False:
        raise ValueError("Historical singleton-token model cannot claim spatial attention")
    if configuration.get("variant") == "spatial_transformer" and configuration.get("claims_spatial_attention") is not True:
        raise ValueError("Spatial Transformer must declare its spatial-attention claim")
    if configuration.get("variant") == "spatial_stem_only" and configuration.get("claims_spatial_attention") is not False:
        raise ValueError("Stem-only ablation cannot claim attention")
    if configuration.get("positional_encoding") == "radial" or "radial" in str(configuration.get("position_policy", "")).lower():
        raise ValueError("Radial positional priors are not authorized in Phase 3D")
    return configuration


class SpatialCustomModel(nn.Module):
    def __init__(self, configuration, classes):
        super().__init__()
        validate_model_configuration(configuration)
        self.variant = configuration["variant"]
        self.input_size = configuration["input_size"]
        self.spatial_token_count = 1 if self.variant == "historical_singleton" else TOKEN_COUNT
        self.claims_spatial_attention = self.variant == "spatial_transformer"

        stem = []
        in_channels = 3
        for out_channels in CHANNELS:
            stem.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
            ])
            in_channels = out_channels
        self.stem = nn.Sequential(*stem)
        # At the fixed 256px input, the stem produces 16x16 features.  These
        # operations are forward-equivalent to adaptive max pooling at 8x8 and
        # 1x1, while avoiding its nondeterministic CUDA backward implementation.
        self.grid_pool = DeterministicGridMaxPool()
        self.global_pool = DeterministicGlobalMaxPool()
        self.token_projection = nn.Sequential(
            nn.Linear(1024, 512), nn.Dropout(p=0.3), nn.ReLU(),
            nn.Linear(512, 256), nn.Dropout(p=0.3), nn.ReLU(),
            nn.Linear(256, 128), nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, classes),
        )
        if self.variant == "historical_singleton":
            layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.01, activation="relu", batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.register_buffer("position_encoding", torch.zeros(1, 1, 128), persistent=True)
        elif self.variant == "spatial_transformer":
            layer = nn.TransformerEncoderLayer(d_model=128, nhead=8, dim_feedforward=512, dropout=0.01, activation="relu", batch_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=2, enable_nested_tensor=False)
            self.register_buffer("position_encoding", fixed_2d_sinusoidal_encoding(), persistent=True)
        else:
            self.transformer = None
            self.register_buffer("position_encoding", torch.empty(1, 0, 128), persistent=True)

    def check_input(self, image):
        if image.ndim != 4 or tuple(image.shape[1:]) != (3, self.input_size, self.input_size):
            raise ValueError("Phase 3D input must match the configured RGB resolution")

    def spatial_tokens(self, features):
        if tuple(features.shape[1:]) != (1024, GRID_SIZE, GRID_SIZE):
            raise ValueError("Spatial stem must provide a native 8x8x1024 feature map")
        tokens = features.flatten(2).transpose(1, 2)
        tokens = self.token_projection(tokens)
        assert tokens.shape[1] == 64
        assert tokens.shape[2] == 128
        return tokens

    def singleton_tokens(self, features):
        pooled = self.global_pool(features).flatten(1)
        tokens = self.token_projection(pooled).unsqueeze(1)
        if tokens.shape[1:] != (1, 128):
            raise AssertionError("Historical-form encoder must receive exactly one 128D token")
        return tokens

    def classify_tokens(self, tokens):
        if self.variant == "spatial_transformer":
            if tokens.shape[1:] != (64, 128):
                raise ValueError("Spatial attention requires exactly [B,64,128]")
            tokens = self.transformer(tokens + self.position_encoding)
        elif self.variant == "historical_singleton":
            if tokens.shape[1:] != (1, 128):
                raise ValueError("Historical Transformer requires exactly [B,1,128]")
            tokens = self.transformer(tokens)
        elif tokens.shape[1:] != (64, 128):
            raise ValueError("Stem-only ablation requires exactly [B,64,128]")
        return self.classifier(tokens.mean(dim=1))

    def forward_features(self, features):
        if self.variant == "historical_singleton":
            return self.classify_tokens(self.singleton_tokens(features))
        return self.classify_tokens(self.spatial_tokens(self.grid_pool(features)))

    def forward(self, image):
        self.check_input(image)
        features = self.stem(image)
        if features.shape[-2] < GRID_SIZE or features.shape[-1] < GRID_SIZE:
            raise ValueError("Configured input resolution must provide at least an 8x8 stem grid")
        return self.forward_features(features)


def architecture(variant, input_size=256):
    if variant not in VARIANTS:
        raise ValueError("Unknown Phase 3D variant")
    configuration = {
        "adapter_version": VERSION,
        "name": f"phase3d_{variant}_v1",
        "variant": variant,
        "channels": list(CHANNELS),
        "embedding": EMBEDDING,
        "grid_size": GRID_SIZE,
        "token_count": 1 if variant == "historical_singleton" else TOKEN_COUNT,
        "transformer_layers": 0 if variant == "spatial_stem_only" else 2,
        "attention_heads": 8 if variant != "spatial_stem_only" else 0,
        "positional_encoding": "fixed_2d_sinusoidal_xy_v1" if variant == "spatial_transformer" else "none",
        "position_policy": "fixed separable 2D sinusoidal x/y encoding; see models_v1.py" if variant == "spatial_transformer" else "none",
        "input_size": input_size,
        "pretrained": False,
        "claims_spatial_attention": variant == "spatial_transformer",
        "purpose": "bounded development pilot only; not a benchmark",
    }
    validate_model_configuration(configuration)
    return configuration


def model_factory(configuration, classes):
    validate_model_configuration(configuration)
    return SpatialCustomModel(configuration, classes)
