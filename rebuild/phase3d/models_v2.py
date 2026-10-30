"""Configurable spatial grids; v1 remains unchanged for frozen historical runs."""
import torch
from torch import nn
from torch.nn import functional as F

from . import models_v1 as original

VERSION = "spatial-grid-ablation-v2"
VARIANTS = ("spatial_stem_only", "spatial_transformer")


class GridPool(nn.Module):
    def __init__(self, source_size, target_size):
        super().__init__()
        self.source_size = source_size
        self.target_size = target_size

    def forward(self, features):
        if features.shape[-2:] != (self.source_size, self.source_size):
            raise ValueError("Feature grid differs from configured CNN output")
        if self.source_size == self.target_size:
            return features
        factor = self.source_size // self.target_size
        return F.max_pool2d(features, factor, factor)


def positional_encoding(grid_size):
    coordinates = torch.arange(grid_size, dtype=torch.float32)
    y, x = torch.meshgrid(coordinates, coordinates, indexing="ij")
    frequencies = 10000.0 ** (-torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    return torch.cat([
        torch.sin(x[..., None] * frequencies), torch.cos(x[..., None] * frequencies),
        torch.sin(y[..., None] * frequencies), torch.cos(y[..., None] * frequencies),
    ], dim=-1).reshape(1, grid_size ** 2, 128)


def architecture(variant, input_size=256, grid_size=8):
    if variant not in VARIANTS:
        raise ValueError("Grid ablation requires a spatial variant")
    if type(input_size) is not int or input_size < 32 or input_size % 16:
        raise ValueError("Input size must be a multiple of 16, at least 32")
    source_size = input_size // 16
    if type(grid_size) is not int or grid_size < 2 or source_size % grid_size:
        raise ValueError("Grid size must divide the native CNN grid and be at least 2")
    configuration = original.architecture(variant, input_size)
    configuration.update(
        adapter_version=VERSION,
        name=f"{VERSION}_{variant}_grid{grid_size}",
        grid_size=grid_size,
        token_count=grid_size ** 2,
        native_grid_size=source_size,
        additional_pooling="identity" if grid_size == source_size else "max_pool",
        pooling_factor=source_size // grid_size,
        purpose="spatial-grid development ablation; no test evaluation",
    )
    return configuration


def validate_configuration(configuration):
    expected = architecture(configuration.get("variant"), configuration.get("input_size"), configuration.get("grid_size"))
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"Spatial-grid architecture field mismatch: {key}")


class SpatialCustomModel(original.SpatialCustomModel):
    def __init__(self, configuration, classes):
        validate_configuration(configuration)
        # Reuse the exact CNN/projection/classifier definitions and initialization.
        super().__init__(original.architecture(configuration["variant"], configuration["input_size"]), classes)
        self.grid_size = configuration["grid_size"]
        self.spatial_token_count = configuration["token_count"]
        self.grid_pool = GridPool(configuration["native_grid_size"], self.grid_size)
        if self.transformer is not None:
            self.position_encoding = positional_encoding(self.grid_size)

    def spatial_tokens(self, features):
        if features.shape[1:] != (1024, self.grid_size, self.grid_size):
            raise ValueError("Expected configured spatial grid with 1024 channels")
        return self.token_projection(features.flatten(2).transpose(1, 2))

    def classify_tokens(self, tokens):
        if tokens.ndim != 3 or tokens.shape[1:] != (self.spatial_token_count, 128):
            raise ValueError(f"Expected [B,{self.spatial_token_count},128] tokens")
        if self.transformer is not None:
            tokens = self.transformer(tokens + self.position_encoding)
        return self.classifier(tokens.mean(dim=1))

    def forward(self, image):
        self.check_input(image)
        return self.forward_features(self.stem(image))


def model_factory(configuration, classes):
    return SpatialCustomModel(configuration, classes)
