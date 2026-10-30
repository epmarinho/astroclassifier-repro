"""Configurable pooling policies for spatial-grid ablations."""
import torch
from torch import nn
from torch.nn import functional as F

from . import models_v2 as spatial

VERSION = "spatial-pooling-ablation-v3"
VARIANTS = spatial.VARIANTS
POOLING_POLICIES = ("max", "avg", "mean_std", "mean_grad", "gated_mix")


class SpatialPool(nn.Module):
    def __init__(self, source_size, target_size, policy):
        super().__init__()
        self.source_size = source_size
        self.target_size = target_size
        if policy not in (*POOLING_POLICIES, "identity"):
            raise ValueError("Unknown pooling policy")
        if (type(source_size) is not int or type(target_size) is not int
                or target_size < 1 or source_size < target_size or source_size % target_size):
            raise ValueError("Target grid must divide the source grid")
        if policy == "identity" and source_size != target_size:
            raise ValueError("Identity pooling requires the native grid")
        self.policy = policy
        if policy == "gated_mix" and source_size != target_size:
            # One content-dependent gate per channel and output window.
            self.gate_mean = nn.Parameter(torch.zeros(1, 1024, 1, 1))
            self.gate_max = nn.Parameter(torch.zeros(1, 1024, 1, 1))
            self.gate_bias = nn.Parameter(torch.zeros(1, 1024, 1, 1))

    def forward(self, features):
        if features.shape[-2:] != (self.source_size, self.source_size):
            raise ValueError("Feature grid differs from configured CNN output")
        if self.source_size == self.target_size:
            return features
        factor = self.source_size // self.target_size
        if self.policy == "max":
            return F.max_pool2d(features, factor, factor)
        mean = F.avg_pool2d(features, factor, factor)
        if self.policy == "avg":
            return mean
        if self.policy == "gated_mix":
            maximum = F.max_pool2d(features, factor, factor)
            alpha = torch.sigmoid(self.gate_mean * mean + self.gate_max * maximum + self.gate_bias)
            return alpha * maximum + (1 - alpha) * mean
        if self.policy == "mean_grad":
            # Forward spatial differences; missing neighbors at bottom/right
            # contribute zero, so constant maps have no artificial border edges.
            dx = F.pad(features[..., 1:] - features[..., :-1], (0, 1, 0, 0))
            dy = F.pad(features[..., 1:, :] - features[..., :-1, :], (0, 0, 0, 1))
            magnitude = torch.sqrt(dx.square() + dy.square() + 1e-6) - 0.001
            return mean + F.avg_pool2d(magnitude, factor, factor)

        # Center before squaring to avoid cancellation in E[x²] - E[x]².
        centered = features - mean.repeat_interleave(factor, -2).repeat_interleave(factor, -1)
        variance = F.avg_pool2d(centered.square(), factor, factor)
        # Smooth at zero; constant windows return their mean exactly.
        deviation = torch.sqrt(variance + 1e-6) - 0.001
        return mean + deviation


def architecture(variant, input_size=256, grid_size=8, pooling="max"):
    if pooling not in (*POOLING_POLICIES, "identity"):
        raise ValueError("Unknown pooling policy")
    configuration = spatial.architecture(variant, input_size, grid_size)
    identity = configuration["additional_pooling"] == "identity"
    if pooling == "identity" and not identity:
        raise ValueError("Identity pooling requires the native CNN grid")
    policy = "identity" if identity else pooling
    configuration.update(
        adapter_version=VERSION,
        name=f"{VERSION}_{variant}_grid{grid_size}_{policy}",
        additional_pooling=policy,
        pooling_policy=policy,
        contrast_weight=1.0 if policy in {"mean_std", "mean_grad"} else 0.0,
        contrast_epsilon=1e-6 if policy in {"mean_std", "mean_grad"} else 0.0,
        gradient_policy="forward_xy_zero_missing_neighbor" if policy == "mean_grad" else "none",
        gate_policy="sigmoid_per_channel_per_window_mean_max" if policy == "gated_mix" else "none",
        pooling_trainable_parameters=3072 if policy == "gated_mix" else 0,
        purpose="pooling/grid development ablation; no test evaluation",
    )
    return configuration


def validate_configuration(configuration):
    expected = architecture(configuration.get("variant"), configuration.get("input_size"),
                            configuration.get("grid_size"), configuration.get("pooling_policy"))
    for key, value in expected.items():
        if configuration.get(key) != value:
            raise ValueError(f"Pooling architecture field mismatch: {key}")


class SpatialCustomModel(spatial.SpatialCustomModel):
    def __init__(self, configuration, classes):
        validate_configuration(configuration)
        super().__init__(spatial.architecture(configuration["variant"], configuration["input_size"], configuration["grid_size"]), classes)
        self.grid_pool = SpatialPool(configuration["native_grid_size"], configuration["grid_size"], configuration["pooling_policy"])


def model_factory(configuration, classes):
    return SpatialCustomModel(configuration, classes)
