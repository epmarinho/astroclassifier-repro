from copy import deepcopy
import json
import math

from .common import digest


DEFAULT_CONFIG = {
    "seed": 3908274,
    "model": {"name": "adapter_required", "claims_spatial_attention": False, "token_count": 1},
    "training": {"epochs": 100, "batch_size": 16, "learning_rate": 9e-5, "weight_decay": 5e-7, "max_grad_norm": 3.0, "patience": 20, "min_delta": 0.0, "scheduler_patience": 5, "scheduler_factor": 0.1},
    "preprocessing": {"size": 256, "interpolation": "lanczos", "aspect_policy": "letterbox", "color": "RGB", "mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5], "rotation_degrees": 15.0, "horizontal_flip_probability": 0.5},
    "runtime": {"device": "cpu", "num_threads": 1, "deterministic": True},
    "evaluation": {"calibration_bins": 10, "bootstrap_samples": 1000, "bootstrap_seed": 3908274, "confidence": 0.95},
}


def validate_config(configuration):
    if set(configuration) != set(DEFAULT_CONFIG):
        raise ValueError("Configuration must contain exactly seed, model, training, preprocessing, runtime, evaluation")
    runtime = configuration["runtime"]
    if set(runtime) != set(DEFAULT_CONFIG["runtime"]) or runtime["device"] not in {"cpu", "cuda"} or type(runtime["num_threads"]) is not int or runtime["num_threads"] < 1 or runtime["deterministic"] is not True:
        raise ValueError("Explicit CPU/CUDA, positive thread count and deterministic execution are required")
    evaluation = configuration["evaluation"]
    if set(evaluation) != set(DEFAULT_CONFIG["evaluation"]):
        raise ValueError("Unknown or missing evaluation fields")
    for name in ["calibration_bins", "bootstrap_samples"]:
        if type(evaluation[name]) is not int or evaluation[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    if type(evaluation["bootstrap_seed"]) is not int or not 0 <= evaluation["bootstrap_seed"] < 2**63 or not 0 < evaluation["confidence"] < 1:
        raise ValueError("Invalid bootstrap seed/confidence")
    if not isinstance(configuration["seed"], int) or not 0 <= configuration["seed"] < 2**63:
        raise ValueError("Seed must be an integer in [0, 2**63)")
    training = configuration["training"]
    if set(training) != set(DEFAULT_CONFIG["training"]):
        raise ValueError("Unknown or missing training fields")
    for name in ["epochs", "batch_size", "patience", "scheduler_patience"]:
        if not isinstance(training[name], int) or training[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    for name in ["learning_rate", "max_grad_norm", "scheduler_factor"]:
        if not isinstance(training[name], (int, float)) or not math.isfinite(training[name]) or training[name] <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if training["scheduler_factor"] >= 1:
        raise ValueError("Scheduler factor must be less than one")
    for name in ["weight_decay", "min_delta"]:
        if not math.isfinite(training[name]) or training[name] < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    preprocessing = configuration["preprocessing"]
    if set(preprocessing) != set(DEFAULT_CONFIG["preprocessing"]):
        raise ValueError("Unknown or missing preprocessing fields")
    if not isinstance(preprocessing["size"], int) or preprocessing["size"] < 1:
        raise ValueError("Image size must be positive")
    if preprocessing["interpolation"] != "lanczos" or preprocessing["aspect_policy"] != "letterbox" or preprocessing["color"] != "RGB":
        raise ValueError("Only the documented Lanczos/RGB/letterbox policy is implemented")
    if not 0 <= preprocessing["horizontal_flip_probability"] <= 1 or not 0 <= preprocessing["rotation_degrees"] <= 180:
        raise ValueError("Invalid augmentation parameters")
    for name in ["mean", "std"]:
        if len(preprocessing[name]) != 3 or not all(math.isfinite(value) for value in preprocessing[name]):
            raise ValueError("Normalization requires three finite channels")
    if any(value <= 0 for value in preprocessing["std"]):
        raise ValueError("Normalization scales must be positive")
    validate_attention(configuration["model"])
    digest(configuration)
    return deepcopy(configuration)


def validate_attention(model_configuration, model=None):
    if not model_configuration.get("name"):
        raise ValueError("Model configuration needs an explicit name")
    if model_configuration.get("claims_spatial_attention"):
        count = model_configuration.get("token_count", 0)
        if not isinstance(count, int) or count <= 1:
            raise ValueError("Spatial attention requires more than one spatial token")
        if model is not None and getattr(model, "spatial_token_count", None) != count:
            raise ValueError("Model adapter must expose its actual spatial_token_count")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate configuration key: {key}")
        result[key] = value
    return result


def load_config(path):
    with open(path, encoding="utf-8") as stream:
        return validate_config(json.load(stream, object_pairs_hook=unique_object))


def run_identity(configuration, manifest_hash, split_hash, class_map):
    return digest({"foundation_version": 3, "configuration": validate_config(configuration), "dataset_manifest_hash": manifest_hash, "split_hash": split_hash, "class_map": class_map})
