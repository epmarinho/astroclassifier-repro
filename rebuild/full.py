"""Full train/validation development runs for the three custom architectures.

This command deliberately excludes the frozen test split.  It is a one-seed
development screen over every eligible train and validation content group, not
a final benchmark or a test-set evaluation.
"""

import argparse
from copy import deepcopy
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from PIL import Image

from . import training
from .common import PROJECT, ROOT, digest, file_hash, read_json, write_json
from .config import DEFAULT_CONFIG, validate_config
from .data import CLASSES, quantitative_records
from .phase3d.models_v1 import VARIANTS, architecture, model_factory
from .phase3d.pilot import PilotDataset, projected_train_validation


VERSION = "full-development-v3"
SEED = 3908274
REPORT = ROOT / "reports/full_v3"
CONFIGS = ROOT / "configs/full_v3"
RUNS = ROOT / "runs/full_v3"
FIGURES = ROOT / "figures/full_v3"
MANIFEST_PATH = REPORT / "train_validation_manifest.json"
SPLIT_PATH = REPORT / "train_validation_split.json"
PREFLIGHT_PATH = REPORT / "preflight.json"
PLAN_PATH = REPORT / "plan.json"


def source_hashes():
    names = [
        "common.py", "config.py", "data.py", "evaluation.py", "training.py",
        "phase3b/data_v1.py", "phase3d/models_v1.py", "phase3d/pilot.py",
        "full.py",
    ]
    return {"rebuild/" + name: file_hash(ROOT / name) for name in names}


def environment():
    capability = list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    return {
        "python_executable": sys.executable,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_cuda_architectures": torch.cuda.get_arch_list(),
        "torchvision": importlib.metadata.version("torchvision"),
        "numpy": importlib.metadata.version("numpy"),
        "Pillow": Image.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_capability": capability,
        "deterministic_algorithms": True,
        "network_used": False,
    }


def full_collection():
    """Project only frozen train/validation representatives; skip test groups."""
    manifest, groups, skipped_test_groups, provenance = projected_train_validation()
    representatives = {
        row["content_group_id"]: row for row in quantitative_records(manifest)
    }
    records = []
    assignments = {}
    for group in groups:
        role = group["split"]
        if role not in {"train", "validation"}:
            raise ValueError("Full development collection accepts train/validation only")
        record = deepcopy(representatives[group["content_group_id"]])
        records.append(record)
        assignments[record["image_id"]] = role
    collection = {
        "schema_version": 2,
        "records": sorted(records, key=lambda row: row["image_id"]),
        "parent_evidence_sha256": provenance["evidence_manifest_sha256"],
        "parent_split_sha256": provenance["split_sha256"],
    }
    split = {
        "manifest_hash": digest(collection),
        "assignments": dict(sorted(assignments.items())),
        "policy": "full_development_train_validation_only_exact_content_groups",
        "object_independence_certified": False,
        "parent_split_sha256": provenance["split_sha256"],
    }
    counts = {
        role: sum(value == role for value in assignments.values())
        for role in ("train", "validation")
    }
    if counts != {"train": 485, "validation": 158}:
        raise AssertionError(f"Unexpected full development counts: {counts}")
    if set(assignments.values()) != {"train", "validation"}:
        raise AssertionError("The full development collection has an unexpected role")
    return collection, split, counts, skipped_test_groups


def full_configuration(variant, preflight):
    configuration = deepcopy(DEFAULT_CONFIG)
    configuration["seed"] = SEED
    configuration["model"] = {
        **architecture(variant),
        "source_digests": preflight["source_digests"],
        "environment_digest": digest(preflight["environment"]),
        "optimizer": "Adam",
        "data_loader_workers": 8,
        "pin_memory": True,
        "purpose": "full development train-validation screen; no test evaluation",
    }
    configuration["training"].update(
        epochs=100,
        batch_size=16,
        learning_rate=9e-5,
        weight_decay=5e-7,
        patience=20,
        min_delta=0.0,
        scheduler_patience=5,
        scheduler_factor=0.1,
    )
    configuration["runtime"] = {
        "device": "cuda",
        "num_threads": 2,
        "deterministic": True,
    }
    configuration["evaluation"].update(
        calibration_bins=10,
        bootstrap_samples=1000,
        bootstrap_seed=SEED,
    )
    return validate_config(configuration)


def write_once_or_verify(path, value):
    if path.exists():
        if read_json(path) != value:
            raise ValueError(f"Existing immutable output differs: {path}")
    else:
        write_json(path, value)


def require_blackwell_cuda(info):
    if not info["cuda_available"]:
        raise RuntimeError("CUDA is unavailable; activate the GPU-capable environment")
    if info["cuda_capability"] != [12, 0]:
        raise RuntimeError(f"Expected RTX 5060 Ti capability 12.0, got {info['cuda_capability']}")
    if "sm_120" not in info["torch_cuda_architectures"]:
        raise RuntimeError("This PyTorch build lacks sm_120 kernels")


def gpu_smoke(configuration):
    training.configure_runtime(configuration)
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(configuration["seed"])
    model = model_factory(configuration["model"], len(CLASSES)).to("cuda")
    model.train()
    batch = torch.zeros(
        configuration["training"]["batch_size"],
        3,
        configuration["preprocessing"]["size"],
        configuration["preprocessing"]["size"],
        device="cuda",
    )
    labels = torch.arange(len(batch), device="cuda") % len(CLASSES)
    loss = torch.nn.functional.cross_entropy(model(batch), labels)
    loss.backward()
    torch.cuda.synchronize()
    result = {
        "loss": float(loss.detach().cpu()),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "batch_size": len(batch),
        "forward_backward": "passed",
    }
    del loss, labels, batch, model
    torch.cuda.empty_cache()
    return result


def preflight():
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    info = environment()
    require_blackwell_cuda(info)
    collection, split, counts, skipped = full_collection()
    for row in collection["records"]:
        path = (PROJECT / row["relative_path"]).resolve()
        if not path.is_relative_to(PROJECT) or file_hash(path) != row["sha256"]:
            raise ValueError(f"Frozen development input differs: {row['relative_path']}")
    preflight_record = {
        "version": VERSION,
        "status": "passed",
        "purpose": "one-seed full development screen; not final benchmark",
        "environment": info,
        "source_digests": source_hashes(),
        "counts": {**counts, "test": 0},
        "skipped_test_groups_before_label_deserialization": skipped,
        "test_dataset_constructed": False,
        "object_independence_certified": False,
    }
    configurations = {
        variant: full_configuration(variant, preflight_record) for variant in VARIANTS
    }
    smoke = {variant: gpu_smoke(configurations[variant]) for variant in VARIANTS}
    plan = {
        "version": VERSION,
        "variants": list(VARIANTS),
        "seed": SEED,
        "counts": preflight_record["counts"],
        "maximum_epochs": 100,
        "batch_size": 16,
        "data_loader_workers": 8,
        "pin_memory": True,
        "selection": "minimum class-weighted validation loss",
        "test_evaluation": False,
        "gpu_smoke": smoke,
        "configuration_digests": {
            variant: digest(configuration) for variant, configuration in configurations.items()
        },
    }
    write_once_or_verify(MANIFEST_PATH, collection)
    write_once_or_verify(SPLIT_PATH, split)
    write_once_or_verify(PREFLIGHT_PATH, preflight_record)
    write_once_or_verify(PLAN_PATH, plan)
    for variant, configuration in configurations.items():
        write_once_or_verify(CONFIGS / f"{variant}_seed{SEED}.json", configuration)
    print("Full GPU preflight passed: 485 train, 158 validation, 0 test.", flush=True)
    for variant in VARIANTS:
        mib = smoke[variant]["peak_memory_bytes"] / 2**20
        print(f"  {variant}: synthetic forward/backward passed; peak {mib:.0f} MiB", flush=True)


def checked_inputs():
    preflight_record = read_json(PREFLIGHT_PATH)
    if environment() != preflight_record["environment"]:
        raise RuntimeError("Python/PyTorch/CUDA environment differs from preflight")
    if source_hashes() != preflight_record["source_digests"]:
        raise RuntimeError("Execution sources differ from preflight")
    collection = read_json(MANIFEST_PATH)
    split = read_json(SPLIT_PATH)
    if split["manifest_hash"] != digest(collection):
        raise RuntimeError("Frozen full manifest/split differ")
    return preflight_record, collection, split


def run(selected_variants):
    preflight_record, collection, split = checked_inputs()
    variants = list(VARIANTS) if selected_variants == ["all"] else selected_variants
    if len(set(variants)) != len(variants):
        raise ValueError("A variant was requested more than once")
    for variant in variants:
        configuration = read_json(CONFIGS / f"{variant}_seed{SEED}.json")
        if configuration != full_configuration(variant, preflight_record):
            raise RuntimeError(f"Frozen configuration differs for {variant}")
        datasets = {
            role: PilotDataset(collection, split, role, configuration["preprocessing"], SEED)
            for role in ("train", "validation")
        }
        identifier = training.run_identity(
            configuration,
            datasets["train"].manifest_hash,
            datasets["train"].split_hash,
            datasets["train"].class_map,
        )
        run_directory = RUNS / identifier
        resume = run_directory.exists()
        action = "validating completed/resuming" if resume else "starting"
        print(f"[{variant}] {action} run {identifier[:12]}", flush=True)

        def progress(item):
            print(
                f"[{variant}] epoch {item['epoch'] + 1:03d}/{item['epochs_maximum']:03d} "
                f"train={item['train_loss']:.5f} val={item['validation_loss']:.5f} "
                f"acc={item['validation_accuracy']:.3f} "
                f"bal_acc={item['validation_balanced_accuracy']:.3f} "
                f"lr={item['learning_rate']:.3g} bad={item['bad_epochs']}",
                flush=True,
            )

        started = time.monotonic()
        result = training.fit(
            model_factory,
            datasets["train"],
            datasets["validation"],
            configuration,
            RUNS,
            resume=resume,
            epoch_callback=progress,
        )
        elapsed = time.monotonic() - started
        metrics = result["validation"]
        print(
            f"[{variant}] complete: best epoch {result['best_epoch'] + 1}, "
            f"balanced accuracy {metrics['balanced_accuracy']:.4f}, "
            f"macro-F1 {metrics['macro_f1']:.4f}, command time {elapsed:.1f}s",
            flush=True,
        )
    print("Requested full development runs completed; test remains locked.", flush=True)


def status():
    if not PREFLIGHT_PATH.exists():
        print("Preflight has not been created.")
        return
    _, collection, split = checked_inputs()
    for variant in VARIANTS:
        configuration = read_json(CONFIGS / f"{variant}_seed{SEED}.json")
        dataset = PilotDataset(collection, split, "train", configuration["preprocessing"], SEED)
        identifier = training.run_identity(
            configuration, dataset.manifest_hash, dataset.split_hash, dataset.class_map
        )
        directory = RUNS / identifier
        state = "completed" if (directory / "result.json").exists() else "interrupted/resumable" if (directory / "last.pt").exists() else "not started"
        print(f"{variant:24s} {state:23s} {identifier[:12]}")


def visualize(force):
    from . import visualize as charts

    runs = charts.load_runs(RUNS)
    output = charts.prepare_output(FIGURES, force)
    label = "Full development train/validation screen"
    charts.plot_learning_curves(runs, output, label=label)
    charts.plot_confusion_matrices(runs, output, label=label)
    charts.write_scores(
        runs,
        output,
        heading="Full development validation scores",
        note=(
            "These one-seed scores use all 485 development-training and 158 "
            "development-validation contents. The locked test split was not evaluated."
        ),
    )
    print(f"Generated full-development figures and tables in {output.relative_to(PROJECT)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    subparsers.add_parser("preflight", help="freeze inputs/configurations and test CUDA memory")
    run_parser = subparsers.add_parser("run", help="run or resume selected architectures")
    run_parser.add_argument(
        "--variant",
        action="append",
        choices=["all", *VARIANTS],
        default=None,
        help="variant to run; repeat the option, or omit it to run all",
    )
    subparsers.add_parser("status", help="show completed/interrupted/not-started runs")
    visual_parser = subparsers.add_parser("visualize", help="generate curves and score tables")
    visual_parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    if arguments.action == "preflight":
        preflight()
    elif arguments.action == "run":
        run(arguments.variant or ["all"])
    elif arguments.action == "status":
        status()
    else:
        visualize(arguments.force)


if __name__ == "__main__":
    main()
