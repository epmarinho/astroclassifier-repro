import argparse
from collections import Counter
from copy import deepcopy
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
import time
import math

import torch
from PIL import Image

from .. import training
from ..common import PROJECT, ROOT, digest, file_hash, read_json, write_json
from ..config import DEFAULT_CONFIG, validate_config
from ..data import CLASSES, Preprocessor, quantitative_records
from ..phase3b.data_v1 import raw_array_objects
from .models_v1 import VARIANTS, architecture, model_factory


REPORT = ROOT / "reports/phase3d_20261016"
CONFIGS = ROOT / "configs/phase3d_v4"
RUNS = ROOT / "runs/phase3d_v4"
SUBSET_SALT = "phase3d-v4-stratified-development-pilot-20261016"
PER_CLASS = {"train": 12, "validation": 6}
SEED = 3908274


def source_hashes():
    names = [
        "common.py", "config.py", "data.py", "evaluation.py", "training.py",
        "phase3b/data_v1.py", "phase3d/__init__.py", "phase3d/models_v1.py",
        "phase3d/pilot.py", "tests/test_phase3d.py",
    ]
    return {"rebuild/" + name: file_hash(ROOT / name) for name in names}


def environment():
    return {
        "python_executable": sys.executable,
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "numpy": importlib.metadata.version("numpy"),
        "Pillow": Image.__version__,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "torch_threads": 2,
        "deterministic_algorithms": True,
        "network_used": False,
    }


def projected_train_validation():
    folder = ROOT / "reports/reaudit_20261016"
    provenance = read_json(folder / "split_provenance.json")
    manifest_path = folder / "evidence_axes_manifest.json"
    split_path = folder / "provisional_content_grouped_split.json"
    if file_hash(manifest_path) != provenance["evidence_manifest_sha256"] or file_hash(split_path) != provenance["split_sha256"]:
        raise ValueError("Frozen Phase 3A source hashes differ")
    groups = []
    selected_ids = set()
    skipped_test_groups = 0
    for raw in raw_array_objects(split_path, "assignments"):
        roles = __import__("re").findall(r'"split"\s*:\s*"([^"]+)"', raw)
        if roles == ["test"]:
            skipped_test_groups += 1
            continue
        if len(roles) != 1 or roles[0] not in PER_CLASS:
            raise ValueError("Unexpected split role in Phase 3A proposal")
        group = __import__("json").loads(raw)
        groups.append(group)
        selected_ids.update(group["member_ids"])
    rows = []
    for raw in raw_array_objects(manifest_path, "records"):
        matches = __import__("re").findall(r'"image_id"\s*:\s*"([^"]+)"', raw)
        if len(matches) != 1:
            raise ValueError("Ambiguous image ID in frozen manifest")
        if matches[0] in selected_ids:
            rows.append(__import__("json").loads(raw))
    manifest = {
        "schema_version": 2,
        "records": rows,
        "parent_evidence_sha256": provenance["evidence_manifest_sha256"],
        "parent_split_sha256": provenance["split_sha256"],
    }
    representatives = {row["content_group_id"]: row for row in quantitative_records(manifest)}
    for group in groups:
        record = representatives[group["content_group_id"]]
        if (record["relative_path"], record["class_label"], record["sha256"]) != (group["representative"], group["label"], group["sha256"]):
            raise ValueError("Phase 3A representative mismatch")
    return manifest, groups, skipped_test_groups, provenance


def select_pilot_subset(manifest, groups):
    representatives = {row["content_group_id"]: row for row in quantitative_records(manifest)}
    chosen = []
    assignment = {}
    for role, count in PER_CLASS.items():
        for label in sorted(CLASSES):
            candidates = [group for group in groups if group["split"] == role and group["label"] == label]
            candidates.sort(key=lambda group: digest([SUBSET_SALT, role, label, group["content_group_id"]]))
            if len(candidates) < count:
                raise ValueError(f"Insufficient {role}/{label} groups for fixed pilot")
            for group in candidates[:count]:
                record = deepcopy(representatives[group["content_group_id"]])
                chosen.append(record)
                assignment[record["image_id"]] = role
    subset = {
        "schema_version": 2,
        "records": sorted(chosen, key=lambda row: row["image_id"]),
        "parent_evidence_sha256": manifest["parent_evidence_sha256"],
        "parent_split_sha256": manifest["parent_split_sha256"],
        "subset_salt": SUBSET_SALT,
    }
    split = {
        "manifest_hash": digest(subset),
        "assignments": dict(sorted(assignment.items())),
        "policy": "phase3d_development_pilot_train_validation_only",
        "object_independence_certified": False,
        "parent_split_sha256": manifest["parent_split_sha256"],
        "per_class": PER_CLASS,
    }
    owners = {}
    for row in subset["records"]:
        role = assignment[row["image_id"]]
        for key in [row["content_group_id"], row.get("provisional_object_id")]:
            if key and key in owners and owners[key] != role:
                raise ValueError("Content/object group crosses pilot splits")
            if key:
                owners[key] = role
    counts = Counter(assignment.values())
    if counts != {"train": 48, "validation": 24}:
        raise AssertionError("Unexpected Phase 3D subset sizes")
    return subset, split


class PilotDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, split, role, preprocessing, seed):
        if role not in PER_CLASS:
            raise ValueError("Phase 3D dataset accepts train/validation only")
        if split["manifest_hash"] != digest(manifest):
            raise ValueError("Pilot subset manifest changed")
        if set(split["assignments"].values()) != {"train", "validation"}:
            raise ValueError("Test split cannot be constructed in Phase 3D")
        self.class_map = {label: index for index, label in enumerate(sorted(CLASSES))}
        self.manifest_hash = digest(manifest)
        self.split_hash = digest(split)
        self.split_policy = split["policy"]
        self.object_independence_certified = False
        self.split = role
        self.seed = seed
        self.epoch = 0
        self.root = PROJECT
        self.records = [row for row in manifest["records"] if split["assignments"][row["image_id"]] == role]
        self.preprocessing = Preprocessor(preprocessing)
        self._ground_truth = self._label_contract()
        self.ground_truth_hash = digest(self._ground_truth)

    def _label_contract(self):
        records = [{
            "image_id": row["image_id"],
            "source_path": row["relative_path"],
            "content_group_id": row["content_group_id"],
            "provisional_object_id": row.get("provisional_object_id"),
            "true_label": self.class_map[row["class_label"]],
        } for row in self.records]
        return {"manifest_hash": self.manifest_hash, "split_id": self.split_hash, "split": self.split, "class_map": self.class_map, "records": records}

    def evaluation_contract(self):
        if digest(self._ground_truth) != self.ground_truth_hash or self._label_contract() != self._ground_truth:
            raise ValueError("Frozen pilot ground truth changed")
        return deepcopy(self._ground_truth)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        row = self.records[index]
        path = (self.root / row["relative_path"]).resolve()
        if not path.is_relative_to(self.root) or file_hash(path) != row["sha256"]:
            raise ValueError("Pilot image path escaped root or source bytes changed")
        augmentation_seed = int(digest([self.seed, self.epoch, row["image_id"]])[:16], 16)
        if path.suffix.lower() == ".avif":
            import pillow_avif
        with Image.open(path) as image:
            image.load()
            tensor = self.preprocessing(image, self.split, augmentation_seed)
        return {
            "image": tensor,
            "label": self.class_map[row["class_label"]],
            "image_id": row["image_id"],
            "source_path": row["relative_path"],
            "content_group_id": row["content_group_id"],
            "provisional_object_id": row.get("provisional_object_id") or "",
            "object_group_id": row["object_group_id"],
            "provenance_status": row["provenance_status"],
            "licence_status": row["licence_status"],
            "provenance_evidence": row.get("provenance_evidence") or "",
        }


class FileBoundary:
    def __init__(self, allowed_images):
        self.allowed_images = {path.resolve() for path in allowed_images}
        self.opened_images = Counter()
        self.denied = []
        self.write_roots = [REPORT.resolve(), CONFIGS.resolve(), RUNS.resolve(), (ROOT / "tmp").resolve()]

    def __call__(self, event, arguments):
        if event != "open" or not isinstance(arguments[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.fsdecode(arguments[0])).resolve()
        mode, flags = arguments[1:3]
        writing = bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)) or isinstance(mode, str) and any(char in mode for char in "wa+")
        if writing and not any(path.is_relative_to(root) for root in self.write_roots):
            self.denied.append(str(path))
            raise PermissionError("Phase 3D writes are limited to new rebuild outputs")
        if path.is_relative_to(PROJECT / "images"):
            if path not in self.allowed_images:
                self.denied.append(str(path))
                raise PermissionError("Phase 3D permits only selected train/validation images")
            self.opened_images[str(path.relative_to(PROJECT))] += 1


def pilot_configuration(variant, seed, preflight):
    configuration = deepcopy(DEFAULT_CONFIG)
    configuration["seed"] = seed
    configuration["model"] = {
        **architecture(variant),
        "source_digests": preflight["source_digests"],
        "environment_digest": digest(preflight["environment"]),
        "optimizer": "Adam",
        "purpose": "bounded development pilot only; no superiority claim",
    }
    configuration["training"].update(epochs=20, batch_size=4, learning_rate=1e-4, weight_decay=5e-7, patience=5, min_delta=0.001, scheduler_patience=2, scheduler_factor=0.5)
    configuration["preprocessing"].update(size=256, rotation_degrees=15.0, horizontal_flip_probability=0.5)
    configuration["runtime"] = {"device": "cpu", "num_threads": 2, "deterministic": True}
    configuration["evaluation"].update(calibration_bins=5, bootstrap_samples=500, bootstrap_seed=seed)
    return validate_config(configuration)


def preflight():
    REPORT.mkdir(parents=True, exist_ok=True)
    CONFIGS.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    (ROOT / "tmp").mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    manifest, groups, skipped_test_groups, source_provenance = projected_train_validation()
    subset, split = select_pilot_subset(manifest, groups)
    allowed = [PROJECT / row["relative_path"] for row in subset["records"]]
    boundary = FileBoundary(allowed)
    sys.addaudithook(boundary)
    inspection = []
    for row in subset["records"]:
        path = PROJECT / row["relative_path"]
        if file_hash(path) != row["sha256"]:
            raise RuntimeError("Selected image bytes differ from frozen manifest")
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            inspection.append({"image_id": row["image_id"], "source_path": row["relative_path"], "format": image.format, "mode": image.mode, "size": list(image.size), "sha256": row["sha256"], "split": split["assignments"][row["image_id"]]})
    environment_info = environment()
    if environment_info["cuda_available"]:
        environment_info["device_note"] = "CUDA was available but CPU was explicitly selected for bounded pilot"
    result = {
        "status": "passed_cpu_preflight",
        "purpose": "bounded development pilot only",
        "environment": environment_info,
        "source_digests": source_hashes(),
        "parent_split_sha256": source_provenance["split_sha256"],
        "parent_manifest_sha256": source_provenance["evidence_manifest_sha256"],
        "projected_test_groups_skipped_before_label_deserialization": skipped_test_groups,
        "subset_counts": {"train": 48, "validation": 24, "test": 0},
        "input_resolution": 128,
        "input_inspection": inspection,
        "image_access_during_preflight": dict(boundary.opened_images),
        "denied_access": boundary.denied,
        "object_independence_certified": False,
        "test_images_opened": 0,
        "test_labels_deserialized": False,
    }
    result["input_resolution"] = 256
    write_json(REPORT / "preflight_256_v4.json", result)
    write_json(REPORT / "subset_manifest_256_v4.json", subset)
    write_json(REPORT / "subset_split_256_v4.json", split)
    configurations = []
    for variant in VARIANTS:
        configuration = pilot_configuration(variant, SEED, result)
        write_json(CONFIGS / f"{variant}_seed{SEED}.json", configuration)
        torch.manual_seed(SEED)
        model = model_factory(configuration["model"], len(CLASSES))
        configurations.append({
            "variant": variant,
            "seed": SEED,
            "configuration": configuration,
            "configuration_digest": digest(configuration),
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
            "spatial_sequence": [1, 1, 128] if variant == "historical_singleton" else [1, 64, 128],
        })
    write_json(REPORT / "pilot_plan_256_v4.json", {"run_count": 3, "variants": list(VARIANTS), "seed": SEED, "epochs_maximum": 20, "configurations": configurations})
    print("Phase 3D CPU preflight passed; train/validation-only subset and three immutable model configurations recorded.", flush=True)


def run():
    preflight_result = read_json(REPORT / "preflight_256_v4.json")
    if environment() != {key: value for key, value in preflight_result["environment"].items() if key != "device_note"}:
        raise RuntimeError("Environment differs from preflight")
    if source_hashes() != preflight_result["source_digests"]:
        raise RuntimeError("Source digests differ from preflight")
    subset = read_json(REPORT / "subset_manifest_256_v4.json")
    split = read_json(REPORT / "subset_split_256_v4.json")
    allowed = [PROJECT / row["relative_path"] for row in subset["records"]]
    boundary = FileBoundary(allowed)
    sys.addaudithook(boundary)
    ledger = []
    for variant in VARIANTS:
        configuration = read_json(CONFIGS / f"{variant}_seed{SEED}.json")
        if configuration != pilot_configuration(variant, SEED, preflight_result):
            raise RuntimeError("Frozen pilot configuration differs")
        datasets = {role: PilotDataset(subset, split, role, configuration["preprocessing"], SEED) for role in ("train", "validation")}
        initial_states = {}
        gradient_checks = []
        def capturing_factory(model_configuration, classes):
            model = model_factory(model_configuration, classes)
            initial_states.update({name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()})
            return model
        original_train_epoch = training.train_epoch
        def checked_train_epoch(model, loader, optimizer, weights=None, max_grad_norm=3.0):
            loss = original_train_epoch(model, loader, optimizer, weights, max_grad_norm)
            gradient_checks.append(all(parameter.grad is None or torch.isfinite(parameter.grad).all().item() for parameter in model.parameters()))
            return loss
        training.train_epoch = checked_train_epoch
        start = time.monotonic()
        run_identifier = training.run_identity(configuration, datasets["train"].manifest_hash, datasets["train"].split_hash, datasets["train"].class_map)
        partial_run = RUNS / run_identifier
        resume_interrupted_run = (partial_run / "last.pt").exists() and not (partial_run / "result.json").exists()
        try:
            result = training.fit(capturing_factory, datasets["train"], datasets["validation"], configuration, RUNS, resume=resume_interrupted_run)
        finally:
            training.train_epoch = original_train_epoch
        run_dir = RUNS / result["run_id"]
        provenance = read_json(run_dir / "run.json")
        checkpoint = training.read_checkpoint(run_dir / "best.pt", provenance)
        predictions = read_json(run_dir / "validation_predictions.json")
        history = read_json(run_dir / "history.json")
        best_state = checkpoint["model_state"]
        parameters_changed = any(not torch.equal(initial_states[name], value.detach().cpu()) for name, value in best_state.items() if name in initial_states and value.is_floating_point())
        train_losses = [entry["train_loss"] for entry in history]
        finite_losses = all(math.isfinite(value) for value in train_losses + [entry["validation"]["loss"] for entry in history])
        train_loss_decreased = bool(train_losses) and train_losses[-1] < train_losses[0]
        train_counts = Counter(row["class_label"] for row in datasets["train"].records)
        priors = [train_counts[label] / len(datasets["train"]) for label in sorted(CLASSES)]
        baseline_nll = -sum(math.log(priors[row["true_label"]]) for row in predictions) / len(predictions)
        validation_nll = result["validation"]["negative_log_likelihood"]
        run_before_resume = {path.name: file_hash(path) for path in run_dir.iterdir() if path.is_file()}
        resumed = training.fit(lambda *arguments: (_ for _ in ()).throw(AssertionError("Completed run was refit")), datasets["train"], datasets["validation"], configuration, RUNS, resume=True)
        resume_consistent = resumed == result and run_before_resume == {path.name: file_hash(path) for path in run_dir.iterdir() if path.is_file()}
        labels_predicted = sorted({row["predicted_label"] for row in predictions})
        record = {
            "variant": variant,
            "seed": SEED,
            "run_id": result["run_id"],
            "configuration_digest": digest(configuration),
            "epochs_completed": result["epochs_completed"],
            "best_epoch": result["best_epoch"],
            "elapsed_seconds": time.monotonic() - start,
            "parameter_count": sum(parameter.numel() for parameter in model_factory(configuration["model"], len(CLASSES)).parameters()),
            "train_count": len(datasets["train"]),
            "validation_count": len(predictions),
            "train_group_ids": sorted({row["content_group_id"] for row in datasets["train"].records}),
            "validation_group_ids": sorted({row["content_group_id"] for row in datasets["validation"].records}),
            "groups_disjoint": not bool({row["content_group_id"] for row in datasets["train"].records} & {row["content_group_id"] for row in datasets["validation"].records}),
            "predicted_class_indices": labels_predicted,
            "train_loss_by_epoch": train_losses,
            "acceptance": {
                "finite_losses": finite_losses,
                "finite_gradients": bool(gradient_checks) and all(gradient_checks),
                "trainable_parameters_changed": parameters_changed,
                "training_loss_decreased": train_loss_decreased,
                "predictions_contain_at_least_two_classes": len(labels_predicted) >= 2,
                "balanced_validation_accuracy_gt_0_25": result["validation"]["balanced_accuracy"] > 0.25,
                "validation_nll_better_than_constant_training_prior": validation_nll < baseline_nll,
                "checkpoint_resume_consistent": resume_consistent,
                "test_accessed": False,
            },
            "constant_training_prior_nll": baseline_nll,
            "validation_nll": validation_nll,
            "metrics_label": "development-pilot diagnostics only; not a benchmark",
            "validation_metrics": result["validation"],
            "checkpoint_epoch": checkpoint["epoch"],
            "checkpoint_provenance_complete": checkpoint["configuration_digest"] == digest(configuration) and checkpoint["split_hash"] == datasets["train"].split_hash,
            "test_accessed": False,
        }
        write_json(REPORT / f"run_{variant}_256_v4.json", record)
        ledger.append(record)
        print(f"Completed {variant}: {result['epochs_completed']} epochs", flush=True)
    write_json(REPORT / "image_access_256_v4.json", {
        "allowed_paths": sorted(str(path.relative_to(PROJECT)) for path in boundary.allowed_images),
        "opened_images": dict(boundary.opened_images),
        "denied_access": boundary.denied,
        "test_images_opened": 0,
        "test_labels_deserialized": False,
    })
    write_json(REPORT / "execution_ledger_256_v4.json", {"purpose": "bounded development pilot only", "run_count": len(ledger), "seed_is_engineering_replication_only": True, "no_model_ranking_or_superiority_claim": True, "runs": ledger})
    print("Phase 3D pilot stopped after the three predeclared runs.", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["preflight", "run"])
    selected_action = parser.parse_args().action
    (preflight if selected_action == "preflight" else run)()
