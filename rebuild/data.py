from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
import importlib
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset

from .common import ROOT, digest, file_hash, read_json
from .config import DEFAULT_CONFIG, validate_config


ROLES = ("train", "validation", "test")
CLASSES = {"galaxies", "globularclusters", "nebulae", "opencluster"}
EVIDENCE_AXES = {
    "file_status": {"readable", "unreadable", "unsupported_decoder", "corrupted"},
    "duplicate_status": {"unique_bytes", "exact_duplicate", "near_duplicate_candidate"},
    "label_status": {"historically_assigned", "internally_consistent", "conflicting_exact_duplicate", "author_confirmed", "externally_verified", "unresolved"},
    "provenance_status": {"locally_documented", "source_identifier_present", "source_unknown", "externally_verified"},
    "licence_status": {"documented", "requires_review", "unknown"},
    "benchmark_status": {"provisionally_eligible", "grouped_duplicate", "excluded_confirmed_problem", "pending_author_decision"},
}


def approved_records(manifest, allow_synthetic=False):
    records = []
    identifiers = set()
    for record in manifest["records"]:
        if record["image_id"] in identifiers:
            raise ValueError("Duplicate image identifiers")
        identifiers.add(record["image_id"])
        if "benchmark_status" in record and any(record.get(axis) not in values for axis, values in EVIDENCE_AXES.items()):
            raise ValueError("Invalid or missing evidence-axis value")
        if record.get("benchmark_status") not in {"provisionally_eligible", "grouped_duplicate"}:
            continue
        if record.get("file_status") != "readable" or record.get("label_status") in {"conflicting_exact_duplicate", "unresolved"}:
            continue
        label = record.get("class_label", record.get("original_directory_label"))
        if label not in CLASSES:
            continue
        if record.get("original_directory_label") not in CLASSES and not (record.get("label_status") in {"author_confirmed", "externally_verified"} and record.get("adjudication_status") == "approved" and record.get("class_evidence")):
            continue
        if record.get("content_status") == "synthetic" and not allow_synthetic:
            continue
        normalized = deepcopy(record)
        normalized["class_label"] = label
        normalized["content_group_id"] = "bytes:" + record["sha256"]
        normalized["provisional_object_id"] = record.get("provisional_object_id") or record.get("object_group_id")
        normalized["object_group_id"] = normalized["provisional_object_id"] or normalized["content_group_id"]
        records.append(normalized)
    all_labels = defaultdict(set)
    blocked_hashes = set()
    for record in manifest["records"]:
        label = record.get("class_label", record.get("original_directory_label"))
        if label in CLASSES:
            all_labels[record["sha256"]].add(label)
            if record.get("benchmark_status") in {"excluded_confirmed_problem", "pending_author_decision"}:
                blocked_hashes.add(record["sha256"])
    records = [record for record in records if len(all_labels[record["sha256"]]) == 1 and record["sha256"] not in blocked_hashes]
    return sorted(records, key=lambda record: record["image_id"])


def apply_adjudications(source_manifest, decisions):
    manifest = deepcopy(source_manifest)
    by_identifier = {record["image_id"]: record for record in manifest["records"]}
    editable = {"class_label", "class_evidence", "provenance_evidence", "licence_evidence", "object_evidence", "object_group_id", "provisional_object_id", "object_identity_status", "provenance_status", "licence_status", "content_status", "benchmark_status", "label_status", "exclusion_reason", "adjudication_status"}
    for identifier, updates in decisions.items():
        if identifier not in by_identifier or not set(updates).issubset(editable):
            raise ValueError("Unknown record or attempted alteration of immutable source fields")
        by_identifier[identifier].update(updates)
    manifest["source_manifest_hash"] = digest(source_manifest)
    manifest["adjudication_hash"] = digest(decisions)
    approved_records(manifest)
    return manifest


def group_components(records):
    parent = {record["image_id"]: record["image_id"] for record in records}

    def find(identifier):
        while parent[identifier] != identifier:
            parent[identifier] = parent[parent[identifier]]
            identifier = parent[identifier]
        return identifier

    owners = {}
    for record in records:
        keys = [("bytes", record["sha256"])]
        object_id = record.get("provisional_object_id") or record.get("object_group_id")
        if object_id:
            keys.append(("object", object_id))
        for key in keys:
            identifier = record["image_id"]
            if key in owners:
                left, right = find(identifier), find(owners[key])
                parent[max(left, right)] = min(left, right)
            else:
                owners[key] = identifier
    groups = defaultdict(list)
    for record in records:
        groups[find(record["image_id"])].append(record)
    return dict(sorted(groups.items()))


def make_splits(manifest, seed, ratios=(0.6, 0.2, 0.2), allow_synthetic=False):
    if len(ratios) != 3 or any(value <= 0 for value in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("Three positive split ratios must sum to one")
    records = approved_records(manifest, allow_synthetic)
    if not records:
        raise ValueError("No eligible historically labeled readable records in the evidence-axis manifest")
    by_class = defaultdict(list)
    for identifier, members in group_components(records).items():
        labels = {record["class_label"] for record in members}
        if len(labels) != 1:
            raise ValueError("An object/duplicate group has conflicting class labels")
        by_class[next(iter(labels))].append((identifier, members))
    assignments = {}
    for label, groups in sorted(by_class.items()):
        groups.sort(key=lambda group: digest([seed, label, group[0]]))
        if len(groups) < 3:
            raise ValueError(f"Class {label} needs at least three content/object groups")
        validation_count = max(1, int(len(groups) * ratios[1]))
        test_count = max(1, int(len(groups) * ratios[2]))
        train_count = len(groups) - validation_count - test_count
        if train_count < 1:
            raise ValueError("Requested ratios leave a class without training groups")
        for position, (_, members) in enumerate(groups):
            role = "train" if position < train_count else "validation" if position < train_count + validation_count else "test"
            for record in members:
                assignments[record["image_id"]] = role
    result = {"schema_version": 3, "manifest_hash": digest(manifest), "seed": seed, "ratios": list(ratios), "assignments": dict(sorted(assignments.items())), "policy": "development_only; exact bytes and supplied object groups; one representative per content group", "object_independence_certified": False}
    validate_splits(manifest, result, allow_synthetic)
    return result


def validate_splits(manifest, split, allow_synthetic=False):
    records = approved_records(manifest, allow_synthetic)
    if split["manifest_hash"] != digest(manifest):
        raise ValueError("Split was derived from a different manifest")
    assignments = split["assignments"]
    if set(assignments) != {record["image_id"] for record in records} or not records:
        raise ValueError("Split must cover exactly all provisional candidate records")
    if any(role not in ROLES for role in assignments.values()):
        raise ValueError("Unknown split role")
    if set(assignments.values()) != set(ROLES):
        raise ValueError("Train, validation and test must all be nonempty")
    for members in group_components(records).values():
        if len({assignments[record["image_id"]] for record in members}) != 1:
            raise ValueError("Duplicate/object group crosses splits")
        if len({record["class_label"] for record in members}) != 1:
            raise ValueError("Conflicting labels within an approved group")


def load_development_data():
    folder = ROOT / "reports/reaudit_20261016"
    manifest = read_json(folder / "evidence_axes_manifest.json")
    proposal = read_json(folder / "provisional_content_grouped_split.json")
    provenance = read_json(folder / "split_provenance.json")
    if file_hash(folder / "evidence_axes_manifest.json") != provenance["evidence_manifest_sha256"] or file_hash(folder / "provisional_content_grouped_split.json") != provenance["split_sha256"]:
        raise ValueError("Reconciled audit bytes differ from recorded provenance")
    assignments = {}
    for group in proposal["assignments"]:
        for identifier in group["member_ids"]:
            if identifier in assignments:
                raise ValueError("Repeated split member")
            assignments[identifier] = group["split"]
    split = {"schema_version": 3, "manifest_hash": digest(manifest), "seed": proposal["seed"], "assignments": assignments, "policy": "development_only_exact_content_groups", "object_independence_certified": False, "source_proposal_sha256": provenance["split_sha256"]}
    validate_splits(manifest, split)
    representatives = quantitative_records(manifest)
    by_hash = {record["sha256"]: record for record in representatives}
    if len(by_hash) != len(proposal["assignments"]):
        raise ValueError("Development proposal does not cover content groups exactly")
    for group in proposal["assignments"]:
        record = by_hash[group["sha256"]]
        if record["relative_path"] != group["representative"] or record["class_label"] != group["label"] or record["content_group_id"] != group["content_group_id"]:
            raise ValueError("Development representative/label differs from frozen proposal")
    return manifest, split


def quantitative_records(manifest, allow_synthetic=False):
    groups = defaultdict(list)
    for record in approved_records(manifest, allow_synthetic):
        groups[record["sha256"]].append(record)
    return sorted((min(members, key=lambda record: (record["relative_path"], record["image_id"])) for members in groups.values()), key=lambda record: record["image_id"])


class Preprocessor:
    def __init__(self, configuration):
        complete = deepcopy(DEFAULT_CONFIG)
        complete["preprocessing"] = configuration
        self.configuration = validate_config(complete)["preprocessing"]

    def __call__(self, image, role, seed=0):
        if role not in ROLES:
            raise ValueError("Unknown split role")
        if image.width <= 0 or image.height <= 0 or getattr(image, "n_frames", 1) != 1:
            raise ValueError("Malformed or multiframe image")
        if image.mode in {"F", "I", "I;16", "I;16B", "I;16L"}:
            raise ValueError("Scientific intensity arrays need a separately specified display mapping")
        image = ImageOps.exif_transpose(image)
        if image.mode == "RGBA" or "transparency" in image.info or image.mode == "LA":
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            image = image.convert("RGB")
        side = max(image.size)
        canvas = Image.new("RGB", (side, side), (0, 0, 0))
        canvas.paste(image, ((side - image.width) // 2, (side - image.height) // 2))
        if role == "train":
            generator = random.Random(seed)
            angle = generator.uniform(-self.configuration["rotation_degrees"], self.configuration["rotation_degrees"])
            canvas = canvas.rotate(angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=(0, 0, 0))
            if generator.random() < self.configuration["horizontal_flip_probability"]:
                canvas = ImageOps.mirror(canvas)
        size = self.configuration["size"]
        canvas = canvas.resize((size, size), resample=Image.Resampling.LANCZOS)
        values = np.array(canvas, dtype=np.float32, copy=True) / 255.0
        if not np.isfinite(values).all():
            raise ValueError("Invalid pixels")
        tensor = torch.from_numpy(values).permute(2, 0, 1)
        mean = torch.tensor(self.configuration["mean"], dtype=torch.float32)[:, None, None]
        deviation = torch.tensor(self.configuration["std"], dtype=torch.float32)[:, None, None]
        return (tensor - mean) / deviation


class ManifestDataset(Dataset):
    def __init__(self, root, manifest, split, role, configuration, seed=3908274, allow_synthetic=False):
        if role not in ROLES:
            raise ValueError("Unknown split role")
        validate_splits(manifest, split, allow_synthetic)
        approved = approved_records(manifest, allow_synthetic)
        self.class_map = {label: index for index, label in enumerate(sorted({record["class_label"] for record in approved}))}
        self.manifest_hash = digest(manifest)
        self.split_hash = digest(split)
        self.split_policy = split.get("policy", "development_only")
        self.object_independence_certified = split.get("object_independence_certified", False)
        self.split = role
        self.seed = seed
        self.epoch = 0
        self.root = Path(root).resolve()
        selected = [record for record in quantitative_records(manifest, allow_synthetic) if split["assignments"][record["image_id"]] == role]
        self.records = sorted(selected, key=lambda record: record["image_id"])
        self.preprocessing = Preprocessor(configuration)
        self._test_access = False
        self._ground_truth = self._label_contract()
        self.ground_truth_hash = digest(self._ground_truth)

    def _label_contract(self):
        rows = [{"image_id": record["image_id"], "source_path": record["relative_path"], "content_group_id": record["content_group_id"], "provisional_object_id": record["provisional_object_id"], "true_label": self.class_map[record["class_label"]]} for record in self.records]
        return {"manifest_hash": self.manifest_hash, "split_id": self.split_hash, "split": self.split, "class_map": self.class_map, "records": rows}

    def evaluation_contract(self):
        if self.split == "test" and not self._test_access:
            raise ValueError("Test labels are inaccessible before a locked final evaluation")
        if digest(self._ground_truth) != self.ground_truth_hash or self._label_contract() != self._ground_truth:
            raise ValueError("Frozen ground truth changed")
        return deepcopy(self._ground_truth)

    @contextmanager
    def final_test_access(self, selection_lock):
        from .training import validate_selection_lock
        context = {"run_id": selection_lock.get("run_id"), "configuration_digest": selection_lock.get("configuration_digest"), "split_id": self.split_hash}
        validate_selection_lock(selection_lock, context)
        if self.split != "test":
            raise ValueError("Final test access requires a test dataset")
        previous = self._test_access
        self._test_access = True
        try:
            yield
        finally:
            self._test_access = previous

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        self.evaluation_contract()
        record = self.records[index]
        path = (self.root / record["relative_path"]).resolve()
        if not path.is_relative_to(self.root) or file_hash(path) != record["sha256"]:
            raise ValueError("Source path escaped root or source bytes changed")
        augmentation_seed = int(digest([self.seed, self.epoch, record["image_id"]])[:16], 16)
        if path.suffix.lower() == ".avif":
            try:
                importlib.import_module("pillow_avif")
            except ImportError as error:
                raise RuntimeError("AVIF decoder unavailable in this interpreter; use a locally installed decoder, do not drop records") from error
        with Image.open(path) as image:
            image.load()
            tensor = self.preprocessing(image, self.split, augmentation_seed)
        return {"image": tensor, "label": self.class_map[record["class_label"]], "image_id": record["image_id"], "source_path": record["relative_path"], "content_group_id": record["content_group_id"], "provisional_object_id": record["provisional_object_id"] or "", "object_group_id": record["object_group_id"], "provenance_status": record["provenance_status"], "licence_status": record["licence_status"], "provenance_evidence": record.get("provenance_evidence") or ""}
