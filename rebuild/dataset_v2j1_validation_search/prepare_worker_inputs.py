"""Prepare a train/validation-only tensor allowlist from frozen v2G/v2H records."""
import hashlib
import json
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "rebuild/dataset_v2j1_validation_search"
V2G = ROOT / "rebuild/dataset_v2g_split_materialization"
V2H = ROOT / "rebuild/dataset_v2h_representation_materialization"
CLASS_ORDER = ["galaxies", "globularclusters", "nebulae", "opencluster"]


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_role(role):
    if role not in {"train", "validation"}:
        raise ValueError("Only training and validation role files are accepted")
    path = V2G / "splits" / f"{role}.json"
    with path.open(encoding="utf-8") as stream:
        data = json.load(stream)
    if data.get("role") != role:
        raise ValueError("Frozen split role mismatch")
    objects = data.get("objects")
    if not isinstance(objects, list) or not objects:
        raise ValueError("Empty or malformed role object list")
    by_id = {}
    for item in objects:
        oid = item["physical_object_id"]
        if oid in by_id or item.get("split") != role:
            raise ValueError("Duplicate object or role mismatch")
        if item.get("frozen_class") not in CLASS_ORDER:
            raise ValueError("Frozen class outside class order")
        by_id[oid] = item
    return path, data, by_id


train_path, train_data, train_objects = load_role("train")
val_path, val_data, val_objects = load_role("validation")
if set(train_objects) & set(val_objects):
    raise ValueError("Train and validation object overlap")

manifest_path = V2H / "representation_A_manifest.json"
if sha_file(manifest_path) != "85e1fea03b573463235ec292dbe65ad618437d32ef0bb07672cc173bfe6cc246":
    raise ValueError("Frozen A manifest hash mismatch")
with manifest_path.open(encoding="utf-8") as stream:
    a_manifest = json.load(stream)
records_by_role = {"train": {}, "validation": {}}
for rec in a_manifest.get("records", []):
    role = rec.get("split")
    if role not in records_by_role:
        continue
    oid = rec.get("physical_object_id")
    if oid in records_by_role[role]:
        raise ValueError("Duplicate A tensor for object/role")
    records_by_role[role][oid] = rec

result = {"phase": "Dataset-v2J1", "class_order": CLASS_ORDER, "representation": "A",
          "split_identity": "1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf",
          "roles": {}}
for role, objects, split_path, split_data in [
    ("train", train_objects, train_path, train_data),
    ("validation", val_objects, val_path, val_data),
]:
    if set(records_by_role[role]) != set(objects):
        raise ValueError(f"A tensor/object membership mismatch in {role}")
    rows = []
    for oid in sorted(objects):
        source = records_by_role[role][oid]
        obj = objects[oid]
        if source.get("representation") != "A" or source.get("split") != role:
            raise ValueError("Representation or role mismatch")
        if source.get("shape") != [3, 224, 224] or source.get("dtype") != "float32":
            raise ValueError("Frozen tensor contract mismatch")
        if source.get("channel_order") != ["blue", "red", "NIR"]:
            raise ValueError("Frozen channel order mismatch")
        if source.get("physical_object_id") != oid or source.get("split_unit_id") != obj.get("split_unit_id"):
            raise ValueError("Object/assignment-unit identity mismatch")
        path = Path(source["path"])
        if path.is_absolute() or ".." in path.parts or "test" in path.parts:
            raise ValueError("Unsafe or non-development tensor path")
        full = ROOT / path
        if not full.is_file() or full.stat().st_size != source.get("size_bytes"):
            raise ValueError("Missing or size-mismatched development tensor")
        if sha_file(full) != source.get("sha256"):
            raise ValueError("Development tensor hash mismatch")
        rows.append({"path": path.as_posix(), "sha256": source["sha256"],
                     "size_bytes": source["size_bytes"], "shape": source["shape"],
                     "dtype": source["dtype"], "physical_object_id": oid,
                     "split_unit_id": obj["split_unit_id"], "split": role,
                     "label": CLASS_ORDER.index(obj["frozen_class"]),
                     "class_name": obj["frozen_class"]})
    if role == "train" and len(rows) != 222:
        raise ValueError("Training object count differs from frozen split")
    if role == "validation" and len(rows) != 75:
        raise ValueError("Validation object count differs from frozen split")
    result["roles"][role] = {"split_manifest_path": split_path.relative_to(ROOT).as_posix(),
                              "split_manifest_sha256": sha_file(split_path),
                              "objects": rows}
if len(result["roles"]["train"]["objects"]) + len(result["roles"]["validation"]["objects"]) != 297:
    raise ValueError("Unexpected development membership size")
serialized = json.dumps(result, sort_keys=True, ensure_ascii=False, allow_nan=False, indent=2) + "\n"
(OUT / "worker_inputs.json").write_text(serialized, encoding="utf-8")
receipt = {"phase": "Dataset-v2J1", "worker_input_file": "worker_inputs.json",
           "worker_input_sha256": sha_file(OUT / "worker_inputs.json"),
           "role_counts": {r: len(result["roles"][r]["objects"]) for r in ["train", "validation"]},
           "tensor_count": sum(len(result["roles"][r]["objects"]) for r in ["train", "validation"]),
           "labels_used": ["train", "validation"], "roles_present": ["train", "validation"],
           "non_development_tensor_paths": 0, "test_tensor_or_label_data_opened": False}
(OUT / "input_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(receipt, sort_keys=True))
