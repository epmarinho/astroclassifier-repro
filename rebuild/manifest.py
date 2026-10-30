import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import itertools
from pathlib import Path
import warnings

from PIL import Image, ImageOps

from .common import PROJECT, ROOT, digest, file_hash, output_path, write_json


AUTHOR_CLAIMS = {
    "potw2030a": {"description": "ESO photograph of ALMA observatory under the Milky Way", "url": "https://eso.org/public/australia/images/potw2030a/"},
    "potw2151a": {"description": "ESO seasonal-greetings artwork", "url": "https://eso.org/public/images/potw2151a/"},
}


def inspect_image(path):
    details = {"width": None, "height": None, "image_mode": None, "detected_format": None, "frames": None, "metadata": {}, "read_error": None, "dhash": None}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                details.update(width=image.width, height=image.height, image_mode=image.mode, detected_format=image.format, frames=getattr(image, "n_frames", 1))
                exif = image.getexif()
                for tag, name in [(270, "description"), (274, "orientation"), (305, "software"), (33432, "copyright")]:
                    value = exif.get(tag)
                    if isinstance(value, (str, int, float)):
                        details["metadata"][name] = value[:2000] if isinstance(value, str) else value
                profile = image.info.get("icc_profile")
                if profile:
                    details["metadata"]["icc_sha256"] = hashlib.sha256(profile).hexdigest()
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                image.load()
                small = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
                pixels = list(small.getdata())
                bits = [pixels[row * 9 + column] > pixels[row * 9 + column + 1] for row in range(8) for column in range(8)]
                details["dhash"] = f"{sum(int(bit) << index for index, bit in enumerate(bits)):016x}"
    except Exception as error:
        details["read_error"] = f"{type(error).__name__}: {error}"
    return details


def build_manifest(root):
    root = Path(root)
    records = []
    for path in sorted((root / "images").rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        parts = Path(relative).parts
        checksum = file_hash(path)
        label = parts[2] if len(parts) > 3 else None
        inspected = inspect_image(path)
        record = {
            "image_id": "img_" + hashlib.sha256(relative.encode()).hexdigest(),
            "relative_path": relative,
            "original_declared_split": parts[1],
            "original_directory_label": label,
            "class_label": label,
            "extension": path.suffix.lower(),
            "byte_size": path.stat().st_size,
            "sha256": checksum,
            "exact_duplicate_group_id": "sha256_" + checksum,
            "provisional_object_group_id": None,
            "object_group_id": None,
            "object_identity_status": "unresolved",
            "class_evidence": None,
            "provenance_evidence": None,
            "licence_evidence": None,
            "object_evidence": None,
            "provenance_status": "unverified",
            "licence_status": "unverified",
            "content_status": "unknown",
            "inclusion_status": "quarantined",
            "exclusion_reason": "insufficient_class_and_object_evidence",
            "adjudication_status": "pending",
            **inspected,
        }
        if inspected["read_error"]:
            record["exclusion_reason"] = "unreadable_or_decoder_unavailable"
        elif inspected["frames"] != 1:
            record["exclusion_reason"] = "multiframe_image_requires_review"
        if path.stem.lower() in AUTHOR_CLAIMS:
            record["author_source_claim"] = AUTHOR_CLAIMS[path.stem.lower()]
            record["provenance_status"] = "author_claim_visual_mismatch_requires_review"
            record["exclusion_reason"] = "provenance_identifier_mismatch_requires_review"
        records.append(record)
    groups = duplicate_groups(records)
    for members in groups.values():
        labels = {record["original_directory_label"] for record in members if record["original_directory_label"]}
        if len(labels) > 1:
            for record in members:
                record["inclusion_status"] = "excluded"
                record["exclusion_reason"] = "conflicting_labels_pending_adjudication"
    return {"schema_version": 1, "records": records}


def duplicate_groups(records):
    groups = defaultdict(list)
    for record in records:
        groups[record["exact_duplicate_group_id"]].append(record)
    return dict(sorted(groups.items()))


def near_candidates(records, threshold=4):
    readable = [record for record in records if record["dhash"] and not record["read_error"]]
    pairs = []
    for left, right in itertools.combinations(readable, 2):
        if left["sha256"] == right["sha256"]:
            continue
        distance = (int(left["dhash"], 16) ^ int(right["dhash"], 16)).bit_count()
        if distance <= threshold:
            pairs.append({"left_id": left["image_id"], "right_id": right["image_id"], "left_path": left["relative_path"], "right_path": right["relative_path"], "dhash_distance": distance, "status": "review_only_no_automatic_merge"})
    return pairs


def write_csv(path, rows, fields):
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in fields} for row in rows)


def create_reports(manifest, destination):
    destination = output_path(destination)
    records = manifest["records"]
    duplicates = {key: members for key, members in duplicate_groups(records).items() if len(members) > 1}
    reports = []
    for key, members in duplicates.items():
        reports.append({"group_id": key, "paths": [record["relative_path"] for record in members], "labels": sorted({record["original_directory_label"] for record in members if record["original_directory_label"]}), "original_splits": sorted({record["original_declared_split"] for record in members})})
    conflicts = [row for row in reports if len(row["labels"]) > 1]
    cross_split = [row for row in reports if len(row["original_splits"]) > 1]
    write_json(destination / "exact_duplicates.json", reports)
    write_json(destination / "cross_split_duplicates.json", cross_split)
    write_json(destination / "conflicting_labels.json", conflicts)
    candidates = near_candidates(records)
    write_csv(destination / "near_duplicate_review.csv", candidates, ["left_id", "right_id", "left_path", "right_path", "dhash_distance", "status"])
    fields = ["image_id", "relative_path", "sha256", "original_directory_label", "class_label", "inclusion_status", "exclusion_reason", "object_group_id", "object_identity_status", "class_evidence", "object_evidence", "provenance_status", "provenance_evidence", "licence_status", "licence_evidence", "content_status", "adjudication_status"]
    write_csv(destination / "manual_adjudication.csv", records, fields)
    write_csv(destination / "source_licence_review.csv", records, ["image_id", "relative_path", "provenance_status", "provenance_evidence", "licence_status", "licence_evidence", "content_status", "object_identity_status"])
    summary = {
        "source_images": len(records), "exact_duplicate_groups": len(duplicates),
        "duplicate_member_images": sum(len(members) for members in duplicates.values()),
        "redundant_exact_copies": sum(len(members) - 1 for members in duplicates.values()),
        "cross_split_groups": len(cross_split), "conflicting_label_groups": len(conflicts),
        "near_duplicate_review_pairs": len(candidates),
        "inclusion_counts": dict(Counter(record["inclusion_status"] for record in records)),
        "reason_counts": dict(Counter(record["exclusion_reason"] for record in records)),
        "read_failures": sum(bool(record["read_error"]) for record in records),
        "manifest_content_hash": digest(manifest),
        "defensible_split_available": False,
        "dataset_decision": "No benchmark split authorized. Curated data with verified labels and object groups is mandatory; obtain a new dataset if review cannot supply it.",
    }
    write_json(destination / "dataset_summary.json", summary)
    return summary


def main():
    argparse.ArgumentParser(description="Read source images and create an immutable inventory under rebuild/.").parse_args()
    target = ROOT / "manifests/source_images.json"
    if target.exists():
        raise SystemExit("Source manifest already exists; choose a separately versioned inventory instead of overwriting it.")
    manifest = build_manifest(PROJECT)
    write_json(target, manifest)
    summary = create_reports(manifest, ROOT / "reports")
    write_json(ROOT / "reports/scientific_input_hashes.json", {record["relative_path"]: record["sha256"] for record in manifest["records"]})
    print(summary)


if __name__ == "__main__":
    main()
