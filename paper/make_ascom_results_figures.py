#!/usr/bin/env python3
"""Build the Astronomy & Computing figures from frozen, hash-checked artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


EXPECTED_SUMMARY_SHA256 = (
    "5a5d4e33a41319a9676a06bb14e00cfc0ac0e93d1888578fba9c2b503c652e89"
)
EXPECTED_MEMBERSHIP_SHA256 = (
    "c01da3d7e93ab224d6d936d6d6c6e427c289bae78f422774477ca3b48ea3d16c"
)
EXPECTED_TENSOR_INDEX_SHA256 = (
    "df06b84e695b9ecdd4f9536571677b1fa8795f20111638f1ef9b0249866f817f"
)
EXPECTED_CONFUSIONS_SHA256 = (
    "b679e26a2349b93a040937685cc71892146cd6dcf2d05ffb1e2d1dec8dadb985"
)
EXPECTED_PREDICTION_MANIFEST_SHA256 = (
    "72aca601cf2cadabfc8606f95fa7ddd69f30088f87c03cb1717cfadb0655d95a"
)
PAPER = Path(__file__).resolve().parent
ROOT = PAPER.parent
SUMMARY = ROOT / (
    "rebuild/dataset_v2k_test_executor_freeze_v3/scoring_attempt_01/"
    "aggregate_scoring_summary.json"
)
OUT = PAPER / "figures_ascom"
MEMBERSHIP = ROOT / "rebuild/dataset_v2g_split_materialization/object_membership.json"
TENSOR_INDEX = ROOT / (
    "rebuild/dataset_v2h_representation_materialization/tensor_hash_index.json"
)
INFERENCE_OUT = ROOT / (
    "rebuild/dataset_v2k_test_executor_freeze_v3/execution_attempt_01/"
    "inference_output"
)
PREDICTION_MANIFEST = INFERENCE_OUT / "prediction_manifest.json"
CONFUSIONS = ROOT / (
    "rebuild/dataset_v2k_test_executor_freeze_v3/scoring_attempt_01/"
    "scoring_output/confusion_matrices.json"
)

FAMILY_ORDER = ["custom_cnn_transformer", "resnet50", "vit_b_16"]
FAMILY_LABEL = {
    "custom_cnn_transformer": "Custom CNN–Transformer",
    "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16",
}
REP_ORDER = ["A", "B", "A_outer", "B_outer"]
REP_LABEL = ["A", "B", r"$A_{outer}$", r"$B_{outer}$"]
COLORS = ["#0072B2", "#D55E00", "#009E73"]
MARKERS = ["o", "s", "^"]
CLASS_ORDER = ["galaxies", "globularclusters", "nebulae", "opencluster"]
CLASS_LABEL = {
    "galaxies": "Galaxy",
    "globularclusters": "Globular cluster",
    "nebulae": "Nebula",
    "opencluster": "Open cluster",
}
REPRESENTATIVE_OBJECTS = {
    "galaxies": "object_28a5d4f9a1313cb4771f",
    "globularclusters": "object_2f44bb74936a476070b2",
    "nebulae": "object_0a20acda342c7794cad3",
    "opencluster": "object_75cc478d7864b8dfcb89",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_verified_json(path: Path, expected_sha256: str) -> dict:
    payload = path.read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(
            f"hash mismatch for {path}: expected {expected_sha256}, got {digest}"
        )
    return json.loads(payload)


def load_summary() -> dict:
    data = load_verified_json(SUMMARY, EXPECTED_SUMMARY_SHA256)
    if data.get("individual_truth_included") is not False:
        raise RuntimeError("aggregate summary unexpectedly contains individual truth")
    return data


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def balanced_accuracy_plot(data: dict) -> None:
    rows = {
        (row["family"], row["representation"]): row
        for row in data["family_representation_summary"]
    }
    x = np.arange(len(REP_ORDER), dtype=float)
    offsets = [-0.18, 0.0, 0.18]

    fig, ax = plt.subplots(figsize=(7.15, 4.15), constrained_layout=True)
    for idx, family in enumerate(FAMILY_ORDER):
        means = [float(rows[(family, rep)]["balanced_accuracy_mean"]) for rep in REP_ORDER]
        sds = [float(rows[(family, rep)]["balanced_accuracy_sample_sd"]) for rep in REP_ORDER]
        ax.errorbar(
            x + offsets[idx],
            means,
            yerr=sds,
            color=COLORS[idx],
            marker=MARKERS[idx],
            markersize=5.2,
            linestyle="none",
            elinewidth=1.25,
            capsize=3,
            label=FAMILY_LABEL[family],
            zorder=3,
        )

    metadata = {row["family"]: float(row["balanced_accuracy"]) for row in data["metadata_controls"]}
    tree = metadata["metadata_decision_tree_depth3"]
    logistic = metadata["metadata_logreg_C1_balanced_lbfgs"]
    ax.axhline(tree, color="#6F6F6F", linestyle="--", linewidth=1.0, zorder=1)
    ax.axhline(logistic, color="#A0A0A0", linestyle=(0, (2, 2)), linewidth=1.0, zorder=1)
    ax.text(3.42, tree + 0.005, "metadata tree", color="#555555", ha="right", va="bottom", fontsize=7.5)
    ax.text(3.42, logistic - 0.007, "metadata logistic", color="#777777", ha="right", va="top", fontsize=7.5)

    ax.axhline(0.25, color="#B0B0B0", linewidth=0.8, zorder=0)
    ax.text(-0.42, 0.258, "chance", color="#777777", fontsize=7.5, ha="left", va="bottom")
    ax.set_xticks(x, REP_LABEL)
    ax.set_xlim(-0.48, 3.48)
    ax.set_ylim(0.20, 0.96)
    ax.set_ylabel("Test balanced accuracy")
    ax.set_xlabel("Representation")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.legend(loc="lower left", frameon=False, ncol=3, bbox_to_anchor=(0.0, 1.01))

    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"test_balanced_accuracy.{suffix}", dpi=300)
    plt.close(fig)


def primary_difference_plot(data: dict) -> None:
    selected = [row for row in data["main_A_tests"] if row["metric"] == "balanced_accuracy"]
    key_order = [
        ("custom_cnn_transformer", "resnet50"),
        ("custom_cnn_transformer", "vit_b_16"),
        ("resnet50", "vit_b_16"),
    ]
    lookup = {(row["first_family"], row["second_family"]): row for row in selected}
    rows = [lookup[key] for key in key_order]
    labels = ["Custom − ResNet-50", "Custom − ViT-B/16", "ResNet-50 − ViT-B/16"]
    y = np.arange(len(rows))[::-1]
    values = np.array([float(row["difference"]) for row in rows])
    lower = np.array([float(row["interval_95"][0]) for row in rows])
    upper = np.array([float(row["interval_95"][1]) for row in rows])
    xerr = np.vstack((values - lower, upper - values))

    fig, ax = plt.subplots(figsize=(4.8, 3.15), constrained_layout=True)
    ax.errorbar(
        values,
        y,
        xerr=xerr,
        fmt="o",
        markersize=5,
        color="#0072B2",
        ecolor="#0072B2",
        capsize=3,
        linewidth=1.25,
        zorder=3,
    )
    ax.axvline(0.0, color="#555555", linewidth=0.9, linestyle="--")
    ax.set_yticks(y, labels)
    ax.set_xlim(-0.18, 0.085)
    ax.set_ylim(-0.55, 2.55)
    ax.set_xlabel("Difference in test balanced accuracy (first − second)")
    ax.grid(axis="x", color="#E5E5E5", linewidth=0.7)
    for yi, row in zip(y, rows):
        p = float(row["holm_adjusted_p"])
        ax.text(0.081, yi, f"Holm p={p:.3f}", ha="right", va="bottom", fontsize=7.5, color="#555555")

    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"primary_A_balanced_accuracy_differences.{suffix}", dpi=300)
    plt.close(fig)


def representation_examples_plot() -> None:
    membership = load_verified_json(MEMBERSHIP, EXPECTED_MEMBERSHIP_SHA256)
    tensor_index = load_verified_json(TENSOR_INDEX, EXPECTED_TENSOR_INDEX_SHA256)
    validation = [row for row in membership["records"] if row["split"] == "validation"]

    selected = {}
    for class_name in CLASS_ORDER:
        rows = [row for row in validation if row["frozen_class"] == class_name]
        median_fov = float(np.median([float(row["FOV_arcmin"]) for row in rows]))
        selected[class_name] = min(
            rows,
            key=lambda row: (
                abs(float(row["FOV_arcmin"]) - median_fov),
                row["physical_object_id"],
            ),
        )
        actual = selected[class_name]["physical_object_id"]
        if actual != REPRESENTATIVE_OBJECTS[class_name]:
            raise RuntimeError(
                f"representative selection changed for {class_name}: {actual}"
            )

    tensor_lookup = {
        (row["physical_object_id"], row["representation"]): row
        for row in tensor_index["records"]
    }
    fig, axes = plt.subplots(4, 4, figsize=(7.15, 7.0), constrained_layout=True)
    for row_index, class_name in enumerate(CLASS_ORDER):
        object_row = selected[class_name]
        object_id = object_row["physical_object_id"]
        for col_index, representation in enumerate(REP_ORDER):
            record = tensor_lookup[(object_id, representation)]
            tensor_path = ROOT / record["path"]
            if sha256(tensor_path) != record["sha256"]:
                raise RuntimeError(f"tensor hash mismatch: {tensor_path}")
            tensor = np.load(tensor_path, allow_pickle=False)
            if tuple(tensor.shape) != (3, 224, 224) or tensor.dtype != np.float32:
                raise RuntimeError(f"unexpected tensor contract: {tensor_path}")
            # Stored channels are blue/red/NIR. Display NIR/red/blue as RGB.
            display = np.clip((tensor[[2, 1, 0]] + 1.0) / 2.0, 0.0, 1.0)
            display = np.moveaxis(display, 0, -1)
            ax = axes[row_index, col_index]
            ax.imshow(display, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.5)
                spine.set_color("#666666")
            if row_index == 0:
                ax.set_title(REP_LABEL[col_index], pad=4)
            if col_index == 0:
                fov = float(object_row["FOV_arcmin"])
                ax.set_ylabel(
                    f"{CLASS_LABEL[class_name]}\n(source FOV {fov:g} arcmin)",
                    rotation=90,
                    labelpad=7,
                )

    fig.suptitle(
        "Input representations (NIR/red/blue pseudo-colour)",
        fontsize=10,
    )
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"representation_examples.{suffix}", dpi=300)
    plt.close(fig)


def confusion_matrices_A_plot(data: dict) -> None:
    manifest = load_verified_json(
        PREDICTION_MANIFEST, EXPECTED_PREDICTION_MANIFEST_SHA256
    )
    confusions = load_verified_json(CONFUSIONS, EXPECTED_CONFUSIONS_SHA256)
    target_hashes = {
        row["target_id"]: (row["path"], row["sha256"])
        for row in manifest["target_records"]
    }
    by_family = {family: [] for family in FAMILY_ORDER}
    for target_id, (relative_path, expected_hash) in target_hashes.items():
        target_path = INFERENCE_OUT / relative_path
        if sha256(target_path) != expected_hash:
            raise RuntimeError(f"prediction-record hash mismatch: {target_path}")
        target = json.loads(target_path.read_bytes())
        family = target.get("family")
        if target.get("representation") == "A" and family in by_family:
            matrix = np.asarray(confusions[target_id], dtype=float)
            if matrix.shape != (4, 4):
                raise RuntimeError(f"unexpected confusion matrix: {target_id}")
            by_family[family].append(matrix)
    if any(len(matrices) != 3 for matrices in by_family.values()):
        raise RuntimeError("expected three A-view confusion matrices per family")

    summary_lookup = {
        (row["family"], row["representation"]): row
        for row in data["family_representation_summary"]
    }
    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.75), constrained_layout=True)
    tick_labels = ["Galaxy", "Globular", "Nebula", "Open"]
    image_artist = None
    for ax, family in zip(axes, FAMILY_ORDER):
        mean_counts = np.mean(np.stack(by_family[family]), axis=0)
        normalized = mean_counts / mean_counts.sum(axis=1, keepdims=True)
        image_artist = ax.imshow(normalized, cmap="Blues", vmin=0.0, vmax=1.0)
        for i in range(4):
            for j in range(4):
                value = normalized[i, j]
                ax.text(
                    j,
                    i,
                    f"{100 * value:.0f}",
                    ha="center",
                    va="center",
                    fontsize=7.5,
                    color="white" if value >= 0.55 else "#222222",
                )
        ba = float(summary_lookup[(family, "A")]["balanced_accuracy_mean"])
        ax.set_title(f"{FAMILY_LABEL[family]}\nBA={ba:.3f}")
        ax.set_xticks(range(4), tick_labels, rotation=40, ha="right")
        ax.set_yticks(range(4), tick_labels)
        ax.set_xlabel("Predicted class")
        if ax is axes[0]:
            ax.set_ylabel("True class")
        else:
            ax.set_yticklabels([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    colorbar = fig.colorbar(image_artist, ax=axes, fraction=0.025, pad=0.02)
    colorbar.set_label("Row-normalized proportion")
    for suffix in ("pdf", "png"):
        fig.savefig(OUT / f"confusion_matrices_A.{suffix}", dpi=300)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    style()
    data = load_summary()
    balanced_accuracy_plot(data)
    primary_difference_plot(data)
    representation_examples_plot()
    confusion_matrices_A_plot(data)
    print("verified frozen artifact SHA-256 values:")
    for digest in (
        EXPECTED_SUMMARY_SHA256,
        EXPECTED_MEMBERSHIP_SHA256,
        EXPECTED_TENSOR_INDEX_SHA256,
        EXPECTED_CONFUSIONS_SHA256,
        EXPECTED_PREDICTION_MANIFEST_SHA256,
    ):
        print(f"  {digest}")
    print(f"wrote figures to {OUT}")


if __name__ == "__main__":
    main()
