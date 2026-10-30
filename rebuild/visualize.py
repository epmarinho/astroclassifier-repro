"""Generate human-readable figures and score tables from completed runs."""

import argparse
import csv
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
DEFAULT_RUNS = ROOT / "runs/phase3d_v4"
DEFAULT_OUTPUT = ROOT / "figures/phase3d_v4"

os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib"))
import matplotlib  # noqa: E402

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


DISPLAY_NAMES = {
    "historical_singleton": "A — historical singleton",
    "spatial_stem_only": "B — spatial stem only",
    "spatial_transformer": "C — spatial Transformer",
}
COLORS = {
    "historical_singleton": "#3366cc",
    "spatial_stem_only": "#dc3912",
    "spatial_transformer": "#109618",
}


def read_json(path):
    with Path(path).open(encoding="utf-8") as stream:
        return json.load(stream)


def load_runs(runs_root):
    completed = []
    for result_path in sorted(Path(runs_root).glob("*/result.json")):
        result = read_json(result_path)
        provenance = result.get("provenance", {})
        model = provenance.get("configuration", {}).get("model", {})
        variant = model.get("variant")
        if variant not in DISPLAY_NAMES:
            continue
        history_path = result_path.parent / "history.json"
        predictions_path = result_path.parent / "validation_predictions.json"
        errors_path = result_path.parent / "validation_errors.json"
        if not all(path.is_file() for path in (history_path, predictions_path, errors_path)):
            raise FileNotFoundError(f"Incomplete run directory: {result_path.parent}")
        completed.append({
            "variant": variant,
            "name": DISPLAY_NAMES[variant],
            "result": result,
            "history": read_json(history_path),
            "predictions": read_json(predictions_path),
            "errors": read_json(errors_path),
            "directory": result_path.parent,
        })
    by_variant = {run["variant"]: run for run in completed}
    missing = set(DISPLAY_NAMES) - set(by_variant)
    if missing:
        raise ValueError(f"Missing completed variants: {sorted(missing)}")
    if len(completed) != len(DISPLAY_NAMES):
        raise ValueError("Expected exactly one completed run per Phase 3D variant")
    return [by_variant[variant] for variant in DISPLAY_NAMES]


def prepare_output(output, force):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    names = [
        "learning_curves.png", "learning_curves.svg",
        "confusion_matrices.png", "confusion_matrices.svg",
        "scores.csv", "per_class_scores.csv", "SCORES.md",
    ]
    existing = [output / name for name in names if (output / name).exists()]
    if existing and not force:
        joined = ", ".join(path.name for path in existing)
        raise FileExistsError(f"Outputs already exist ({joined}); pass --force to regenerate")
    return output


def plot_learning_curves(runs, output, label="Phase 3D development pilot"):
    figure, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    panels = [
        ("accuracy", "Validation accuracy", (0.0, 1.0)),
        ("balanced_accuracy", "Validation balanced accuracy", (0.0, 1.0)),
        ("macro_f1", "Validation macro-F1", (0.0, 1.0)),
    ]
    loss_axis = axes[0, 0]
    for run in runs:
        epochs = [entry["epoch"] + 1 for entry in run["history"]]
        color = COLORS[run["variant"]]
        loss_axis.plot(epochs, [entry["train_loss"] for entry in run["history"]],
                       color=color, linewidth=2, label=f"{run['name']} — train")
        loss_axis.plot(epochs, [entry["validation"]["loss"] for entry in run["history"]],
                       color=color, linewidth=2, linestyle="--", label=f"{run['name']} — validation")
        loss_axis.axvline(run["result"]["best_epoch"] + 1, color=color, alpha=0.18, linewidth=1)
    loss_axis.set_title("Training and validation loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Weighted cross-entropy")
    loss_axis.grid(alpha=0.25)
    loss_axis.legend(fontsize=8, ncol=2)

    for axis, (metric, title, limits) in zip((axes[0, 1], axes[1, 0], axes[1, 1]), panels):
        for run in runs:
            epochs = [entry["epoch"] + 1 for entry in run["history"]]
            values = [entry["validation"][metric] for entry in run["history"]]
            axis.plot(epochs, values, marker="o", markersize=3, linewidth=2,
                      color=COLORS[run["variant"]], label=run["name"])
            best_epoch = run["result"]["best_epoch"] + 1
            axis.scatter([best_epoch], [values[best_epoch - 1]], s=70,
                         facecolors="none", edgecolors=COLORS[run["variant"]], linewidths=2)
        axis.axhline(0.25, color="#777777", linestyle=":", linewidth=1, label="chance reference")
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylim(*limits)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)

    figure.suptitle(f"{label} learning curves\nOpen circles mark the checkpoint selected by validation loss", fontsize=14)
    figure.savefig(output / "learning_curves.png", dpi=180)
    figure.savefig(output / "learning_curves.svg")
    plt.close(figure)


def plot_confusion_matrices(runs, output, label="Phase 3D development pilot"):
    class_map = runs[0]["result"]["provenance"]["class_map"]
    classes = [name for name, _ in sorted(class_map.items(), key=lambda item: item[1])]
    short = ["galaxies", "globular", "nebulae", "open"]
    figure, axes = plt.subplots(1, 3, figsize=(14, 4.6), constrained_layout=True)
    maximum = max(
        value
        for run in runs
        for row in run["result"]["validation"]["confusion_matrix"]
        for value in row
    )
    image = None
    for axis, run in zip(axes, runs):
        matrix = run["result"]["validation"]["confusion_matrix"]
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=maximum)
        for row, values in enumerate(matrix):
            for column, value in enumerate(values):
                axis.text(column, row, str(value), ha="center", va="center",
                          color="white" if value >= 4 else "black", fontsize=11)
        axis.set_title(run["name"])
        axis.set_xticks(range(len(classes)), short, rotation=35, ha="right")
        axis.set_yticks(range(len(classes)), short)
        axis.set_xlabel("Predicted class")
        axis.set_ylabel("True class")
    if image is not None:
        figure.colorbar(image, ax=axes, shrink=0.75, label="Images")
    sample_count = len(runs[0]["predictions"])
    figure.suptitle(f"{label} confusion matrices ({sample_count} images)", fontsize=14)
    figure.savefig(output / "confusion_matrices.png", dpi=180)
    figure.savefig(output / "confusion_matrices.svg")
    plt.close(figure)


def write_scores(runs, output, heading="Phase 3D development-pilot scores", note=None):
    fields = [
        "variant", "display_name", "epochs_completed", "best_epoch",
        "validation_images", "errors", "accuracy", "balanced_accuracy",
        "macro_f1", "weighted_f1", "validation_loss",
        "negative_log_likelihood", "expected_calibration_error", "brier_score",
    ]
    rows = []
    per_class_rows = []
    for run in runs:
        result = run["result"]
        metrics = result["validation"]
        row = {
            "variant": run["variant"],
            "display_name": run["name"],
            "epochs_completed": result["epochs_completed"],
            "best_epoch": result["best_epoch"] + 1,
            "validation_images": len(run["predictions"]),
            "errors": len(run["errors"]),
            "accuracy": metrics["accuracy"],
            "balanced_accuracy": metrics["balanced_accuracy"],
            "macro_f1": metrics["macro_f1"],
            "weighted_f1": metrics["weighted_f1"],
            "validation_loss": metrics["loss"],
            "negative_log_likelihood": metrics["negative_log_likelihood"],
            "expected_calibration_error": metrics["expected_calibration_error"],
            "brier_score": metrics["brier_score"],
        }
        rows.append(row)
        for class_name, values in metrics["per_class"].items():
            per_class_rows.append({"variant": run["variant"], "class": class_name, **values})

    with (output / "scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with (output / "per_class_scores.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["variant", "class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(per_class_rows)

    note = note or "These scores describe one small train/validation pilot. They are not a model ranking or final test result."
    lines = [
        f"# {heading}",
        "",
        note,
        "",
        "| Variant | Epochs | Best epoch | Accuracy | Balanced accuracy | Macro-F1 | NLL | ECE | Errors |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['display_name']} | {row['epochs_completed']} | {row['best_epoch']} | "
            f"{row['accuracy']:.3f} | {row['balanced_accuracy']:.3f} | {row['macro_f1']:.3f} | "
            f"{row['negative_log_likelihood']:.3f} | {row['expected_calibration_error']:.3f} | "
            f"{row['errors']}/{row['validation_images']} |"
        )
    lines.extend([
        "",
        "Generated from the immutable `result.json`, `history.json`, prediction and error records under `rebuild/runs/phase3d_v4/`.",
        "",
    ])
    (output / "SCORES.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Plot completed Phase 3D learning curves and scores without training models.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="directory containing completed run directories")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="new directory for figures and score tables")
    parser.add_argument("--force", action="store_true", help="replace previously generated visualization files")
    arguments = parser.parse_args()
    runs = load_runs(arguments.runs)
    output = prepare_output(arguments.output, arguments.force)
    plot_learning_curves(runs, output)
    plot_confusion_matrices(runs, output)
    write_scores(runs, output)
    print(f"Generated learning curves, confusion matrices and score tables in {output.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
