import math

import numpy as np

from .common import digest


METRIC_VERSION = "phase3a-v1"


def validate_predictions(predictions, class_map, unique=True):
    if not class_map or sorted(class_map.values()) != list(range(len(class_map))):
        raise ValueError("Class map must use contiguous indices")
    if not predictions:
        raise ValueError("Empty prediction table")
    identifiers = set()
    groups = set()
    for record in predictions:
        for field in ["image_id", "source_path", "content_group_id", "split_id", "run_id", "configuration_digest"]:
            if not record.get(field):
                raise ValueError(f"Prediction is missing {field}")
        if unique and (record["image_id"] in identifiers or record["content_group_id"] in groups):
            raise ValueError("Duplicate image/content group would receive extra evaluation weight")
        identifiers.add(record["image_id"])
        groups.add(record["content_group_id"])
        for field in ["true_label", "predicted_label"]:
            if type(record[field]) is not int or not 0 <= record[field] < len(class_map):
                raise ValueError("Invalid class index")
        probabilities = record.get("probabilities")
        if probabilities is None or len(probabilities) != len(class_map) or not all(math.isfinite(value) and 0 <= value <= 1 for value in probabilities) or not math.isclose(sum(probabilities), 1.0, abs_tol=1e-6):
            raise ValueError("Expected finite normalized class probabilities")
        if record["predicted_label"] != int(np.argmax(probabilities)):
            raise ValueError("Predicted class does not match probabilities")
    for field in ["split_id", "run_id", "configuration_digest"]:
        if len({record[field] for record in predictions}) != 1:
            raise ValueError("Mixed prediction provenance")


def _metrics(predictions, class_map, bins):
    count = len(class_map)
    truth = np.asarray([record["true_label"] for record in predictions])
    guessed = np.asarray([record["predicted_label"] for record in predictions])
    probabilities = np.asarray([record["probabilities"] for record in predictions], dtype=np.float64)
    matrix = np.zeros((count, count), dtype=np.int64)
    np.add.at(matrix, (truth, guessed), 1)
    support = matrix.sum(axis=1)
    predicted = matrix.sum(axis=0)
    positive = matrix.diagonal()
    precision = np.divide(positive, predicted, out=np.zeros(count), where=predicted != 0)
    recall = np.divide(positive, support, out=np.zeros(count), where=support != 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros(count), where=precision + recall != 0)
    per_class = {name: {"support": int(support[index]), "precision": float(precision[index]), "recall": float(recall[index]), "f1": float(f1[index])} for name, index in sorted(class_map.items(), key=lambda item: item[1])}
    confidence = probabilities.max(axis=1)
    correct = truth == guessed
    bin_indices = np.minimum((confidence * bins).astype(int), bins - 1)
    calibration_bins = []
    ece = 0.0
    for index in range(bins):
        members = bin_indices == index
        size = int(members.sum())
        mean_confidence = float(confidence[members].mean()) if size else None
        accuracy = float(correct[members].mean()) if size else None
        if size:
            ece += size / len(predictions) * abs(accuracy - mean_confidence)
        calibration_bins.append({"lower": index / bins, "upper": (index + 1) / bins, "count": size, "accuracy": accuracy, "mean_confidence": mean_confidence})
    nll = -np.log(np.maximum(probabilities[np.arange(len(truth)), truth], 1e-15)).mean()
    targets = np.eye(count)[truth]
    return {"sample_count": len(predictions), "accuracy": float(correct.mean()), "balanced_accuracy": float(recall[support > 0].mean()), "macro_f1": float(f1.mean()), "weighted_f1": float(np.dot(f1, support) / len(predictions)), "per_class": per_class, "confusion_matrix": matrix.tolist(), "negative_log_likelihood": float(nll), "expected_calibration_error": float(ece), "brier_score": float(np.square(probabilities - targets).sum(axis=1).mean()), "calibration_bins": calibration_bins}


def classification_metrics(predictions, class_map, bins=10):
    if type(bins) is not int or bins < 1:
        raise ValueError("Calibration bins must be a positive integer")
    validate_predictions(predictions, class_map)
    return _metrics(predictions, class_map, bins)


def group_bootstrap(predictions, class_map, samples=1000, seed=3908274, confidence=0.95, bins=10):
    validate_predictions(predictions, class_map)
    if type(samples) is not int or samples < 1 or not 0 < confidence < 1:
        raise ValueError("Invalid bootstrap configuration")
    groups = {}
    for record in predictions:
        groups.setdefault(record["content_group_id"], []).append(record)
    groups = [groups[key] for key in sorted(groups)]
    generator = np.random.default_rng(seed)
    keys = ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1", "negative_log_likelihood", "expected_calibration_error", "brier_score"]
    distributions = {key: [] for key in keys}
    for _ in range(samples):
        records = [record for index in generator.integers(0, len(groups), len(groups)) for record in groups[index]]
        scores = _metrics(records, class_map, bins)
        for key in keys:
            distributions[key].append(scores[key])
    tail = (1 - confidence) / 2
    intervals = {key: {"lower": float(np.quantile(values, tail)), "upper": float(np.quantile(values, 1 - tail))} for key, values in distributions.items()}
    return {"method": "percentile content-group cluster bootstrap, equal groups sampled with replacement", "group_count": len(groups), "samples": samples, "seed": seed, "confidence": confidence, "object_independence_certified": False, "intervals": intervals}


def metric_bundle(predictions, class_map, settings, bootstrap=True):
    result = classification_metrics(predictions, class_map, settings["calibration_bins"])
    result["metric_version"] = METRIC_VERSION
    if bootstrap:
        result["bootstrap"] = group_bootstrap(predictions, class_map, settings["bootstrap_samples"], settings["bootstrap_seed"], settings["confidence"], settings["calibration_bins"])
    return result


def verify_cached_evaluation(evaluation, predictions, class_map, settings):
    expected = metric_bundle(predictions, class_map, settings)
    for key, value in expected.items():
        if evaluation.get(key) != value:
            raise ValueError(f"Corrupted cached metrics: {key}")
    if evaluation.get("prediction_digest") != digest(predictions):
        raise ValueError("Corrupted cached prediction digest")
    losses = [record.get("weighted_loss") for record in predictions]
    if any(value is None or not math.isfinite(value) or value < 0 for value in losses) or not math.isclose(evaluation.get("loss", math.nan), sum(losses) / len(losses), abs_tol=1e-12, rel_tol=1e-12):
        raise ValueError("Corrupted cached metrics: loss")
