from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.nn import functional as functional
from torch.utils.data import DataLoader

from .common import digest, file_hash, output_path, read_json, write_json
from .config import run_identity, validate_attention, validate_config
from .evaluation import metric_bundle, verify_cached_evaluation


@dataclass
class EarlyStopping:
    patience: int
    min_delta: float = 0.0
    best: float = math.inf
    bad_epochs: int = 0

    def update(self, loss):
        if not math.isfinite(loss):
            raise ValueError("Validation loss is not finite")
        improved = loss < self.best - self.min_delta
        if improved:
            self.best = loss
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return improved, self.bad_epochs >= self.patience


def atomic_write(path, value, binary=False):
    target = output_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=target.parent, prefix=".pending-", delete=False) as stream:
            temporary_path = Path(stream.name)
            if binary:
                torch.save(value, stream)
            else:
                stream.write((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def batch_losses(model, batch, weights):
    device = next(model.parameters(), torch.empty(0)).device
    logits = model(batch["image"].to(device, non_blocking=True))
    if logits.ndim != 2 or not torch.isfinite(logits).all():
        raise ValueError("Expected finite classification logits [batch, classes]")
    losses = functional.cross_entropy(logits, batch["label"].to(device, non_blocking=True), weight=weights.to(device) if weights is not None else None, reduction="none")
    if not torch.isfinite(losses).all():
        raise ValueError("Nonfinite loss")
    return logits, losses


def train_epoch(model, loader, optimizer, weights=None, max_grad_norm=3.0):
    if getattr(loader.dataset, "split", None) != "train":
        raise ValueError("Optimization requires the training split")
    model.train()
    loss_sum = 0.0
    sample_count = 0
    for batch in loader:
        optimizer.zero_grad(set_to_none=True)
        _, losses = batch_losses(model, batch, weights)
        losses.mean().backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True)
        optimizer.step()
        loss_sum += losses.detach().double().sum().item()
        sample_count += len(losses)
    if not sample_count:
        raise ValueError("Empty training loader")
    return loss_sum / sample_count


def evaluate(model, loader, class_map, weights=None, purpose="selection", *, context, settings, bootstrap=False, selection_lock=None):
    role = getattr(loader.dataset, "split", None)
    if purpose not in {"selection", "final"} or role not in {"validation", "test"}:
        raise ValueError("Evaluation requires validation/test and an explicit purpose")
    if purpose == "selection" and role == "test":
        raise ValueError("Test labels cannot be used for model selection")
    if role == "test":
        validate_selection_lock(selection_lock, context)
    if not context.get("run_id") or not context.get("configuration_digest") or not context.get("split_id"):
        raise ValueError("Evaluation requires run/configuration/split identities")
    contract = loader.dataset.evaluation_contract()
    if contract["class_map"] != class_map or contract["split_id"] != context["split_id"] or contract["split"] != role:
        raise ValueError("Frozen ground truth provenance mismatch")
    frozen = {record["image_id"]: record for record in contract["records"]}
    ground_truth_digest = digest(contract)
    if len(frozen) != len(contract["records"]):
        raise ValueError("Repeated ground truth identifiers")
    was_training = model.training
    model.eval()
    predictions = []
    loss_sum = 0.0
    try:
        with torch.no_grad():
            for batch in loader:
                logits, losses = batch_losses(model, batch, weights)
                loss_sum += losses.double().sum().item()
                guessed = logits.argmax(dim=1).tolist()
                probabilities = logits.softmax(dim=1).tolist()
                for index, predicted in enumerate(guessed):
                    identifier = batch["image_id"][index]
                    truth = frozen.get(identifier)
                    if truth is None or int(batch["label"][index]) != truth["true_label"] or batch["source_path"][index] != truth["source_path"] or batch["content_group_id"][index] != truth["content_group_id"]:
                        raise ValueError("Batch differs from frozen ground truth")
                    predictions.append({**truth, **context, "split": role, "object_group_id": batch["object_group_id"][index], "provenance_status": batch["provenance_status"][index], "licence_status": batch.get("licence_status", ["unknown"] * len(guessed))[index], "predicted_label": predicted, "probabilities": probabilities[index], "ground_truth_digest": ground_truth_digest})
                    predictions[-1]["weighted_loss"] = float(losses[index])
                    if "provenance_evidence" in batch:
                        predictions[-1]["provenance_evidence"] = batch["provenance_evidence"][index]
    finally:
        model.train(was_training)
    if not predictions:
        raise ValueError("Empty evaluation loader")
    if {record["image_id"] for record in predictions} != set(frozen):
        raise ValueError("Evaluation did not cover exactly the frozen ground truth")
    return {**context, "loss": loss_sum / len(predictions), "split": role, "purpose": purpose, "prediction_digest": digest(predictions), "ground_truth_digest": ground_truth_digest, **metric_bundle(predictions, class_map, settings, bootstrap), "predictions": predictions}


def read_checkpoint(path, expected_provenance):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint["provenance"] != expected_provenance:
        raise ValueError("Checkpoint provenance mismatch")
    required = {"schema_version", "epoch", "seed", "model_configuration", "class_map", "class_weights", "dataset_manifest_hash", "split_hash", "configuration_digest", "ground_truth_hashes", "model_state", "optimizer_state", "scheduler_state", "early_stopping", "torch_rng_state", "python_rng_state", "numpy_rng_state", "cuda_rng_states", "history", "runtime", "torch_version"}
    if not required.issubset(checkpoint):
        raise ValueError("Incomplete checkpoint")
    for name in ["seed", "model_configuration", "class_map", "dataset_manifest_hash", "split_hash", "configuration_digest", "ground_truth_hashes", "runtime"]:
        if checkpoint[name] != expected_provenance[name]:
            raise ValueError(f"Checkpoint provenance mismatch: {name}")
    return checkpoint


def configure_runtime(configuration):
    runtime = configuration["runtime"]
    if runtime["device"] == "cuda" and not torch.cuda.is_available():
        raise ValueError("Requested CUDA is unavailable; no automatic device substitution")
    torch.set_num_threads(runtime["num_threads"])
    torch.use_deterministic_algorithms(runtime["deterministic"])
    if runtime["device"] == "cuda":
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed_all(configuration["seed"])


def completed_run(run_directory, provenance):
    result = read_json(run_directory / "result.json")
    if result.get("status") != "completed" or result.get("provenance") != provenance:
        raise ValueError("Completed result provenance mismatch")
    if result.get("result_digest") != digest({key: value for key, value in result.items() if key != "result_digest"}):
        raise ValueError("Corrupted cached metrics/result digest")
    for name, expected in result["artifacts"].items():
        if Path(name).name != name or file_hash(run_directory / name) != expected:
            raise ValueError("Corrupted cached run artifact")
    predictions = read_json(run_directory / "validation_predictions.json")
    verify_cached_evaluation(result["validation"], predictions, provenance["class_map"], provenance["configuration"]["evaluation"])
    validate_saved_truth(predictions, read_json(run_directory / "validation_ground_truth.json"), provenance)
    return result


def validate_saved_truth(predictions, contract, provenance, expected_hash=None):
    expected_hash = expected_hash or provenance["ground_truth_hashes"].get(contract["split"])
    if digest(contract) != expected_hash or contract["manifest_hash"] != provenance["dataset_manifest_hash"] or contract["split_id"] != provenance["split_hash"] or contract["class_map"] != provenance["class_map"]:
        raise ValueError("Frozen ground truth provenance mismatch")
    truth = {record["image_id"]: record for record in contract["records"]}
    if set(truth) != {record["image_id"] for record in predictions}:
        raise ValueError("Prediction coverage mismatch")
    for record in predictions:
        if any(record.get(key) != value for key, value in truth[record["image_id"]].items()) or record["run_id"] != provenance["run_id"] or record["split_id"] != provenance["split_hash"] or record["configuration_digest"] != provenance["configuration_digest"]:
            raise ValueError("Prediction differs from frozen ground truth/provenance")


def fit(model_factory, train_dataset, validation_dataset, configuration, runs_root, resume=False, epoch_callback=None):
    configuration = validate_config(configuration)
    settings = configuration["training"]
    workers = configuration["model"].get("data_loader_workers", 0)
    pin_memory = configuration["model"].get("pin_memory", False)
    if type(workers) is not int or workers < 0 or type(pin_memory) is not bool:
        raise ValueError("Model configuration has invalid data-loader settings")
    if pin_memory and configuration["runtime"]["device"] != "cuda":
        raise ValueError("Pinned-memory loading is reserved for CUDA runs")
    loader_options = {"num_workers": workers, "pin_memory": pin_memory}
    if workers:
        loader_options["prefetch_factor"] = 2
    if train_dataset.split != "train" or validation_dataset.split != "validation":
        raise ValueError("Model selection accepts only train and validation datasets")
    if (train_dataset.manifest_hash, train_dataset.split_hash, train_dataset.class_map) != (validation_dataset.manifest_hash, validation_dataset.split_hash, validation_dataset.class_map):
        raise ValueError("Dataset provenance differs between splits")
    for dataset in [train_dataset, validation_dataset]:
        if dataset.seed != configuration["seed"] or dataset.preprocessing.configuration != configuration["preprocessing"]:
            raise ValueError("Dataset preprocessing/seed differs from run configuration")
    class_map = train_dataset.class_map
    ground_truth = {dataset.split: dataset.evaluation_contract() for dataset in [train_dataset, validation_dataset]}
    identifier = run_identity(configuration, train_dataset.manifest_hash, train_dataset.split_hash, class_map)
    provenance = {"run_id": identifier, "configuration": configuration, "configuration_digest": digest(configuration), "model_configuration": configuration["model"], "class_map": class_map, "dataset_manifest_hash": train_dataset.manifest_hash, "split_hash": train_dataset.split_hash, "seed": configuration["seed"], "runtime": configuration["runtime"], "ground_truth_hashes": {role: digest(contract) for role, contract in ground_truth.items()}, "split_policy": getattr(train_dataset, "split_policy", "development_only"), "object_independence_certified": False, "selection_split": "validation", "selection_metric": "weighted_validation_loss", "loss_reduction": "mean over samples of class-weighted negative log likelihood"}
    context = {"run_id": identifier, "configuration_digest": digest(configuration), "split_id": train_dataset.split_hash}
    run_directory = output_path(Path(runs_root) / identifier)
    result_path = run_directory / "result.json"
    if run_directory.exists():
        if not resume:
            raise FileExistsError("Run exists; request resume explicitly")
        if read_json(run_directory / "run.json") != provenance:
            raise ValueError("Existing run provenance mismatch")
        if result_path.exists():
            return completed_run(run_directory, provenance)
        if not (run_directory / "last.pt").exists():
            raise ValueError("Incomplete run has no resumable checkpoint")
    else:
        if resume:
            raise FileNotFoundError("Requested run does not exist; resume cannot start a new run")
        run_directory.mkdir(parents=True)
        write_json(run_directory / "run.json", provenance)
        for role, contract in ground_truth.items():
            write_json(run_directory / f"{role}_ground_truth.json", contract)
    for role, contract in ground_truth.items():
        if read_json(run_directory / f"{role}_ground_truth.json") != contract:
            raise ValueError("Immutable frozen ground truth changed")
    configure_runtime(configuration)
    random.seed(configuration["seed"])
    np.random.seed(configuration["seed"] % 2**32)
    torch.manual_seed(configuration["seed"])
    model = model_factory(json.loads(json.dumps(configuration["model"])), len(class_map)).to(configuration["runtime"]["device"])
    validate_attention(configuration["model"], model)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=settings["scheduler_factor"], patience=settings["scheduler_patience"])
    stopper = EarlyStopping(settings["patience"], settings["min_delta"])
    counts = torch.bincount(torch.tensor([class_map[record["class_label"]] for record in train_dataset.records]), minlength=len(class_map)).float()
    if (counts == 0).any():
        raise ValueError("Every class must have training examples")
    weights = counts.reciprocal()
    weights = weights / weights.mean()
    validation_loader = DataLoader(validation_dataset, batch_size=settings["batch_size"], shuffle=False, **loader_options)
    start_epoch = 0
    history = []
    if resume and (run_directory / "last.pt").exists():
        checkpoint = read_checkpoint(run_directory / "last.pt", provenance)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        stopper = EarlyStopping(**checkpoint["early_stopping"])
        start_epoch = checkpoint["epoch"] + 1
        history = checkpoint["history"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if configuration["runtime"]["device"] == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        random.setstate(checkpoint["python_rng_state"])
        numpy_state = checkpoint["numpy_rng_state"]
        np.random.set_state((numpy_state[0], np.array(numpy_state[1], dtype=np.uint32), *numpy_state[2:]))
    for epoch in range(start_epoch, settings["epochs"]):
        if stopper.bad_epochs >= stopper.patience:
            break
        train_dataset.set_epoch(epoch)
        generator = torch.Generator().manual_seed(configuration["seed"] + epoch)
        loader = DataLoader(train_dataset, batch_size=settings["batch_size"], shuffle=True, generator=generator, **loader_options)
        training_loss = train_epoch(model, loader, optimizer, weights, settings["max_grad_norm"])
        validation = evaluate(model, validation_loader, class_map, weights, purpose="selection", context=context, settings=configuration["evaluation"])
        scheduler.step(validation["loss"])
        improved, stop = stopper.update(validation["loss"])
        history.append({"epoch": epoch, "train_loss": training_loss, "validation": validation})
        checkpoint = {"schema_version": 3, "provenance": provenance, "epoch": epoch, "seed": configuration["seed"], "model_configuration": configuration["model"], "class_map": class_map, "dataset_manifest_hash": train_dataset.manifest_hash, "split_hash": train_dataset.split_hash, "configuration_digest": digest(configuration), "ground_truth_hashes": provenance["ground_truth_hashes"], "runtime": configuration["runtime"], "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(), "early_stopping": vars(stopper), "torch_rng_state": torch.get_rng_state(), "cuda_rng_states": torch.cuda.get_rng_state_all() if configuration["runtime"]["device"] == "cuda" else [], "python_rng_state": random.getstate(), "history": history, "torch_version": str(torch.__version__)}
        checkpoint["class_weights"] = weights.tolist()
        numpy_state = np.random.get_state()
        checkpoint["numpy_rng_state"] = [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]]
        if improved:
            atomic_write(run_directory / "best.pt", checkpoint, binary=True)
        atomic_write(run_directory / "last.pt", checkpoint, binary=True)
        if epoch_callback is not None:
            epoch_callback({
                "epoch": epoch,
                "epochs_maximum": settings["epochs"],
                "train_loss": training_loss,
                "validation_loss": validation["loss"],
                "validation_accuracy": validation["accuracy"],
                "validation_balanced_accuracy": validation["balanced_accuracy"],
                "learning_rate": optimizer.param_groups[0]["lr"],
                "improved": improved,
                "bad_epochs": stopper.bad_epochs,
            })
        if stop:
            break
    best = read_checkpoint(run_directory / "best.pt", provenance)
    model.load_state_dict(best["model_state"], strict=True)
    validation = evaluate(model, validation_loader, class_map, weights, purpose="selection", context=context, settings=configuration["evaluation"], bootstrap=True)
    atomic_write(run_directory / "validation_errors.json", [record for record in validation["predictions"] if record["true_label"] != record["predicted_label"]])
    atomic_write(run_directory / "validation_predictions.json", validation.pop("predictions"))
    result = {"status": "completed", "run_id": identifier, "provenance": provenance, "best_epoch": best["epoch"], "epochs_completed": len(history), "validation": validation, "test_evaluated": False}
    atomic_write(run_directory / "history.json", history)
    result["artifacts"] = {name: file_hash(run_directory / name) for name in ["run.json", "train_ground_truth.json", "validation_ground_truth.json", "best.pt", "last.pt", "history.json", "validation_predictions.json", "validation_errors.json"]}
    result["result_digest"] = digest(result)
    atomic_write(result_path, result)
    return result


def freeze_final_selection(run_directory, rationale):
    run_directory = output_path(run_directory)
    provenance = read_json(run_directory / "run.json")
    result = completed_run(run_directory, provenance)
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("A pre-test selection rationale is required")
    lock = {"run_directory": str(run_directory), "run_id": provenance["run_id"], "configuration_digest": provenance["configuration_digest"], "split_id": provenance["split_hash"], "checkpoint_sha256": result["artifacts"]["best.pt"], "result_digest": result["result_digest"], "rationale": rationale, "selection_split": "validation", "threshold_policy": "argmax, no test tuning", "evaluation_settings": provenance["configuration"]["evaluation"], "development_only": True}
    lock["lock_digest"] = digest(lock)
    write_json(run_directory / "final_selection_lock.json", lock)
    return lock


def validate_selection_lock(lock, context):
    if not isinstance(lock, dict) or lock.get("lock_digest") != digest({key: value for key, value in lock.items() if key != "lock_digest"}):
        raise ValueError("Test evaluation requires an immutable final selection lock")
    run_directory = output_path(lock["run_directory"])
    if read_json(run_directory / "final_selection_lock.json") != lock:
        raise ValueError("Final selection lock changed")
    if any(context.get(key) != lock.get(key) for key in ["run_id", "configuration_digest", "split_id"]):
        raise ValueError("Final selection provenance mismatch")
    result = completed_run(run_directory, read_json(run_directory / "run.json"))
    if result["result_digest"] != lock["result_digest"] or result["artifacts"]["best.pt"] != lock["checkpoint_sha256"]:
        raise ValueError("Locked selection changed")


def final_evaluate(model_factory, run_directory, test_dataset_factory):
    run_directory = output_path(run_directory)
    lock = read_json(run_directory / "final_selection_lock.json")
    context = {key: lock[key] for key in ["run_id", "configuration_digest", "split_id"]}
    validate_selection_lock(lock, context)
    provenance = read_json(run_directory / "run.json")
    configuration = validate_config(provenance["configuration"])
    destination = run_directory / "final_evaluation"
    if destination.exists():
        result = read_json(destination / "result.json")
        if result.get("result_digest") != digest({key: value for key, value in result.items() if key != "result_digest"}) or result.get("selection_lock_digest") != lock["lock_digest"]:
            raise ValueError("Corrupted cached final result")
        for name, expected in result["artifacts"].items():
            if Path(name).name != name or file_hash(destination / name) != expected:
                raise ValueError("Corrupted cached final artifact")
        predictions = read_json(destination / "predictions.json")
        verify_cached_evaluation(result, predictions, provenance["class_map"], configuration["evaluation"])
        validate_saved_truth(predictions, read_json(destination / "ground_truth.json"), provenance, result["ground_truth_digest"])
        return result
    dataset = test_dataset_factory()
    if dataset.split != "test" or dataset.manifest_hash != provenance["dataset_manifest_hash"] or dataset.split_hash != provenance["split_hash"] or dataset.class_map != provenance["class_map"] or dataset.preprocessing.configuration != configuration["preprocessing"] or dataset.seed != configuration["seed"]:
        raise ValueError("Final dataset differs from locked provenance")
    with dataset.final_test_access(lock):
        contract = dataset.evaluation_contract()
        destination.mkdir()
        write_json(destination / "ground_truth.json", contract)
        write_json(destination / "selection_lock.json", lock)
        configure_runtime(configuration)
        torch.manual_seed(configuration["seed"])
        model = model_factory(configuration["model"], len(provenance["class_map"])).to(configuration["runtime"]["device"])
        validate_attention(configuration["model"], model)
        checkpoint = read_checkpoint(run_directory / "best.pt", provenance)
        model.load_state_dict(checkpoint["model_state"], strict=True)
        weights = torch.tensor(checkpoint["class_weights"])
        result = evaluate(model, DataLoader(dataset, batch_size=configuration["training"]["batch_size"], shuffle=False, num_workers=0), provenance["class_map"], weights, purpose="final", context=context, settings=configuration["evaluation"], bootstrap=True, selection_lock=lock)
    predictions = result.pop("predictions")
    write_json(destination / "predictions.json", predictions)
    write_json(destination / "errors.json", [record for record in predictions if record["true_label"] != record["predicted_label"]])
    result["selection_lock_digest"] = lock["lock_digest"]
    result["development_only"] = True
    result["artifacts"] = {name: file_hash(destination / name) for name in ["ground_truth.json", "selection_lock.json", "predictions.json", "errors.json"]}
    result["result_digest"] = digest(result)
    write_json(destination / "result.json", result)
    return result
