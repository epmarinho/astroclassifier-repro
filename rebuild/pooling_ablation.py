"""Frozen full-development pooling/grid/attention ablation; never constructs test data."""
import argparse
import csv
import time

from . import full, training
from .common import ROOT, PROJECT, digest, file_hash, read_json
from .config import validate_config
from .data import CLASSES
from .phase3d.pilot import PilotDataset
from .phase3d.models_v3 import VARIANTS, POOLING_POLICIES, architecture, model_factory

BASE = ROOT / "reports/pooling_ablation_v1"
CONFIGS = ROOT / "configs/pooling_ablation_v1"
RUNS = ROOT / "runs/pooling_ablation_v1"
FIGURES = ROOT / "figures/pooling_ablation_v1"


def source_hashes():
    return {**full.source_hashes(), **{
        f"rebuild/{name}": file_hash(ROOT / name)
        for name in ("ablation.py", "phase3d/models_v2.py", "pooling_ablation.py", "phase3d/models_v3.py")
    }}


def configurations(grids, seeds, preflight, pools=POOLING_POLICIES):
    if not grids or not seeds or len(set(grids)) != len(grids) or len(set(seeds)) != len(seeds):
        raise ValueError("Grids and seeds must be nonempty and unique")
    if not pools or len(set(pools)) != len(pools) or any(pool not in POOLING_POLICIES for pool in pools):
        raise ValueError("Pooling policies must be nonempty, unique and supported")
    result = {}
    for seed in seeds:
        for grid in grids:
            policies = ["identity"] if grid == 16 else pools
            for policy in policies:
                for variant in VARIANTS:
                    configuration = full.full_configuration(variant, preflight)
                    configuration["model"].update(architecture(variant, grid_size=grid, pooling=policy))
                    configuration["seed"] = seed
                    configuration["evaluation"]["bootstrap_seed"] = seed
                    key = f"{variant}_grid{grid}_{policy}_seed{seed}"
                    result[key] = validate_config(configuration)
    return result


def preflight(grids, seeds, pools):
    # Reject changing a frozen experiment before any GPU work or writes.
    if (BASE / "plan.json").exists():
        plan = checked_plan()
        if plan["grids"] != grids or plan["seeds"] != seeds or plan["pools"] != list(pools):
            raise ValueError("This ablation plan is frozen; use its existing grids/seeds")
        print("Existing ablation preflight verified; plan unchanged.", flush=True)
        return
    info = full.environment()
    full.require_blackwell_cuda(info)
    collection, split, counts, skipped = full.full_collection()
    record = {
        "environment": info, "source_digests": source_hashes(),
        "counts": {**counts, "test": 0},
        "skipped_test_groups_before_label_deserialization": skipped,
        "test_dataset_constructed": False,
    }
    configs = configurations(grids, seeds, record, pools)
    for row in collection["records"]:
        path = (PROJECT / row["relative_path"]).resolve()
        if not path.is_relative_to(PROJECT) or file_hash(path) != row["sha256"]:
            raise ValueError(f"Frozen input differs: {row['relative_path']}")
    smoke = {}
    for key, configuration in configs.items():
        training.configure_runtime(configuration)
        import torch
        torch.manual_seed(configuration["seed"])
        torch.cuda.reset_peak_memory_stats()
        model = model_factory(configuration["model"], len(CLASSES)).to("cuda")
        batch = torch.zeros(configuration["training"]["batch_size"], 3, 256, 256, device="cuda")
        labels = torch.arange(len(batch), device="cuda") % len(CLASSES)
        loss = torch.nn.functional.cross_entropy(model(batch), labels)
        loss.backward()
        torch.cuda.synchronize()
        if not torch.isfinite(loss) or any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise RuntimeError(f"Nonfinite synthetic forward/backward: {key}")
        smoke[key] = {
            "forward_backward": "passed", "peak_memory_bytes": torch.cuda.max_memory_allocated(),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            "pooling_trainable_parameters": sum(p.numel() for p in model.grid_pool.parameters() if p.requires_grad),
        }
        del model, batch, labels, loss
        torch.cuda.empty_cache()
        print(f"{key}: CUDA forward/backward passed", flush=True)
    for path, value in ((BASE / "manifest.json", collection), (BASE / "split.json", split)):
        full.write_once_or_verify(path, value)
    for key, configuration in configs.items():
        full.write_once_or_verify(CONFIGS / f"{key}.json", configuration)
    full.write_once_or_verify(BASE / "plan.json", {
        "version": "pooling-ablation-v1", "grids": grids, "seeds": seeds, "pools": list(pools),
        "preflight": record, "gpu_smoke": smoke,
        "configurations": {key: digest(value) for key, value in configs.items()},
        "manifest_digest": digest(collection), "split_digest": digest(split),
    })
    print(f"Frozen {len(configs)} runs: 485 train / 158 validation / 0 test.", flush=True)


def checked_plan():
    plan = read_json(BASE / "plan.json")
    record = plan["preflight"]
    if source_hashes() != record["source_digests"]:
        raise RuntimeError("Execution sources differ from ablation preflight")
    if full.environment() != record["environment"]:
        raise RuntimeError("Environment differs from ablation preflight")
    expected = configurations(plan["grids"], plan["seeds"], record, plan["pools"])
    if {key: digest(value) for key, value in expected.items()} != plan["configurations"]:
        raise RuntimeError("Frozen ablation configurations differ")
    for key, value in expected.items():
        if read_json(CONFIGS / f"{key}.json") != value:
            raise RuntimeError(f"Configuration differs: {key}")
    for name in ("manifest", "split"):
        if digest(read_json(BASE / f"{name}.json")) != plan[f"{name}_digest"]:
            raise RuntimeError(f"Frozen {name} differs")
    return plan


def entries(plan):
    collection = read_json(BASE / "manifest.json")
    split = read_json(BASE / "split.json")
    for key in plan["configurations"]:
        configuration = read_json(CONFIGS / f"{key}.json")
        datasets = {role: PilotDataset(collection, split, role, configuration["preprocessing"], configuration["seed"])
                    for role in ("train", "validation")}
        train = datasets["train"]
        identifier = training.run_identity(configuration, train.manifest_hash, train.split_hash, train.class_map)
        yield key, configuration, datasets, RUNS / identifier


def run():
    for key, configuration, datasets, directory in entries(checked_plan()):
        print(f"[{key}] run {directory.name[:12]}; pooling={configuration['model']['additional_pooling']}", flush=True)
        def progress(item):
            print(f"[{key}] epoch {item['epoch'] + 1:03d}/{item['epochs_maximum']} "
                  f"train={item['train_loss']:.5f} val={item['validation_loss']:.5f} "
                  f"acc={item['validation_accuracy']:.3f} bal_acc={item['validation_balanced_accuracy']:.3f} "
                  f"lr={item['learning_rate']:.3g} bad={item['bad_epochs']}", flush=True)
        start = time.monotonic()
        result = training.fit(model_factory, datasets["train"], datasets["validation"], configuration,
                              RUNS, resume=directory.exists(), epoch_callback=progress)
        print(f"[{key}] complete: best epoch {result['best_epoch'] + 1}, "
              f"balanced accuracy {result['validation']['balanced_accuracy']:.4f}, "
              f"time {time.monotonic() - start:.1f}s", flush=True)
    print("Ablation completed; test remains locked.")


def status():
    for key, _, _, directory in entries(checked_plan()):
        state = "completed" if (directory / "result.json").exists() else "resumable" if (directory / "last.pt").exists() else "not started"
        print(f"{key}: {state}")


def visualize(force=False):
    from . import visualize as charts
    plan = read_json(BASE / "plan.json")
    runs = []
    for key, configuration, _, directory in entries(plan):
        result = read_json(directory / "result.json")
        runs.append({"variant": configuration["model"]["variant"], "name": key,
                     "result": result, "history": read_json(directory / "history.json"),
                     "predictions": read_json(directory / "validation_predictions.json"),
                     "errors": read_json(directory / "validation_errors.json")})
    output = charts.prepare_output(FIGURES, force)
    charts.write_scores(runs, output, heading="Pooling/grid/attention ablation — validation",
                        note="485 train / 158 validation; checkpoint selected by validation loss; no test evaluation.")
    # Correct the legacy writer's fixed pilot provenance caption in this new output only.
    path = output / "SCORES.md"
    path.write_text(path.read_text().replace("rebuild/runs/phase3d_v4/", "rebuild/runs/pooling_ablation_v1/"))
    charts.plot_learning_curves(runs, output, label="Pooling/grid/attention ablation")
    from matplotlib import pyplot as plt
    columns = 2
    rows = (len(runs) + columns - 1) // columns
    figure, axes = plt.subplots(rows, columns, figsize=(12, 4 * rows), squeeze=False, constrained_layout=True)
    for axis, item in zip(axes.flat, runs):
        matrix = item["result"]["validation"]["confusion_matrix"]
        axis.imshow(matrix, cmap="Blues")
        for i, row in enumerate(matrix):
            for j, value in enumerate(row):
                axis.text(j, i, str(value), ha="center", va="center", color="red")
        axis.set_title(item["name"], fontsize=8)
        class_map = item["result"]["provenance"]["class_map"]
        classes = sorted(class_map, key=class_map.get)
        axis.set_xticks(range(4), classes, rotation=30, ha="right")
        axis.set_yticks(range(4), classes)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    for axis in list(axes.flat)[len(runs):]:
        axis.set_visible(False)
    for extension in ("png", "svg"):
        figure.savefig(output / f"confusion_matrices.{extension}")
    plt.close(figure)
    write_experiment_table(runs, output, plan)
    print(f"Generated ablation figures and scores in {output}")


def write_experiment_table(runs, output, plan):
    with (output / "experiments.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["experiment", "seed", "grid_size", "token_count", "additional_pooling", "run_id", "balanced_accuracy", "macro_f1", "validation_loss", "pooling_trainable_parameters", "trainable_parameters"])
        for item in runs:
            config = item["result"]["provenance"]["configuration"]
            model = config["model"]
            writer.writerow([item["name"], config["seed"], model["grid_size"], model["token_count"], model["additional_pooling"], item["result"]["provenance"].get("run_id", ""), item["result"]["validation"]["balanced_accuracy"], item["result"]["validation"]["macro_f1"], item["result"]["validation"]["loss"], model["pooling_trainable_parameters"], plan["gpu_smoke"][item["name"]]["trainable_parameters"]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    pre = sub.add_parser("preflight")
    pre.add_argument("--pools", nargs="+", choices=POOLING_POLICIES, default=list(POOLING_POLICIES))
    pre.add_argument("--grids", type=int, nargs="+", default=[8, 16])
    pre.add_argument("--seeds", type=int, nargs="+", default=[full.SEED])
    sub.add_parser("run")
    sub.add_parser("status")
    vis = sub.add_parser("visualize")
    vis.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.action == "preflight":
        preflight(args.grids, args.seeds, args.pools)
    elif args.action == "visualize":
        visualize(args.force)
    elif args.action == "status":
        status()
    else:
        run()


if __name__ == "__main__":
    main()
