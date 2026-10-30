"""Build a test-blind, hash-bound Dataset-v2K final-evaluation plan."""
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "rebuild/dataset_v2k_test_evaluation_plan"
REBUILD = ROOT / "rebuild"

PROTOCOL = REBUILD / "dataset_v2i_training_protocol/protocol.json"
TEST_LOCK = REBUILD / "dataset_v2i_training_protocol/test_lock.json"
CANDIDATES = REBUILD / "dataset_v2i_training_protocol/candidate_manifest.json"
STAT_PLAN = REBUILD / "dataset_v2i_training_protocol/statistical_analysis_plan.md"
MEMBERSHIP = REBUILD / "dataset_v2g_split_materialization/assignment_unit_membership.json"
SPLIT_MANIFEST = REBUILD / "dataset_v2g_split_materialization/split_manifest.json"
LABEL_MANIFEST = REBUILD / "dataset_v2h_representation_materialization/sealed_label_manifest.json"
R4_SELECTION = REBUILD / "dataset_v2j1_validation_search/repair_r4/frozen_selection_r4.json"
R4_CKPT_MANIFEST = REBUILD / "dataset_v2j1_validation_search/repair_r4/selected_checkpoint_manifest_r4.json"
J2_CKPT_MANIFEST = REBUILD / "dataset_v2j2_control_fits/control_checkpoint_manifest.json"
METADATA_RESULTS = REBUILD / "dataset_v2j1_validation_search/metadata_controls.json"


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def file_record(path, label=None):
    path = Path(path)
    return {"label": label or path.name, "repository_relative_path": path.relative_to(ROOT).as_posix(),
            "size_bytes": path.stat().st_size, "sha256": sha_file(path)}


def verify_inventory(path):
    inv = read_json(path)
    entries = inv.get("files", [])
    for item in entries:
        target = ROOT / item["path"]
        if not target.is_file() or target.stat().st_size != item["size_bytes"] or sha_file(target) != item["sha256"]:
            raise RuntimeError(f"Frozen preservation inventory mismatch: {item['path']}")
    return {"inventory_path": path.relative_to(ROOT).as_posix(), "inventory_file_sha256": sha_file(path),
            "inventory_canonical_sha256": inv.get("canonical_sha256", inv.get("inventory_entries_sha256")),
            "entries_verified": len(entries), "all_listed_files_match": True}


def main():
    protocol = read_json(PROTOCOL)
    lock = read_json(TEST_LOCK)
    candidates = read_json(CANDIDATES)
    selection = read_json(R4_SELECTION)
    r4_manifest = read_json(R4_CKPT_MANIFEST)
    j2_manifest = read_json(J2_CKPT_MANIFEST)

    if protocol["canonical_sha256"] != "6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737":
        raise RuntimeError("Frozen v2I canonical protocol digest mismatch")
    if sha_file(PROTOCOL) != "fcbc29993adf83c86f5f4cab445e8e61bb88cd312f390fdfa253c37dad8d2b22":
        raise RuntimeError("Frozen v2I protocol byte hash mismatch")
    if lock["training_protocol_sha256"] != protocol["canonical_sha256"] or lock["test_access_allowed_now"] is not False:
        raise RuntimeError("Frozen test lock identity or current access state mismatch")
    if sha_file(CANDIDATES) != "e5032c7194e0f8479c0df6a355d6651f1fb5b90e8811429b992ebb3fbcb13572":
        raise RuntimeError("Frozen v2I candidate manifest byte hash mismatch")
    if sha_file(R4_SELECTION) != "52643419e9ad8a15480ee821baf19bf655d6d9e62c374386b544b8f38fe236a2":
        raise RuntimeError("R4 selection byte hash mismatch")
    if selection.get("selection_digest") != "cd45b0c9fbb1e6310b31410c5f3be5edf9e694a7737f17a6c95c0788381a580b":
        raise RuntimeError("R4 selection canonical digest mismatch")
    if sha_file(R4_CKPT_MANIFEST) != "ab9d653923ee8b42af97fa99f4901d6ecf979cd7f83791bbec61dbb3d56fbb32":
        raise RuntimeError("R4 selected-checkpoint manifest byte hash mismatch")
    if r4_manifest.get("canonical_sha256") != "ec3ef679ef9fa20504e83aa091b4b3e0a859f4db12e860380f1f2da695f723e7":
        raise RuntimeError("R4 selected-checkpoint manifest canonical digest mismatch")
    if sha_file(J2_CKPT_MANIFEST) != "14c823362d2f294fe1e6ad77542ae4ff407143694194acbfbd75cadf4ac44f56":
        raise RuntimeError("v2J2 checkpoint manifest byte hash mismatch")
    if j2_manifest.get("canonical_sha256") != "542fd4f42dc116c4806e3ef5ff96c0ca0abf2c25c4340bdaadda3bae87fba162":
        raise RuntimeError("v2J2 checkpoint manifest canonical digest mismatch")
    if j2_manifest.get("selection_digest") != selection["selection_digest"] or j2_manifest.get("test_data_accessed") is not False:
        raise RuntimeError("v2J2 manifest is not bound to R4 or reports test access")

    # Membership and truth files are deliberately hashed as raw bytes only.
    # Do not decode, parse, or inspect their records in this planning phase.
    membership_sha = sha_file(MEMBERSHIP)
    if membership_sha != "0440a32d8ef8d314fcea828e14ba517a5621fd0b0a67ca750ae6ddb99a424209":
        raise RuntimeError("Frozen v2G membership byte hash mismatch")
    label_manifest_sha = sha_file(LABEL_MANIFEST)
    if label_manifest_sha != "8355ae20da64b27eaa1506bbde960e7f3b2b34c8c148fe73b68fff8ff43c3121":
        raise RuntimeError("Frozen v2H sealed-label manifest byte hash mismatch")
    if sha_file(SPLIT_MANIFEST) != "85335f9f6db49689c9a68200c7d2b77d0a8912e9acd8a63fbaf8cc6fe20f33c7":
        raise RuntimeError("Frozen v2G split-manifest byte hash mismatch")

    # Preserve v2J1 and v2J2 against their already frozen inventories.
    j1_preservation = verify_inventory(REBUILD / "dataset_v2j1_validation_search/repair_r4/file_inventory_r4.json")
    j2_preservation = verify_inventory(REBUILD / "dataset_v2j2_control_fits/file_inventory_v2j2_final.json")

    a_by_family = {}
    a_rows = r4_manifest["checkpoints"]
    if len(a_rows) != 9:
        raise RuntimeError("R4 must bind exactly nine A checkpoints")
    for row in a_rows:
        if row["representation"] != "A":
            raise RuntimeError("R4 selected checkpoint has a non-A representation")
        a_by_family.setdefault(row["family"], []).append(row)
    expected_families = ["custom_cnn_transformer", "resnet50", "vit_b_16"]
    if set(a_by_family) != set(expected_families) or any(len(a_by_family[f]) != 3 for f in expected_families):
        raise RuntimeError("R4 must bind three seeds for each of the three architectures")

    network_targets = []
    for row in a_rows:
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != row["size_bytes"] or sha_file(path) != row["sha256"]:
            raise RuntimeError(f"R4 checkpoint bytes/size mismatch: {row['path']}")
        network_targets.append({"family": row["family"], "representation": "A",
                                "seed": row["seed"], "fit_id": row["fit_id"],
                                "configuration_id": row["configuration_id"],
                                "configuration_sha256": row["configuration_sha256"],
                                "effective_parameters_sha256": digest(row["effective_parameters"]),
                                "checkpoint_path": row["path"], "checkpoint_size_bytes": row["size_bytes"],
                                "checkpoint_sha256": row["sha256"], "manifest_role": "R4 selected A winner"})

    control_rows = j2_manifest["checkpoints"]
    if len(control_rows) != 27:
        raise RuntimeError("v2J2 must bind exactly 27 controls")
    required_reps = {"B", "A_outer", "B_outer"}
    seen_controls = set()
    for row in control_rows:
        slot = (row["family"], row["representation"], row["seed"])
        if slot in seen_controls or row["representation"] not in required_reps:
            raise RuntimeError("v2J2 duplicate or unauthorized representation slot")
        seen_controls.add(slot)
        winner = a_by_family[row["family"]][0]
        if (row["configuration_id"] != winner["configuration_id"] or
                row["configuration_sha256"] != winner["configuration_sha256"] or
                row["effective_parameters"] != winner["effective_parameters"]):
            raise RuntimeError(f"Control does not match frozen R4 configuration: {slot}")
        path = ROOT / row["checkpoint_path"]
        if not path.is_file() or path.stat().st_size != row["checkpoint_size_bytes"] or sha_file(path) != row["checkpoint_sha256"]:
            raise RuntimeError(f"v2J2 checkpoint bytes/size mismatch: {row['checkpoint_path']}")
        network_targets.append({"family": row["family"], "representation": row["representation"],
                                "seed": row["seed"], "fit_id": row["fit_id"],
                                "configuration_id": row["configuration_id"],
                                "configuration_sha256": row["configuration_sha256"],
                                "effective_parameters_sha256": digest(row["effective_parameters"]),
                                "checkpoint_path": row["checkpoint_path"],
                                "checkpoint_size_bytes": row["checkpoint_size_bytes"],
                                "checkpoint_sha256": row["checkpoint_sha256"], "manifest_role": "v2J2 control fit"})
    expected_control_slots = {(f, r, s) for f in expected_families for r in required_reps
                              for s in (3908274, 3908275, 3908276)}
    if seen_controls != expected_control_slots or len(network_targets) != 36:
        raise RuntimeError("The frozen network test-evaluation design is not exactly 36 checkpoints")
    if len({x["checkpoint_sha256"] for x in network_targets}) != 36:
        raise RuntimeError("Network checkpoint hashes are not unique")

    metadata = read_json(METADATA_RESULTS)
    if metadata.get("fit_count") != 2 or metadata.get("status") != "completed" or metadata.get("test_paths_or_labels_accessed") is not False:
        raise RuntimeError("Frozen metadata controls are incomplete or report test access")
    metadata_targets = []
    for model in metadata["models"]:
        path = REBUILD / "dataset_v2j1_validation_search" / model["model_artifact"]
        if not path.is_file() or path.stat().st_size != model["model_artifact_size_bytes"] or sha_file(path) != model["model_artifact_sha256"]:
            raise RuntimeError(f"Metadata-control artifact identity mismatch: {model['configuration_id']}")
        metadata_targets.append({"control_id": model["configuration_id"], "fit_id": model["fit_id"],
                                 "model_type": model["family"], "representation": "metadata_FOV_only",
                                 "seed": model["seed"], "artifact_path": model["model_artifact"],
                                 "artifact_size_bytes": model["model_artifact_size_bytes"],
                                 "artifact_sha256": model["model_artifact_sha256"],
                                 "features": model["features"],
                                 "test_evaluations": 1, "manifest_role": "fixed v2I metadata/FOV control"})
    if len(metadata_targets) != 2:
        raise RuntimeError("Exactly two fixed metadata/FOV controls are required")

    # Bind frozen custom-source files and the exact local inference implementations.
    custom_sources = a_by_family["custom_cnn_transformer"][0]["effective_parameters"]["model"]["source_digests"]
    code_sources = []
    for rel, expected in sorted(custom_sources.items()):
        actual = sha_file(ROOT / rel)
        if actual != expected:
            raise RuntimeError(f"Frozen custom-model source mismatch: {rel}")
        code_sources.append({"path": rel, "sha256": actual, "authority": "R4 effective_parameters.model.source_digests"})
    explicit_code = [
        ("rebuild/dataset_v2j1_validation_search/repair_r3/executor_r3/train_worker_r3.py",
         "899afb65fb8dfba4193447e1f536410279cd97ed07dcec0ff1c5be2c7cb67d2c", "frozen J1 A model/validation worker source"),
        ("rebuild/dataset_v2j2_control_fits/executor/train_worker_v2j2.py",
         "90a370f329d0a12b929bf796d5ba763fd1e90af53515036113c12fef6fadd6aa", "frozen J2 model/control worker source"),
        ("rebuild/dataset_v2j1_validation_search/metadata_controls.py",
         None, "fixed metadata-control inference implementation"),
        ("rebuild/dataset_v2h_representation_materialization/materialize.py",
         None, "frozen tensor representation implementation"),
        ("rebuild/dataset_v2i_training_protocol/statistical_analysis_plan.md",
         None, "frozen statistical scoring rules"),
    ]
    for rel, expected, role in explicit_code:
        actual = sha_file(ROOT / rel)
        if expected is not None and actual != expected:
            raise RuntimeError(f"Frozen source hash mismatch: {rel}")
        code_sources.append({"path": rel, "sha256": actual, "authority": role})
    torchvision_root = Path("/home/emarinho/miniconda3/envs/astro_env/lib/python3.12/site-packages/torchvision/models")
    torchvision_code = []
    for filename in ("resnet.py", "vision_transformer.py"):
        path = torchvision_root / filename
        if path.is_file():
            torchvision_code.append({"path": str(path), "sha256": sha_file(path)})

    network_targets.sort(key=lambda x: (expected_families.index(x["family"]),
                                       ["A", "B", "A_outer", "B_outer"].index(x["representation"]), x["seed"]))
    metadata_targets.sort(key=lambda x: x["control_id"])
    target_manifest = {"phase": "Dataset-v2K-plan", "status": "frozen_targets_not_evaluated",
                       "selection_digest": selection["selection_digest"],
                       "network_evaluations": 36, "metadata_control_evaluations": 2,
                       "total_evaluations": 38, "observations_per_evaluation": 73,
                       "network_targets": network_targets, "metadata_targets": metadata_targets,
                       "test_membership_rows_opened": False, "test_labels_opened": False,
                       "test_tensors_opened": False, "inference_or_scoring_performed": False}
    target_manifest["canonical_sha256"] = digest(target_manifest)

    bindings = {
        "frozen_protocols_and_manifests": [
            file_record(PROTOCOL, "v2I protocol bytes"), file_record(TEST_LOCK, "v2I test lock bytes"),
            file_record(CANDIDATES, "v2I candidate manifest bytes"), file_record(STAT_PLAN, "v2I statistical analysis plan"),
            file_record(SPLIT_MANIFEST, "v2G split manifest bytes"), file_record(LABEL_MANIFEST, "v2H sealed label manifest bytes"),
            file_record(R4_SELECTION, "R4 frozen selection bytes"), file_record(R4_CKPT_MANIFEST, "R4 selected checkpoint manifest bytes"),
            file_record(J2_CKPT_MANIFEST, "v2J2 control checkpoint manifest bytes"),
            file_record(METADATA_RESULTS, "v2I fixed metadata control record bytes"),
            file_record(REBUILD / "dataset_v2j1_validation_search/repair_r4/file_inventory_r4.json", "v2J1 R4 inventory bytes"),
            file_record(REBUILD / "dataset_v2j2_control_fits/file_inventory_v2j2_final.json", "v2J2 final inventory bytes"),
            file_record(REBUILD / "dataset_v2j1_validation_search/environment.json", "J1 frozen environment receipt"),
            file_record(REBUILD / "dataset_v2j2_control_fits/environment.json", "J2 frozen environment receipt"),
            file_record(REBUILD / "dataset_v2h_representation_materialization/representation_A_manifest.json", "v2H A tensor manifest bytes"),
            file_record(REBUILD / "dataset_v2h_representation_materialization/representation_B_manifest.json", "v2H B tensor manifest bytes"),
            file_record(REBUILD / "dataset_v2h_representation_materialization/control_manifest.json", "v2H static control manifest bytes"),
        ],
        "protected_content_identity_raw_hashes_only": {
            "assignment_unit_membership": {"path": MEMBERSHIP.relative_to(ROOT).as_posix(), "sha256_bytes": membership_sha,
                                            "parsed_or_deserialized": False, "rows_or_individual_ids_read": False},
            "sealed_test_truth_manifest": {"path": LABEL_MANIFEST.relative_to(ROOT).as_posix(), "sha256_bytes": label_manifest_sha,
                                           "parsed_or_deserialized": False, "labels_read": False},
            "frozen_membership_digest_from_v2I_lock": lock["test_membership_digest"],
            "frozen_test_split_digest": lock["test_split_sha256"],
            "v2H_dataset_digest": protocol["source_digests"]["v2H_dataset_digest"],
            "v2H_tensor_merkle_digest": protocol["source_digests"]["v2H_tensor_merkle"],
        },
        "selection_bindings": {"protocol_canonical_sha256": protocol["canonical_sha256"],
                               "protocol_file_sha256": sha_file(PROTOCOL),
                               "candidate_manifest_sha256": sha_file(CANDIDATES),
                               "R4_selection_digest": selection["selection_digest"],
                               "R4_selection_file_sha256": sha_file(R4_SELECTION),
                               "R4_checkpoint_manifest_canonical_sha256": r4_manifest["canonical_sha256"],
                               "R4_checkpoint_manifest_file_sha256": sha_file(R4_CKPT_MANIFEST),
                               "J2_checkpoint_manifest_canonical_sha256": j2_manifest["canonical_sha256"],
                               "J2_checkpoint_manifest_file_sha256": sha_file(J2_CKPT_MANIFEST)},
        "code_hashes": {"project_and_protocol_code": code_sources,
                        "torchvision_model_source_files": torchvision_code,
                        "final_inference_executor": {"status": "not yet implemented or frozen",
                                                      "sha256": None,
                                                      "required_gate": "Create a read-only, test-isolated executor in the separately authorized evaluation phase; audit and hash it before opening any test tensor or membership row."},
                        "final_scoring_executor": {"status": "not yet implemented or frozen",
                                                   "sha256": None,
                                                   "required_gate": "Scoring source and hash must be frozen before predictions are sealed and truth is released; the scoring process receives truth only after all prediction hashes are immutable."}},
        "preservation_verification": {"v2J1": j1_preservation, "v2J2": j2_preservation,
                                      "writes_to_v2J1_or_v2J2_during_plan": False},
    }

    plan = {
        "phase": "Dataset-v2K test-evaluation plan",
        "version": "v2K-plan-01",
        "status": "plan_frozen_execution_not_authorized_by_this_phase",
        "evaluation_authorized_now": False,
        "test_pixels_membership_rows_or_labels_opened_now": False,
        "inference_or_scoring_performed_now": False,
        "frozen_identity": {"split_digest": lock["test_split_sha256"],
                            "test_membership_digest": lock["test_membership_digest"],
                            "test_observation_count": lock["test_observation_count"],
                            "assignment_unit_count": lock["test_assignment_unit_count"],
                            "v2G_membership_file_sha256_bytes": membership_sha,
                            "v2G_split_manifest_file_sha256_bytes": sha_file(SPLIT_MANIFEST),
                            "sealed_label_manifest_sha256_bytes": label_manifest_sha,
                            "class_order": protocol["data_and_preprocessing"]["class_order"]},
        "frozen_targets": {"path": (OUT / "evaluation_targets.json").relative_to(ROOT).as_posix(),
                           "canonical_sha256": target_manifest["canonical_sha256"],
                           "network_checkpoints": 36, "metadata_controls": 2, "total_evaluations": 38},
        "selection_independence": {"network_targets_are_exact_union_of_R4_A_and_v2J2_B_A_outer_B_outer": True,
                                   "metadata_targets_are_the_two_frozen_v2I_controls": True,
                                   "no_test_based_selection_or_best_seed": True,
                                   "all_declared_targets_reported_regardless_of_performance": True,
                                   "no_retraining_replacement_reselection_or_threshold_change": True,
                                   "control_results_do_not_change_primary_representation_A": True},
        "single_pass": {"one_inference_evaluation_per_target": True,
                        "observations_per_target": 73, "network_model_observation_pairs": 2628,
                        "metadata_model_observation_pairs": 146, "total_model_observation_predictions": 2774,
                        "successful_evaluation_must_not_be_repeated": True,
                        "no_training_optimizer_or_scheduler": True},
        "inference_contract": {"input_source": "frozen v2H tensors for exactly the common ordered test membership",
                               "representations": ["A", "B", "A_outer", "B_outer"],
                               "channels": ["blue", "red", "NIR"],
                               "class_order": protocol["data_and_preprocessing"]["class_order"],
                               "normalization": "exact v2H frozen normalization; no additional family/ImageNet normalization",
                               "custom_input_adapter": protocol["custom_input_adapter"]["operation"],
                               "adapter_applies": ["A", "B", "A_outer", "B_outer"],
                               "augmentation": "identity only; no D4, random transforms, TTA, crop, threshold or fitted transform",
                               "model_mode": "eval() and inference_mode/no_grad; float32; no AMP",
                               "batch_size": protocol["optimization_proposal"]["batch_size"],
                               "decision_rule": "four-class argmax in frozen class order; no calibrated or tuned threshold",
                               "determinism": protocol["determinism"]["cuda"],
                               "metadata_controls": "load the two hash-verified frozen joblib artifacts; predict with the frozen feature order and train-only preprocessing embedded in those artifacts; do not refit/impute/scale using held-out data"},
        "isolation_and_order": [
            "Before test content access, recheck all frozen protocol, selection, checkpoint, code, tensor-manifest, split, membership-byte and label-manifest hashes.",
            "Verify 73 eligible observations and 61 frozen assignment units against the already sealed membership digest; do not include/expose truth to inference workers.",
            "Construct a truth-free ordered membership view for inference. The worker can read only identifiers, unit IDs and tensors; it cannot read the sealed label manifest or class labels.",
            "Run every one of the 38 targets once in the frozen target order, on the same ordered membership, then write and SHA-256 seal all prediction files and their manifest.",
            "After all prediction hashes are immutable, release the frozen truth to a separate scorer exactly once; never expose truth back to prediction workers.",
            "Compute the declared metrics, summaries, bootstrap, randomization and Holm corrections once; do not act on interim scores."
        ],
        "metrics_and_aggregation": {"primary": ["balanced_accuracy", "macro_f1", "accuracy"],
                                    "also": ["per_class_precision_recall_f1_support", "confusion_matrix_true_rows_predicted_columns", "object_level_prediction_records", "FOV_and_extent_stratified_metrics"],
                                    "per_seed": "report every network seed separately for each architecture and representation",
                                    "family_summary": "for each architecture × representation, report the three seeds and mean ± sample SD (ddof=1); never select best seed",
                                    "metadata": "report both fixed metadata/FOV controls separately as descriptive controls; no averaging with neural families",
                                    "metric_conventions": protocol["statistical_analysis"]["metric_conventions"],
                                    "FOV_bins_arcmin": ["5 exactly", "(5,10]", "(10,20]", "(20,40]", "(40,80]", "(80,120]"],
                                    "extent_pixel_bins": ["<2", "[2,8)", "[8,32)", "[32,128)", ">=128", "unknown"],
                                    "empty_or_degenerate_strata": "report zero/undefined support explicitly; do not omit or merge bins after viewing results"},
        "paired_statistics": {"unit": "61 frozen v2G assignment units; all objects in a unit stay together",
                              "bootstrap": {"generator": "NumPy PCG64", "seed": 20261021, "replicates": 20000,
                                            "interval": "two-sided 95% percentile; linear quantile",
                                            "same_draws_across": "all paired models, representations, metrics and seeds",
                                            "estimand": "seed-level paired metric differences and mean of the three matched-seed differences; conditional on frozen checkpoints"},
                              "randomization": {"generator": "NumPy PCG64", "seed": 20261021, "replicates": 20000,
                                                "unit": "assignment unit; jointly swap model predictions for every object in sampled unit and matched seeds",
                                                "two_sided_p": "(1 + count(replicates at least as extreme))/(20001)",
                                                "same_swaps_across": "related models, representations, metrics and seeds"},
                              "multiplicity": {"method": "Holm step-down, alpha 0.05",
                                               "families": ["9 hypotheses: three architecture pairs × three primary metrics on A",
                                                            "27 hypotheses: three architecture pairs × three metrics across B/A_outer/B_outer",
                                                            "27 hypotheses for each frozen network control representation versus A, as declared in v2I"],
                                               "intervals": "marginal percentile intervals; not simultaneous after Holm",
                                               "no_t_test_n3": True},
                              "metadata_control_comparisons": "paired assignment-unit bootstrap differences versus each A architecture; descriptive only as frozen in v2I"},
        "prediction_schema": {"one_file_per_target": True,
                              "required_fields_per_row": ["ordered_position", "physical_object_id", "split_unit_id", "predicted_class", "four_class_probabilities"],
                              "truth_in_prediction_record": False,
                              "membership_order_shared_by_all_38_targets": True,
                              "record_identity": ["target_id", "family/control_id", "representation", "seed", "configuration_id", "checkpoint_or_artifact_sha256", "protocol_sha256", "split_digest", "preprocessing_digest", "environment_digest"],
                              "sealing": "SHA-256 each target prediction file and canonical manifest before scorer receives truth"},
        "required_future_outputs": ["execution_manifest.json", "membership_hash_receipt.json", "sealed_truth_receipt.json",
                                    "predictions/<target_id>.json × 38", "prediction_manifest.json", "per_seed_metrics.csv",
                                    "family_representation_summary.csv", "metadata_control_metrics.csv", "per_class_metrics.csv",
                                    "confusion_matrices.json", "paired_comparisons.csv", "bootstrap_intervals.csv",
                                    "randomization_tests.csv", "fov_extent_stratified_metrics.csv", "validation.json",
                                    "REPORT.md", "file_inventory.json"],
        "stopping_rules": [
            "Any protocol, digest, checkpoint, code, environment, split, membership, tensor-manifest or sealed-label hash mismatch: stop before decoding/inference.",
            "Any missing/extra/duplicate observation, assignment unit, checkpoint, model control, or unequal ordered membership across targets: stop before inference.",
            "Any class-order, tensor shape/range, preprocessing, metadata-feature, deterministic-runtime or truth-isolation mismatch: stop before inference.",
            "Any nonfinite probability, malformed probability row, inference exception, partial prediction, hash mismatch or non-deterministic output: preserve partial evidence and stop; do not retry or repeat a completed target under this plan.",
            "Any metric/scoring discrepancy, support mismatch, confusion-matrix total mismatch, or analysis implementation/hash mismatch: stop scoring and preserve sealed predictions/truth separation; no altered analysis or rerun without separate authorization."
        ],
        "scientific_limits": ["No test outcome may trigger retraining, replacement, tuning, representation choice, seed selection, threshold changes, or preprocessing changes.",
                              "Report all target results, including failures, degeneracies and low-support strata.",
                              "Inference uncertainty is conditional on frozen checkpoints; seeds are not an inferential n=3 sample.",
                              "The inference/statistical unit is the frozen assignment unit, not each FITS image or treating 73 objects as independent.",
                              "No broad astronomical population generalization claim follows from this single frozen held-out collection."],
        "code_and_execution_gates": {"preprocessing_and_model_source_hashes_recorded": True,
                                      "future_inference_and_scoring_source_hashes_must_be_frozen_before_access": True,
                                      "test_evaluator_code_created_in_this_plan_phase": False,
                                      "test_evaluator_code_sha256": None,
                                      "test_scoring_code_created_in_this_plan_phase": False,
                                      "test_scoring_code_sha256": None,
                                      "no_network_or_installation": True},
        "preservation": {"v2J1_inventory_verified": j1_preservation, "v2J2_inventory_verified": j2_preservation,
                         "no_v2J1_or_v2J2_write_performed": True},
        "plan_bindings_sha256": digest(bindings),
        "plan_created_without_test_content": True,
    }
    plan["canonical_sha256"] = digest(plan)

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "evaluation_targets.json").write_text(json.dumps(target_manifest, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    (OUT / "binding_hashes.json").write_text(json.dumps(bindings, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    (OUT / "evaluation_plan.json").write_text(json.dumps(plan, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    validation = {"phase": "Dataset-v2K-plan", "status": "passed_plan_only", "target_count_network": 36,
                  "target_count_metadata": 2, "target_count_total": 38, "unique_network_checkpoint_hashes": 36,
                  "test_membership_rows_opened": False, "test_labels_opened": False, "test_tensors_opened": False,
                  "inference_performed": False, "scoring_performed": False,
                  "v2J1_inventory_entries_verified": j1_preservation["entries_verified"],
                  "v2J2_inventory_entries_verified": j2_preservation["entries_verified"],
                  "evaluation_executor_hash_pending_future_preaccess_gate": True,
                  "plan_canonical_sha256": plan["canonical_sha256"],
                  "target_manifest_canonical_sha256": target_manifest["canonical_sha256"]}
    validation["canonical_sha256"] = digest(validation)
    (OUT / "validation.json").write_text(json.dumps(validation, sort_keys=True, indent=2, ensure_ascii=False) + "\n")

    target_lines = "\n".join(f"| {x['family']} | {x['representation']} | {x['seed']} | `{x['configuration_id']}` | `{x['checkpoint_sha256']}` |" for x in network_targets)
    metadata_lines = "\n".join(f"| {x['control_id']} | `{x['artifact_sha256']}` |" for x in metadata_targets)
    report = f"""# Dataset-v2K — plano versionado da avaliação final

## Estado

Plano `v2K-plan-01` congelado, ainda sem autorização para executar a avaliação. Nenhum membership individual, tensor ou rótulo de teste foi desserializado; não houve inferência nem scoring. v2J1 e v2J2 foram verificados contra seus inventários congelados ({j1_preservation['entries_verified']} e {j2_preservation['entries_verified']} arquivos listados, respectivamente) e não foram modificados.

## Alvos congelados

Serão avaliados exatamente 36 checkpoints neurais: nove vencedores A da seleção R4 e 27 controles B/A_outer/B_outer do v2J2, com três seeds por arquitetura e representação. Somam-se os dois artefatos de controle metadata/FOV previstos em v2I: 38 avaliações totais, cada uma sobre as mesmas 73 observações e 61 unidades congeladas. Os 36 checkpoints têm hashes distintos e foram verificados contra os respectivos manifestos.

| Arquitetura | Representação | Seed | Configuração | SHA-256 do checkpoint |
|---|---|---:|---|---|
{target_lines}

Controles metadata/FOV congelados:

| Controle | SHA-256 do artefato |
|---|---|
{metadata_lines}

Manifesto completo com caminhos, tamanhos, configurações e hashes: `evaluation_targets.json`.

## Passagem, isolamento e análise

O plano exige uma única passagem: uma avaliação por alvo × 73 observações (2.628 pares modelo/objeto para redes e 146 para controles metadata). Não há treino, otimização, threshold, seleção por resultado ou repetição de avaliação concluída. A ordem de membership será congelada sem truth; workers recebem somente IDs de objeto/unidade e tensores. Os 38 arquivos de predição, com probabilidades de quatro classes e identidades ordenadas, serão hasheados antes de liberar truth ao scorer isolado.

Serão reportados balanced accuracy, macro-F1 e accuracy por seed; métricas por classe, matrizes de confusão e estratos FOV/extensão; média ± SD amostral (ddof=1) dos três seeds por arquitetura × representação; e os dois controles metadata separadamente. Comparações pareadas, intervalos bootstrap e randomização seguirão literalmente o protocolo v2I: 61 unidades inteiras, PCG64 seed 20261021, 20.000 replicatas compartilhadas e Holm. Não haverá teste t com n=3 seeds nem escolha do melhor seed.

## Integridade e parada

`binding_hashes.json` registra hashes dos protocolos, seleções, manifestos, membership (somente SHA-256 dos bytes), truth manifest (somente SHA-256 dos bytes), checkpoints e código. Antes de qualquer conteúdo de teste, a futura execução terá de congelar e auditar os hashes do executor de inferência e do scorer, hoje ainda inexistentes. Qualquer divergência de identidade, hash, membership, ambiente, preprocessing, probabilidade, selagem ou scoring interrompe a etapa; nenhuma correção ou repetição automática é permitida.

Digest canônico do plano: `{plan['canonical_sha256']}`. Digest canônico do manifesto de alvos: `{target_manifest['canonical_sha256']}`.
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")

    git = subprocess.run(["git", "status", "--short", "--untracked-files=normal", "--", OUT.relative_to(ROOT).as_posix()],
                         text=True, capture_output=True, check=True)
    (OUT / "final_git_status.txt").write_text("Restricted final status for v2K plan directory\n" + git.stdout)

    inventory_files = []
    for path in sorted(x for x in OUT.rglob("*") if x.is_file() and x.name != "file_inventory.json"):
        inventory_files.append({"path": path.relative_to(ROOT).as_posix(), "size_bytes": path.stat().st_size,
                                "sha256": sha_file(path)})
    inventory = {"phase": "Dataset-v2K-plan", "file_count_excluding_inventory": len(inventory_files),
                 "files": inventory_files}
    inventory["canonical_sha256"] = digest(inventory)
    (OUT / "file_inventory.json").write_text(json.dumps(inventory, sort_keys=True, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"status": "plan_frozen_execution_not_authorized", "network_targets": 36,
                      "metadata_controls": 2, "total_evaluations": 38,
                      "plan_canonical_sha256": plan["canonical_sha256"],
                      "target_manifest_canonical_sha256": target_manifest["canonical_sha256"],
                      "inventory_canonical_sha256": inventory["canonical_sha256"],
                      "test_content_accessed": False, "inference_or_scoring": False,
                      "preserved_j1_files": j1_preservation["entries_verified"],
                      "preserved_j2_files": j2_preservation["entries_verified"]}, sort_keys=True))


if __name__ == "__main__":
    main()
