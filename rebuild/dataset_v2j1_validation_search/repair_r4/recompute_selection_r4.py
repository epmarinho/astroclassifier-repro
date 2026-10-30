"""R4: select from frozen fit summaries; never load data or recompute metrics."""
import argparse,csv,hashlib,json,math,statistics
from collections import defaultdict
from pathlib import Path

ROOT=Path.cwd().resolve();OUT=ROOT/"rebuild/dataset_v2j1_validation_search";V1=ROOT/"rebuild/dataset_v2i_training_protocol";R4=OUT/"repair_r4"
PROTOCOL_SHA="6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737"
SEEDS=[3908274,3908275,3908276];FAMILIES=["custom_cnn_transformer","resnet50","vit_b_16"]

def sha(b):return hashlib.sha256(b).hexdigest()
def file_sha(p):
 h=hashlib.sha256()
 with Path(p).open("rb") as f:
  for x in iter(lambda:f.read(1<<20),b""):h.update(x)
 return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(x):return sha(canonical(x))
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n")
protocol=read(V1/"protocol.json")
if protocol["canonical_sha256"]!=PROTOCOL_SHA:raise RuntimeError("Protocol digest changed")
candidates=read(V1/"candidate_manifest.json");slots=candidates["mandatory_search_fits"]
if len(slots)!=90:raise RuntimeError("Search budget mismatch")
if len(list((OUT/"fit_attempts").glob("*/attempt_03/fit_complete.json")))!=90:raise RuntimeError("Not all 90 completed fit records exist")
metadata=read(OUT/"metadata_controls.json")
if metadata.get("status")!="completed" or len(metadata.get("models",[]))!=2 or metadata.get("test_paths_or_labels_accessed") is not False:raise RuntimeError("Two frozen metadata controls are not valid")
all_records=[];by_family_config=defaultdict(list);parameters_by_family_config={};fit_record_hashes={}
for index,slot in enumerate(slots):
 d=OUT/"fit_attempts"/slot["fit_id"]/"attempt_03";complete=read(d/"fit_complete.json")
 receipt=read(d/"configuration.json")
 if complete.get("status")!="completed" or complete.get("manifest_index")!=index or complete.get("fit_id")!=slot["fit_id"]:raise RuntimeError("Fit record does not match candidate manifest order")
 if complete.get("representation")!="A" or complete.get("candidate_configuration_sha256")!=slot["configuration_sha256"] or complete.get("protocol_canonical_sha256")!=PROTOCOL_SHA:raise RuntimeError("Fit identity differs from frozen declaration")
 if complete["complete_record_sha256"]!=digest({k:v for k,v in complete.items() if k!="complete_record_sha256"}):raise RuntimeError("Fit completion canonical digest mismatch")
 if complete.get("candidate_manifest_sha256")!=file_sha(V1/"candidate_manifest.json"):raise RuntimeError("Candidate manifest digest differs from frozen fit receipt")
 if complete.get("configuration_sha256")!=digest(receipt) or complete["artifacts"].get("configuration.json")!=file_sha(d/"configuration.json"):raise RuntimeError("Configuration receipt hash mismatch")
 frozen_slot=receipt.get("slot")
 if frozen_slot!=slot:raise RuntimeError("Per-fit configuration slot differs from the frozen candidate declaration")
 if frozen_slot.get("configuration_sha256")!=digest(frozen_slot["effective_parameters"]):raise RuntimeError("Effective-parameter digest mismatch")
 if complete.get("best_checkpoint_sha256")!=complete["artifacts"].get("best.pt") or complete.get("history_sha256")!=complete["artifacts"].get("history.json") or complete.get("validation_prediction_sha256")!=complete["artifacts"].get("best_validation_predictions.json"):raise RuntimeError("Frozen result artifact hashes disagree")
 metrics=complete.get("validation_metrics",{})
 for k in ["accuracy","balanced_accuracy","macro_f1","weighted_validation_loss"]:
  if k not in metrics or not math.isfinite(float(metrics[k])):raise RuntimeError(f"Missing or nonfinite frozen metric {slot['fit_id']}/{k}")
 if not 0<=metrics["accuracy"]<=1 or not 0<=metrics["balanced_accuracy"]<=1 or not 0<=metrics["macro_f1"]<=1 or metrics["weighted_validation_loss"]<0:raise RuntimeError("Frozen metric outside its valid range")
 if not (d/"best.pt").is_file():raise RuntimeError("Selected checkpoint file is missing")
 key=(slot["family"],slot["configuration_id"]);params=frozen_slot["effective_parameters"]
 if key in parameters_by_family_config and canonical(parameters_by_family_config[key])!=canonical(params):raise RuntimeError("Effective parameters differ across frozen seeds")
 parameters_by_family_config[key]=params
 record={"manifest_index":index,"fit_id":slot["fit_id"],"family":slot["family"],"configuration_id":slot["configuration_id"],
         "configuration_sha256":slot["configuration_sha256"],"seed":slot["seed"],"representation":"A",
         "effective_parameters":params,"best_epoch":complete["best_epoch_zero_based"]+1,"epochs_completed":complete["epochs_completed"],
         **{k:metrics[k] for k in ["accuracy","balanced_accuracy","macro_f1","weighted_validation_loss"]},
         "best_checkpoint_path":(d/"best.pt").relative_to(ROOT).as_posix(),"best_checkpoint_sha256":complete["best_checkpoint_sha256"],
         "last_checkpoint_sha256":complete["last_checkpoint_sha256"],"history_sha256":complete["history_sha256"],
         "validation_predictions_sha256":complete["validation_prediction_sha256"],"completed_record_sha256":complete["complete_record_sha256"],
         "fit_wall_seconds":complete["fit_wall_seconds"],"gpu_seconds":complete["gpu_seconds"]}
 all_records.append(record);by_family_config[key].append(record)

selection_rule=protocol["model_selection"]["selection_rule"] if "selection_rule" in protocol["model_selection"] else "highest mean balanced accuracy; highest mean macro-F1; lowest mean weighted validation loss; lowest sample variance ddof=1 of balanced accuracy; lexicographically smallest configuration ID"
family_ranks=[];winners=[]
for fam in FAMILIES:
 groups=[]
 for (family,cid),rows in by_family_config.items():
  if family!=fam:continue
  rows=sorted(rows,key=lambda x:SEEDS.index(x["seed"]))
  if [x["seed"] for x in rows]!=SEEDS:raise RuntimeError("Candidate lacks exactly three frozen seeds")
  bas=[x["balanced_accuracy"] for x in rows];mfs=[x["macro_f1"] for x in rows];loss=[x["weighted_validation_loss"] for x in rows]
  original=next(x for x in candidates["candidate_configurations_by_family"][fam] if x["configuration_id"]==cid)
  if original["configuration_sha256"]!=rows[0]["configuration_sha256"]:raise RuntimeError("Grouped candidate digest differs from per-fit declaration")
  groups.append({"family":fam,"configuration_id":cid,"configuration_sha256":original["configuration_sha256"],
                 "effective_parameters":parameters_by_family_config[(fam,cid)],"seed_ids":SEEDS,
                 "per_seed":[{k:r[k] for k in ["seed","fit_id","accuracy","balanced_accuracy","macro_f1","weighted_validation_loss","best_epoch","best_checkpoint_path","best_checkpoint_sha256","history_sha256","validation_predictions_sha256"]} for r in rows],
                 "mean_balanced_accuracy":statistics.mean(bas),"mean_macro_f1":statistics.mean(mfs),
                 "mean_weighted_validation_loss":statistics.mean(loss),"sample_variance_balanced_accuracy_ddof1":statistics.variance(bas),
                 "sample_sd_balanced_accuracy_ddof1":statistics.stdev(bas)})
 groups.sort(key=lambda x:(-x["mean_balanced_accuracy"],-x["mean_macro_f1"],x["mean_weighted_validation_loss"],x["sample_variance_balanced_accuracy_ddof1"],x["configuration_id"]))
 for rank,g in enumerate(groups,1):g["validation_rank"]=rank
 family_ranks.append({"family":fam,"candidates":groups,"winner_configuration_id":groups[0]["configuration_id"],"winner_configuration_sha256":groups[0]["configuration_sha256"]})
 winners.append(groups[0])

# Freeze only after all candidate and metadata artifacts have passed their hash gates.
selected=[]
for win in winners:
 for seedrow in win["per_seed"]:
  d=ROOT/seedrow["best_checkpoint_path"]
  if not d.is_file():raise RuntimeError("Selected checkpoint file is missing")
  selected.append({"family":win["family"],"configuration_id":win["configuration_id"],"configuration_sha256":win["configuration_sha256"],
                   "effective_parameters":win["effective_parameters"],"seed":seedrow["seed"],"fit_id":seedrow["fit_id"],
                   "representation":"A","path":seedrow["best_checkpoint_path"],"size_bytes":d.stat().st_size,
                   "sha256":seedrow["best_checkpoint_sha256"],"history_path":str(d.parent.relative_to(ROOT)/"history.json"),
                   "history_sha256":seedrow["history_sha256"],"validation_predictions_path":str(d.parent.relative_to(ROOT)/"best_validation_predictions.json"),
                   "validation_predictions_sha256":seedrow["validation_predictions_sha256"],
                   "validation":{"accuracy":seedrow["accuracy"],"balanced_accuracy":seedrow["balanced_accuracy"],"macro_f1":seedrow["macro_f1"],"weighted_validation_loss":seedrow["weighted_validation_loss"]}})
if len(selected)!=9 or len({x["sha256"] for x in selected})!=9:raise RuntimeError("Selection must bind nine unique A checkpoints")
env=read(OUT/"environment.json");lock=read(V1/"test_lock.json")
selection_core={"phase":"Dataset-v2J1","status":"frozen","protocol_canonical_sha256":PROTOCOL_SHA,
                "candidate_manifest_sha256":file_sha(V1/"candidate_manifest.json"),"split_sha256":protocol["data_and_preprocessing"]["split_sha256"],
                "representation":"A","selection_rule":selection_rule,"families":family_ranks,
                "selected_checkpoints":selected,"class_order":protocol["data_and_preprocessing"]["class_order"],
                "preprocessing":protocol["data_and_preprocessing"]["normalization"],
                "augmentation":protocol["augmentation"],"environment_sha256":file_sha(OUT/"environment.json"),
                "test_lock_sha256":file_sha(V1/"test_lock.json"),"network_search_fit_count":90,
                "metadata_controls_fit_count":2,"test_accessed":False}
selection_core["selection_digest"]=digest(selection_core)
parser=argparse.ArgumentParser();parser.add_argument("--output",required=True);args=parser.parse_args()
target=Path(args.output).resolve()
if R4.resolve() not in target.parents:raise RuntimeError("Output must remain inside repair_r4")
write(target,selection_core)
print(json.dumps({"status":"selection_recomputed_from_frozen_metrics","winners":{x["family"]:x["winner_configuration_id"] for x in family_ranks},"selection_digest":selection_core["selection_digest"],"selected_checkpoints":len(selected)},sort_keys=True))
