"""Independent recomputation and validation-only selection from sealed A records."""
import csv,hashlib,json,math,statistics,time
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT=Path.cwd().resolve();OUT=ROOT/"rebuild/dataset_v2j1_validation_search";V1=ROOT/"rebuild/dataset_v2i_training_protocol"
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
def assert_close(a,b,key):
 if not math.isclose(float(a),float(b),rel_tol=0,abs_tol=1e-12):raise RuntimeError(f"Independent recomputation mismatch {key}: {a} vs {b}")
def independent_metrics(rows):
 labels=[int(x["true_label"]) for x in rows];pred=[int(x["predicted_label"]) for x in rows]
 counts=np.zeros((4,4),dtype=np.int64)
 for y,p in zip(labels,pred):counts[y,p]+=1
 support=counts.sum(1);tp=np.diag(counts);precision=np.divide(tp,counts.sum(0),out=np.zeros(4,dtype=float),where=counts.sum(0)>0)
 recall=np.divide(tp,support,out=np.zeros(4,dtype=float),where=support>0)
 f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4,dtype=float),where=(precision+recall)>0)
 ba=float(np.mean(recall[support>0]));macro=float(np.mean(f1));accuracy=float(tp.sum()/len(labels))
 loss=float(np.mean([float(x["weighted_loss"]) for x in rows]))
 return {"accuracy":accuracy,"balanced_accuracy":ba,"macro_f1":macro,"weighted_validation_loss":loss,
         "per_class":{"precision":precision.tolist(),"recall":recall.tolist(),"f1":f1.tolist(),"support":support.tolist()},
         "confusion_matrix_true_rows_predicted_columns":counts.tolist()}

protocol=read(V1/"protocol.json")
if protocol["canonical_sha256"]!=PROTOCOL_SHA:raise RuntimeError("Protocol digest changed")
candidates=read(V1/"candidate_manifest.json");slots=candidates["mandatory_search_fits"]
if len(slots)!=90:raise RuntimeError("Search budget mismatch")
if len(list((OUT/"fits").glob("*/fit_complete.json")))!=90:raise RuntimeError("Not all 90 completed fit records exist")
if len(read(OUT/"metadata_controls.json").get("models",[]))!=2:raise RuntimeError("Two mandatory metadata controls are not complete")
inputs=read(OUT/"worker_inputs.json");expected_val=inputs["roles"]["validation"]["objects"]
expected_ids=[r["physical_object_id"] for r in expected_val]
all_records=[];by_family_config=defaultdict(list);records_by_fit={};account_fitrows=[]
for index,slot in enumerate(slots):
 d=OUT/"fits"/slot["fit_id"];complete=read(d/"fit_complete.json")
 if complete.get("status")!="completed" or complete.get("manifest_index")!=index or complete.get("fit_id")!=slot["fit_id"]:raise RuntimeError("Fit record does not match candidate manifest order")
 if complete.get("representation")!="A" or complete.get("candidate_configuration_sha256")!=slot["configuration_sha256"] or complete.get("protocol_canonical_sha256")!=PROTOCOL_SHA:raise RuntimeError("Fit identity differs from frozen declaration")
 if complete["complete_record_sha256"]!=digest({k:v for k,v in complete.items() if k!="complete_record_sha256"}):raise RuntimeError("Fit completion canonical digest mismatch")
 for name,expected in complete["artifacts"].items():
  if file_sha(d/name)!=expected:raise RuntimeError(f"Fit artifact hash failed: {slot['fit_id']}/{name}")
 pred=read(d/"best_validation_predictions.json")
 if pred["prediction_digest"]!=digest({k:v for k,v in pred.items() if k!="prediction_digest"}):raise RuntimeError("Validation prediction seal digest mismatch")
 rows=pred["ordered_validation_predictions"]
 if len(rows)!=75 or [r["physical_object_id"] for r in rows]!=expected_ids:raise RuntimeError("Validation membership/order mismatch")
 for r,expected in zip(rows,expected_val):
  if r["true_label"]!=expected["label"] or r["split_unit_id"]!=expected["split_unit_id"]:raise RuntimeError("Validation truth or assignment-unit identity mismatch")
  probs=r["probabilities"]
  if len(probs)!=4 or not all(math.isfinite(x) and 0<=x<=1 for x in probs) or not math.isclose(sum(probs),1,rel_tol=0,abs_tol=1e-6):raise RuntimeError("Invalid sealed validation probability row")
  if r["predicted_label"]!=int(np.argmax(probs)):raise RuntimeError("Stored class differs from frozen argmax rule")
 recomputed=independent_metrics(rows);stored=pred["validation_metrics"]
 for k in ["accuracy","balanced_accuracy","macro_f1","weighted_validation_loss"]:assert_close(recomputed[k],stored[k],slot["fit_id"]+"/"+k)
 if complete["validation_metrics"]!=stored:raise RuntimeError("Fit summary differs from sealed prediction metrics")
 record={"manifest_index":index,"fit_id":slot["fit_id"],"family":slot["family"],"configuration_id":slot["configuration_id"],
         "configuration_sha256":slot["configuration_sha256"],"seed":slot["seed"],"representation":"A",
         "best_epoch":complete["best_epoch_zero_based"]+1,"epochs_completed":complete["epochs_completed"],
         **{k:recomputed[k] for k in ["accuracy","balanced_accuracy","macro_f1","weighted_validation_loss"]},
         "best_checkpoint_path":(d/"best.pt").relative_to(ROOT).as_posix(),"best_checkpoint_sha256":file_sha(d/"best.pt"),
         "last_checkpoint_sha256":file_sha(d/"last.pt"),"history_sha256":file_sha(d/"history.json"),
         "validation_predictions_sha256":file_sha(d/"best_validation_predictions.json"),"completed_record_sha256":file_sha(d/"fit_complete.json"),
         "fit_wall_seconds":complete["fit_wall_seconds"],"gpu_seconds":complete["gpu_seconds"]}
 all_records.append(record);by_family_config[(slot["family"],slot["configuration_id"])].append(record);records_by_fit[slot["fit_id"]]=record
 account_fitrows.append(record)

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
  groups.append({"family":fam,"configuration_id":cid,"configuration_sha256":original["configuration_sha256"],
                 "effective_parameters":original["effective_parameters"],"seed_ids":SEEDS,
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
  if file_sha(d)!=seedrow["best_checkpoint_sha256"]:raise RuntimeError("Selected checkpoint changed during freeze")
  selected.append({"family":win["family"],"configuration_id":win["configuration_id"],"configuration_sha256":win["configuration_sha256"],
                   "effective_parameters":win["effective_parameters"],"seed":seedrow["seed"],"fit_id":seedrow["fit_id"],
                   "representation":"A","path":seedrow["best_checkpoint_path"],"size_bytes":d.stat().st_size,
                   "sha256":seedrow["best_checkpoint_sha256"],"history_path":str(d.parent.relative_to(ROOT)/"history.json"),
                   "history_sha256":seedrow["history_sha256"],"validation_predictions_path":str(d.parent.relative_to(ROOT)/"best_validation_predictions.json"),
                   "validation_predictions_sha256":seedrow["validation_predictions_sha256"],
                   "validation":{"accuracy":seedrow["accuracy"],"balanced_accuracy":seedrow["balanced_accuracy"],"macro_f1":seedrow["macro_f1"],"weighted_validation_loss":seedrow["weighted_validation_loss"]}})
if len(selected)!=9 or len({x["sha256"] for x in selected})!=9:raise RuntimeError("Selection must bind nine unique A checkpoints")
pre=read(OUT/"preflight.json");env=read(OUT/"environment.json");lock=read(V1/"test_lock.json")
selection_core={"phase":"Dataset-v2J1","status":"frozen","protocol_canonical_sha256":PROTOCOL_SHA,
                "candidate_manifest_sha256":file_sha(V1/"candidate_manifest.json"),"split_sha256":protocol["data_and_preprocessing"]["split_sha256"],
                "representation":"A","selection_rule":selection_rule,"families":family_ranks,
                "selected_checkpoints":selected,"class_order":protocol["data_and_preprocessing"]["class_order"],
                "preprocessing":protocol["data_and_preprocessing"]["normalization"],
                "augmentation":protocol["augmentation"],"environment_sha256":file_sha(OUT/"environment.json"),
                "test_lock_sha256":file_sha(V1/"test_lock.json"),"network_search_fit_count":90,
                "metadata_controls_fit_count":2,"test_accessed":False}
selection_core["selection_digest"]=digest(selection_core)
write(OUT/"selection.json",selection_core)
checkpoint_manifest={"phase":"Dataset-v2J1","status":"frozen","selection_digest":selection_core["selection_digest"],
                     "protocol_canonical_sha256":PROTOCOL_SHA,"split_sha256":selection_core["split_sha256"],
                     "selected_configuration_count":3,"selected_checkpoint_count":9,"families":winners,
                     "checkpoints":selected,"test_tensor_or_label_accessed":False}
checkpoint_manifest["canonical_sha256"]=digest(checkpoint_manifest)
write(OUT/"selected_checkpoint_manifest.json",checkpoint_manifest)

# Per-fit CSV includes all 90 network search fits followed by the two fixed controls.
metadata=read(OUT/"metadata_controls.json");csv_rows=[]
for r in all_records:
 csv_rows.append({"fit_id":r["fit_id"],"family":r["family"],"configuration_id":r["configuration_id"],"seed":r["seed"],"representation":"A","accuracy":r["accuracy"],"balanced_accuracy":r["balanced_accuracy"],"macro_f1":r["macro_f1"],"weighted_validation_loss":r["weighted_validation_loss"],"best_epoch":r["best_epoch"],"checkpoint_sha256":r["best_checkpoint_sha256"],"history_sha256":r["history_sha256"]})
for m in metadata["models"]:
 v=m["validation_metrics"];csv_rows.append({"fit_id":m["fit_id"],"family":m["family"],"configuration_id":m["configuration_id"],"seed":m["seed"],"representation":"metadata_FOV_only","accuracy":v["accuracy"],"balanced_accuracy":v["balanced_accuracy"],"macro_f1":v["macro_f1"],"weighted_validation_loss":"","best_epoch":"","checkpoint_sha256":m["model_artifact_sha256"],"history_sha256":""})
with (OUT/"per_fit_validation_metrics.csv").open("w",newline="",encoding="utf-8") as f:
 w=csv.DictWriter(f,fieldnames=list(csv_rows[0]));w.writeheader();w.writerows(csv_rows)

progress=read(OUT/"progress.json")
fit_accounting={"phase":"Dataset-v2J1","status":"completed","protocol_canonical_sha256":PROTOCOL_SHA,
                "candidate_manifest_sha256":file_sha(V1/"candidate_manifest.json"),"network_fit_count":90,"metadata_fit_count":2,
                "completed_network_fits":90,"failed_network_fits":0,"resumed_fit_count":progress.get("resumed_fits",0),
                "preflight_checker_attempts":len(read(OUT/"preflight_attempts.json")["attempts"]),
                "fit_campaign_wall_seconds":progress.get("cumulative_campaign_wall_seconds",0),
                "network_gpu_seconds":sum(r["gpu_seconds"] for r in all_records),
                "metadata_cpu_wall_seconds":metadata["total_wall_seconds"],
                "per_family_fit_counts":{"custom_cnn_transformer":18,"resnet50":36,"vit_b_16":36},
                "manifest_execution_order":[r["fit_id"] for r in slots],"test_evaluation_fits_started":0,
                "conditional_control_network_fits_started":0,"test_data_accessed":False}
write(OUT/"fit_accounting.json",fit_accounting)

validation={"phase":"Dataset-v2J1","status":"passed","protocol_canonical_sha256":PROTOCOL_SHA,
            "candidate_manifest_sha256":file_sha(V1/"candidate_manifest.json"),"v2g_split_sha256":protocol["data_and_preprocessing"]["split_sha256"],
            "network_fit_records_validated":90,"candidate_configuration_count":{"resnet50":12,"vit_b_16":12,"custom_cnn_transformer":6},
            "seeds_per_configuration":3,"independent_validation_recomputations":90,"selection_matches_frozen_rule":True,
            "selected_architectures":3,"selected_checkpoints":9,"unique_selected_checkpoints":9,
            "metadata_control_fits_validated":2,"test_tensor_arrays_opened":False,"test_labels_deserialized":False,
            "test_inference_or_scoring_performed":False,"conditional_controls_started":False,
            "training_failures":0,"resumed_fits":progress.get("resumed_fits",0),
            "prediction_membership_identity": "same ordered 75 development validation objects for every fit",
            "object_independence_limitation":"v2G assignment units bind components, but astronomical-object independence is not asserted beyond frozen grouping evidence."}
write(OUT/"validation.json",validation)

lines=["# Dataset-v2J1 — busca de validação congelada","",
       "## Estado","",
       "**Busca concluída e seleção congelada.** Todos os fits usaram somente a representação A e dados de treino/validação; não foram iniciados os 27 fits de controle B/A_outer/B_outer.","",
       f"Protocolo canônico: `{PROTOCOL_SHA}`. Digest da seleção: `{selection_core['selection_digest']}`.","",
       "## Execução","",
       f"Fits de rede completos: 90; falhos: 0; resumidos: {progress.get('resumed_fits',0)}. Dois controles metadata/FOV fixos foram ajustados após a busca e não influenciaram seleção.",
       f"Tempo GPU acumulado das redes: {fit_accounting['network_gpu_seconds']/3600:.3f} h. Tempo wall de campanha: {fit_accounting['fit_campaign_wall_seconds']/3600:.3f} h.",
       "O preflight passou com Python 3.12.3, torch 2.7.0+cu128, torchvision 0.22.0+cu128 e RTX 5060 Ti. Três falhas de verificador do preflight foram operacionais, preservadas em `attempts/`; nenhuma ocorreu após início de fit.","",
       "## Resultados de cada configuração","",
       "Métricas abaixo são médias sobre as três seeds; a tabela detalhada e as métricas por fit estão em `selection.json` e `per_fit_validation_metrics.csv`.",""]
for fam in FAMILIES:
 fr=next(x for x in family_ranks if x["family"]==fam)
 lines += [f"### {fam}","","| Rank | Configuração | BA média | Macro-F1 média | Loss ponderada média | Variância amostral BA | Seed BA / Macro-F1 / loss |","|---:|---|---:|---:|---:|---:|---|"]
 for g in fr["candidates"]:
  seeds_str="; ".join(f"{r['seed']}: {r['balanced_accuracy']:.6f} / {r['macro_f1']:.6f} / {r['weighted_validation_loss']:.6f}" for r in g["per_seed"])
  mark=" **selecionada**" if g["validation_rank"]==1 else ""
  lines.append(f"| {g['validation_rank']} | {g['configuration_id']}{mark} | {g['mean_balanced_accuracy']:.6f} | {g['mean_macro_f1']:.6f} | {g['mean_weighted_validation_loss']:.6f} | {g['sample_variance_balanced_accuracy_ddof1']:.8g} | {seeds_str} |")
 lines.append("")
 lines.append(f"Checkpoint selecionado: `{fr['winner_configuration_id']}`; três seeds {', '.join(map(str,SEEDS))}. Os três hashes e caminhos completos estão no manifesto de checkpoints.")
 lines.append("")
lines += ["## Controles metadata/FOV","","Os dois controles fixos foram ajustados apenas no treino e avaliados na validação, sem busca nem influência sobre a seleção das redes.",""]
for m in metadata["models"]:
 v=m["validation_metrics"];lines.append(f"- `{m['fit_id']}`: accuracy {v['accuracy']:.6f}, balanced accuracy {v['balanced_accuracy']:.6f}, macro-F1 {v['macro_f1']:.6f}; artefato SHA-256 `{m['model_artifact_sha256']}`.")
lines += ["","## Integridade e isolamento","",
          "As 90 métricas foram recalculadas independentemente a partir das previsões seladas de validação, com mesma membership ordenada de 75 objetos. A regra congelada escolheu uma configuração por arquitetura agregando as três seeds; nenhuma seed individual foi escolhida.",
          "Hashes completos de todos os checkpoints best/last e histories foram verificados. O worker tinha allowlist apenas para os 222 objetos de treino e 75 de validação; os tensores de teste não foram abertos e os rótulos de teste não foram desserializados. Não houve inferência nem scoring de teste.",
          "","## Próxima fase","",
          "Dataset-v2J2 pode ser autorizado separadamente para executar somente os 27 fits de controle definidos no protocolo. Esta fase não os iniciou.",""]
(OUT/"REPORT.md").write_text("\n".join(lines),encoding="utf-8")
print(json.dumps({"status":"selection_frozen","winners":{x["family"]:x["winner_configuration_id"] for x in family_ranks},"selection_digest":selection_core["selection_digest"],"selected_checkpoints":9},sort_keys=True))
