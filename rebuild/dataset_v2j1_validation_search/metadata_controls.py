"""Two fixed train/validation-only metadata/FOV controls; no network or GPU use."""
import hashlib,json,math,time,warnings
from pathlib import Path
import numpy as np
from sklearn.compose import TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,balanced_accuracy_score,confusion_matrix,f1_score,precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,export_text
import joblib

ROOT=Path.cwd();OUT=ROOT/"rebuild/dataset_v2j1_validation_search";G=ROOT/"rebuild/dataset_v2g_split_materialization"
CLASS_ORDER=["galaxies","globularclusters","nebulae","opencluster"]

def sha(b):return hashlib.sha256(b).hexdigest()
def sha_file(p):return sha(Path(p).read_bytes())
def canonical(x):return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def load_role(role):
 if role not in {"train","validation"}:raise ValueError("Metadata controls only accept training and validation")
 p=G/"splits"/f"{role}.json"
 d=json.loads(p.read_text())
 if d.get("role")!=role:raise ValueError("Frozen metadata split role mismatch")
 rows=d["objects"]
 if not rows:raise ValueError("Empty role")
 return p,d,rows

def features(rows):
 result=[]
 for o in rows:
  fov=float(o["FOV_arcmin"])
  if not math.isfinite(fov) or fov<=0:raise ValueError("Invalid frozen FOV")
  extent=o.get("catalogue_extent_arcmin")
  missing=extent is None or not math.isfinite(float(extent)) or float(extent)<=0
  if missing:log_extent=float("nan")
  else:log_extent=math.log(float(extent))
  pixel_scale=math.degrees(2*math.tan(math.radians(fov/60.0)/2.0))/512.0*3600.0
  capped=bool((not missing) and float(extent)>60.0)
  result.append([math.log(fov),math.log(pixel_scale),log_extent,float(missing),float(capped)])
 return np.asarray(result,dtype=np.float64)

def metrics(y,pred):
 pr,re,ff,s=precision_recall_fscore_support(y,pred,labels=list(range(4)),zero_division=0)
 return {"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),
         "macro_f1":float(f1_score(y,pred,labels=list(range(4)),average="macro",zero_division=0)),
         "per_class":{"precision":pr.tolist(),"recall":re.tolist(),"f1":ff.tolist(),"support":s.tolist()},
         "confusion_matrix_true_rows_predicted_columns":confusion_matrix(y,pred,labels=list(range(4))).tolist()}

train_path,train_data,train_rows=load_role("train");val_path,val_data,val_rows=load_role("validation")
if len(train_rows)!=222 or len(val_rows)!=75:raise RuntimeError("Frozen metadata control role counts mismatch")
if {x["physical_object_id"] for x in train_rows}&{x["physical_object_id"] for x in val_rows}:raise RuntimeError("Metadata train/validation overlap")
Xtr=features(train_rows);Xva=features(val_rows)
ytr=np.asarray([CLASS_ORDER.index(x["frozen_class"]) for x in train_rows],dtype=np.int64)
yva=np.asarray([CLASS_ORDER.index(x["frozen_class"]) for x in val_rows],dtype=np.int64)
base_steps=[("median_imputer",SimpleImputer(strategy="median")),("standard_scaler",StandardScaler())]
models=[
 ("metadata_logreg_C1_balanced_lbfgs",LogisticRegression(C=1,class_weight="balanced",max_iter=1000,solver="lbfgs")),
 ("metadata_decision_tree_depth3",DecisionTreeClassifier(max_depth=3,random_state=3908274)),
]
rows=[];start=time.monotonic();outdir=OUT/"metadata_controls";outdir.mkdir(parents=True,exist_ok=True)
for name,estimator in models:
 pipe=Pipeline([*base_steps,("classifier",estimator)])
 with warnings.catch_warnings(record=True) as caught:
  warnings.simplefilter("always");pipe.fit(Xtr,ytr)
 pred=pipe.predict(Xva).astype(int).tolist();m=metrics(yva,pred)
 model_path=outdir/f"{name}.joblib";joblib.dump(pipe,model_path,compress=0)
 warning_text=[str(w.message) for w in caught]
 row={"fit_id":name,"family":name,"configuration_id":name,"representation":"metadata_FOV_only","seed":None if "logreg" in name else 3908274,
      "train_objects":222,"validation_objects":75,"features":["log(source_FOV_arcmin)","log(source_pixel_scale_arcsec)","log(recorded_catalogue_extent_arcmin)","catalogue_extent_missing","source_FOV_capped"],
      "train_split_manifest_sha256":sha_file(train_path),"validation_split_manifest_sha256":sha_file(val_path),
      "fit_preprocessing":"Median imputation and centering/scaling fit on training objects only; no held-out fitting.",
      "validation_metrics":m,"validation_predictions":[{"physical_object_id":r["physical_object_id"],"split_unit_id":r["split_unit_id"],"true_label":int(y),"predicted_label":int(pr)} for r,y,pr in zip(val_rows,yva,pred)],
      "warnings":warning_text,"logistic_iterations":pipe.named_steps["classifier"].n_iter_.tolist() if "logreg" in name else None,
      "decision_tree_text":export_text(pipe.named_steps["classifier"],feature_names=["log_FOV","log_pixel_scale","log_extent","extent_missing","FOV_capped"]) if "tree" in name else None,
      "model_artifact":model_path.relative_to(OUT).as_posix(),"model_artifact_sha256":sha_file(model_path),"model_artifact_size_bytes":model_path.stat().st_size,
      "test_data_accessed":False}
 row["record_sha256"]=sha(canonical(row));rows.append(row)
record={"phase":"Dataset-v2J1","status":"completed","fit_count":2,"models":rows,
        "total_wall_seconds":time.monotonic()-start,"frozen_rule":"fixed v2F models; no tuning; no influence on network configuration selection",
        "test_paths_or_labels_accessed":False}
(OUT/"metadata_controls.json").write_text(json.dumps(record,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n")
print(json.dumps({"status":"completed","fit_count":2,"results":[{"id":r["fit_id"],**r["validation_metrics"]} for r in rows]},sort_keys=True))
