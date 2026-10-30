#!/usr/bin/env python3
"""Dataset-v2K scoring-only executor; no model, checkpoint, or tensor imports."""
from __future__ import annotations
import argparse, hashlib, json, math, sys
import csv
from pathlib import Path
import numpy as np

if np.__version__!="1.26.4": raise RuntimeError("Frozen NumPy scoring dependency mismatch")

CLASS_ORDER = ["galaxies", "globularclusters", "nebulae", "opencluster"]
PHASE = Path(__file__).resolve().parent
EXPECTED_MEMBERSHIP = "ed396faf242a2ec183a7c9b5a906f46e4a0b4d373d452371b90f6fe2c266a53d"
EXPECTED_PROTOCOL = "6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737"
EXPECTED_SPLIT = "1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf"
EXPECTED_PREPROCESSING = "a213a5c01d3e8d31ad31197f365b888d1b01aea297a658947c72fbd4c435bce7"
EXPECTED_TARGET_ID_SET_SHA256 = "6a53aa19c7877093f47525b7884b212431889b5450fd888de021188460d285d1"
EXPECTED_TARGET_METADATA_SHA256 = "96defa7f1b45237846a88381f7134124c5a2757f30e8076a19396e5a30526438"
N_BOOT = 20000
N_RANDOM = 20000
SEED = 20261021
FORBIDDEN_MODEL_FIELDS = {"checkpoint_path", "model_path", "tensor_path", "tensor_paths", "weights_path"}

def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def install_offline_guard():
    def deny(event,args):
        if event.startswith("socket.") or event in {"urllib.Request","http.client.connect"}:
            raise RuntimeError(f"Offline scoring guard blocked {event}")
    sys.addaudithook(deny)
def write_csv(path,rows,fields):
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader()
        for row in rows:w.writerow({k:(json.dumps(row[k],sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False) if isinstance(row.get(k),(dict,list)) else row.get(k)) for k in fields})
def sha_file(p):
    h=hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""):h.update(b)
    return h.hexdigest()
def reject_model_refs(x,where="root"):
    if isinstance(x,dict):
        for k,v in x.items():
            if str(k).lower() in FORBIDDEN_MODEL_FIELDS: raise RuntimeError(f"model/tensor field prohibited in scorer input: {where}.{k}")
            reject_model_refs(v,f"{where}.{k}")
    elif isinstance(x,list):
        for i,v in enumerate(x): reject_model_refs(v,f"{where}[{i}]")

def validate_membership_rows(rows,expected_digest=EXPECTED_MEMBERSHIP):
    if not isinstance(rows,list) or len(rows)!=73: raise RuntimeError("Scoring membership must contain exactly 73 observations")
    ids=[r.get("physical_object_id") for r in rows]
    units=[r.get("split_unit_id") for r in rows]
    if any(not isinstance(x,str) or not x for x in ids+units) or len(set(ids))!=73 or len(set(units))!=61:
        raise RuntimeError("Scoring membership object/unit identities are malformed")
    membership_rows=[]
    for r in rows:
        if not isinstance(r.get("path"),str) or not isinstance(r.get("sha256"),str): raise RuntimeError("Scoring membership lacks frozen primary path/hash")
        membership_rows.append({"physical_object_id":r["physical_object_id"],"split_unit_id":r["split_unit_id"],"path":r["path"],"sha256":r["sha256"]})
    if hashlib.sha256(canonical(membership_rows)).hexdigest()!=expected_digest:
        raise RuntimeError("Scoring membership does not reproduce the frozen ordered membership digest")
    return ids,units

def metric_bundle(y,p):
    y=np.asarray(y,dtype=np.int64);p=np.asarray(p,dtype=np.int64)
    if y.ndim!=1 or p.shape!=y.shape or not len(y) or np.any((y<0)|(y>3)) or np.any((p<0)|(p>3)): raise ValueError("invalid metric vectors")
    cm=np.zeros((4,4),dtype=np.int64)
    np.add.at(cm,(y,p),1)
    support=cm.sum(1);predicted=cm.sum(0);tp=np.diag(cm)
    recall=np.divide(tp,support,out=np.zeros(4,dtype=float),where=support>0)
    precision=np.divide(tp,predicted,out=np.zeros(4,dtype=float),where=predicted>0)
    f1=np.divide(2*precision*recall,precision+recall,out=np.zeros(4,dtype=float),where=(precision+recall)>0)
    present=support>0
    return {"accuracy":float(tp.sum()/len(y)),"balanced_accuracy":float(recall[present].mean()) if present.any() else 0.0,
            "macro_f1":float(f1.mean()),"per_class":[{"class":CLASS_ORDER[i],"precision":float(precision[i]),"recall":float(recall[i]),"f1":float(f1[i]),"support":int(support[i])} for i in range(4)],
            "confusion_matrix_true_rows_predicted_columns":cm.tolist(),"n":int(len(y)),"undefined_recall_classes":[CLASS_ORDER[i] for i in range(4) if support[i]==0]}

def target_key(record): return (record["family"],record["representation"],record.get("seed"))
def metric_delta(y,p1,p2):
    a=metric_bundle(y,p1);b=metric_bundle(y,p2)
    return {k:a[k]-b[k] for k in ("accuracy","balanced_accuracy","macro_f1")}

def expand_units(unit_order, unit_index_draws, y, p1, p2):
    ix=np.concatenate([unit_index_draws[u] for u in unit_order])
    return y[ix],p1[ix],p2[ix]

def percentile_pair(values): return [float(np.quantile(values,.025,method="linear")),float(np.quantile(values,.975,method="linear"))]

def paired_cluster_analysis(records_by_seed, y, units, bootstrap_draws, random_swaps):
    # records_by_seed: three aligned (p_left,p_right) pairs; draws retain all
    # objects of each sampled assignment unit and are shared across seeds.
    unit_order=list(dict.fromkeys(units)); groups={u:np.flatnonzero(np.asarray(units)==u) for u in unit_order}
    seed_diffs=[metric_delta(y,a,b) for a,b in records_by_seed]
    observed={m:float(np.mean([d[m] for d in seed_diffs])) for m in seed_diffs[0]}
    boot={m:np.empty(N_BOOT,dtype=np.float64) for m in observed}
    boot_seed=[{m:np.empty(N_BOOT,dtype=np.float64) for m in observed} for _ in records_by_seed]
    random_extreme={m:0 for m in observed}
    for bi,draw in enumerate(bootstrap_draws):
        sampled=[unit_order[i] for i in draw]
        per_seed=[]
        for si,(a,b) in enumerate(records_by_seed):
            yy,p1,p2=expand_units(sampled,groups,y,a,b)
            per_seed.append(metric_delta(yy,p1,p2))
            for m in boot:boot_seed[si][m][bi]=per_seed[-1][m]
        for m in boot:boot[m][bi]=float(np.mean([d[m] for d in per_seed]))
    # Joint assignment-unit swaps; for each sampled unit, the same swap applies
    # to all three matched seeds and all its object rows.
    for ri,swap in enumerate(random_swaps):
        row_swap=np.zeros(len(units),dtype=bool)
        for ui,u in enumerate(unit_order): row_swap[groups[u]]=swap[ui]
        vals=[]
        for a,b in records_by_seed:
            aa=np.where(row_swap,b,a);bb=np.where(row_swap,a,b)
            vals.append({m:metric_delta(y,aa,bb)[m] for m in observed})
        for m in observed:
            if abs(float(np.mean([v[m] for v in vals]))) >= abs(observed[m])-1e-15: random_extreme[m]+=1
    result={m:{"mean_matched_seed_difference":observed[m],"bootstrap_95_percentile":percentile_pair(boot[m]),
               "randomization_p_two_sided":(1+random_extreme[m])/(N_RANDOM+1)} for m in observed}
    for si,diff in enumerate(seed_diffs):
        for m in observed:
            result[m].setdefault("seed_level",[]).append({"seed":(3908274,3908275,3908276)[si],"difference":diff[m],"bootstrap_95_percentile":percentile_pair(boot_seed[si][m])})
    return result,seed_diffs

def holm(pvalues,alpha=.05):
    n=len(pvalues);order=sorted(range(n),key=lambda i:(pvalues[i],i));adj=[0.0]*n;running=0.0
    for rank,i in enumerate(order):
        running=max(running,(n-rank)*pvalues[i]);adj[i]=min(1.0,running)
    return [{"raw_p":float(pvalues[i]),"holm_adjusted_p":adj[i],"reject_alpha_0_05":bool(adj[i]<=alpha)} for i in range(n)]

def load_inputs(pred_dir, membership_path, truth_path, authorized_manifest_sha256):
    pred_dir=Path(pred_dir).resolve(strict=True)
    membership_path=Path(membership_path).resolve(strict=True);truth_path=Path(truth_path).resolve(strict=True)
    if not pred_dir.is_relative_to(PHASE) or not membership_path.is_relative_to(PHASE) or not truth_path.is_relative_to(PHASE):
        raise RuntimeError("Scoring may read only phase-local sealed predictions, membership, and released truth")
    if membership_path.name!="scoring_membership.json" or truth_path.name!="released_truth.json":
        raise RuntimeError("Scoring input filenames differ from the frozen truth-release interface")
    seal_path=pred_dir/"prediction_manifest.json";seal_bytes=seal_path.read_bytes()
    if hashlib.sha256(seal_bytes).hexdigest()!=authorized_manifest_sha256: raise RuntimeError("Prediction seal changed after scorer authorization")
    seal=json.loads(seal_bytes)
    if seal.get("target_count")!=38 or len(seal.get("target_records",[]))!=38: raise RuntimeError("Prediction seal does not contain 38 targets")
    if seal.get("canonical_sha256")!=hashlib.sha256(canonical({k:v for k,v in seal.items() if k!="canonical_sha256"})).hexdigest():
        raise RuntimeError("Prediction manifest canonical seal mismatch")
    if seal.get("membership_digest")!=EXPECTED_MEMBERSHIP or seal.get("ordered_observations")!=73: raise RuntimeError("Prediction membership binding mismatch")
    predictions={}
    for item in seal["target_records"]:
        p=(pred_dir/item["path"]).resolve(strict=True)
        if not p.is_relative_to(pred_dir): raise RuntimeError("Prediction path escaped sealed output directory")
        prediction_bytes=p.read_bytes()
        if len(prediction_bytes)!=item["size_bytes"] or hashlib.sha256(prediction_bytes).hexdigest()!=item["sha256"]: raise RuntimeError("Prediction hash mismatch")
        d=json.loads(prediction_bytes); reject_model_refs(d)
        if d.get("target_id")!=item["target_id"] or d.get("ordered_membership_digest")!=EXPECTED_MEMBERSHIP or len(d.get("predictions",[]))!=73: raise RuntimeError("Malformed sealed prediction record")
        if d.get("protocol_sha256")!=EXPECTED_PROTOCOL or d.get("split_digest")!=EXPECTED_SPLIT or d.get("preprocessing_digest")!=EXPECTED_PREPROCESSING:
            raise RuntimeError("Prediction protocol/split/preprocessing identity mismatch")
        if not isinstance(d.get("environment"),dict) or hashlib.sha256(canonical(d["environment"])).hexdigest()!=d.get("environment_digest"):
            raise RuntimeError("Prediction environment identity mismatch")
        predictions[item["target_id"]]=d
    if len(predictions)!=38: raise RuntimeError("Duplicate prediction target")
    if hashlib.sha256(canonical(sorted(predictions))).hexdigest()!=EXPECTED_TARGET_ID_SET_SHA256:
        raise RuntimeError("Sealed prediction targets differ from the exact frozen target set")
    target_metadata=[]
    for target_id,d in predictions.items():
        target_metadata.append({"target_id":target_id,"family":d.get("family"),"representation":d.get("representation"),"seed":d.get("seed"),"checkpoint_sha256":d.get("checkpoint_sha256")})
    if hashlib.sha256(canonical(sorted(target_metadata,key=lambda x:x["target_id"]))).hexdigest()!=EXPECTED_TARGET_METADATA_SHA256:
        raise RuntimeError("Prediction architecture/representation/seed/checkpoint identities differ from frozen targets")
    membership_bytes=membership_path.read_bytes();truth_bytes=truth_path.read_bytes()
    members=json.loads(membership_bytes); truth=json.loads(truth_bytes)
    if members.get("membership_digest")!=EXPECTED_MEMBERSHIP or truth.get("membership_digest")!=EXPECTED_MEMBERSHIP: raise RuntimeError("Scoring membership/truth identity mismatch")
    rows=members.get("observations"); labels=truth.get("labels")
    if not isinstance(rows,list) or not isinstance(labels,dict) or len(rows)!=73: raise RuntimeError("Scoring input schema/count mismatch")
    ids,_=validate_membership_rows(rows)
    if len(set(ids))!=73 or set(ids)!=set(labels): raise RuntimeError("Truth and membership identities differ")
    if any("class" in r or "label" in r for r in rows): raise RuntimeError("Membership input must not duplicate truth")
    for oid,label in labels.items():
        if not isinstance(label,int) or label not in range(4): raise RuntimeError(f"Invalid frozen truth label for {oid}")
    ordered_ids=seal.get("ordered_object_ids_sha256")
    if hashlib.sha256(canonical(ids)).hexdigest()!=ordered_ids: raise RuntimeError("Scoring membership order differs from sealed predictions")
    membership_identity=hashlib.sha256(canonical([{"physical_object_id":r["physical_object_id"],"split_unit_id":r["split_unit_id"]} for r in rows])).hexdigest()
    aligned=[]
    for target_id,d in predictions.items():
        pr=d["predictions"]
        if [x.get("physical_object_id") for x in pr]!=ids or [x.get("ordered_position") for x in pr]!=list(range(73)):
            raise RuntimeError("Prediction order differs across targets/membership")
        if d.get("ordered_membership_identity_sha256")!=membership_identity: raise RuntimeError("Ordered membership identity binding mismatch")
        probs=np.asarray([x.get("probabilities") for x in pr],dtype=np.float64)
        if probs.shape!=(73,4) or not np.isfinite(probs).all() or np.any(probs<0) or not np.allclose(probs.sum(1),1,rtol=0,atol=1e-6): raise RuntimeError("Invalid probability matrix")
        pred=np.asarray([x.get("predicted_class_index") for x in pr],dtype=np.int64)
        if not np.array_equal(pred,probs.argmax(1)) or any(x.get("predicted_class")!=CLASS_ORDER[int(x["predicted_class_index"])] for x in pr):
            raise RuntimeError("Frozen argmax/class-order prediction mismatch")
        aligned.append((target_id,d,pred))
    receipt={"membership_file_sha256":hashlib.sha256(membership_bytes).hexdigest(),"released_truth_file_sha256":hashlib.sha256(truth_bytes).hexdigest(),
      "prediction_manifest_sha256":hashlib.sha256(seal_bytes).hexdigest(),"prediction_target_count":38,"predictions_verified_before_truth_read":True}
    return rows,np.asarray([labels[x] for x in ids],dtype=np.int64),aligned,receipt

def fov_bin(x):
    if not math.isfinite(float(x)) or float(x)<5 or float(x)>120: raise ValueError("FOV value falls outside frozen strata")
    x=float(x)
    if x==5:return "5 exactly"
    bounds=((10,"(5,10]"),(20,"(10,20]"),(40,"(20,40]"),(80,"(40,80]"),(120,"(80,120]"))
    for hi,name in bounds:
        if 0<x<=hi:return name
    return "out_of_range"
def extent_bin(x):
    if x is None:return "unknown"
    if not math.isfinite(float(x)) or float(x)<0: raise ValueError("Extent value invalid under frozen strata")
    x=float(x)
    if x<2:return "<2"
    if x<8:return "[2,8)"
    if x<32:return "[8,32)"
    if x<128:return "[32,128)"
    return ">=128"

def score(pred_dir,membership_path,truth_path,out_dir,authorized_manifest_sha256):
    rows,y,aligned,receipts=load_inputs(pred_dir,membership_path,truth_path,authorized_manifest_sha256)
    output=Path(out_dir).resolve()
    if not output.is_relative_to(PHASE) or output.exists(): raise RuntimeError("Scoring output must be new and inside the v2K executor phase")
    output.mkdir(parents=True)
    by_key={}
    per_target=[];conf={};fov_rows=[]
    for tid,d,pred in aligned:
        metrics=metric_bundle(y,pred);key=target_key(d);by_key[key]=pred
        if sum(map(sum,metrics["confusion_matrix_true_rows_predicted_columns"]))!=73 or sum(metrics["per_class"][i]["support"] for i in range(4))!=73:
            raise RuntimeError("Confusion matrix/support total differs from frozen 73 observations")
        per_target.append({"target_id":tid,"family":d["family"],"representation":d["representation"],"seed":d.get("seed"),**metrics})
        conf[tid]=metrics["confusion_matrix_true_rows_predicted_columns"]
        for attr,bin_fn,all_bins in (("source_fov_arcmin",fov_bin,["5 exactly","(5,10]","(10,20]","(20,40]","(40,80]","(80,120]"]),
                                      ("catalogue_extent_pixels",extent_bin,["<2","[2,8)","[8,32)","[32,128)",">=128","unknown"])):
            if any(attr not in r for r in rows): raise RuntimeError(f"Missing frozen stratification field {attr}")
            for b in all_bins:
                ix=[i for i,r in enumerate(rows) if bin_fn(r[attr])==b]
                fov_rows.append({"target_id":tid,"stratifier":attr,"bin":b,"support":len(ix),"metrics":metric_bundle(y[ix],pred[ix]) if ix else None})
    # Neural summaries: all seeds are retained and family means use sample SD.
    summaries=[]
    for fam in ("custom_cnn_transformer","resnet50","vit_b_16"):
      for rep in ("A","B","A_outer","B_outer"):
        rec=[r for r in per_target if r["family"]==fam and r["representation"]==rep]
        if len(rec)!=3 or sorted(x["seed"] for x in rec)!=[3908274,3908275,3908276]: raise RuntimeError(f"Incomplete frozen seed set: {fam}/{rep}")
        row={"family":fam,"representation":rep,"seed_metrics":[{k:x[k] for k in ("seed","accuracy","balanced_accuracy","macro_f1")} for x in rec]}
        for metric in ("accuracy","balanced_accuracy","macro_f1"):
            vals=np.asarray([x[metric] for x in rec]);row[metric+"_mean"]=float(vals.mean());row[metric+"_sample_sd"]=float(vals.std(ddof=1))
        summaries.append(row)
    if len(per_target)!=38: raise RuntimeError("Scorer did not consume exactly 38 targets")
    if len(summaries)!=12 or sum(x["representation"]=="metadata_FOV_only" for x in per_target)!=2:
        raise RuntimeError("Scoring architecture/representation/metadata accounting mismatch")
    # Paired architecture comparisons, matched by frozen seed and object; the
    # component bootstrap/randomization is conditional on the sealed models.
    pairs=[];rng=np.random.Generator(np.random.PCG64(SEED))
    unit_count=len(set(r["split_unit_id"] for r in rows))
    shared_bootstrap_draws=rng.integers(0,unit_count,size=(N_BOOT,unit_count),dtype=np.int16)
    shared_random_swaps=rng.integers(0,2,size=(N_RANDOM,unit_count),dtype=np.int8).astype(bool)
    for rep in ("A","B","A_outer","B_outer"):
      fams=("custom_cnn_transformer","resnet50","vit_b_16")
      for ai in range(3):
       for bi in range(ai+1,3):
        left,right=fams[ai],fams[bi]
        trip=[]
        for seed in (3908274,3908275,3908276):
            la=next(p for tid,d,p in aligned if d["family"]==left and d["representation"]==rep and d.get("seed")==seed)
            rb=next(p for tid,d,p in aligned if d["family"]==right and d["representation"]==rep and d.get("seed")==seed)
            trip.append((la,rb))
        stat,seed_diff=paired_cluster_analysis(trip,y,[r["split_unit_id"] for r in rows],shared_bootstrap_draws,shared_random_swaps)
        pairs.append({"representation":rep,"first_family":left,"second_family":right,"seed_differences":seed_diff,"family_level":stat})
    if len(pairs)!=12: raise RuntimeError("Expected twelve architecture-pair × representation comparisons")
    representation_vs_A=[]
    for rep in ("B","A_outer","B_outer"):
      for fam in ("custom_cnn_transformer","resnet50","vit_b_16"):
        trip=[]
        for seed in (3908274,3908275,3908276):
            ctrl=next(p for _,d,p in aligned if d["family"]==fam and d["representation"]==rep and d.get("seed")==seed)
            primary=next(p for _,d,p in aligned if d["family"]==fam and d["representation"]=="A" and d.get("seed")==seed)
            trip.append((ctrl,primary))
        stat,seed_diff=paired_cluster_analysis(trip,y,[r["split_unit_id"] for r in rows],shared_bootstrap_draws,shared_random_swaps)
        representation_vs_A.append({"family":fam,"control_representation":rep,"reference_representation":"A","seed_differences":seed_diff,"family_level":stat})
    if len(representation_vs_A)!=9: raise RuntimeError("Expected nine control-representation versus A comparisons")
    # Multiplicity families are exactly those frozen: A; 27 contrasts in each
    # network control representation versus A, and pooled 27 across controls.
    p_A=[next(x for x in pairs if x["representation"]=="A" and x["first_family"]==a and x["second_family"]==b)["family_level"][m]["randomization_p_two_sided"] for a,b in (("custom_cnn_transformer","resnet50"),("custom_cnn_transformer","vit_b_16"),("resnet50","vit_b_16")) for m in ("balanced_accuracy","macro_f1","accuracy")]
    correction={"A_9_hypotheses":holm(p_A)}
    control_arch_keys=[(x["representation"],x["first_family"],x["second_family"],m) for x in pairs if x["representation"] in {"B","A_outer","B_outer"} for m in ("balanced_accuracy","macro_f1","accuracy")]
    control_arch_p=[next(x for x in pairs if x["representation"]==rep and x["first_family"]==a and x["second_family"]==b)["family_level"][m]["randomization_p_two_sided"] for rep,a,b,m in control_arch_keys]
    correction["architecture_pairs_across_B_A_outer_B_outer_27_hypotheses"]=holm(control_arch_p)
    for rep in ("B","A_outer","B_outer"):
        pp2=[x["family_level"][m]["randomization_p_two_sided"] for x in representation_vs_A if x["control_representation"]==rep for m in ("balanced_accuracy","macro_f1","accuracy")]
        correction[rep+"_vs_A_27_hypotheses"]=holm(pp2)
    correction_lookup={}
    def bind_holm(family,keys):
        for key,result in zip(keys,correction[family]): correction_lookup[key]=dict(result,family=family)
    metric_order=("balanced_accuracy","macro_f1","accuracy")
    bind_holm("A_9_hypotheses",[("architecture_pair","A",a,b,m) for a,b in (("custom_cnn_transformer","resnet50"),("custom_cnn_transformer","vit_b_16"),("resnet50","vit_b_16")) for m in metric_order])
    bind_holm("architecture_pairs_across_B_A_outer_B_outer_27_hypotheses",[("architecture_pair",rep,a,b,m) for rep,a,b,m in control_arch_keys])
    for rep in ("B","A_outer","B_outer"):
        bind_holm(rep+"_vs_A_27_hypotheses",[("representation_control_vs_A",rep,x["family"],"A",m) for x in representation_vs_A if x["control_representation"]==rep for m in metric_order])
    metadata_comparisons=[]
    unit_names=list(dict.fromkeys(r["split_unit_id"] for r in rows))
    unit_groups={u:np.flatnonzero(np.asarray([r["split_unit_id"] for r in rows])==u) for u in unit_names}
    meta_targets=[(tid,d,pred) for tid,d,pred in aligned if d["representation"]=="metadata_FOV_only"]
    for mtid,md,mpred in meta_targets:
      for fam in ("custom_cnn_transformer","resnet50","vit_b_16"):
        a_preds=[next(p for _,d,p in aligned if d["family"]==fam and d["representation"]=="A" and d.get("seed")==seed) for seed in (3908274,3908275,3908276)]
        observed={m:metric_bundle(y,mpred)[m]-float(np.mean([metric_bundle(y,p)[m] for p in a_preds])) for m in ("accuracy","balanced_accuracy","macro_f1")}
        draws={m:np.empty(N_BOOT,dtype=np.float64) for m in observed}
        for bi,draw in enumerate(shared_bootstrap_draws):
            sampled=[unit_names[int(i)] for i in draw];ix=np.concatenate([unit_groups[u] for u in sampled]);yy=y[ix]
            for m in draws:draws[m][bi]=metric_bundle(yy,mpred[ix])[m]-float(np.mean([metric_bundle(yy,p[ix])[m] for p in a_preds]))
        metadata_comparisons.append({"metadata_control_id":mtid,"architecture":fam,"reference":"mean of three A seed-level metrics; no prediction ensemble","descriptive_only":True,
          "differences":{m:{"estimate":observed[m],"bootstrap_95_percentile":percentile_pair(draws[m])} for m in observed}})
    write_csv(output/"per_seed_metrics.csv",per_target,["target_id","family","representation","seed","accuracy","balanced_accuracy","macro_f1","n","undefined_recall_classes"])
    write_csv(output/"family_representation_summary.csv",summaries,["family","representation","seed_metrics","accuracy_mean","accuracy_sample_sd","balanced_accuracy_mean","balanced_accuracy_sample_sd","macro_f1_mean","macro_f1_sample_sd"])
    write_csv(output/"metadata_control_metrics.csv",[x for x in per_target if x["representation"]=="metadata_FOV_only"],["target_id","family","representation","seed","accuracy","balanced_accuracy","macro_f1","n","undefined_recall_classes"])
    per_class=[{"target_id":x["target_id"],"family":x["family"],"representation":x["representation"],"seed":x["seed"],**pc} for x in per_target for pc in x["per_class"]]
    write_csv(output/"per_class_metrics.csv",per_class,["target_id","family","representation","seed","class","precision","recall","f1","support"])
    (output/"confusion_matrices.json").write_bytes(canonical(conf)+b"\n")
    flat_strata=[{"target_id":x["target_id"],"stratifier":x["stratifier"],"bin":x["bin"],"support":x["support"],"metrics":x["metrics"]} for x in fov_rows]
    write_csv(output/"fov_extent_stratified_metrics.csv",flat_strata,["target_id","stratifier","bin","support","metrics"])
    paired_rows=[];bootstrap_rows=[];randomization_rows=[]
    for scope,coll in (("architecture_pair",pairs),("representation_control_vs_A",representation_vs_A)):
        for item in coll:
            left=item.get("first_family",item.get("family"));right=item.get("second_family",item.get("reference_representation"))
            rep=item.get("representation",item.get("control_representation"))
            for metric,stats in item["family_level"].items():
                paired_rows.append({"comparison_scope":scope,"representation":rep,"first":left,"second":right,"metric":metric,
                    "difference":stats["mean_matched_seed_difference"],"seed_differences":item["seed_differences"]})
                bootstrap_rows.append({"comparison_scope":scope,"representation":rep,"first":left,"second":right,"metric":metric,
                    "estimate":stats["mean_matched_seed_difference"],"lower_95":stats["bootstrap_95_percentile"][0],"upper_95":stats["bootstrap_95_percentile"][1],"unit":"assignment_unit_id"})
                for seed_row in stats["seed_level"]:
                    paired_rows.append({"comparison_scope":scope+"_seed_level","representation":rep,"first":left,"second":right,"metric":metric,"seed":seed_row["seed"],"difference":seed_row["difference"]})
                    bootstrap_rows.append({"comparison_scope":scope+"_seed_level","representation":rep,"first":left,"second":right,"metric":metric,"seed":seed_row["seed"],"estimate":seed_row["difference"],"lower_95":seed_row["bootstrap_95_percentile"][0],"upper_95":seed_row["bootstrap_95_percentile"][1],"unit":"assignment_unit_id"})
                randomization_rows.append({"comparison_scope":scope,"representation":rep,"first":left,"second":right,"metric":metric,
                    "raw_p":stats["randomization_p_two_sided"],**correction_lookup.get((scope,rep,left,right,metric),{"holm_adjusted_p":None,"reject_alpha_0_05":None,"family":"not_applicable"})})
    for item in metadata_comparisons:
        for metric,stats in item["differences"].items():
            paired_rows.append({"comparison_scope":"metadata_descriptive","representation":"metadata_FOV_only", "first":item["metadata_control_id"],"second":item["architecture"],"metric":metric,"difference":stats["estimate"]})
            bootstrap_rows.append({"comparison_scope":"metadata_descriptive","representation":"metadata_FOV_only", "first":item["metadata_control_id"],"second":item["architecture"],"metric":metric,"estimate":stats["estimate"],"lower_95":stats["bootstrap_95_percentile"][0],"upper_95":stats["bootstrap_95_percentile"][1],"unit":"assignment_unit_id"})
    write_csv(output/"paired_comparisons.csv",paired_rows,["comparison_scope","representation","first","second","metric","seed","difference","seed_differences"])
    write_csv(output/"bootstrap_intervals.csv",bootstrap_rows,["comparison_scope","representation","first","second","metric","seed","estimate","lower_95","upper_95","unit"])
    write_csv(output/"randomization_tests.csv",randomization_rows,["comparison_scope","representation","first","second","metric","raw_p","holm_adjusted_p","reject_alpha_0_05","family"])
    (output/"paired_statistics.json").write_bytes(canonical({"architecture_pairs":pairs,"representation_controls_vs_A":representation_vs_A,"metadata_control_descriptive_comparisons":metadata_comparisons,"holm":correction,"bootstrap_replicates":N_BOOT,"randomization_replicates":N_RANDOM,"seed":SEED,"unit":"assignment_unit_id"})+b"\n")
    (output/"sealed_truth_receipt.json").write_bytes(canonical(receipts)+b"\n")

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--predictions",required=True);ap.add_argument("--membership",required=True);ap.add_argument("--truth",required=True);ap.add_argument("--output-dir",required=True)
    ap.add_argument("--authorize-scoring",action="store_true",help="requires separate future scoring authorization")
    ap.add_argument("--authorized-scorer-sha256",required=True);ap.add_argument("--authorized_prediction_manifest_sha256",required=True)
    a=ap.parse_args()
    install_offline_guard()
    pred_dir=Path(a.predictions).resolve(strict=True)
    if not pred_dir.is_relative_to(PHASE): raise RuntimeError("Scoring predictions must be phase-local sealed outputs")
    seal_path=pred_dir/"prediction_manifest.json"
    if not a.authorize_scoring or a.authorized_scorer_sha256!=sha_file(Path(__file__)) or a.authorized_prediction_manifest_sha256!=sha_file(seal_path):
        raise RuntimeError("Missing/mismatched explicit scoring authorization and prediction-seal digest")
    output=Path(a.output_dir).resolve()
    if not output.is_relative_to(PHASE) or output.exists(): raise RuntimeError("Scoring output must be a new directory within this phase")
    if (PHASE/"scoring.lock").exists() or any(PHASE.rglob("validation.json")):
        raise RuntimeError("Scoring attempt already exists; never repeat or resume final scoring")
    with (PHASE/"scoring.lock").open("x",encoding="utf-8") as f:
        f.write(json.dumps({"scorer_sha256":sha_file(Path(__file__)),"prediction_manifest_sha256":sha_file(seal_path),"one_pass_lock":True})+"\n")
    score(a.predictions,a.membership,a.truth,a.output_dir,a.authorized_prediction_manifest_sha256)
if __name__=="__main__":main()
