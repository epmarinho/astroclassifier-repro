#!/usr/bin/env python3
"""Dataset-v2K inference-only executor. Frozen before any test-content access.

Inputs at a future authorized run: frozen protocol/target manifests, the
selected checkpoint artifacts, and one truth-free ordered input manifest that
points only to the frozen test tensors. This program has no truth/membership
label input, no metric code, and no training path.
"""
from __future__ import annotations
import argparse, hashlib, io, json, math, os, random, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
PHASE = ROOT / "rebuild/dataset_v2k_test_executor_freeze_v3"
PLAN = ROOT / "rebuild/dataset_v2k_test_evaluation_plan/evaluation_plan.json"
TARGETS = ROOT / "rebuild/dataset_v2k_test_evaluation_plan/evaluation_targets.json"
EXPECTED_PLAN = "c0fd829094c6ad02a9cd4b0099c7accec18743ef576cbedffbce96735941c231"
EXPECTED_PLAN_BYTES = "6d78ab9676b5ad7da1751c40fae596bfecfc2059c50e446342a5253fb958fe41"
EXPECTED_TARGETS = "ccb33c9a90b867bf859a1b99db29c87c3b02a8d4cc3fab2c9f48465c775c8942"
EXPECTED_TARGETS_BYTES = "4815c40d3a84c8b7b80748cc56b19fc637316104d14499e2d2abfc9fe9392370"
EXPECTED_TARGET_ID_SET = "6a53aa19c7877093f47525b7884b212431889b5450fd888de021188460d285d1"
EXPECTED_MEMBERSHIP = "ed396faf242a2ec183a7c9b5a906f46e4a0b4d373d452371b90f6fe2c266a53d"
EXPECTED_SELECTION_BYTES = "52643419e9ad8a15480ee821baf19bf655d6d9e62c374386b544b8f38fe236a2"
EXPECTED_R4_MANIFEST_BYTES = "ab9d653923ee8b42af97fa99f4901d6ecf979cd7f83791bbec61dbb3d56fbb32"
EXPECTED_J2_MANIFEST_BYTES = "14c823362d2f294fe1e6ad77542ae4ff407143694194acbfbd75cadf4ac44f56"
CLASS_ORDER = ["galaxies", "globularclusters", "nebulae", "opencluster"]
NETWORK_FAMILIES = {"custom_cnn_transformer", "resnet50", "vit_b_16"}
FORBIDDEN_INPUT_KEYS = {"label", "labels", "true_label", "truth", "target", "targets", "y", "ground_truth"}

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()

def sha_file(p):
    h = hashlib.sha256()
    with Path(p).open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()

def digest(x): return hashlib.sha256(canonical(x)).hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding="utf-8"))

def install_offline_guard():
    def deny(event,args):
        if event.startswith("socket.") or event in {"urllib.Request","http.client.connect"}:
            raise RuntimeError(f"Offline inference guard blocked {event}")
    sys.addaudithook(deny)

def reject_truth_fields(x, where="root"):
    if isinstance(x, dict):
        for k, v in x.items():
            key=str(k).lower()
            if key in FORBIDDEN_INPUT_KEYS or any(s in key for s in ("truth", "label", "ground_truth")):
                raise RuntimeError(f"forbidden truth-like input field at {where}.{k}")
            reject_truth_fields(v, f"{where}.{k}")
    elif isinstance(x, list):
        for i, v in enumerate(x): reject_truth_fields(v, f"{where}[{i}]")

def verify_plan():
    plan, targets = read_json(PLAN), read_json(TARGETS)
    if sha_file(PLAN)!=EXPECTED_PLAN_BYTES or sha_file(TARGETS)!=EXPECTED_TARGETS_BYTES:
        raise RuntimeError("Frozen plan/target byte hash mismatch")
    if plan.get("canonical_sha256") != EXPECTED_PLAN or targets.get("canonical_sha256") != EXPECTED_TARGETS:
        raise RuntimeError("Frozen plan/target digest mismatch")
    if digest({k:v for k,v in plan.items() if k!="canonical_sha256"})!=EXPECTED_PLAN or digest({k:v for k,v in targets.items() if k!="canonical_sha256"})!=EXPECTED_TARGETS:
        raise RuntimeError("Frozen plan/target canonical digest recomputation failed")
    net, meta = targets.get("network_targets", []), targets.get("metadata_targets", [])
    if len(net) != 36 or len(meta) != 2 or len(net) + len(meta) != 38:
        raise RuntimeError("Frozen target count is not exactly 38")
    if len({x["checkpoint_sha256"] for x in net}) != 36:
        raise RuntimeError("Network checkpoint identities are not 36 unique hashes")
    target_ids=sorted([x["fit_id"] for x in net]+[x["control_id"] for x in meta])
    if hashlib.sha256(canonical(target_ids)).hexdigest()!=EXPECTED_TARGET_ID_SET:
        raise RuntimeError("Exact frozen 38-target identity mismatch")
    if any(x.get("representation") not in {"A", "B", "A_outer", "B_outer"} for x in net):
        raise RuntimeError("Unexpected network representation")
    if any(x.get("family") not in NETWORK_FAMILIES for x in net):
        raise RuntimeError("Unexpected network family")
    combos={(x["family"],x["representation"],x["seed"]) for x in net}
    expected={(f,r,s) for f in NETWORK_FAMILIES for r in ("A","B","A_outer","B_outer") for s in (3908274,3908275,3908276)}
    if combos!=expected or len(combos)!=36: raise RuntimeError("Network target matrix is not exactly 3×4×3")
    if {x["control_id"] for x in meta}!={"metadata_decision_tree_depth3","metadata_logreg_C1_balanced_lbfgs"}:
        raise RuntimeError("Frozen metadata target identities changed")
    contract=plan["inference_contract"]
    if contract["batch_size"]!=16 or contract["channels"]!=["blue","red","NIR"] or contract["class_order"]!=CLASS_ORDER or contract["augmentation"]!="identity only; no D4, random transforms, TTA, crop, threshold or fitted transform":
        raise RuntimeError("Frozen inference contract mismatch")
    return plan, targets

def verify_release(args):
    if not args.authorize_test_access or args.authorized_plan_sha256!=EXPECTED_PLAN:
        raise RuntimeError("Explicit test-access authorization flag/digest is required")
    if args.authorized_inference_executor_sha256!=sha_file(Path(__file__)):
        raise RuntimeError("Inference release receipt is not bound to this executor")
    if args.authorized_scoring_executor_sha256!=sha_file(PHASE/"scoring_executor.py"):
        raise RuntimeError("Inference/scoring executor pair hash mismatch")
    lock=PHASE/"dependency_lock.json"
    if not lock.is_file() or args.authorized_dependency_lock_sha256!=sha_file(lock):
        raise RuntimeError("Missing/mismatched local dependency lock")

def verify_dependency_lock():
    import importlib.metadata
    lock=read_json(PHASE/"dependency_lock.json")
    for row in lock["source_dependencies"]:
        p=Path(row["path"])
        if not p.is_file() or sha_file(p)!=row["sha256"]: raise RuntimeError(f"Frozen source dependency changed: {p}")
    if sys.version.split()[0]!=lock["python"]["version"] or sha_file(sys.executable)!=lock["python"]["executable_sha256"]:
        raise RuntimeError("Frozen Python executable identity mismatch")
    for name,entry in lock["packages"].items():
        dist=importlib.metadata.distribution(name)
        record=Path(dist._path)/"RECORD"
        if dist.version!=entry["version"] or str(record)!=entry["record_path"] or sha_file(record)!=entry["record_sha256"]:
            raise RuntimeError(f"Frozen dependency identity mismatch: {name}")

def verify_all_artifact_hashes(targets):
    # Hash every frozen model artifact before any input-manifest or tensor read.
    for t in targets["network_targets"]:
        p = (ROOT / t["checkpoint_path"]).resolve(strict=True)
        if not p.is_relative_to(ROOT) or p.stat().st_size != t["checkpoint_size_bytes"] or sha_file(p) != t["checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint identity mismatch: {t['fit_id']}")
    for t in targets["metadata_targets"]:
        p = (ROOT / "rebuild/dataset_v2j1_validation_search" / t["artifact_path"]).resolve(strict=True)
        if not p.is_relative_to(ROOT) or p.stat().st_size != t["artifact_size_bytes"] or sha_file(p) != t["artifact_sha256"]:
            raise RuntimeError(f"metadata model identity mismatch: {t['control_id']}")

def verify_selected_manifests(targets):
    selection_path=ROOT/"rebuild/dataset_v2j1_validation_search/repair_r4/frozen_selection_r4.json"
    r4_path=ROOT/"rebuild/dataset_v2j1_validation_search/repair_r4/selected_checkpoint_manifest_r4.json"
    j2_path=ROOT/"rebuild/dataset_v2j2_control_fits/control_checkpoint_manifest.json"
    if sha_file(selection_path)!=EXPECTED_SELECTION_BYTES or sha_file(r4_path)!=EXPECTED_R4_MANIFEST_BYTES or sha_file(j2_path)!=EXPECTED_J2_MANIFEST_BYTES:
        raise RuntimeError("Frozen selection/checkpoint manifest byte identity mismatch")
    selection=read_json(selection_path);r4=read_json(r4_path);j2=read_json(j2_path)
    if selection.get("selection_digest")!="cd45b0c9fbb1e6310b31410c5f3be5edf9e694a7737f17a6c95c0788381a580b" or r4.get("canonical_sha256")!="ec3ef679ef9fa20504e83aa091b4b3e0a859f4db12e860380f1f2da695f723e7" or j2.get("canonical_sha256")!="542fd4f42dc116c4806e3ef5ff96c0ca0abf2c25c4340bdaadda3bae87fba162":
        raise RuntimeError("Frozen selection/checkpoint manifest internal identity mismatch")
    effective={}
    for row in r4.get("checkpoints",[])+j2.get("checkpoints",[]):
        fid=row.get("fit_id")
        if not isinstance(fid,str) or fid in effective or not isinstance(row.get("effective_parameters"),dict):
            raise RuntimeError("Malformed/duplicate selected configuration manifest row")
        effective[fid]=row
    for t in targets["network_targets"]:
        m=effective.get(t["fit_id"])
        manifest_path=m.get("checkpoint_path",m.get("path")) if m else None
        manifest_hash=m.get("checkpoint_sha256",m.get("sha256")) if m else None
        manifest_size=m.get("checkpoint_size_bytes",m.get("size_bytes")) if m else None
        if m is None or m.get("configuration_id")!=t["configuration_id"] or m.get("configuration_sha256")!=t["configuration_sha256"] or manifest_hash!=t["checkpoint_sha256"] or manifest_path!=t["checkpoint_path"] or manifest_size!=t["checkpoint_size_bytes"] or m.get("representation")!=t["representation"] or m.get("seed")!=t["seed"]:
            raise RuntimeError(f"Target differs from frozen selected manifest: {t['fit_id']}")
    return {k:v["effective_parameters"] for k,v in effective.items()}

def validate_input_manifest(d, input_path):
    reject_truth_fields(d)
    if d.get("schema") != "v2k_truth_free_input_v1": raise RuntimeError("Input manifest schema mismatch")
    if d.get("membership_digest") != EXPECTED_MEMBERSHIP: raise RuntimeError("Frozen membership digest mismatch")
    rows = d.get("observations")
    if not isinstance(rows, list) or len(rows) != 73: raise RuntimeError("Expected exactly 73 input observations")
    ids = [r.get("physical_object_id") for r in rows]
    units = [r.get("split_unit_id") for r in rows]
    if any(not isinstance(x, str) or not x for x in ids + units) or len(set(ids)) != 73 or len(set(units)) != 61:
        raise RuntimeError("Malformed/duplicate frozen observation or assignment-unit identity")
    membership_rows=[]
    for r in rows:
        if not isinstance(r.get("path"),str) or not isinstance(r.get("sha256"),str): raise RuntimeError("Missing frozen primary membership identity")
        membership_rows.append({"physical_object_id":r["physical_object_id"],"split_unit_id":r["split_unit_id"],"path":r["path"],"sha256":r["sha256"]})
        if r["path"]!=r.get("tensors",{}).get("A",{}).get("path") or r["sha256"]!=r.get("tensors",{}).get("A",{}).get("sha256"):
            raise RuntimeError("Primary A tensor does not match the frozen membership identity")
    if digest(membership_rows)!=EXPECTED_MEMBERSHIP: raise RuntimeError("Exact frozen ordered test membership digest mismatch")
    if d.get("ordered_object_ids_sha256") != hashlib.sha256(canonical(ids)).hexdigest():
        raise RuntimeError("Ordered object identity digest mismatch")
    for r in rows:
        if set(r.get("tensors", {})) != {"A", "B", "A_outer", "B_outer"}:
            raise RuntimeError("Each observation must provide exactly the four frozen representations")
        if set(r.get("metadata_features", {})) != {
            "log_source_fov_arcmin", "log_source_pixel_scale_arcsec", "log_recorded_catalogue_extent_arcmin",
            "catalogue_extent_missing", "source_fov_capped"}:
            raise RuntimeError("Frozen metadata feature schema mismatch")
        mf=r["metadata_features"]
        for k in ("log_source_fov_arcmin","log_source_pixel_scale_arcsec"):
            if not isinstance(mf[k],(int,float)) or not math.isfinite(float(mf[k])): raise RuntimeError("Nonfinite frozen metadata feature")
        if mf["catalogue_extent_missing"] not in (0,1) or mf["source_fov_capped"] not in (0,1): raise RuntimeError("Invalid frozen metadata indicator")
        extent=mf["log_recorded_catalogue_extent_arcmin"]
        if mf["catalogue_extent_missing"]==0 and (not isinstance(extent,(int,float)) or not math.isfinite(float(extent))):
            raise RuntimeError("Required frozen catalogue extent feature is invalid")
        if mf["catalogue_extent_missing"]==1 and extent is not None and isinstance(extent,(int,float)) and math.isfinite(float(extent)):
            raise RuntimeError("Missingness indicator conflicts with frozen extent feature")
    return rows

def resolve_tensor(rows, rep, i, tensors_root):
    r = rows[i]["tensors"][rep]
    p = (ROOT / r["path"]).resolve(strict=True)
    root = tensors_root.resolve(strict=True)
    if not p.is_relative_to(root):
        raise RuntimeError(f"tensor path/hash mismatch at observation index {i}, representation {rep}")
    return p

def prehash_all_tensors(rows,tensors_root):
    import numpy as np
    root=Path(tensors_root).resolve(strict=True);verified={};seen=set()
    for i,row in enumerate(rows):
        for rep in ("A","B","A_outer","B_outer"):
            r=row["tensors"][rep];p=(ROOT/r["path"]).resolve(strict=True)
            if not p.is_relative_to(root) or p in seen or p.stat().st_size!=r["size_bytes"]:
                raise RuntimeError(f"Frozen test tensor identity/path mismatch at row={i}, representation={rep}")
            if p.suffix!=".npy": raise RuntimeError("Frozen tensor product must be a NumPy tensor file")
            raw=p.read_bytes()
            if len(raw)!=r["size_bytes"] or hashlib.sha256(raw).hexdigest()!=r["sha256"]:
                raise RuntimeError(f"Frozen test tensor byte hash mismatch at row={i}, representation={rep}")
            array=np.load(io.BytesIO(raw),allow_pickle=False)
            if array.shape!=(3,224,224) or array.dtype!=np.float32 or not np.isfinite(array).all() or float(array.min()) < -1.0 or float(array.max()) > 1.0:
                raise RuntimeError(f"Frozen tensor value contract mismatch at row={i}, representation={rep}")
            seen.add(p);verified[(i,rep)]=np.ascontiguousarray(array)
    if len(verified)!=292 or len(seen)!=292: raise RuntimeError("Expected four unique frozen tensors for each of 73 observations")
    return verified

def make_network(target, effective, protocol):
    # Imports happen only after a future external authorization invokes this CLI.
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import models
    fam = target["family"]
    spec = effective["model"]
    if fam == "custom_cnn_transformer":
        source=ROOT/"rebuild/phase3d/models_v3.py"
        if sha_file(source)!="e304b8ada45aef6143ce8acec4e3d1c441a6350d478927d7f68d4706cdf4a82a":
            raise RuntimeError("Frozen custom architecture source hash mismatch")
        from rebuild.phase3d.models_v3 import model_factory
        return model_factory(json.loads(json.dumps(spec)), 4)
    if torchvision.__version__ != "0.22.0+cu128": raise RuntimeError("Frozen torchvision version mismatch")
    source_name="resnet.py" if fam=="resnet50" else "vision_transformer.py"
    source=(Path(torchvision.__file__).resolve().parent/"models"/source_name)
    expected="7499a5059ad7b1a1f8f7543c04b4a13795140fad837c38530b48ff5d9610bf6e" if fam=="resnet50" else "3b899d063605b29f874d903d6d77eec7fc09cc83d1bf5b92e373e3b8d879db37"
    if sha_file(source)!=expected: raise RuntimeError("Frozen torchvision architecture source hash mismatch")
    if fam == "resnet50": net = models.resnet50(weights=None)
    elif fam == "vit_b_16": net = models.vit_b_16(weights=None)
    else: raise RuntimeError("Unknown family")
    if fam == "resnet50": net.fc = nn.Linear(net.fc.in_features, 4)
    else: net.heads.head = nn.Linear(net.heads.head.in_features, 4)
    # The frozen J1/J2 BaselineModel wrapper prefixes all baseline state_dict
    # keys with "network.". Recreate that inference-only state layout exactly.
    class FrozenBaseline(nn.Module):
        def __init__(self, network, family):
            super().__init__(); self.network=network; self.family=family
        def forward(self, x): return self.network(x)
    return FrozenBaseline(net,fam)

def run_network(target, effective, protocol, rows, tensors_root, verified_tensors):
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["OMP_NUM_THREADS"] = "2"
    os.environ["MKL_NUM_THREADS"] = "2"
    import numpy as np
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1 or torch.cuda.get_device_name(0)!="NVIDIA GeForce RTX 5060 Ti":
        raise RuntimeError("Frozen CUDA device unavailable or identity differs; no CPU fallback")
    if str(torch.__version__)!="2.7.0+cu128" or sys.version_info[:3]!=(3,12,3):
        raise RuntimeError("Frozen Python/PyTorch environment mismatch")
    if np.__version__!="1.26.4" or torch.version.cuda!="12.8": raise RuntimeError("Frozen NumPy/CUDA runtime mismatch")
    torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True; torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    if not torch.are_deterministic_algorithms_enabled() or torch.get_num_threads()!=2 or not torch.backends.cudnn.deterministic or torch.backends.cudnn.benchmark or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32:
        raise RuntimeError("Frozen deterministic CUDA settings failed")
    rep, fam = target["representation"], target["family"]
    checkpoint = (ROOT / target["checkpoint_path"]).resolve(strict=True)
    checkpoint_bytes=checkpoint.read_bytes()
    if len(checkpoint_bytes)!=target["checkpoint_size_bytes"] or hashlib.sha256(checkpoint_bytes).hexdigest()!=target["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed after pre-access hash gate")
    payload = torch.load(io.BytesIO(checkpoint_bytes), map_location="cpu", weights_only=True)
    del checkpoint_bytes
    context=payload.get("context",{})
    expected={"fit_id":target["fit_id"],"family":target["family"],"representation":rep,
              "configuration_id":target["configuration_id"],"configuration_sha256":target["configuration_sha256"],
              "seed":target["seed"],"protocol_canonical_sha256":"6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737"}
    if any(context.get(k)!=v for k,v in expected.items()) or payload.get("seed")!=target["seed"] or payload.get("configuration_sha256")!=target["configuration_sha256"]:
        raise RuntimeError("Checkpoint embedded identity differs from frozen target")
    model = make_network(target, effective, protocol)
    state = payload.get("model_state")
    if not isinstance(state, dict): raise RuntimeError("Frozen checkpoint lacks model_state")
    model.load_state_dict(state, strict=True)
    model.eval()
    device = torch.device("cuda")
    model.to(device=device, dtype=torch.float32)
    result = []
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            chunk = rows[start:start+16]
            arrays = [verified_tensors[(i,rep)] for i in range(start, min(start+16, len(rows)))]
            x = torch.from_numpy(np.stack(arrays)).to(dtype=torch.float32)
            if x.ndim != 4 or tuple(x.shape[1:]) != (3, 224, 224) or not torch.isfinite(x).all() or float(x.min()) < -1 or float(x.max()) > 1:
                raise RuntimeError("Frozen tensor contract mismatch")
            if fam == "custom_cnn_transformer":
                x = torch.stack([F.interpolate(z.unsqueeze(0), size=(256,256), mode="bilinear", align_corners=False, antialias=True).squeeze(0).contiguous() for z in x])
            logits = model(x.to(device=device, dtype=torch.float32))
            if tuple(logits.shape) != (len(chunk),4) or logits.dtype != torch.float32 or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid model output")
            probs = logits.softmax(dim=1).cpu().numpy()
            if not np.isfinite(probs).all() or not np.allclose(probs.sum(1),1.0,rtol=0,atol=1e-6): raise RuntimeError("Invalid probability row")
            preds = probs.argmax(axis=1)
            for j, row in enumerate(chunk):
                result.append({"physical_object_id":row["physical_object_id"], "split_unit_id":row["split_unit_id"],
                               "predicted_class_index":int(preds[j]), "probabilities":[float(v) for v in probs[j]]})
    del model, payload
    return result

def run_metadata(target, rows):
    import numpy as np
    import joblib
    import sklearn
    if np.__version__!="1.26.4" or joblib.__version__!="1.4.2" or sklearn.__version__!="1.4.2":
        raise RuntimeError("Frozen metadata inference environment mismatch")
    artifact = (ROOT / "rebuild/dataset_v2j1_validation_search" / target["artifact_path"]).resolve(strict=True)
    artifact_bytes=artifact.read_bytes()
    if hashlib.sha256(artifact_bytes).hexdigest()!=target["artifact_sha256"]: raise RuntimeError("Metadata artifact changed after pre-access hash gate")
    model = joblib.load(io.BytesIO(artifact_bytes))
    del artifact_bytes
    order = ["log_source_fov_arcmin", "log_source_pixel_scale_arcsec", "log_recorded_catalogue_extent_arcmin", "catalogue_extent_missing", "source_fov_capped"]
    xrows=[]
    for r in rows:
        f=r["metadata_features"]
        vals=[]
        for k in order:
            v=f[k]
            vals.append(float("nan") if k=="log_recorded_catalogue_extent_arcmin" and f["catalogue_extent_missing"]==1 and v is None else v)
        xrows.append(vals)
    x = np.asarray(xrows,dtype=np.float64)
    if not np.isfinite(x[:,[0,1,3,4]]).all(): raise RuntimeError("Invalid frozen metadata features")
    probs = model.predict_proba(x)
    if list(map(int, model.classes_)) != [0,1,2,3] or probs.shape != (73,4) or not np.isfinite(probs).all():
        raise RuntimeError("Metadata control output/class order mismatch")
    return [{"physical_object_id":r["physical_object_id"], "split_unit_id":r["split_unit_id"],
             "predicted_class_index":int(p.argmax()), "probabilities":[float(v) for v in p]} for r,p in zip(rows,probs)]

def environment_identity(network):
    import numpy, importlib.metadata
    import torch, torchvision
    obj={"python":sys.version.split()[0],"numpy":numpy.__version__,"torch":torch.__version__,
         "torchvision":torchvision.__version__,"cuda_runtime":torch.version.cuda,
         "device":torch.cuda.get_device_name(0) if network else "CPU metadata inference",
         "precision":"float32","amp":False,"batch_size":16 if network else None,"workers":0}
    if not network:
        obj["scikit_learn"]=importlib.metadata.version("scikit-learn")
        obj["joblib"]=importlib.metadata.version("joblib")
    return obj,digest(obj)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--input-manifest",required=True,help="future truth-free ordered input manifest")
    ap.add_argument("--tensor-root",required=True,help="future root containing only allowlisted frozen test tensors")
    ap.add_argument("--output-dir",required=True,help="must be a new directory below this phase")
    ap.add_argument("--authorize-test-access",action="store_true",help="requires separate future authorization")
    ap.add_argument("--authorized-plan-sha256",required=True)
    ap.add_argument("--authorized-inference-executor-sha256",required=True)
    ap.add_argument("--authorized-scoring-executor-sha256",required=True)
    ap.add_argument("--authorized-dependency-lock-sha256",required=True)
    a=ap.parse_args()
    verify_release(a)
    plan, targets=verify_plan(); verify_dependency_lock(); target_effective=verify_selected_manifests(targets); verify_all_artifact_hashes(targets)
    install_offline_guard()
    input_path=Path(a.input_manifest).resolve(strict=True)
    if not input_path.is_relative_to(PHASE): raise RuntimeError("Truth-free input manifest must be versioned inside the executor phase")
    tensors_root=Path(a.tensor_root).resolve(strict=True)
    if tensors_root!=(ROOT/"rebuild/dataset_v2h_representation_materialization").resolve():
        raise RuntimeError("Tensor root differs from frozen v2H materialization directory")
    d=read_json(input_path); rows=validate_input_manifest(d,input_path)
    input_manifest_sha=sha_file(input_path)
    verified_tensors=prehash_all_tensors(rows,tensors_root)
    out=Path(a.output_dir).resolve()
    if not out.is_relative_to(PHASE) or out.exists(): raise RuntimeError("Output must be a new directory under executor phase")
    if any(PHASE.rglob("prediction_manifest.json")) or (PHASE/"inference.lock").exists():
        raise RuntimeError("A v2K inference attempt already exists; never repeat or resume this one-pass evaluation")
    # Persistent exclusive lock: any later failure requires a new authorization
    # and a new versioned phase; this executor never retries or resumes.
    with (PHASE/"inference.lock").open("x",encoding="utf-8") as lock:
        lock.write(json.dumps({"executor_sha256":sha_file(Path(__file__)),"plan_sha256":EXPECTED_PLAN,"one_pass_lock":True})+"\n")
    out.mkdir(parents=True)
    protocol_path=ROOT/"rebuild/dataset_v2i_training_protocol/protocol.json"
    if sha_file(protocol_path)!="fcbc29993adf83c86f5f4cab445e8e61bb88cd312f390fdfa253c37dad8d2b22": raise RuntimeError("v2I protocol byte hash mismatch")
    protocol=read_json(protocol_path)
    if protocol.get("canonical_sha256") != "6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737" or digest({k:v for k,v in protocol.items() if k!="canonical_sha256"})!="6a4593c9e470f8eb61819a13d8fc82e74a4bfb662b5bb22ce9ca221d179b3737":
        raise RuntimeError("Frozen v2I protocol identity mismatch")
    preprocessing_digest=digest(plan["inference_contract"])
    protocol_digest=protocol["canonical_sha256"]
    ordered_membership_identity=hashlib.sha256(canonical([{"physical_object_id":r["physical_object_id"],"split_unit_id":r["split_unit_id"]} for r in rows])).hexdigest()
    produced=[]
    for t in targets["network_targets"]:
        ep=target_effective.get(t["fit_id"])
        if not isinstance(ep,dict): raise RuntimeError("Missing exact effective parameters for target")
        preds=run_network(t,ep,protocol,rows,tensors_root,verified_tensors); env,env_digest=environment_identity(True)
        for i,pred in enumerate(preds):
            pred["ordered_position"]=i; pred["predicted_class"]=CLASS_ORDER[pred["predicted_class_index"]]
        item={"schema":"v2k_prediction_record_v1","target_id":t["fit_id"],"family":t["family"],"representation":t["representation"],"seed":t["seed"],
              "configuration_id":t["configuration_id"],"checkpoint_sha256":t["checkpoint_sha256"],"protocol_sha256":protocol_digest,
              "split_digest":"1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf","preprocessing_digest":preprocessing_digest,
              "environment_digest":env_digest,"environment":env,"ordered_membership_digest":EXPECTED_MEMBERSHIP,
              "ordered_membership_identity_sha256":ordered_membership_identity,"ordered_object_ids_sha256":hashlib.sha256(canonical([r["physical_object_id"] for r in rows])).hexdigest(),"predictions":preds}
        raw=canonical(item)+b"\n"; p=out/(t["fit_id"]+".json"); p.write_bytes(raw)
        produced.append({"target_id":t["fit_id"],"path":p.name,"sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":len(raw)})
    for t in targets["metadata_targets"]:
        preds=run_metadata(t,rows); env,env_digest=environment_identity(False)
        for i,pred in enumerate(preds):
            pred["ordered_position"]=i; pred["predicted_class"]=CLASS_ORDER[pred["predicted_class_index"]]
        item={"schema":"v2k_prediction_record_v1","target_id":t["control_id"],"family":t["control_id"],"representation":"metadata_FOV_only","seed":t["seed"],
              "configuration_id":t["control_id"],"checkpoint_sha256":t["artifact_sha256"],"protocol_sha256":protocol_digest,
              "split_digest":"1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf","preprocessing_digest":preprocessing_digest,
              "environment_digest":env_digest,"environment":env,"ordered_membership_digest":EXPECTED_MEMBERSHIP,
              "ordered_membership_identity_sha256":ordered_membership_identity,"ordered_object_ids_sha256":hashlib.sha256(canonical([r["physical_object_id"] for r in rows])).hexdigest(),"predictions":preds}
        raw=canonical(item)+b"\n"; p=out/(t["control_id"]+".json"); p.write_bytes(raw)
        produced.append({"target_id":t["control_id"],"path":p.name,"sha256":hashlib.sha256(raw).hexdigest(),"size_bytes":len(raw)})
    if len(produced)!=38 or len({x["target_id"] for x in produced})!=38: raise RuntimeError("Prediction target set incomplete")
    manifest={"schema":"v2k_prediction_seal_v1","target_count":38,"membership_digest":EXPECTED_MEMBERSHIP,
              "ordered_observations":73,"ordered_object_ids_sha256":hashlib.sha256(canonical([r["physical_object_id"] for r in rows])).hexdigest(),
              "target_records":produced,"scoring_not_performed":True}
    manifest["canonical_sha256"]=digest(manifest)
    (out/"prediction_manifest.json").write_bytes(canonical(manifest)+b"\n")
    (out/"membership_hash_receipt.json").write_bytes(canonical({"membership_digest":EXPECTED_MEMBERSHIP,"ordered_membership_identity_sha256":ordered_membership_identity,
      "ordered_test_observations":73,"assignment_units":61,"truth_fields_in_worker":False,"truth_source_opened":False,
      "truth_free_input_manifest_sha256":input_manifest_sha})+b"\n")
    (out/"execution_manifest.json").write_bytes(canonical({"phase":"Dataset-v2K","plan_canonical_sha256":EXPECTED_PLAN,"protocol_canonical_sha256":protocol_digest,
      "split_digest":"1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf","preprocessing_digest":preprocessing_digest,
      "ordered_membership_digest":EXPECTED_MEMBERSHIP,"target_count":38,"network_checkpoint_evaluations":36,"metadata_control_evaluations":2,
      "observations_per_target":73,"prediction_manifest_sha256":sha_file(out/"prediction_manifest.json"),
      "inference_executor_sha256":sha_file(Path(__file__)),"scoring_not_started":True,"test_truth_opened":False})+b"\n")

if __name__=="__main__": main()
