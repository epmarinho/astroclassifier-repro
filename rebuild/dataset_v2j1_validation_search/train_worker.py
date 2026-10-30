"""Single-fit, train/validation-only worker for Dataset-v2J1."""
from __future__ import annotations
import argparse, csv, hashlib, json, math, os, random, resource, shutil, socket, sys, tempfile, time, traceback
from pathlib import Path

ROOT=Path.cwd().resolve()
OUT=(ROOT/"rebuild/dataset_v2j1_validation_search").resolve()
V1=(ROOT/"rebuild/dataset_v2i_training_protocol").resolve()
ENV_PATH=OUT/"environment.json"
PROTOCOL_PATH=V1/"protocol.json"
CANDIDATE_PATH=V1/"candidate_manifest.json"
INPUT_PATH=OUT/"worker_inputs.json"

# Set caches before importing PyTorch; no model or data is loaded in this section.
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["CUDA_CACHE_PATH"] = str(OUT/"cache/cuda")
os.environ["TORCHINDUCTOR_CACHE_DIR"] = str(OUT/"cache/inductor")
os.environ["TRITON_CACHE_DIR"] = str(OUT/"cache/triton")
os.environ["TMPDIR"] = str(OUT/"tmp")
os.environ["TMP"] = str(OUT/"tmp")
os.environ["TEMP"] = str(OUT/"tmp")
Path(os.environ["TMPDIR"]).mkdir(parents=True,exist_ok=True)

import numpy as np
import torch
import torchvision
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import models

SEEDS=[3908274,3908275,3908276]
CLASSES=["galaxies","globularclusters","nebulae","opencluster"]
WEIGHT_PATHS={
    "resnet50":Path("/home/emarinho/.cache/torch/hub/checkpoints/resnet50-0676ba61.pth"),
    "vit_b_16":Path("/home/emarinho/.cache/torch/hub/checkpoints/vit_b_16-c867db91.pth"),
}


def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""): h.update(block)
    return h.hexdigest()
def canonical(x): return json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=False,allow_nan=False).encode()
def digest(x): return sha_bytes(canonical(x))
def read_json(path):
    with Path(path).open(encoding="utf-8") as f:return json.load(f)
def atomic_json(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT):raise RuntimeError(f"Write outside phase blocked: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+f".pending-{os.getpid()}")
    with pending.open("x",encoding="utf-8") as f:
        json.dump(value,f,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False);f.write("\n");f.flush();os.fsync(f.fileno())
    os.replace(pending,path)
def atomic_torch(path,value):
    path=Path(path).resolve()
    if not path.is_relative_to(OUT):raise RuntimeError(f"Checkpoint write outside phase blocked: {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    pending=path.with_name(path.name+f".pending-{os.getpid()}")
    with pending.open("xb") as f:
        torch.save(value,f);f.flush();os.fsync(f.fileno())
    os.replace(pending,path)
def append_event(kind,**fields):
    event={"utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),"event":kind,**fields}
    with (OUT/"events.jsonl").open("a",encoding="utf-8") as f:
        f.write(json.dumps(event,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n");f.flush();os.fsync(f.fileno())
    print(json.dumps(event,sort_keys=True,ensure_ascii=False,allow_nan=False),flush=True)


def install_access_guard(allowed_tensors):
    allowed={Path(x).resolve() for x in allowed_tensors}
    def guard(event,args):
        if event in {"socket.connect","socket.getaddrinfo","socket.bind"}:
            raise RuntimeError("Network access prohibited in fit worker")
        if event=="open" and args and isinstance(args[0],(str,bytes,os.PathLike)):
            path=Path(os.fsdecode(args[0])).resolve()
            mode=args[1] if len(args)>1 else "r"
            flags=args[2] if len(args)>2 and isinstance(args[2],int) else 0
            writing=(isinstance(mode,str) and any(c in mode for c in "wax+")) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC))
            if writing and not path.is_relative_to(OUT):
                raise RuntimeError(f"Worker write outside authorized phase blocked: {path}")
            if not writing:
                low=str(path).lower()
                if path.name.lower() in {"test.json","test.csv","sealed_label_manifest.json"} or "sealed_test_label" in low:
                    raise RuntimeError("Protected membership/labels are inaccessible to fit worker")
                if path.suffix.lower()==".npy" and path.is_relative_to(ROOT/"rebuild/dataset_v2h_representation_materialization") and path not in allowed:
                    raise RuntimeError("Tensor not in train/validation A allowlist")
        if event in {"os.rename","os.remove","os.rmdir"}:
            vals=[args[0],args[1]] if event=="os.rename" else [args[0]]
            for v in vals:
                if isinstance(v,(str,bytes,os.PathLike)):
                    path=Path(os.fsdecode(v))
                    if not path.is_absolute():path=(Path.cwd()/path)
                    if not path.resolve().is_relative_to(OUT):raise RuntimeError("Worker mutation outside authorized phase blocked")
    sys.addaudithook(guard)


def d4_code(object_id,epoch):
    key=f"Dataset-v2F|augment|2026102303|{object_id}|{epoch}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8],"big")%8

def apply_d4(x,code):
    y=torch.flip(x,(-1,)) if code>=4 else x
    return torch.rot90(y,k=code%4,dims=(-2,-1))

class FrozenObjects(Dataset):
    def __init__(self,rows,arrays,split,seed):
        if split not in {"train","validation"}:raise ValueError("Worker accepts only training and validation data")
        self.rows=rows;self.arrays=arrays;self.split=split;self.seed=seed;self.epoch=0
    def __len__(self):return len(self.rows)
    def set_epoch(self,epoch):
        if self.split=="train":self.epoch=int(epoch)
    def __getitem__(self,index):
        row=self.rows[index]
        x=torch.from_numpy(self.arrays[row["physical_object_id"]])
        if self.split=="train":x=apply_d4(x,d4_code(row["physical_object_id"],self.epoch))
        return x,int(row["label"]),row["physical_object_id"],row["split_unit_id"]

def seed_worker(worker_id):
    seed=torch.initial_seed()%2**32;random.seed(seed);np.random.seed(seed)

class BaselineModel(nn.Module):
    def __init__(self,network,family,warmup):
        super().__init__();self.network=network;self.family=family;self.warmup=int(warmup);self.epoch=0
        self.set_epoch(0)
    @property
    def head(self):return self.network.fc if self.family=="resnet50" else self.network.heads.head
    def backbone_parameters(self):
        head_ids={id(p) for p in self.head.parameters()}
        return [p for p in self.parameters() if id(p) not in head_ids]
    def set_epoch(self,epoch):
        self.epoch=int(epoch)
        for p in self.backbone_parameters():p.requires_grad_(self.epoch>=self.warmup)
        for p in self.head.parameters():p.requires_grad_(True)
        self.train(self.training)
    def train(self,mode=True):
        super().train(mode)
        if mode and self.epoch<self.warmup:
            self.network.eval();self.head.train(True)
        return self
    def forward(self,x):return self.network(x)

class FixedRatioPlateau(torch.optim.lr_scheduler.ReduceLROnPlateau):
    def _reduce_lr(self,epoch):
        super()._reduce_lr(epoch)
        self.optimizer.param_groups[1]["lr"]=10.0*self.optimizer.param_groups[0]["lr"]


def configure_runtime():
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1:raise RuntimeError("Frozen CUDA runtime unavailable; no CPU fallback")
    if torch.cuda.get_device_name(0)!="NVIDIA GeForce RTX 5060 Ti":raise RuntimeError("Frozen GPU identity mismatch")
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if not (torch.are_deterministic_algorithms_enabled() and torch.get_num_threads()==2 and torch.backends.cudnn.deterministic and not torch.backends.cudnn.benchmark and not torch.backends.cuda.matmul.allow_tf32 and not torch.backends.cudnn.allow_tf32):
        raise RuntimeError("Frozen deterministic settings were not applied")


def load_development_arrays(worker_input):
    arrays={};all_rows=[];allow=[]
    for role in ["train","validation"]:
        rows=worker_input["roles"][role]["objects"]
        for row in rows:
            if row["split"]!=role or row["label"] not in range(4):raise RuntimeError("Development input label/role mismatch")
            path=(ROOT/row["path"]).resolve()
            if path not in { (ROOT/r["path"]).resolve() for r in rows }:
                raise RuntimeError("Tensor path escaped role manifest")
            if path.stat().st_size!=row["size_bytes"] or sha_file(path)!=row["sha256"]:
                raise RuntimeError(f"Development tensor identity mismatch: {row['physical_object_id']}")
            allow.append(path)
            arr=np.load(path,allow_pickle=False)
            if arr.shape!=(3,224,224) or arr.dtype!=np.float32 or not np.isfinite(arr).all() or float(arr.min()) < -1.0 or float(arr.max())>1.0:
                raise RuntimeError("Frozen v2H tensor contract failed")
            arrays[row["physical_object_id"]]=np.ascontiguousarray(arr)
            all_rows.append(row)
    if len(allow)!=297 or len(set(allow))!=297:raise RuntimeError("Development A tensor set is incomplete or repeated")
    return arrays,allow


def data_loader(ds,batch_size,shuffle,seed):
    gen=torch.Generator();gen.manual_seed(seed)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,generator=gen if shuffle else None,
                      num_workers=8,pin_memory=True,prefetch_factor=2,worker_init_fn=seed_worker,
                      persistent_workers=False,drop_last=False)


def prepare_images(images,family):
    if family=="custom_cnn_transformer":
        # Frozen per-sample CPU adapter; exact v2I operation before CUDA transfer.
        images=torch.stack([F.interpolate(x.unsqueeze(0),size=(256,256),mode="bilinear",align_corners=False,antialias=True).squeeze(0).contiguous() for x in images])
    return images.to(device="cuda",dtype=torch.float32,non_blocking=True)


def metric_result(labels,preds,probabilities,weighted_loss):
    labels=np.asarray(labels,dtype=np.int64);preds=np.asarray(preds,dtype=np.int64)
    p,r,f,s=precision_recall_fscore_support(labels,preds,labels=list(range(4)),zero_division=0)
    cm=confusion_matrix(labels,preds,labels=list(range(4)))
    return {"accuracy":float(accuracy_score(labels,preds)),"balanced_accuracy":float(balanced_accuracy_score(labels,preds)),
            "macro_f1":float(f1_score(labels,preds,labels=list(range(4)),average="macro",zero_division=0)),
            "weighted_validation_loss":float(weighted_loss),
            "per_class":{"precision":p.tolist(),"recall":r.tolist(),"f1":f.tolist(),"support":s.tolist()},
            "confusion_matrix_true_rows_predicted_columns":cm.tolist(),
            "probability_rows":len(probabilities)}


def evaluate_validation(model,loader,weights,family,context):
    model.eval();loss_sum=0.0;count=0;labels=[];preds=[];probs=[];rows=[]
    with torch.inference_mode():
        for x,y,oids,units in loader:
            x=prepare_images(x,family);y=y.to(device="cuda",dtype=torch.long,non_blocking=True)
            logits=model(x)
            if logits.ndim!=2 or tuple(logits.shape)!=(len(y),4) or logits.dtype!=torch.float32 or not torch.isfinite(logits).all():
                raise RuntimeError("Invalid validation logits")
            losses=F.cross_entropy(logits,y,weight=weights,reduction="none")
            if not torch.isfinite(losses).all():raise RuntimeError("Nonfinite validation loss")
            pp=logits.softmax(dim=1).cpu().numpy().tolist();gg=logits.argmax(dim=1).cpu().numpy().tolist()
            yy=y.cpu().numpy().tolist();ll=losses.cpu().numpy().tolist()
            loss_sum+=float(losses.double().sum().item());count+=len(y)
            for oid,unit,label,pred,prob,loss in zip(oids,units,yy,gg,pp,ll):
                rows.append({"physical_object_id":oid,"split_unit_id":unit,"true_label":int(label),"predicted_label":int(pred),"probabilities":[float(z) for z in prob],"weighted_loss":float(loss),"fit_id":context["fit_id"],"seed":context["seed"],"configuration_id":context["configuration_id"]})
            labels.extend(yy);preds.extend(gg);probs.extend(pp)
    if count!=75:raise RuntimeError("Validation coverage differs from frozen 75 objects")
    metrics=metric_result(labels,preds,probs,loss_sum/count)
    return metrics,rows


def build_model(slot,seed,protocol):
    family=slot["family"];effective=slot["effective_parameters"];spec=effective["model"]
    if family=="custom_cnn_transformer":
        from rebuild.phase3d.models_v3 import model_factory
        if spec.get("input_size")!=256:raise RuntimeError("Custom 256px input contract mismatch")
        torch.manual_seed(seed)
        model=model_factory(json.loads(json.dumps(spec)),4)
    else:
        expected=protocol["initialization"][family]["sha256"]
        path=WEIGHT_PATHS[family]
        if sha_file(path)!=expected:raise RuntimeError(f"Frozen pretrained weight bytes changed: {family}")
        payload=torch.load(path,map_location="cpu",weights_only=True)
        torch.manual_seed(seed)
        network=models.resnet50(weights=None) if family=="resnet50" else models.vit_b_16(weights=None)
        network.load_state_dict(payload,strict=True)
        del payload
        torch.manual_seed(seed)
        if family=="resnet50":network.fc=nn.Linear(network.fc.in_features,4)
        else:network.heads.head=nn.Linear(network.heads.head.in_features,4)
        model=BaselineModel(network,family,spec["head_only_warmup_epochs"])
    return model


def build_optimizer(model,slot,protocol):
    fam=slot["family"];spec=slot["effective_parameters"];ms=spec["model"];ts=spec["training"]
    lr=float(ts["learning_rate"]);wd=float(ts["weight_decay"])
    if fam=="custom_cnn_transformer":
        if ms.get("optimizer")!="Adam":raise RuntimeError("Custom optimizer identity mismatch")
        custom_settings=protocol["optimization_proposal"]["family_settings"][family]
        opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=wd,
                             betas=tuple(custom_settings["betas"]),eps=custom_settings["epsilon"])
        return opt,torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",factor=ts["scheduler_factor"],patience=ts["scheduler_patience"])
    if ms.get("optimizer")!="AdamW" or ms.get("head_lr_multiplier")!=10.0:raise RuntimeError("Baseline optimizer identity mismatch")
    back=model.backbone_parameters();head=list(model.head.parameters())
    opt=torch.optim.AdamW([{"params":back,"lr":lr,"name":"backbone"},{"params":head,"lr":10*lr,"name":"head"}],
                          betas=tuple(ms["optimizer_betas"]),eps=ms["optimizer_epsilon"],weight_decay=wd)
    all_ids=[id(x) for x in model.parameters()];group_ids=[id(x) for g in opt.param_groups for x in g["params"]]
    if len(group_ids)!=len(set(group_ids)) or set(group_ids)!=set(all_ids):raise RuntimeError("Optimizer parameter membership mismatch")
    scheduler=FixedRatioPlateau(opt,mode="min",factor=ts["scheduler_factor"],patience=ts["scheduler_patience"])
    return opt,scheduler


def rng_snapshot():
    ns=np.random.get_state()
    return {"python":random.getstate(),"numpy":[ns[0],ns[1].tolist(),int(ns[2]),int(ns[3]),float(ns[4])],
            "torch_cpu":torch.get_rng_state(),"torch_cuda":torch.cuda.get_rng_state_all()}
def restore_rng(state):
    random.setstate(state["python"])
    ns=state["numpy"];np.random.set_state((ns[0],np.asarray(ns[1],dtype=np.uint32),int(ns[2]),int(ns[3]),float(ns[4])))
    torch.set_rng_state(state["torch_cpu"]);torch.cuda.set_rng_state_all(state["torch_cuda"])

class Stopper:
    def __init__(self,patience,min_delta):self.patience=patience;self.min_delta=min_delta;self.best=math.inf;self.bad_epochs=0
    def update(self,value):
        if not math.isfinite(value):raise RuntimeError("Nonfinite weighted validation loss")
        improved=value<self.best-self.min_delta
        if improved:self.best=value;self.bad_epochs=0
        else:self.bad_epochs+=1
        return improved,self.bad_epochs>=self.patience

def checkpoint_payload(context,epoch,model,optimizer,scheduler,stopper,history,weights,best_epoch,environment_sha,input_sha):
    return {"schema_version":1,"context":context,"epoch":int(epoch),"seed":int(context["seed"]),
            "model_configuration":context["effective_parameters"],"class_order":CLASSES,
            "class_weights":weights.detach().cpu().tolist(),"configuration_sha256":context["configuration_sha256"],
            "protocol_canonical_sha256":context["protocol_canonical_sha256"],"environment_sha256":environment_sha,
            "input_receipt_sha256":input_sha,"model_state":model.state_dict(),
            "optimizer_state":optimizer.state_dict(),"scheduler_state":scheduler.state_dict(),
            "early_stopping":{"patience":stopper.patience,"min_delta":stopper.min_delta,"best":stopper.best,"bad_epochs":stopper.bad_epochs},
            "best_epoch":int(best_epoch),"history":history,"rng":rng_snapshot(),
            "sampler_state":{"next_epoch":int(epoch+1),"shuffle_seed_base":int(context["seed"]),"last_epoch_seed":int(context["seed"]+epoch)},
            "runtime":{"device":"cuda","precision":"float32","amp":False,"batch_size":16,"loader_workers":8,"cpu_threads":2}}

def fit_one(index,resume):
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    import torchvision as vision
    if str(torch.__version__)!="2.7.0+cu128" or str(vision.__version__)!="0.22.0+cu128" or sys.version_info[:3]!=(3,12,3):
        raise RuntimeError("Frozen software environment mismatch")
    configure_runtime()
    protocol=read_json(PROTOCOL_PATH);protocol_sha=protocol.pop("canonical_sha256")
    if digest(protocol)!=protocol_sha:raise RuntimeError("Frozen protocol digest mismatch")
    protocol["canonical_sha256"]=protocol_sha
    candidate=read_json(CANDIDATE_PATH);candidate_sha=sha_file(CANDIDATE_PATH)
    if candidate_sha!=protocol["candidate_manifest_sha256"]:raise RuntimeError("Candidate manifest changed")
    slot=candidate["mandatory_search_fits"][index]
    if slot["representation"]!="A" or slot["stage"]!="mandatory_primary_A_search":raise RuntimeError("Worker only executes frozen A search slots")
    worker_input=read_json(INPUT_PATH);input_sha=sha_file(INPUT_PATH)
    if worker_input["split_identity"]!=protocol["data_and_preprocessing"]["split_sha256"]:raise RuntimeError("Worker input split identity mismatch")
    rows_train=worker_input["roles"]["train"]["objects"];rows_val=worker_input["roles"]["validation"]["objects"]
    arrays,allowed=load_development_arrays(worker_input)
    install_access_guard(allowed)
    fit_id=slot["fit_id"];directory=(OUT/"fits"/fit_id).resolve()
    if not directory.is_relative_to(OUT):raise RuntimeError("Fit directory outside phase")
    directory.mkdir(parents=True,exist_ok=True)
    env=read_json(ENV_PATH);env_sha=sha_file(ENV_PATH)
    config_record={"slot":slot,"protocol_canonical_sha256":protocol_sha,"candidate_manifest_sha256":candidate_sha,
                   "input_receipt_sha256":sha_file(OUT/"input_receipt.json"),"worker_input_sha256":input_sha,
                   "environment_sha256":env_sha}
    config_sha=digest(config_record)
    config_path=directory/"configuration.json"
    if config_path.exists():
        if read_json(config_path)!=config_record:raise RuntimeError("Existing fit configuration differs")
    else:atomic_json(config_path,config_record)
    if not (directory/"environment_receipt.json").exists():atomic_json(directory/"environment_receipt.json",env)
    if not (directory/"input_receipt.json").exists():atomic_json(directory/"input_receipt.json",read_json(OUT/"input_receipt.json"))
    completed_path=directory/"fit_complete.json"
    if completed_path.exists():
        complete=read_json(completed_path)
        for name,expected in complete["artifacts"].items():
            if sha_file(directory/name)!=expected:raise RuntimeError("Completed fit artifact hash mismatch")
        print(json.dumps({"event":"completed_fit_verified_no_restart","fit_id":fit_id,"index":index},sort_keys=True),flush=True)
        return complete
    if directory.joinpath("last.pt").exists() and not resume:raise RuntimeError("Incomplete fit exists; explicit exact resume required")
    if resume and not directory.joinpath("last.pt").exists():raise RuntimeError("Resume requested without an epoch checkpoint")
    context={"fit_id":fit_id,"manifest_index":index,"family":slot["family"],"representation":"A",
             "configuration_id":slot["configuration_id"],"configuration_sha256":slot["configuration_sha256"],
             "seed":slot["seed"],"protocol_canonical_sha256":protocol_sha,"candidate_manifest_sha256":candidate_sha,
             "effective_parameters":slot["effective_parameters"],"input_sha256":input_sha,"environment_sha256":env_sha}
    provenance_sha=digest(context);family=slot["family"];seed=slot["seed"];effective=slot["effective_parameters"]
    model=build_model(slot,seed,protocol).to("cuda",dtype=torch.float32)
    optimizer,scheduler=build_optimizer(model,slot,protocol)
    if family=="custom_cnn_transformer":
        total_params=sum(p.numel() for p in model.parameters())
    else:total_params=sum(p.numel() for p in model.parameters())
    weights_cpu=torch.tensor([sum(r["label"]==i for r in rows_train) for i in range(4)],dtype=torch.float32)
    if (weights_cpu==0).any():raise RuntimeError("Training split missing frozen class")
    class_weights=weights_cpu.reciprocal();class_weights=class_weights/class_weights.mean();weights=class_weights.to("cuda")
    train_ds=FrozenObjects(rows_train,arrays,"train",seed);val_ds=FrozenObjects(rows_val,arrays,"validation",seed)
    val_loader=data_loader(val_ds,16,False,seed)
    stopper=Stopper(effective["training"]["patience"],effective["training"]["min_delta"])
    history=[];start_epoch=0;best_epoch=-1
    state_path=directory/"fit_state.json"
    if resume:
        if not state_path.is_file():raise RuntimeError("Resume state receipt missing")
        state=read_json(state_path)
        for filename,key in [("last.pt","last_sha256"),("history.json","history_sha256"),("best.pt","best_sha256"),("best_validation_predictions.json","best_predictions_sha256")]:
            if not (directory/filename).is_file() or sha_file(directory/filename)!=state[key]:raise RuntimeError(f"Resume artifact hash mismatch: {filename}")
        if state["configuration_sha256"]!=config_sha or state["provenance_sha256"]!=provenance_sha:raise RuntimeError("Resume configuration/environment/input identity mismatch")
        checkpoint=torch.load(directory/"last.pt",map_location="cpu",weights_only=True)
        if checkpoint["context"]!=context or checkpoint["configuration_sha256"]!=slot["configuration_sha256"] or checkpoint["protocol_canonical_sha256"]!=protocol_sha:raise RuntimeError("Resume checkpoint provenance mismatch")
        model.load_state_dict(checkpoint["model_state"],strict=True);optimizer.load_state_dict(checkpoint["optimizer_state"]);scheduler.load_state_dict(checkpoint["scheduler_state"])
        stopper=Stopper(checkpoint["early_stopping"]["patience"],checkpoint["early_stopping"]["min_delta"])
        stopper.best=checkpoint["early_stopping"]["best"];stopper.bad_epochs=checkpoint["early_stopping"]["bad_epochs"]
        start_epoch=checkpoint["epoch"]+1;best_epoch=checkpoint["best_epoch"];history=checkpoint["history"]
        if checkpoint["sampler_state"]!={"next_epoch":start_epoch,"shuffle_seed_base":seed,"last_epoch_seed":seed+start_epoch-1}:raise RuntimeError("Resume sampler/epoch state mismatch")
        restore_rng(checkpoint["rng"])
        if len(history)!=start_epoch:raise RuntimeError("Resume history epoch count mismatch")
        append_event("fit_resumed",fit_id=fit_id,index=index,next_epoch=start_epoch)
    elif state_path.exists():raise RuntimeError("Fit state exists but explicit resume was not requested")
    else:
        append_event("fit_started",fit_id=fit_id,index=index,family=family,configuration_id=slot["configuration_id"],seed=seed,representation="A")
    begin=time.monotonic();gpu_seconds=0.0;stop=False
    max_epochs=effective["training"]["epochs"]
    for epoch in range(start_epoch,max_epochs):
        if stopper.bad_epochs>=stopper.patience:break
        if hasattr(model,"set_epoch"):model.set_epoch(epoch)
        train_ds.set_epoch(epoch)
        train_loader=data_loader(train_ds,16,True,seed+epoch)
        model.train();train_sum=0.0;train_count=0
        torch.cuda.synchronize();g0=torch.cuda.Event(enable_timing=True);g1=torch.cuda.Event(enable_timing=True);g0.record()
        for x,y,oids,units in train_loader:
            optimizer.zero_grad(set_to_none=True)
            xx=prepare_images(x,family);yy=y.to("cuda",dtype=torch.long,non_blocking=True)
            logits=model(xx)
            if logits.ndim!=2 or tuple(logits.shape)!=(len(yy),4) or not torch.isfinite(logits).all():raise RuntimeError("Invalid training logits")
            losses=F.cross_entropy(logits,yy,weight=weights,reduction="none")
            if not torch.isfinite(losses).all():raise RuntimeError("Nonfinite training loss")
            losses.mean().backward()
            norm=effective["training"]["max_grad_norm"]
            torch.nn.utils.clip_grad_norm_(model.parameters(),norm,error_if_nonfinite=True)
            optimizer.step();train_sum+=float(losses.detach().double().sum().item());train_count+=len(yy)
        g1.record();torch.cuda.synchronize();train_gpu=g0.elapsed_time(g1)/1000.0
        if train_count!=222:raise RuntimeError("Training epoch did not visit all frozen objects")
        torch.cuda.synchronize();v0=torch.cuda.Event(enable_timing=True);v1=torch.cuda.Event(enable_timing=True);v0.record()
        metrics,preds=evaluate_validation(model,val_loader,weights,family,context)
        v1.record();torch.cuda.synchronize();val_gpu=v0.elapsed_time(v1)/1000.0;gpu_seconds+=train_gpu+val_gpu
        scheduler.step(metrics["weighted_validation_loss"])
        improved,stop=stopper.update(metrics["weighted_validation_loss"])
        if improved:best_epoch=epoch
        epoch_record={"epoch":epoch,"train_weighted_loss":train_sum/train_count,"validation":metrics,
                      "learning_rates":[float(g["lr"]) for g in optimizer.param_groups],
                      "warmup":bool(family!="custom_cnn_transformer" and epoch<effective["model"]["head_only_warmup_epochs"]),
                      "bad_epochs":stopper.bad_epochs,"train_gpu_seconds":train_gpu,"validation_gpu_seconds":val_gpu,
                      "peak_allocated_bytes":int(torch.cuda.max_memory_allocated()),"peak_reserved_bytes":int(torch.cuda.max_memory_reserved()),
                      "process_peak_rss_kib":int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}
        history.append(epoch_record)
        gpu_total=int(torch.cuda.get_device_properties(0).total_memory)
        if epoch_record["peak_reserved_bytes"]>0.90*gpu_total:raise RuntimeError("90% GPU memory resource ceiling exceeded")
        if shutil.disk_usage(OUT).free<100*(1<<30):raise RuntimeError("Frozen disk reserve threshold crossed")
        if improved:
            pred_record={"fit_id":fit_id,"family":family,"configuration_id":slot["configuration_id"],"seed":seed,
                         "representation":"A","protocol_canonical_sha256":protocol_sha,"configuration_sha256":slot["configuration_sha256"],
                         "validation_metrics":metrics,"ordered_validation_predictions":preds}
            pred_record["prediction_digest"]=digest(pred_record)
            atomic_json(directory/"best_validation_predictions.json",pred_record)
        atomic_json(directory/"history.json",history)
        payload=checkpoint_payload(context,epoch,model,optimizer,scheduler,stopper,history,weights,best_epoch,env_sha,input_sha)
        if improved:atomic_torch(directory/"best.pt",payload)
        atomic_torch(directory/"last.pt",payload)
        state={"fit_id":fit_id,"configuration_sha256":config_sha,"provenance_sha256":provenance_sha,"last_epoch":epoch,
               "last_sha256":sha_file(directory/"last.pt"),"history_sha256":sha_file(directory/"history.json"),
               "best_sha256":sha_file(directory/"best.pt"),"best_predictions_sha256":sha_file(directory/"best_validation_predictions.json")}
        atomic_json(state_path,state)
        current={"index":index,"fit_id":fit_id,"family":family,"configuration_id":slot["configuration_id"],"seed":seed,
                 "representation":"A","epoch":epoch+1,"maximum_epochs":max_epochs,"train_weighted_loss":train_sum/train_count,
                 "validation_balanced_accuracy":metrics["balanced_accuracy"],"validation_macro_f1":metrics["macro_f1"],
                 "validation_weighted_loss":metrics["weighted_validation_loss"],"best_epoch":best_epoch+1,
                 "bad_epochs":stopper.bad_epochs,"fit_wall_seconds":time.monotonic()-begin,"gpu_seconds":gpu_seconds,
                 "checkpoint_sha256":state["last_sha256"],"history_sha256":state["history_sha256"],
                 "utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())}
        atomic_json(OUT/"current.json",current)
        progress=read_json(OUT/"progress.json");progress.update({"state":"running","current_fit_index":index,"current_fit_id":fit_id,"current_epoch":epoch+1,"completed_fits":len(list((OUT/"fits").glob("*/fit_complete.json"))),"failed_fits":0})
        atomic_json(OUT/"progress.json",progress)
        append_event("epoch_complete",fit_id=fit_id,index=index,epoch=epoch+1,best_epoch=best_epoch+1,
                     validation_balanced_accuracy=metrics["balanced_accuracy"],validation_macro_f1=metrics["macro_f1"],
                     validation_weighted_loss=metrics["weighted_validation_loss"],gpu_seconds=gpu_seconds,
                     checkpoint_sha256=state["last_sha256"],history_sha256=state["history_sha256"])
        print(json.dumps({"epoch":epoch+1,"fit_id":fit_id,"val_ba":metrics["balanced_accuracy"],"val_macro_f1":metrics["macro_f1"],"val_loss":metrics["weighted_validation_loss"]},sort_keys=True),flush=True)
        if stop:break
    if best_epoch<0 or not (directory/"best.pt").is_file():raise RuntimeError("No finite best checkpoint was produced")
    history=read_json(directory/"history.json")
    best_predictions=read_json(directory/"best_validation_predictions.json")
    if best_predictions["prediction_digest"]!=digest({k:v for k,v in best_predictions.items() if k!="prediction_digest"}):raise RuntimeError("Best validation prediction digest mismatch")
    metrics=best_predictions["validation_metrics"]
    state=read_json(state_path)
    artifact_names=["configuration.json","environment_receipt.json","input_receipt.json","history.json","last.pt","best.pt","best_validation_predictions.json","fit_state.json"]
    artifacts={name:sha_file(directory/name) for name in artifact_names}
    complete={"status":"completed","fit_id":fit_id,"manifest_index":index,"family":family,"configuration_id":slot["configuration_id"],
              "seed":seed,"representation":"A","best_epoch_zero_based":best_epoch,"epochs_completed":len(history),
              "validation_metrics":metrics,"configuration_sha256":config_sha,"candidate_configuration_sha256":slot["configuration_sha256"],
              "protocol_canonical_sha256":protocol_sha,"candidate_manifest_sha256":candidate_sha,"input_sha256":input_sha,
              "environment_sha256":env_sha,"fit_wall_seconds":time.monotonic()-begin,"gpu_seconds":gpu_seconds,
              "resumed":bool(resume),"total_parameters":total_params,"artifacts":artifacts,
              "history_sha256":artifacts["history.json"],"best_checkpoint_sha256":artifacts["best.pt"],
              "last_checkpoint_sha256":artifacts["last.pt"],"validation_prediction_sha256":artifacts["best_validation_predictions.json"],
              "cuda_peak_allocated_bytes":max(x["peak_allocated_bytes"] for x in history),
              "cuda_peak_reserved_bytes":max(x["peak_reserved_bytes"] for x in history),"training_updates":True,"test_data_accessed":False}
    complete["complete_record_sha256"]=digest(complete)
    atomic_json(completed_path,complete)
    append_event("fit_complete",fit_id=fit_id,index=index,best_epoch=best_epoch+1,epochs=len(history),
                 gpu_seconds=gpu_seconds,wall_seconds=complete["fit_wall_seconds"],checkpoint_sha256=complete["best_checkpoint_sha256"],
                 history_sha256=complete["history_sha256"])
    return complete


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--index",type=int,required=True);parser.add_argument("--resume",action="store_true");args=parser.parse_args()
    try:
        complete=fit_one(args.index,args.resume)
        print(json.dumps({"status":"completed","fit_id":complete["fit_id"],"index":args.index,"best_epoch":complete["best_epoch_zero_based"]+1},sort_keys=True),flush=True)
    except BaseException as exc:
        append_event("fit_failed",index=args.index,error_type=type(exc).__name__,error=str(exc))
        traceback.print_exc()
        raise

if __name__=="__main__":main()
