"""Synthetic-only proof that the frozen D4 operation commutes with v2H A materialization."""
import hashlib, importlib.util, json
from pathlib import Path
import numpy as np

ROOT=Path.cwd(); OUT=ROOT/"rebuild/dataset_v2j1_validation_search"
SOURCE=ROOT/"rebuild/dataset_v2h_representation_materialization/materialize.py"

def sha(b): return hashlib.sha256(b).hexdigest()
def file_sha(p): return sha(Path(p).read_bytes())

source_hash=file_sha(SOURCE)
spec=importlib.util.spec_from_file_location("frozen_v2h_materializer",SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
n=512
y,x=np.mgrid[:n,:n]
raw=np.stack([(x+3*y)/2048.0,(2*x-y+1024)/2048.0,np.sin(x/37.0)+np.cos(y/53.0)])
# Fixed per-band monotone mappings produce bounded u values; this is synthetic only.
u=[]
for band in raw:
    low=np.quantile(band,.01,method="linear"); high=np.quantile(band,.99,method="linear")
    z=np.clip((band-low)/(high-low),0,1)
    u.append(np.arcsinh(10*z)/np.arcsinh(10))
u=np.stack(u).astype(np.float64)
base=module.tensor_from_u(u,"A")
records=[]
for code in range(8):
    source_aug=module.tensor_from_u(module.d4(u,code),"A")
    tensor_aug=module.d4(base,code)
    exact=bool(np.array_equal(source_aug,tensor_aug))
    records.append({"d4_code":code,"exact_array_equality":exact,
                    "source_then_materialize_sha256":sha(source_aug.tobytes()),
                    "materialize_then_tensor_d4_sha256":sha(tensor_aug.tobytes()),
                    "max_abs_difference":float(np.max(np.abs(source_aug-tensor_aug)))})
    if not exact: raise RuntimeError(f"Frozen D4 source-lattice equivalence failed at code {code}")
receipt={"phase":"Dataset-v2J1","status":"passed","synthetic_only":True,
         "frozen_v2h_materialize_py_sha256":source_hash,"synthetic_source_shape":[3,512,512],
         "materialized_A_shape":[3,224,224],"d4_codes":records,
         "channel_lock":"D4 acts only on the last two spatial axes; all three channels remain in order blue/red/NIR.",
         "v2h_files_modified":False,"training_or_inference_performed":False}
(OUT/"augmentation_conformance.json").write_text(json.dumps(receipt,indent=2,sort_keys=True)+"\n")
print(json.dumps({"status":"passed","codes":len(records),"all_exact":all(x["exact_array_equality"] for x in records),"source_sha256":source_hash},sort_keys=True))
