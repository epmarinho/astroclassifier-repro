#!/usr/bin/env python3
"""Read-only verification and prospective fit-slot enumeration for v2I."""
import hashlib, json, math
from pathlib import Path
import numpy as np

ROOT=Path.cwd(); OUT=ROOT/'rebuild/dataset_v2i_training_protocol'
H=ROOT/'rebuild/dataset_v2h_representation_materialization'
G=ROOT/'rebuild/dataset_v2g_split_materialization'
F=ROOT/'rebuild/dataset_v2f_representation_split_protocol'
V1=ROOT/'rebuild/baseline_comparison_v1'; P4C=V1/'phase4c_selection_freeze'
P4B=V1/'phase4b/attempt_4b1a'; P4E=V1/'phase4e_object_independence'
def canon(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def sha_bytes(x): return hashlib.sha256(x).hexdigest()
def sha_file(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def dump(name,obj): (OUT/name).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')

# v2H inventory and every listed file verified by streaming hash. Do not parse labels.
h_inv_path=H/'file_inventory.json'; h_inv_sha=sha_file(h_inv_path)
assert h_inv_sha=='8fb890200277fe91ba62e683f4de58c20f809586aa0644334f2c6cf1d2075c4d'
h_inv=json.loads(h_inv_path.read_text()); assert h_inv['file_count']==1496
for rec in h_inv['files']:
 p=H/rec['path']
 assert p.is_file() and p.stat().st_size==rec['size_bytes'] and sha_file(p)==rec['sha256'],rec['path']
pres=json.loads((H/'preservation_input_manifest.json').read_text())
for rec in pres['files']:
 p=ROOT/rec['path']
 assert p.is_file() and p.stat().st_size==rec['size_bytes'] and sha_file(p)==rec['sha256'],rec['path']
digest=json.loads((H/'dataset_digest.json').read_text())
assert sha_bytes(canon(digest['binding']))==digest['canonical_dataset_sha256']
split=json.loads((G/'split_manifest.json').read_text()); split_sha=sha_bytes(canon(split['binding']))
assert split_sha=='1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf'
assert digest['binding']['split_sha256']==split_sha
A=json.loads((H/'representation_A_manifest.json').read_text())
B=json.loads((H/'representation_B_manifest.json').read_text())
controls=json.loads((H/'control_manifest.json').read_text())
assert len(A['records'])==len(B['records'])==370
assert {x['physical_object_id'] for x in A['records']}=={x['physical_object_id'] for x in B['records']}
label_rec=next(x for x in h_inv['files'] if x['path']=='sealed_label_manifest.json')
assert label_rec['sha256']==digest['binding']['labels_sha256']

v1_protocol=V1/'protocol.json'; frozen_path=P4C/'frozen_selection.json'
baseline_path=P4C/'baseline_audit.json'; custom_path=P4C/'custom_audit.json'
phase4b_selection=P4B/'selection.json'; phase4e_protocol=P4E/'analysis_protocol.json'
source_paths=[v1_protocol,frozen_path,baseline_path,custom_path,phase4b_selection,phase4e_protocol,
 F/'representation_protocol.json',F/'shortcut_control_protocol.json',G/'split_manifest.json',
 H/'file_inventory.json',H/'dataset_digest.json',H/'representation_A_manifest.json',
 H/'representation_B_manifest.json',H/'control_manifest.json']
source_hashes={str(p):sha_file(p) for p in source_paths}
vp=json.loads(v1_protocol.read_text()); frozen=json.loads(frozen_path.read_text())
ba=json.loads(baseline_path.read_text()); ca=json.loads(custom_path.read_text())
p4bs=json.loads(phase4b_selection.read_text()); p4e=json.loads(phase4e_protocol.read_text())
assert p4bs.get('test_evaluated') is False and frozen.get('test_access_authorized') is False
assert ba['completed_fits']==72 and ca['completed_fits']==36 and len(frozen['families'])==3
seeds=[3908274,3908275,3908276]; reps=['A','B','A_outer','B_outer']
families=['custom_cnn_transformer','resnet50','vit_b_16']
def effective(c):
 # v1 image preprocessing is deliberately omitted: v2H tensors own the frozen input transform.
 out={k:c[k] for k in ['model','training','runtime'] if k in c}
 # The archived candidate configs contain a stale preflight `available=false` field;
 # actual Phase 4C execution identity records the verified weight SHA separately.
 if 'model' in out and 'pretrained_weights' in out['model']:
  out['model']=dict(out['model']);out['model'].pop('pretrained_weights',None)
 return out
def cdigest(x): return sha_bytes(canon(x))
winners={}; winner_cfg={}; winner_meta={}
for row in frozen['families']:
 fam=row['family']; cfg=effective(row['complete_effective_configuration_by_seed'][str(seeds[0])]); dh=cdigest(cfg)
 for seed in seeds: assert cdigest(effective(row['complete_effective_configuration_by_seed'][str(seed)]))==dh
 winners[fam]=row['configuration_id']; winner_cfg[fam]=cfg
 winner_meta[fam]={'configuration_id':row['configuration_id'],'configuration_sha256':dh,
   'architecture_identity':row['source_resolved_architecture_parameters'],'source_identity':row['source_identity'],
   'environment_digest':row['environment_digest'],'pretrained_initialization':row['executed_pretrained_initialization']}

# Frozen candidate grids contain config identity/parameters only, never result metrics.
grid={f:{} for f in families}
for fit in ba['fit_evidence']:
 c=fit['configuration']; fam=c['model']['family']; cfg=effective(c); cid=c['model']['name']; dh=cdigest(cfg)
 assert fam in grid
 if cid in grid[fam]: assert grid[fam][cid]['configuration_sha256']==dh
 grid[fam][cid]={'configuration_id':cid,'configuration_sha256':dh,'effective_parameters':cfg,
                 'candidate_scope':'frozen v1 Phase 4B candidate'}
for fit in ca['fit_evidence']:
 c=fit['configuration']; m=c['model']
 if m.get('variant')!='spatial_transformer': continue
 cfg=effective(c); cid=m['name']; dh=cdigest(cfg)
 if cid in grid['custom_cnn_transformer']: assert grid['custom_cnn_transformer'][cid]['configuration_sha256']==dh
 grid['custom_cnn_transformer'][cid]={'configuration_id':cid,'configuration_sha256':dh,
   'effective_parameters':cfg,'candidate_scope':'frozen v1 custom Transformer-bearing candidate'}
assert {k:len(v) for k,v in grid.items()}=={'custom_cnn_transformer':6,'resnet50':12,'vit_b_16':12}
for fam in families:
 if fam=='custom_cnn_transformer': assert any(x['configuration_sha256']==winner_meta[fam]['configuration_sha256'] for x in grid[fam].values())
 else: assert winners[fam] in grid[fam]

def fit_slot(scenario,fam,rep,cfgid,cfgsha,seed,stage,params=None):
 identity={'scenario':scenario,'family':fam,'representation':rep,'configuration_id':cfgid,
           'configuration_sha256':cfgsha,'seed':seed,'split_sha256':split_sha}
 out={**identity,'fit_id':'fit_'+sha_bytes(canon(identity))[:24],'stage':stage}
 if params is not None: out['effective_parameters']=params
 return out
rows_A=[]; rows_B=[]
for fam in families:
 for rep in reps:
  for seed in seeds: rows_A.append(fit_slot('A_reuse_v1_winner',fam,rep,winners[fam],winner_meta[fam]['configuration_sha256'],seed,'mandatory',winner_cfg[fam]))
for fam in families:
 for cand in sorted(grid[fam].values(),key=lambda x:x['configuration_id']):
  for seed in seeds: rows_B.append(fit_slot('B_v2_validation_search',fam,'A',cand['configuration_id'],cand['configuration_sha256'],seed,'mandatory_primary_A_search',cand['effective_parameters']))
  for rep in ['B','A_outer','B_outer']:
   for seed in seeds: rows_B.append(fit_slot('B_v2_validation_search',fam,rep,cand['configuration_id'],cand['configuration_sha256'],seed,'conditional_if_configuration_wins_on_A',cand['effective_parameters']))
metadata_specs=[
 {'model':'metadata_logistic_regression','config_id':'metadata_logreg_C1_balanced_lbfgs','seed':None,
  'features':['log_source_FOV_arcmin','log_source_pixel_scale_arcsec','log_catalogue_extent_arcmin','catalogue_extent_missing','source_FOV_capped'],
  'fit_policy':'C=1; class_weight=balanced; max_iter=1000; solver=lbfgs; train-only median imputation and scaling'},
 {'model':'metadata_depth3_tree','config_id':'metadata_decision_tree_depth3','seed':3908274,
  'features':['log_source_FOV_arcmin','log_source_pixel_scale_arcsec','log_catalogue_extent_arcmin','catalogue_extent_missing','source_FOV_capped'],
  'fit_policy':'max_depth=3; random_state=3908274; no search'}]
metadata_rows=[]
for spec in metadata_specs:
 key={'scenario':'shared_metadata_FOV_control','model':spec['model'],'config_id':spec['config_id'],'seed':spec['seed'],'split_sha256':split_sha}
 metadata_rows.append({**key,'fit_id':'fit_'+sha_bytes(canon(key))[:24],'features':spec['features'],'fit_policy':spec['fit_policy'],'train_only_fit_statistics':True})
candidate={'phase':'Dataset-v2I','status':'prospective_no_results','split_sha256':split_sha,'seeds':seeds,
 'representations':reps,'families':families,
 'alternatives':{'A':{'deep_model_fit_count':len(rows_A),'total_fit_count_including_metadata_controls':38,'rule':'one frozen v1 winning config per architecture, all four representations and three seeds','fits':rows_A},
 'B':{'candidate_configurations_by_family':{f:[{'configuration_id':x['configuration_id'],'configuration_sha256':x['configuration_sha256']} for x in sorted(grid[f].values(),key=lambda x:x['configuration_id'])] for f in families},
 'potential_fit_slots':len(rows_B),'expected_executed_core_fit_count_after_selection':117,'total_fit_count_including_metadata_controls':119,'mandatory_A_search_fits':90,
 'conditional_selected_control_fits':27,'selected_factorial_fits':36,'fits':rows_B},
 },'shared_mandatory_metadata_control_fits':metadata_rows,'metadata_control_fit_count':2,
 'mutually_exclusive':True,'contains_no_results_prediction_hashes_or_timings':True}
dump('candidate_manifest.json',candidate)

# Compute v1 timing distributions for ResNet/ViT from frozen epoch timing artifacts.
runtime={f:[] for f in ['resnet50','vit_b_16']}; epoch={f:[] for f in runtime}
for fit in ba['fit_evidence']:
 fam=fit['configuration']['model']['family']; p=fit['artifacts'].get('epoch_timing.json',{}).get('path')
 if not p: continue
 arr=json.loads(Path(p).read_text()); tr=sum(float(x['train_pipeline_wall_seconds']) for x in arr); va=sum(float(x['validation_pipeline_wall_seconds']) for x in arr)
 runtime[fam].append({'train_s':tr,'validation_s':va,'epochs':fit['epochs_completed']})
 epoch[fam].extend([(float(x['train_pipeline_wall_seconds']),float(x['validation_pipeline_wall_seconds'])) for x in arr])
assert len(runtime['resnet50'])==len(runtime['vit_b_16'])==36
custom_ep=[int(x['epochs_completed']) for x in ca['fit_evidence'] if x['configuration']['model'].get('variant')=='spatial_transformer']
def q(v,p): return float(np.quantile(np.asarray(v,dtype=float),p,method='linear'))
tr_scale=222/485; va_scale=75/158; profiles={}
for fam,rows in runtime.items():
 scaled=[x['train_s']*tr_scale+x['validation_s']*va_scale for x in rows]
 total=[x['train_s']+x['validation_s'] for x in rows]
 profiles[fam]={'v1_completed_fits':len(rows),'v1_wall_seconds_p10_median_p90':[q(total,p) for p in [.1,.5,.9]],
   'v2_scaled_seconds_per_fit_p10_median_p90':[q(scaled,p) for p in [.1,.5,.9]],'train_size_factor':tr_scale,'validation_size_factor':va_scale}
vittr=q([x[0] for x in epoch['vit_b_16']],.5); vitva=q([x[1] for x in epoch['vit_b_16']],.5)
vittr90=q([x[0] for x in epoch['vit_b_16']],.9);vitva90=q([x[1] for x in epoch['vit_b_16']],.9)
custom_proxy={'v1_transformer_fit_count':len(custom_ep),'epochs_p10_median_p90':[q(custom_ep,p) for p in [.1,.5,.9]],
 'seconds_per_fit_low':q(custom_ep,.1)*(vittr*tr_scale+vitva*va_scale)*.5,
 'seconds_per_fit_central':q(custom_ep,.5)*(vittr*tr_scale+vitva*va_scale),
 'seconds_per_fit_high':100*(vittr90*tr_scale+vitva90*va_scale)*2,
 'evidence':'No custom per-epoch runtime was saved. Broad proxy: measured ViT-B/16 epoch wall time, 0.5x–2x uncertainty; high assumes 100 epochs.'}
def cost(fam,n):
 if fam=='custom_cnn_transformer': return [custom_proxy[k]*n/3600 for k in ['seconds_per_fit_low','seconds_per_fit_central','seconds_per_fit_high']]
 return [x*n/3600 for x in profiles[fam]['v2_scaled_seconds_per_fit_p10_median_p90']]
scenario_counts={'A_reuse_v1_winner':{f:12 for f in families},'B_v2_validation_search':{'resnet50':45,'vit_b_16':45,'custom_cnn_transformer':27}}
estimates={}
for name,counts in scenario_counts.items():
 sums=[0.,0.,0.]
 for fam,n in counts.items(): sums=[a+b for a,b in zip(sums,cost(fam,n))]
 estimates[name]={'GPU_wall_hours_low_central_high':sums,'fit_counts_by_architecture':counts}
fit_accounting={'seeds':seeds,'training_fit_counts':{'alternative_A_total_including_metadata_controls':38,'alternative_A_deep_models':36,'alternative_B_total_including_metadata_controls':119,'alternative_B_deep_models':117,'shared_metadata_controls':2,'selected_factorial_deep_models':36},
 'alternative_B_breakdown':{'ResNet50':45,'ViT-B/16':45,'Custom_CNN_Transformer':27,'mandatory_primary_A_search_fits':90,'conditional_selected_control_fits':27},
 'one_time_final_test_evaluations':38,'deep_checkpoint_test_evaluations':36,'metadata_control_test_evaluations':2,'test_predictions_object_checkpoint':2628,'metadata_control_object_predictions':146,'total_model_output_records':2774,'test_objects':73,'test_assignment_units':61,
 'runtime_evidence':{'baseline_profiles':profiles,'custom_proxy':custom_proxy,'estimates':estimates,
 'assumptions':'v1 RTX 5060 Ti epoch data; scale train/validation counts by v2 222/485 and 75/158 for low/central baseline estimate. High uses 90th-percentile v1 fit time without sample-count scaling. Custom timing is a low-confidence 0.5x–2x ViT proxy; high allows 100 epochs. Excludes setup, serialization and test inference; wall time includes data pipeline.'}}
dump('fit_accounting.json',fit_accounting)

dump('input_verification.json',{'v2A_to_v2G_files_verified':len(pres['files']),'v2H_files_verified':len(h_inv['files']),
 'v2H_sealed_label_manifest_hash_only':label_rec,'split_sha256':split_sha,'dataset_digest':digest['canonical_dataset_sha256'],
 'v1_protocol_source_hashes':source_hashes,'Phase4D_test_predictions_or_truth_opened':False,
 'tensor_arrays_loaded_or_reconstructed':False,'v2A_to_v2H_content_modified':False})
print(json.dumps({'verified_v2h_files':len(h_inv['files']),'preserved_prior_files':len(pres['files']),
 'grid_counts':{f:len(x) for f,x in grid.items()},'fit_counts':{'A_deep':len(rows_A),'A_total':len(rows_A)+2,'B_potential':len(rows_B),'B_core_expected':117,'B_total':119},
 'runtime_estimates':estimates,'custom_proxy':custom_proxy},indent=2))
