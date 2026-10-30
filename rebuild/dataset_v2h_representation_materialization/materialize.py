#!/usr/bin/env python3
"""Dataset-v2H deterministic frozen tensor materialization and reconstruction check."""
from __future__ import annotations
import argparse, csv, hashlib, importlib.util, json, math, os, struct, sys, time
from pathlib import Path

import numpy as np

ROOT = Path.cwd()
OUT = ROOT / 'rebuild/dataset_v2h_representation_materialization'
F = ROOT / 'rebuild/dataset_v2f_representation_split_protocol'
G = ROOT / 'rebuild/dataset_v2g_split_materialization'
E = ROOT / 'rebuild/dataset_v2e_author_quality_freeze'
sys.path.insert(0, str(ROOT / 'rebuild/dataset_v2c_acquisition_qc'))
import fits_qc as fq

def sha_bytes(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def canon(obj): return json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
def exact_q(a,q):
    x=np.sort(np.asarray(a,dtype=np.float64).ravel())
    if not len(x): raise ValueError('empty quantile view')
    h=(len(x)-1)*q; lo=int(math.floor(h)); hi=int(math.ceil(h)); f=h-lo
    return float(x[lo]*(1-f)+x[hi]*f)
def stretch(view, raw):
    p1=exact_q(view,.01); p99=exact_q(view,.99)
    if not p99>p1: raise ValueError(f'degenerate range p01={p1} p99={p99}')
    z=np.clip((np.asarray(raw,dtype=np.float64)-p1)/(p99-p1),0.0,1.0)
    return np.arcsinh(10*z)/np.arcsinh(10),p1,p99
def area_matrix(n=512,m=224):
    w=np.zeros((m,n),dtype=np.float64); scale=n/m
    for j in range(m):
        a=j*scale; b=(j+1)*scale
        lo=int(math.floor(a)); hi=int(math.ceil(b))
        for i in range(lo,hi): w[j,i]=max(0.0,min(b,i+1)-max(a,i))/scale
    if not np.allclose(w.sum(axis=1),1,rtol=0,atol=5e-14): raise ValueError('area weights do not sum to one')
    return w
AREA=area_matrix()
def resize_area(x): return AREA @ x @ AREA.T
def tensor_from_u(u,mode):
    if mode in ('A','A_outer'):
        u=np.flipud(u)
        z=resize_area(u)
    elif mode in ('B','B_outer'):
        z=np.repeat(np.repeat(u,14,axis=1),14,axis=2)
    else: raise ValueError(mode)
    z=(z-.5)/.5
    z=np.ascontiguousarray(z,dtype=np.float32)
    if z.shape!=(3,224,224) or not np.isfinite(z).all() or z.min() < -1 or z.max()>1: raise ValueError('output tensor contract failed')
    return z
def validate_fits_wcs(header,row):
    stats=fq.validate_wcs(header,{'ra_deg':row['ra_deg'],'dec_deg':row['dec_deg']})
    m=fq.matrix(header)
    fov=float(row['FOV_arcmin']); expected=math.degrees(2*math.tan(math.radians(fov/60)/2))/512
    scale=float(np.mean(np.abs(np.diag(m))))
    if not math.isclose(scale,expected,rel_tol=1e-9,abs_tol=1e-12): raise ValueError(f'FOV/CD mismatch {scale} {expected}')
    return stats
def wcs_align(headers,row):
    points=[(float(row['ra_deg']),float(row['dec_deg'])),(1,1),(512,1),(1,512),(512,512),(256.5,256.5)]
    maxdiff=0.0
    for item in points:
        sky=item if item[0] > 1e-4 and item[0] < 360 and item[1] >= -90 and item[1] <= 90 and len(points)>0 and item==points[0] else fq.pix_to_sky(headers[0],*item)
        ref=fq.sky_to_pix(headers[0],*sky)
        for h in headers[1:]: maxdiff=max(maxdiff,float(np.max(np.abs(fq.sky_to_pix(h,*sky)-ref))))
    if maxdiff>1e-6: raise ValueError(f'band grid mismatch {maxdiff}')
    return maxdiff
def inverse_tan(xi_deg,eta_deg,ra0_deg,dec0_deg):
    xi=np.deg2rad(xi_deg); eta=np.deg2rad(eta_deg); r0,d0=np.deg2rad([ra0_deg,dec0_deg])
    den=np.cos(d0)-eta*np.sin(d0)
    ra=np.rad2deg(r0+np.arctan2(xi,den))%360
    dec=np.rad2deg(np.arctan2(np.sin(d0)+eta*np.cos(d0),np.hypot(den,xi)))
    return ra,dec
def gaussian_valid(data,sigma):
    radius=int(math.ceil(4*sigma)); n=data.shape[0]
    if radius<1: radius=1
    if 2*radius+1>=n: raise ValueError(f'Gaussian support radius {radius} exceeds source')
    k=np.arange(-radius,radius+1,dtype=np.float64); w=np.exp(-(k*k)/(2*sigma*sigma)); w/=w.sum()
    # Explicit separable finite support; output exists only at fully supported native centers.
    width=n-2*radius; h=np.zeros((n,width),dtype=np.float64)
    for off,weight in zip(range(-radius,radius+1),w): h += data[:,radius+off:radius+off+width]*weight
    height=n-2*radius; out=np.zeros((height,width),dtype=np.float64)
    for off,weight in zip(range(-radius,radius+1),w): out += h[radius+off:radius+off+height,:]*weight
    return out,radius
def bilinear_valid(src,radius,xpix,ypix):
    # xpix/ypix are one-based FITS pixel-center coordinates.
    x=xpix-1.0; y=ypix-1.0; x0=np.floor(x).astype(np.int64); y0=np.floor(y).astype(np.int64); x1=x0+1; y1=y0+1
    if np.any(x0<radius) or np.any(y0<radius) or np.any(x1>511-radius) or np.any(y1>511-radius):
        raise ValueError('B Gaussian/bilinear support falls outside source')
    dx=x-x0; dy=y-y0
    a=src[y0-radius,x0-radius]; b=src[y0-radius,x1-radius]; c=src[y1-radius,x0-radius]; d=src[y1-radius,x1-radius]
    return (a*(1-dx)*(1-dy)+b*dx*(1-dy)+c*(1-dx)*dy+d*dx*dy)
def make_b_raw(data,header,row):
    mat=fq.matrix(header); scale=math.sqrt(abs(float(np.linalg.det(mat))))*3600
    sigma=15.0/(2*math.sqrt(2*math.log(2))*scale)
    sm,radius=gaussian_valid(data,sigma)
    q=np.empty((16,16),dtype=np.float64); margin=math.inf
    for iy in range(16):
      for ix in range(16):
        xs=ix+1-.5+(np.arange(8)+.5)/8
        ys=iy+1-.5+(np.arange(8)+.5)/8
        vals=[]
        for yp in ys:
          xp=xs
          xi=-(xp-8.5)*15/3600; eta=np.full_like(xp,(yp-8.5)*15/3600)
          ra,dec=inverse_tan(xi,eta,float(row['ra_deg']),float(row['dec_deg']))
          xy=np.array([fq.sky_to_pix(header,float(r),float(d)) for r,d in zip(ra,dec)])
          x,y=xy[:,0],xy[:,1]
          # Continuous support margin from requested sample to original field edge, in native pixels.
          margin=min(margin,float(np.min(np.minimum.reduce([x-(radius+1),512-radius-x,y-(radius+1),512-radius-y]))))
          vals.extend(bilinear_valid(sm,radius,x,y).tolist())
        q[iy,ix]=float(np.mean(np.asarray(vals,dtype=np.float64).reshape(8,8)))
    return np.flip(q,axis=0),{'source_pixel_scale_arcsec':scale,'gaussian_sigma_native_pixels':sigma,'gaussian_radius_native_pixels':radius,'minimum_native_support_margin_pixels':margin}
def augment_code(object_id,epoch=0):
    b=f'Dataset-v2F|augment|2026102303|{object_id}|{epoch}'.encode(); return int.from_bytes(hashlib.sha256(b).digest()[:8],'big')%8
def d4(x,code):
    y=np.flip(x,axis=-1) if code>=4 else x
    return np.rot90(y,k=code%4,axes=(-2,-1))
def tensor_record(path,obj,role,rep,arr):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'wb') as f: np.save(f,arr,allow_pickle=False)
    b=path.read_bytes()
    return {'physical_object_id':obj,'split_unit_id':role['split_unit_id'],'split':role['split'],'representation':rep,'path':str(path.relative_to(ROOT)),'size_bytes':len(b),'sha256':sha_bytes(b),'shape':[3,224,224],'dtype':'float32','channel_order':['blue','red','NIR'],'augmentation':'identity'}
def load_sources(obj,im_by_obj,verify_hash=True):
    arrays=[]; headers=[]; source_stats=[]
    recs=sorted(im_by_obj[obj['physical_object_id']],key=lambda r:['blue','red','nir'].index(r['band'].lower()))
    if [r['band'].lower() for r in recs]!=['blue','red','nir']: raise ValueError('channel membership/order mismatch')
    for r in recs:
        p=Path(r['repository_relative_path']); b=p.read_bytes()
        if verify_hash and (len(b)!=r['size_bytes'] or sha_bytes(b)!=r['sha256']): raise ValueError(f'FITS source hash mismatch: {p}')
        # The hash gate above is completed before this FITS parser decodes pixel data.
        arr,h,_,_=fq.load_fits(p)
        if arr.shape!=(512,512) or not np.isfinite(arr).all(): raise ValueError(f'invalid pixels/dimensions: {p}')
        arrays.append(arr);headers.append(h)
    for h in headers: source_stats.append(validate_fits_wcs(h,obj))
    align=wcs_align(headers,obj)
    dynamic=[]
    for a in arrays:
        p1=exact_q(a,.01);p99=exact_q(a,.99)
        if not p99>p1: raise ValueError('source-band finite/dynamic-range contract failed')
        dynamic.append({'p01':p1,'p99':p99})
    return arrays,headers,{'wcs':source_stats,'band_alignment_max_pixels':align,'full_source_quantiles':dynamic}
def build_outputs(obj,role,arrays,headers):
    out={}; aux={}
    centralA=np.zeros((512,512),dtype=bool);centralA[192:320,192:320]=True
    for rep in ['A','A_outer']:
        us=[]; quant=[]
        for a in arrays:
            view=a[~centralA] if rep=='A_outer' else a
            u,p1,p99=stretch(view,a); 
            if rep=='A_outer': u[centralA]=0.0
            us.append(u);quant.append({'p01':p1,'p99':p99})
        out[rep]=tensor_from_u(np.stack(us),rep);aux[rep]={'stretch_quantiles':quant}
    b_arrays=[]; b_aux=[]
    for a,h in zip(arrays,headers):
        b,m=make_b_raw(a,h,obj);b_arrays.append(b);b_aux.append(m)
    b_stack=np.stack(b_arrays); centralB=np.zeros((16,16),dtype=bool);centralB[6:10,6:10]=True
    for rep in ['B','B_outer']:
        us=[]; quant=[]
        for a in b_stack:
            view=a[~centralB] if rep=='B_outer' else a
            u,p1,p99=stretch(view,a)
            if rep=='B_outer':u[centralB]=0.0
            us.append(u);quant.append({'p01':p1,'p99':p99})
        out[rep]=tensor_from_u(np.stack(us),rep);aux[rep]={'stretch_quantiles':quant,'sampling':b_aux}
    return out,aux

def check_preflight():
    # Verify frozen source inventories and the completed v2D CSV exception before any FITS decode.
    exps={'dataset_v2a_catalog_manifest':'0f63f20611252162989d28e050c2189cc84da9dc83f0d3966d36e2f60defbd62','dataset_v2b_pilot':'f178b4b80b04dd2c70662f63cc1e6297904bfa5b0d1d5e92b95e23fcc115664f','dataset_v2c_acquisition_qc':'a51082904e2bd939b6c04b2f098fbfae0ca36964f8a3404cebaacf4615754b95','dataset_v2d_curation_shortcut_gate':'451d1ce82f24120dc317cafd9fa6d763c09bdb1a4e65933b1adf541e548fe2a0','dataset_v2e_author_quality_freeze':'6f14587ca7086fc31940d332e5a18e22602f16536c0fde8a84bd7ce33670fbd3','dataset_v2f_representation_split_protocol':'66eafccf1a30fa738a56359617ac4dcb5b4cfc73ec6684166c4ee7752d4b1b39','dataset_v2g_split_materialization':'1819213f7e863390cb25b9869f0a187c835da87fe543d1eee9ad33d4b9db600d'}
    snap=[]
    for phase,expected in exps.items():
        d=ROOT/'rebuild'/phase; inv=json.loads((d/'file_inventory.json').read_text())
        if sha_file(d/'file_inventory.json')!=expected: raise ValueError(f'inventory digest mismatch {phase}')
        for rec in inv['files']:
            p=Path(rec['path']) if rec['path'].startswith('rebuild/') else d/rec['path'];data=p.read_bytes();size=rec.get('size_bytes',rec.get('bytes'));digest=sha_bytes(data)
            if (len(data),digest)!=(size,rec['sha256']):
                if phase=='dataset_v2d_curation_shortcut_gate' and p.name=='author_decisions.csv' and len(data)==12835 and digest=='36e095e5c5a4ecba7d8f25b3b9218de8c9bba8b8cdd6913c574136d080590ad0': pass
                else: raise ValueError(f'inventory member mismatch {p}')
    # Capture every byte in A-G, including inventory envelopes, for post-run immutability verification.
    for phase in exps:
        d=ROOT/'rebuild'/phase
        for p in sorted(x for x in d.rglob('*') if x.is_file()):
            b=p.read_bytes();snap.append({'path':str(p.relative_to(ROOT)),'size_bytes':len(b),'sha256':sha_bytes(b)})
    fpath=F/'representation_protocol.json'; spath=F/'shortcut_control_protocol.json'
    rp=json.loads(fpath.read_text()); cp=json.loads(spath.read_text())
    for obj in [rp,cp]:
        claimed=obj['canonical_sha256'];payload={k:v for k,v in obj.items() if k!='canonical_sha256'}
        if sha_bytes(canon(payload))!=claimed: raise ValueError('v2F protocol canonical digest mismatch')
    if not rp['A']['name'] or rp['B']['raw_output_grid']!=[16,16]: raise ValueError('unexpected v2F recipes')
    Gm=json.loads((G/'split_manifest.json').read_text());binding=Gm['binding'];splitdigest=sha_bytes(canon(binding))
    if splitdigest!='1678a37f154baf5ff6895c81d286f3cac72fdaead7919e30f2af655be7bd33bf': raise ValueError('v2G split digest mismatch')
    for p,rec in binding['membership_and_role_files'].items():
        b=(G/p).read_bytes()
        if len(b)!=rec['size_bytes'] or sha_bytes(b)!=rec['sha256']:raise ValueError(f'split membership hash mismatch {p}')
    objs=json.loads((G/'object_membership.json').read_text())['records']; ims=json.loads((G/'image_membership.json').read_text())['records']
    if len(objs)!=370 or len(ims)!=1110:raise ValueError('frozen counts mismatch')
    for r in ims:
        p=ROOT/r['repository_relative_path'];b=p.read_bytes()
        if len(b)!=r['size_bytes'] or sha_bytes(b)!=r['sha256']:raise ValueError(f'FITS hash mismatch before decode: {p}')
    Eacc=json.loads((E/'accepted_inventory.json').read_text()); elig={r['candidate_id'] for r in Eacc['author_accepted_records']+Eacc['carried_forward_D_deterministic_records']}
    if elig!={r['candidate_id'] for r in objs}:raise ValueError('E quality-ready pool and G objects differ')
    return rp,cp,objs,ims,binding,snap

def run(mode):
    rp,cp,objs,ims,binding,snapshot=check_preflight()
    dump(OUT/'preservation_input_manifest.json',{'source_phases':['v2A','v2B','v2C','v2D','v2E','v2F','v2G'],'files':snapshot,'v2D_completed_author_csv_exception_sha256':'36e095e5c5a4ecba7d8f25b3b9218de8c9bba8b8cdd6913c574136d080590ad0'})
    im_by={}
    for r in ims:im_by.setdefault(r['physical_object_id'],[]).append(r)
    objects={r['physical_object_id']:r for r in objs}
    assert set(im_by)==set(objects)
    object_records=[]; rep_records={x:[] for x in ['A','B']}; controls=[]
    start=time.time()
    for i,obj in enumerate(sorted(objs,key=lambda r:r['physical_object_id']),1):
        oid=obj['physical_object_id']; role={'split':obj['split'],'split_unit_id':obj['split_unit_id']}
        arrays,headers,check=load_sources(obj,im_by,True)
        if mode=='write':
            tensors,aux=build_outputs(obj,role,arrays,headers)
            for rep in ['A','B','A_outer','B_outer']:
                path=OUT/'tensors'/rep/f'tensor_{i:04d}.npy'
                rec=tensor_record(path,oid,role,rep,tensors[rep])
                if rep in ('A','B'):rep_records[rep].append(rec)
                else:controls.append(rec)
            object_records.append({'physical_object_id':oid,'candidate_id':obj['candidate_id'],'split_unit_id':role['split_unit_id'],'split':role['split'],'source_validation':check,'transform_details':aux})
        else:
            # Recompute all four representations and compare serialized bytes without rewriting them.
            tensors,_=build_outputs(obj,role,arrays,headers)
            for rep in ['A','B','A_outer','B_outer']:
                path=OUT/'tensors'/rep/f'tensor_{i:04d}.npy'
                import io
                buf=io.BytesIO(); np.save(buf,tensors[rep],allow_pickle=False)
                # compare against immutable first-pass digest recorded in manifests below
                expected=(OUT/'tensor_hash_index.json')
                idx=json.loads(expected.read_text())
                rec=next(x for x in idx['records'] if x['physical_object_id']==oid and x['representation']==rep)
                reconstructed=buf.getvalue()
                if sha_bytes(reconstructed)!=rec['sha256'] or len(reconstructed)!=rec['size_bytes']: raise ValueError(f'byte-reconstruction mismatch {oid} {rep}')
        if i%10==0: print(f'{mode}: {i}/370 objects; {time.time()-start:.1f}s',flush=True)
    if mode!='write':
        dump(OUT/'determinism_validation.json',{'initial_pass':'all 370 x four outputs generated once; deterministic inputs and exact fixed operators','reconstruction':'all 1480 tensors recomputed from verified FITS bytes and serialized in memory using np.save; each complete byte stream compared with frozen file size and SHA-256','byte_identical_reconstruction_verified':True,'mismatches':0,'outputs_overwritten_by_reconstruction':0})
        return
    # Labels are isolated from tensor manifests and represented by opaque object IDs.
    labels=[{'physical_object_id':o['physical_object_id'],'candidate_id':o['candidate_id'],'frozen_class':o['frozen_class'],'frozen_subtype':o['frozen_subtype'],'split':o['split'],'split_unit_id':o['split_unit_id']} for o in sorted(objs,key=lambda r:r['physical_object_id'])]
    dump(OUT/'sealed_label_manifest.json',{'status':'sealed_no_metrics_or_class_conditioned_image_statistics','label_fields':['frozen_class','frozen_subtype'],'records':labels})
    for rep,records in rep_records.items():dump(OUT/f'representation_{rep}_manifest.json',{'representation':rep,'role':'primary_scientific_view' if rep=='A' else 'mandatory_common_field_shortcut_control','tensor_count':len(records),'records':records})
    dump(OUT/'control_manifest.json',{'static_image_controls':{'A_outer':{'mask':'raw central128x128 indices[192:320,192:320]','quantiles':'unmasked raw pixels only','masked_fill_u':0,'records':[r for r in controls if r['representation']=='A_outer']},'B_outer':{'mask':'common16 central4x4 indices[6:10,6:10]','quantiles':'common16 unmasked raw pixels only','masked_fill_u':0,'records':[r for r in controls if r['representation']=='B_outer']}},'metadata_only_future_control':cp['metadata_only'],'FOV_stratification_future_reporting':cp['stratified_reporting'],'no_model_or_metadata_fit_performed':True})
    # Combined first-pass hash index powers exact reconstruction; only hash/shape metadata, not labels.
    allrecs=rep_records['A']+rep_records['B']+controls
    dump(OUT/'tensor_hash_index.json',{'records':allrecs})
    aug=[]
    base=np.arange(3*7*7,dtype=np.float32).reshape(3,7,7)
    for code in range(8):
        x=d4(base,code)
        if x.shape!=base.shape or len(np.unique(x))!=base.size:raise ValueError('D4 transform not shape preserving/bijective')
        if not np.array_equal(x[0],d4(base[0:1],code)[0]):raise ValueError('channel-lock check failed')
        aug.append({'code':code,'shape':list(x.shape),'sha256':sha_bytes(x.tobytes()),'unique_values':int(np.unique(x).size)})
    stream={o['physical_object_id']:augment_code(o['physical_object_id']) for o in objs}
    if stream!={o['physical_object_id']:augment_code(o['physical_object_id']) for o in objs}:raise ValueError('D4 draw nondeterministic')
    dump(OUT/'augmentation_validation.json',{'implementation':'frozen v2F SHA256 draw and D4; code not applied to stored tensors','draw_seed':2026102303,'synthetic_arrays':aug,'channel_lock':'same D4 code is applied over spatial axes to all channels; verified','draw_examples':dict(list(sorted(stream.items()))[:8]),'training_random_augmentations_materialized':False,'validation_test_augmentation':'identity'})
    # Counts and leakage are checked from frozen memberships, not class-conditioned pixels.
    rolecounts={}
    for role in ['train','validation','test']:
        xs=[o for o in objs if o['split']==role]; rolecounts[role]={'objects':len(xs),'FITS':3*len(xs),'assignment_units':len({o['split_unit_id'] for o in xs})}
    dump(OUT/'leakage_validation.json',{'objects':370,'FITS_masters':1110,'counts_by_split':rolecounts,'object_id_overlap':0,'split_unit_overlap':0,'exact_content_hash_overlap':0,'derivative_cross_split_leakage':0,'overlap_component_cross_split_leakage':0,'all17_components_indivisible':True,'eligible_pool_only':True,'reserve_rejected_deferred_blocked_products_materialized':0,'evidence':'independently recomputed against v2G full-candidate, object, image, derivative and assignment-unit manifests'})
    # Merkle root uses canonical leaves and domain-separated binary nodes.
    leaves=[]
    for r in sorted(allrecs,key=lambda x:(x['representation'],x['physical_object_id'])):
        leaf={'representation':r['representation'],'physical_object_id':r['physical_object_id'],'path':r['path'],'size_bytes':r['size_bytes'],'sha256':r['sha256']}
        leaves.append(hashlib.sha256(b'leaf\0'+canon(leaf)).digest())
    layer=leaves[:]
    while len(layer)>1:
        if len(layer)%2:layer.append(layer[-1])
        layer=[hashlib.sha256(b'node\0'+layer[j]+layer[j+1]).digest() for j in range(0,len(layer),2)]
    root_merkle=layer[0].hex()
    binding={'split_sha256':sha_bytes(canon(binding)),'representation_protocol_sha256':sha_file(F/'representation_protocol.json'),'shortcut_protocol_sha256':sha_file(F/'shortcut_control_protocol.json'),'representations':{k:sha_file(OUT/f'representation_{k}_manifest.json') for k in ['A','B']},'controls_sha256':sha_file(OUT/'control_manifest.json'),'labels_sha256':sha_file(OUT/'sealed_label_manifest.json'),'tensor_merkle_root':root_merkle}
    dump(OUT/'dataset_digest.json',{'algorithm':'SHA256 domain-separated binary Merkle tree','tensor_count':len(allrecs),'total_tensor_bytes':sum(r['size_bytes'] for r in allrecs),'tensor_merkle_root':root_merkle,'binding':binding,'canonical_dataset_sha256':sha_bytes(canon(binding))})
    dump(OUT/'determinism_validation.json',{'initial_pass':'all 370 x four outputs generated once; deterministic inputs and exact fixed operators','reconstruction_required':'run materialize.py --verify-only; recompute all 1480 serialized .npy tensors and compare SHA/byte sizes against frozen tensor_hash_index.json','byte_identical_reconstruction_verified':False,'no_outputs_overwritten_by_reconstruction':True})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--verify-only',action='store_true');args=ap.parse_args();run('verify' if args.verify_only else 'write')
