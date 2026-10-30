"""Resume-safe retrieval of the exact sixty predeclared DSS2 products."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

OUT=Path(__file__).resolve().parent
PLAN=json.loads((OUT/'retrieval_plan.json').read_text())
REQUESTS=PLAN['requests'];URLS={r['url'] for r in REQUESTS}
assert len(REQUESTS)==len(URLS)==60 and len({r['candidate_id'] for r in REQUESTS})==20
for name,key in [('pilot_protocol.json','protocol_sha256'),('pilot_selection.json','selection_sha256'),('survey_reuse_policy.json','policy_sha256')]:
 assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==PLAN[key]
assert {r['survey_id'] for r in REQUESTS}=={'CDS/P/DSS2/blue','CDS/P/DSS2/red','CDS/P/DSS2/NIR'}
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
class NoRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,*args,**kwargs):return None
def main():
 for name in ['fits','request_records','failed_responses']:(OUT/name).mkdir(exist_ok=True)
 opener=urllib.request.build_opener(NoRedirect());last=0.0
 for index,row in enumerate(REQUESTS,start=1):
  meta=OUT/'request_records'/(row['candidate_id']+'_'+row['band']+'.json')
  if meta.exists():
   old=json.loads(meta.read_text());assert old['url']==row['url']
   if old.get('response_path'):assert sha(OUT/old['response_path'])==old['response_sha256']
   if index%6==0:print(json.dumps({'completed_slots':index,'resume_zero_network_for_existing':True}),flush=True)
   continue
  delay=max(0,1-(time.monotonic()-last))
  if delay:time.sleep(delay)
  last=time.monotonic();begin=last
  record={**row,'request_index':index,'started_utc':now(),'method':'GET','http_status':None,'response_path':None,'response_sha256':None,'size_bytes':None,
    'request_headers':{'User-Agent':'Astroclassifier-Dataset-v2B-20-object-research-pilot/1.0','Accept':'application/fits,application/octet-stream'},
    'redirects_followed':0,'automatic_retries':0,'error':None}
  response=None;data=None
  try:response=opener.open(urllib.request.Request(row['url'],headers=record['request_headers']),timeout=60)
  except urllib.error.HTTPError as exc:response=exc
  except (urllib.error.URLError,TimeoutError,OSError) as exc:record['error']=type(exc).__name__+': '+str(exc)
  if response is not None:
   with response:
    record['http_status']=response.status;record['response_headers']=list(response.headers.items());record['content_type']=response.headers.get('Content-Type','')
    try:data=response.read(20000001)
    except (TimeoutError,OSError) as exc:record['error']='read_failed: '+str(exc)
    if record['http_status'] in {301,302,303,307,308}:record['location_not_followed']=response.headers.get('Location')
  if data is not None:
   is_fits=record['http_status']==200 and data.startswith(b'SIMPLE  =') and len(data)<=20000000
   path=OUT/row['output'] if is_fits else OUT/'failed_responses'/(row['candidate_id']+'_'+row['band']+'.response.bin')
   with path.open('xb') as stream:stream.write(data)
   path.chmod(0o444)
   record.update({'response_path':str(path.relative_to(OUT)),'response_sha256':hashlib.sha256(data).hexdigest(),'size_bytes':len(data),'fits_master_retained':is_fits})
   if len(data)>20000000:record['error']='response_exceeds_frozen_limit'
  record['finished_utc']=now();record['elapsed_seconds']=time.monotonic()-begin
  with meta.open('x') as stream:json.dump(record,stream,sort_keys=True,indent=2,ensure_ascii=False);stream.write('\n')
  meta.chmod(0o444)
  with (OUT/'network_request_log.jsonl').open('a') as stream:stream.write(json.dumps(record,sort_keys=True,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
  assert len(list((OUT/'fits').glob('*.fits')))<=60
  if index%6==0 or not record.get('fits_master_retained'):
   print(json.dumps({'completed_slots':index,'total':60,'last_status':record['http_status'],'last_bytes':record['size_bytes'],'retained_fits':len(list((OUT/'fits').glob('*.fits'))),'error':record['error']}),flush=True)
  if record['http_status'] in {401,403,429}:raise SystemExit('Access/rate restriction: no bypass or additional download')
if __name__=='__main__':
 sys.dont_write_bytecode=True
 def boundary(event,args):
  if event=='urllib.Request' and args[0] not in URLS:raise RuntimeError('URL outside sixty frozen requests')
  if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
   p=Path(os.fsdecode(args[0])).resolve();mode,flags=args[1:3]
   if ((isinstance(mode,str) and any(c in mode for c in 'wax+')) or flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC)) and not p.is_relative_to(OUT):raise RuntimeError('Write outside v2B')
   if 'baseline_comparison_v1' in p.parts or p.suffix.lower() in {'.pt','.pth','.h5','.hdf5','.safetensors'}:raise RuntimeError('No v1/model/checkpoint access')
 sys.addaudithook(boundary);main()
