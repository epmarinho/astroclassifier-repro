"""Unchanged pure v2B FITS/WCS/QC functions; source hash recorded in protocol inputs."""
import math
from pathlib import Path
import numpy as np

def value(text):
 text=text.strip()
 if text.startswith("'"):
  result='';i=1
  while i<len(text):
   if text[i]=="'":
    if i+1<len(text) and text[i+1]=="'":result+="'";i+=2;continue
    return result.rstrip()
   result+=text[i];i+=1
  raise ValueError('Unterminated FITS string')
 token=text.split('/',1)[0].strip()
 if token in {'T','F'}:return token=='T'
 try:return int(token)
 except ValueError:
  try:return float(token.replace('D','E'))
  except ValueError:return token

def load_fits(path):
 raw=Path(path).read_bytes();assert len(raw)%2880==0 and raw.startswith(b'SIMPLE  =')
 cards=[];header={};last=None;offset=None
 for start in range(0,len(raw),80):
  card=raw[start:start+80].decode('ascii');cards.append(card);key=card[:8].strip()
  if key=='END':offset=((start+80+2879)//2880)*2880;break
  if key in {'HISTORY','COMMENT'}:header.setdefault(key,[]).append(card[8:].strip())
  elif key=='CONTINUE' and last:
   suffix=value(card[8:]);assert isinstance(suffix,str)
   assert isinstance(header[last],str) and header[last].endswith('&')
   header[last]=header[last][:-1]+suffix
  elif card[8:10]=='= ':
   assert key not in header,'Unexpected repeated primary keyword'
   header[key]=value(card[10:]);last=key
 assert offset is not None and header['SIMPLE'] is True and header['NAXIS']==2
 assert header['NAXIS1']==header['NAXIS2']==512
 bitpix=header['BITPIX'];dtype={8:'>u1',16:'>i2',32:'>i4',64:'>i8',-32:'>f4',-64:'>f8'}[bitpix]
 n=header['NAXIS1']*header['NAXIS2'];size=n*abs(bitpix)//8
 assert offset+size<=len(raw)
 assert offset+((size+2879)//2880)*2880==len(raw),'Unexpected extensions/trailing data: retain and flag, never repair'
 native=np.frombuffer(raw,dtype=dtype,count=n,offset=offset).reshape((512,512))
 invalid=~np.isfinite(native)
 if bitpix>0 and 'BLANK' in header:invalid|=native==header['BLANK']
 data=native.astype(np.float64)*float(header.get('BSCALE',1))+float(header.get('BZERO',0))
 data[invalid]=np.nan
 return data,header,cards,{'header_bytes':offset,'data_bytes':size,'padding_bytes':len(raw)-offset-size,'total_bytes':len(raw)}

def matrix(header):
 if 'CD1_1' in header:return np.array([[header['CD1_1'],header.get('CD1_2',0)],[header.get('CD2_1',0),header['CD2_2']]],dtype=float)
 pc=np.array([[header.get('PC1_1',1),header.get('PC1_2',0)],[header.get('PC2_1',0),header.get('PC2_2',1)]],dtype=float)
 return np.diag([header['CDELT1'],header['CDELT2']])@pc

def pix_to_sky(header,x,y):
 xi,eta=np.deg2rad(matrix(header)@np.array([x-header['CRPIX1'],y-header['CRPIX2']]))
 ra0,dec0=np.deg2rad([header['CRVAL1'],header['CRVAL2']]);den=math.cos(dec0)-eta*math.sin(dec0)
 return (math.degrees(ra0+math.atan2(xi,den))%360,math.degrees(math.atan2(math.sin(dec0)+eta*math.cos(dec0),math.hypot(den,xi))))

def sky_to_pix(header,ra,dec):
 ra0,dec0=np.deg2rad([header['CRVAL1'],header['CRVAL2']]);r,d=np.deg2rad([ra,dec]);dr=r-ra0
 den=math.sin(d)*math.sin(dec0)+math.cos(d)*math.cos(dec0)*math.cos(dr)
 assert den>0
 xi=math.cos(d)*math.sin(dr)/den;eta=(math.sin(d)*math.cos(dec0)-math.cos(d)*math.sin(dec0)*math.cos(dr))/den
 xy=np.linalg.solve(matrix(header),np.rad2deg([xi,eta]))
 return xy+np.array([header['CRPIX1'],header['CRPIX2']])

def separation(a,b):
 r1,d1,r2,d2=np.deg2rad([a[0],a[1],b[0],b[1]])
 h=math.sin((d2-d1)/2)**2+math.cos(d1)*math.cos(d2)*math.sin((r2-r1)/2)**2
 return math.degrees(2*math.asin(math.sqrt(min(1,max(0,h)))))*3600

def validate_wcs(h,row):
 assert h['CTYPE1']=='RA---TAN' and h['CTYPE2']=='DEC--TAN'
 m=matrix(h);assert np.all(np.isfinite(m)) and abs(np.linalg.det(m))>0
 assert h.get('RADESYS','ICRS')=='ICRS'
 centre_error=separation((h['CRVAL1'],h['CRVAL2']),(row['ra_deg'],row['dec_deg']))
 xy=sky_to_pix(h,row['ra_deg'],row['dec_deg']);pixel_error=float(np.linalg.norm(xy-[256.5,256.5]))
 assert centre_error<=1 and pixel_error<=1
 max_error=0
 for x,y in [(1,1),(512,1),(1,512),(512,512),(256.5,256.5)]:
  sky=pix_to_sky(h,x,y);back=sky_to_pix(h,*sky);max_error=max(max_error,float(np.max(np.abs(back-[x,y]))))
 assert max_error<1e-6
 assert m[0,0]<0 and m[1,1]>0 and abs(m[0,1])+abs(m[1,0])<1e-12
 return {'matrix_deg_per_pixel':m.tolist(),'reference_centre_error_arcsec':centre_error,'catalogue_position_fits_1based_xy':xy.tolist(),
   'catalogue_to_geometric_centre_pixels':pixel_error,'TAN_roundtrip_max_error_pixels':max_error,'pixel_scale_arcsec':math.sqrt(abs(np.linalg.det(m)))*3600,'north_up_east_left_after_preview_flip':True}

def screen(data):
 finite=np.isfinite(data);values=data[finite];n=data.size
 assert len(values)>0,'All pixels missing'
 p1,p50,p99=np.percentile(values,[1,50,99]);invalid=1-len(values)/n;zero=float(np.count_nonzero(data==0)/n)
 lower,upper=float(np.min(values)),float(np.max(values));maxfrac=float(np.count_nonzero(data==upper)/len(values))
 medrow=np.nanmedian(data,axis=1);medcol=np.nanmedian(data,axis=0)
 def outlier(v):
  centre=np.nanmedian(v);mad=np.nanmedian(np.abs(v-centre))
  return float(np.max(np.abs(v-centre))/(1.4826*mad)) if mad>0 else None
 rz,cz=outlier(medrow),outlier(medcol)
 flags=[]
 if invalid>.01:flags.append('invalid_or_declared_blank_fraction_above_1percent')
 if zero>.95:flags.append('mostly_zero_requires_blank_coverage_review')
 if p99<=p1:flags.append('degenerate_percentile_dynamic_range')
 if maxfrac>.005:flags.append('high_empirical_max_plateau_possible_clipping')
 if any(z is not None and z>8 for z in [rz,cz]):flags.append('row_or_column_median_artifact_proxy')
 return {'finite_fraction':len(values)/n,'invalid_or_declared_blank_fraction':invalid,'zero_fraction_not_automatically_missing':zero,
  'minimum':lower,'p01':float(p1),'median':float(p50),'p99':float(p99),'maximum':upper,'empirical_max_fraction':maxfrac,
  'row_median_outlier_mad':rz,'column_median_outlier_mad':cz,'screening_flags':flags,
  'coverage_caveat':'Finite/BLANK mask and zeros are pixel-level proxies; no independent footprint or visibility certification.'}

def render(data):
 finite=np.isfinite(data);values=data[finite]
 if not len(values):return np.zeros(data.shape,dtype=np.uint8)
 lo,hi=np.percentile(values,[1,99])
 if hi<=lo:return np.zeros(data.shape,dtype=np.uint8)
 scaled=np.clip((data-lo)/(hi-lo),0,1);scaled=np.arcsinh(10*scaled)/np.arcsinh(10);scaled[~finite]=0
 return np.flipud(np.rint(255*scaled).astype(np.uint8))
