from pathlib import Path
import json,time,hashlib
import numpy as np
from PIL import Image
class VisionClient:
 def __init__(self,root,emit):self.root=Path(root);self.index=0;self.emit=emit
 def request(self,op,obs,meta,world_prior=None):
  self.index+=1;assert self.index<=24;folder=self.root/('%02d'%self.index);folder.mkdir(exist_ok=False);rgb=np.asarray(obs['agentview_image']);assert rgb.shape==(256,256,3) and rgb.dtype==np.uint8
  p=folder/'front.png';Image.fromarray(rgb).save(p);request=dict(index=self.index,op=op,meta=meta,png_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),world_prior=None if world_prior is None else np.asarray(world_prior).tolist())
  q=folder/'REQUEST.tmp';q.write_text(json.dumps(request,indent=2,allow_nan=False));q.rename(folder/'REQUEST.json');started=time.time();deadline=time.monotonic()+240
  while not (folder/'RESPONSE.json').exists():
   assert not (self.root/'FAILED.json').exists(),(self.root/'FAILED.json').read_text() if (self.root/'FAILED.json').exists() else ''
   assert time.monotonic()<deadline,'Finite RGB request timeout';time.sleep(.05)
  response=json.loads((folder/'RESPONSE.json').read_text());assert response['index']==self.index and response['rgb_sha256']==request['rgb_sha256'] and response['GT_read'] is False
  assert response['request_sha256']==hashlib.sha256((folder/'REQUEST.json').read_bytes()).hexdigest();self.emit(dict(event='public_rgb_request_closed',step=meta['step'],op=op,index=self.index,seconds=time.time()-started,request=request,response=response))
  return None if response['pose'] is None else np.asarray(response['pose']),response
 def close(self):
  with (self.root/'CLOSE').open('x') as f:f.write('EXACT_OWNER_DONE\n')
