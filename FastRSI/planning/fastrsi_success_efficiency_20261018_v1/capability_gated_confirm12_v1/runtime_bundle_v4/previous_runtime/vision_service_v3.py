"""Separate ABI/process: only public images, static CAD/calibration and priors enter."""
from pathlib import Path
import json,time,hashlib,traceback
import numpy as np
from PIL import Image
from online_vision import Vision
R=Path('/rpc')
def write(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,indent=2,allow_nan=False));t.rename(p)
def emit(v):
 with (R/'NETWORK_LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**v))+'\n')
def main():
 start=time.time();v=None;error=None;requests=0
 try:
  assert not Path('/core').exists() and not Path('/history').exists() and not Path('/eval').exists()
  v=Vision('/input',R/'model_outputs',emit);write(R/'READY.json',dict(time=time.time(),status='READY',no_simulator_access=True))
  deadline=time.monotonic()+1200
  for index in range(1,25):
   root=R/('%02d'%index);p=root/'REQUEST.json'
   while not p.exists():
    if (R/'CLOSE').exists():return
    assert time.monotonic()<deadline,'Finite visual service deadline';time.sleep(.05)
   q=json.loads(p.read_text());assert q['index']==index and q['op'] in ['subject','basket'];path=root/'front.png';assert hashlib.sha256(path.read_bytes()).hexdigest()==q['png_sha256'];rgb=np.asarray(Image.open(str(path)).convert('RGB'));assert hashlib.sha256(rgb.tobytes()).hexdigest()==q['rgb_sha256'];obs=dict(agentview_image=rgb)
   if q['op']=='subject':pose=v.subject(obs,q['meta'],q.get('world_prior'));proof=dict(method='NEURAL_RGB',GT_read=False)
   else:pose,proof=v.basket(obs,q['meta'])
   write(root/'RESPONSE.json',dict(time=time.time(),index=index,request_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),rgb_sha256=q['rgb_sha256'],pose=None if pose is None else np.asarray(pose).tolist(),proof=proof,GT_read=False));requests+=1
  raise RuntimeError('Fixed visual request count exhausted')
 except BaseException:error=traceback.format_exc();write(R/'FAILED.json',dict(time=time.time(),error=error));raise
 finally:
  cleanup=None
  if v is not None:
   try:v.close()
   except BaseException:cleanup=traceback.format_exc()
  write(R/'SERVICE_RESULT.json',dict(time=time.time(),seconds=time.time()-start,requests=requests,error=error,cleanup_error=cleanup,physics_steps=0,money=None))
if __name__=='__main__':main()
