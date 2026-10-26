"""Bounded frozen neural RGB inference. Only public RGB, source crops, CAD and static calibration are mounted."""
from pathlib import Path
import json,hashlib,time,traceback
import numpy as np,cv2
I=Path('/input');O=Path('/out')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def plain(v):
 if isinstance(v,np.ndarray):return v.tolist()
 if isinstance(v,np.generic):return v.item()
 if isinstance(v,dict):return {k:plain(x) for k,x in v.items()}
 if isinstance(v,(list,tuple)):return [plain(x) for x in v]
 return v
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(plain(v),f,indent=2,allow_nan=False)
def event(v):
 with (O/'NETWORK_LEDGER.jsonl').open('a') as f:f.write(json.dumps(plain(dict(time=time.time(),**v)))+'\n')
def detect(image,label,manifest):
 peaks=[]
 for t in manifest['templates'][label]:
  p=I/t['path'];assert sha(p)==t['sha256'];source=cv2.imread(str(p))
  for scale in [.4+.05*i for i in range(23)]:
   size=(max(3,round(source.shape[1]*scale)),max(3,round(source.shape[0]*scale)))
   if size[0]>256 or size[1]>256:continue
   q=cv2.resize(source,size,interpolation=cv2.INTER_AREA if scale<1 else cv2.INTER_LINEAR)
   response=cv2.matchTemplate(image,q,cv2.TM_CCOEFF_NORMED);_,score,_,loc=cv2.minMaxLoc(response)
   peaks.append(dict(template=t['path'],scale=scale,score=float(score),bbox=[loc[0],loc[1],loc[0]+size[0],loc[1]+size[1]]))
 assert peaks;return max(peaks,key=lambda x:x['score']),peaks

def main():
 begin=time.time();error=None;cleanup=None;est=None;handles=[];counts=dict(started=0,completed=0,rows=0);results=[];closures=[]
 try:
  assert not Path('/eval').exists() and not Path('/history').exists() and not Path('/core').exists()
  job=json.loads((I/'JOB.json').read_text());manifest=json.loads((I/'IMAGE_MANIFEST.json').read_text());assert 1<=len(job['identities'])<=2
  import torch,torch.multiprocessing as mp
  mp.set_start_method('spawn',force=True);torch.set_num_threads(8);torch.manual_seed(7);np.random.seed(7)
  from PIL import Image
  from megapose.datasets.object_dataset import RigidObject,RigidObjectDataset
  from megapose.datasets.scene_dataset import ObjectData
  from megapose.inference.types import ObservationTensor
  from megapose.inference.utils import make_detections_from_object_data
  from megapose.utils.load_model import load_named_model,NAMED_MODELS
  objects=[RigidObject(label='cream_cheese_1',mesh_path=Path('/geometry/stable_hope_objects/cream_cheese/cream_cheese.obj'),mesh_units='m',scaling_factor=0.008),RigidObject(label='salad_dressing_1',mesh_path=Path('/geometry/stable_hope_objects/salad_dressing/textured.obj'),mesh_units='m',scaling_factor=0.01),RigidObject(label='ketchup_1',mesh_path=Path('/geometry/stable_hope_objects/ketchup/textured.obj'),mesh_units='m',scaling_factor=0.01),RigidObject(label='butter_1',mesh_path=Path('/geometry/stable_hope_objects/butter/butter.obj'),mesh_units='m',scaling_factor=0.0075),RigidObject(label='milk_1',mesh_path=Path('/geometry/stable_hope_objects/milk/textured.obj'),mesh_units='m',scaling_factor=0.0075),RigidObject(label='orange_juice_1',mesh_path=Path('/geometry/stable_hope_objects/orange_juice/textured.obj'),mesh_units='m',scaling_factor=0.0075),RigidObject(label='alphabet_soup_1',mesh_path=Path('/geometry/stable_hope_objects/alphabet_soup/textured.obj'),mesh_units='m',scaling_factor=.01),RigidObject(label='bbq_sauce_1',mesh_path=Path('/geometry/stable_hope_objects/bbq_sauce/bbq_sauce.obj'),mesh_units='m',scaling_factor=.0077),
   RigidObject(label='tomato_sauce_1',mesh_path=Path('/geometry/stable_hope_objects/tomato_sauce/textured.obj'),mesh_units='m',scaling_factor=.01),
   RigidObject(label='chocolate_pudding_1',mesh_path=Path('/geometry/stable_hope_objects/chocolate_pudding/textured.obj'),mesh_units='m',scaling_factor=.01),
   RigidObject(label='basket_1',mesh_path=Path('/geometry/stable_scanned_objects/basket/basket.obj'),mesh_units='m',scaling_factor=1.)]
  event(dict(event='load_begin'));t=time.time();est=load_named_model('megapose-1.0-RGB-multi-hypothesis',RigidObjectDataset(objects),n_workers=2,bsz_images=8).cpu().eval();event(dict(event='load_complete',seconds=time.time()-t))
  for role,model in [('coarse',est.coarse_model),('refiner',est.refiner_model)]:
   def pre(m,inputs,role=role):
    assert inputs[0].device.type=='cpu';m._started=time.perf_counter();m._rows=int(inputs[0].shape[0]);counts['started']+=1;event(dict(event='forward_start',role=role,rows=m._rows))
   def post(m,inputs,output,role=role):
    counts['completed']+=1;counts['rows']+=m._rows;event(dict(event='forward_complete',role=role,rows=m._rows,seconds=time.perf_counter()-m._started))
   handles.extend([model.backbone.register_forward_pre_hook(pre),model.backbone.register_forward_hook(post)])
  for identity in job['identities']:
   item=next(x for x in manifest['images'] if x['identity']==identity);path=I/'images'/identity/'image.png';assert sha(path)==item['image_sha256']
   original=cv2.imread(str(path));rgb=np.asarray(Image.open(str(path)).convert('RGB'))[:,::-1].copy();cal=json.loads((I/'calibration'/('CALIBRATION_'+item['variant']+'_task'+str(item['task_id'])+'.json')).read_text())['cameras']['agentview']
   for label in [item['subject']]:
    selected,peaks=detect(original,label,manifest);x1,y1,x2,y2=selected['bbox'];box=np.array([255-x2,y1,255-x1,y2],np.float32)
    obs=ObservationTensor.from_numpy(rgb,None,np.asarray(cal['K'],np.float32));det=make_detections_from_object_data([ObjectData(label=label,bbox_modal=box)]).cpu();start=time.time();before=dict(counts)
    with torch.no_grad():pred,extra=est.run_inference_pipeline(obs,detections=det,**NAMED_MODELS['megapose-1.0-RGB-multi-hypothesis']['inference_parameters'])
    poses=extra['scoring']['preds'].poses.cpu().numpy();infos=extra['scoring']['preds'].infos.to_dict('records');chosen=pred.poses.cpu().numpy();assert len(poses)==len(infos)==5 and len(chosen)==1
    best=max(float(x['pose_logit']) for x in infos);assert any(float(x['pose_logit'])==best and np.array_equal(chosen[0],poses[i]) for i,x in enumerate(infos))
    v=dict(time=time.time(),identity=identity,label=label,input_rgb_sha256=sha(path),detection=selected,detection_peaks=peaks,
     T_camera_object=chosen[0],T_world_object=np.asarray(cal['T_world_camera_cv'])@chosen[0],hypotheses=[dict(info=q,T_world_object=np.asarray(cal['T_world_camera_cv'])@p) for q,p in zip(infos,poses)],
     selection='FROZEN_LEARNED_ARGMAX_NO_GT_RERANK',GT_read=False,weights_updated=False,seconds=time.time()-start,counts={k:counts[k]-before[k] for k in counts})
    write(O/identity/(label+'.json'),v);results.append(dict(identity=identity,label=label,seconds=v['seconds'],counts=v['counts']));print(json.dumps(results[-1]),flush=True)
 except BaseException:error=traceback.format_exc()
 finally:
  for h in handles:h.remove()
  if est is not None:
   unique={id(m.renderer):m.renderer for m in [est.coarse_model,est.refiner_model]}
   for renderer in unique.values():
    try:
     renderer.stop();after=[dict(pid=p.pid,alive=p.is_alive(),exitcode=p.exitcode) for p in renderer._renderers];assert all(not p['alive'] and p['exitcode']==0 for p in after);closures.append(after)
    except BaseException:cleanup=(cleanup or '')+traceback.format_exc()
  write(O/'RESULT.json',dict(time=time.time(),error=error,cleanup_error=cleanup,seconds=time.time()-begin,counts=counts,results=results,renderer_shutdown=closures,physical_attempts=0,training=False,money=None))
 if error or cleanup:raise RuntimeError((error or '')+(cleanup or ''))
if __name__=='__main__':main()
