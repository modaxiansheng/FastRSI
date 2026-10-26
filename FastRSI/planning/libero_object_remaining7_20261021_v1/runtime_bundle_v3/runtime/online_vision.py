"""Current RGB inference in the physical worker; no simulator/target-truth argument."""
from pathlib import Path
import json,hashlib,time
import numpy as np,cv2
from pose_subject_batch import detect

class Vision:
 def __init__(self,root,output,emit):
  import torch,torch.multiprocessing as mp
  from megapose.datasets.object_dataset import RigidObject,RigidObjectDataset
  from megapose.utils.load_model import load_named_model
  mp.set_start_method('spawn',force=True);torch.set_num_threads(8);torch.manual_seed(7);np.random.seed(7)
  self.torch=torch;self.root=Path(root);self.out=Path(output);self.out.mkdir(exist_ok=False);self.emit=emit;self.requests=0;self.forward_count=0;self.hooks=[]
  self.manifest=json.loads((self.root/'IMAGE_MANIFEST.json').read_text());self.bank=np.load(self.root/'bank/BANK.npz',allow_pickle=False);self.sift=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.02);self.bf=cv2.BFMatcher(cv2.NORM_L2)
  objects=[RigidObject(label='bbq_sauce_1',mesh_path=Path('/geometry/stable_hope_objects/bbq_sauce/bbq_sauce.obj'),mesh_units='m',scaling_factor=.0077),RigidObject(label='tomato_sauce_1',mesh_path=Path('/geometry/stable_hope_objects/tomato_sauce/textured.obj'),mesh_units='m',scaling_factor=.01),RigidObject(label='chocolate_pudding_1',mesh_path=Path('/geometry/stable_hope_objects/chocolate_pudding/textured.obj'),mesh_units='m',scaling_factor=.01)]
  t=time.time();self.est=load_named_model('megapose-1.0-RGB-multi-hypothesis',RigidObjectDataset(objects),n_workers=2,bsz_images=8).cpu().eval();emit(dict(event='visual_model_loaded',seconds=time.time()-t))
  for role,model in [('coarse',self.est.coarse_model),('refiner',self.est.refiner_model)]:
   def pre(m,inputs,role=role):
    assert inputs[0].device.type=='cpu';self.emit(dict(event='visual_forward_begin',role=role,request=self.requests,rows=int(inputs[0].shape[0])))
   def post(m,inputs,value,role=role):self.forward_count+=1;self.emit(dict(event='visual_forward_end',role=role,request=self.requests,rows=int(inputs[0].shape[0])))
   self.hooks.extend([model.backbone.register_forward_pre_hook(pre),model.backbone.register_forward_hook(post)])
 def camera(self,variant,task):return json.loads((self.root/'calibration'/('CALIBRATION_'+variant+'_task'+str(task)+'.json')).read_text())['cameras']['agentview']
 def subject(self,observation,meta,world_prior=None):
  from megapose.inference.types import ObservationTensor
  from megapose.datasets.scene_dataset import ObjectData
  from megapose.inference.utils import make_detections_from_object_data
  from megapose.utils.load_model import NAMED_MODELS
  self.requests+=1;assert self.requests<=8,'Fixed per-attempt RGB neural request budget'
  t=time.time();rgb=np.asarray(observation['agentview_image']);assert rgb.shape==(256,256,3) and rgb.dtype==np.uint8
  from PIL import Image
  folder=self.out/('%02d'%self.requests);folder.mkdir();Image.fromarray(rgb).save(folder/'front.png');cal=self.camera(meta['variant'],meta['task_id']);wc=np.asarray(cal['T_world_camera_cv']);obs=ObservationTensor.from_numpy(rgb[:,::-1].copy(),None,np.asarray(cal['K'],np.float32));label=meta['subject'];before=self.forward_count;box=None
  if world_prior is None:
   # Restrict action0 detection to the source initial view; no held gripper template.
   manifest=dict(self.manifest);manifest['templates']={k:[x for x in v if Path(x['path']).stem.endswith('step0')] for k,v in self.manifest['templates'].items()}
   # detect uses module /input constant; runtime input is deliberately mounted there.
   selected,_=detect(rgb[:,:,::-1].copy(),label,manifest);x1,y1,x2,y2=selected['bbox'];box=np.array([255-x2,y1,255-x1,y2],np.float32);det=make_detections_from_object_data([ObjectData(label=label,bbox_modal=box)]).cpu()
   with self.torch.no_grad():pred,extra=self.est.run_inference_pipeline(obs,detections=det,**NAMED_MODELS['megapose-1.0-RGB-multi-hypothesis']['inference_parameters'])
  else:
   import pandas as pd
   from megapose.utils.tensor_collection import PandasTensorCollection
   prior=np.linalg.inv(wc)@np.asarray(world_prior);coarse=PandasTensorCollection(pd.DataFrame([dict(label=label,batch_im_id=0,instance_id=0,hypothesis_id=0)]),poses=self.torch.tensor(prior[None],dtype=self.torch.float32))
   with self.torch.no_grad():pred,extra=self.est.run_inference_pipeline(obs,coarse_estimates=coarse,n_pose_hypotheses=1,n_refiner_iterations=5,run_depth_refiner=False)
   assert extra['coarse']['data'] is None
  allposes=extra['scoring']['preds'].poses.cpu().numpy();infos=extra['scoring']['preds'].infos.to_dict('records');chosen=pred.poses.cpu().numpy()[0];best=max(float(x['pose_logit']) for x in infos)
  assert any(float(info['pose_logit'])==best and np.array_equal(chosen,p) for p,info in zip(allposes,infos))
  value=dict(event='current_rgb_subject_pose',request=self.requests,step=meta['step'],phase=meta['phase'],subject=label,seconds=time.time()-t,input_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),png_sha256=hashlib.sha256((folder/'front.png').read_bytes()).hexdigest(),world_pose=(wc@chosen).tolist(),prior=None if world_prior is None else np.asarray(world_prior).tolist(),forward_calls=self.forward_count-before,GT_read=False)
  (folder/'PREDICTION.json').write_text(json.dumps(value,indent=2));self.emit(value);return wc@chosen
 def basket(self,observation,meta):
  rgb=np.asarray(observation['agentview_image']);cal=self.camera(meta['variant'],meta['task_id']);keys,des=self.sift.detectAndCompute(rgb[:,:,::-1].copy(),None)
  if des is None:return None,dict(status='UNKNOWN',reason='NO_BASKET_FEATURES')
  points=self.bank['points'];desc=self.bank['descriptors'];matches=[]
  for source in np.unique(self.bank['source_ids']):
   inds=np.where(self.bank['source_ids']==source)[0]
   for a,b in self.bf.knnMatch(des,desc[inds],k=2):
    if a.distance<.75*b.distance:a.trainIdx=int(inds[a.trainIdx]);matches.append(a)
  used=set();queries=set();unique=[]
  for m in sorted(matches,key=lambda x:x.distance):
   k=tuple(np.round(points[m.trainIdx],5))
   if k in used or m.queryIdx in queries:continue
   used.add(k);queries.add(m.queryIdx);unique.append(m)
  if len(unique)<8:return None,dict(status='UNKNOWN',reason='INSUFFICIENT_BASKET_FEATURES',matches=len(unique))
  xyz=np.asarray([points[m.trainIdx] for m in unique],float);uv=np.asarray([[255-keys[m.queryIdx].pt[0],keys[m.queryIdx].pt[1]] for m in unique],float);K=np.asarray(cal['K'],float);cv2.setRNGSeed(7);ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=1000,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
  if not ok or inliers is None or len(inliers)<8:return None,dict(status='UNKNOWN',reason='BASKET_GEOMETRY_NOT_CONFIRMED')
  ind=inliers.ravel();rvec,tvec=cv2.solvePnPRefineLM(xyz[ind],uv[ind],K,None,rvec,tvec);R,_=cv2.Rodrigues(rvec);T=np.eye(4);T[:3,:3]=R;T[:3,3]=tvec.ravel();proj,_=cv2.projectPoints(xyz[ind],rvec,tvec,K,None);error=float(np.sqrt(np.mean(np.sum((proj.reshape(-1,2)-uv[ind])**2,axis=1))))
  if tvec[2,0]<=0 or error>2.:return None,dict(status='UNKNOWN',reason='BASKET_INVALID_DEPTH_OR_REPROJECTION')
  world=np.asarray(cal['T_world_camera_cv'])@T;proof=dict(event='current_rgb_basket_pose',status='VISUAL_PNP_BASKET_POSE',step=meta['step'],matches=len(unique),inliers=len(inliers),reprojection_rmse_px=error,world_pose=world.tolist(),GT_read=False);self.emit(proof);return world,proof
 def close(self):
  for h in self.hooks:h.remove()
  for r in {id(m.renderer):m.renderer for m in [self.est.coarse_model,self.est.refiner_model]}.values():
   r.stop();assert all(not p.is_alive() and p.exitcode==0 for p in r._renderers)
  self.emit(dict(event='visual_model_closed',requests=self.requests,forward_calls=self.forward_count))
