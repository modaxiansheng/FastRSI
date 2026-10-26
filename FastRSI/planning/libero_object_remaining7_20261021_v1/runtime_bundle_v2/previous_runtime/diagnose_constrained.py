from pathlib import Path
import json,time,traceback,hashlib
import numpy as np
from PIL import Image
from online_vision import Vision
from pose_subject_batch import detect
from constrained_localization import detect_supported
R=Path('/out')
def emit(x):
 with (R/'NETWORK_LEDGER.jsonl').open('a') as f:f.write(json.dumps(dict(time=time.time(),**x))+'\n')
def main():
 start=time.time();v=None;error=None;rows=[]
 try:
  v=Vision('/input',R/'model_outputs',emit)
  import pandas as pd
  from megapose.utils.tensor_collection import PandasTensorCollection
  from megapose.inference.types import ObservationTensor
  source=json.loads(Path('/candidate/STATIC_SOURCE_POSE.json').read_text());manifest=v.manifest;template=next(t for t in manifest['templates']['chocolate_pudding_1'] if t['path'].endswith('step0.png'))
  for i in [20,21,22,23]:
   path=Path('/images')/('pudding%02d.png'%i);rgb=np.asarray(Image.open(path).convert('RGB'));variant='position' if i<22 else 'environment';cal=v.camera(variant,8);selected,seed,proof,peaks=detect_supported(rgb[:,:,::-1].copy(),__import__('cv2').imread('/input/'+template['path']),template,source['skill']['source_object_pose'],source['calibration']['environment']['cameras']['agentview'],cal,-.445 if variant=='position' else 0.)
   coarse=PandasTensorCollection(pd.DataFrame([dict(label='chocolate_pudding_1',batch_im_id=0,instance_id=0,hypothesis_id=0)]),poses=v.torch.tensor(seed[None],dtype=v.torch.float32));obs=ObservationTensor.from_numpy(rgb[:,::-1].copy(),None,np.asarray(cal['K'],np.float32));before=v.forward_count;t=time.time()
   with v.torch.no_grad():pred,extra=v.est.run_inference_pipeline(obs,coarse_estimates=coarse,n_pose_hypotheses=1,n_refiner_iterations=5,run_depth_refiner=False)
   assert extra['coarse']['data'] is None
   world=np.asarray(cal['T_world_camera_cv'])@pred.poses.cpu().numpy()[0];src=np.asarray(source['skill']['source_object_pose']);up=src[:3,:3].T@np.array([0.,0.,1.]);tilt=float(np.degrees(np.arccos(np.clip((world[:3,:3]@up)[2],-1,1))))
   row=dict(identity='crossmodel%02d'%i,input_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),detector=selected,seed_camera=seed.tolist(),world_pose=world.tolist(),tilt_deg=tilt,unchanged_gate_15deg=tilt<=15,seconds=time.time()-t,forward_calls=v.forward_count-before,initialization=proof,GT_read=False);rows.append(row);(R/('PREDICTION_%02d.json'%i)).write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
 except BaseException:error=traceback.format_exc();raise
 finally:
  cleanup=None
  if v:
   try:v.close()
   except BaseException:cleanup=traceback.format_exc()
  (R/'RESULT.json').write_text(json.dumps(dict(time=time.time(),seconds=time.time()-start,rows=rows,error=error,cleanup_error=cleanup,physical_steps=0,policy_calls=0,training=False,money=None),indent=2))
if __name__=='__main__':main()
