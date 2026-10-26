"""Source-supervised crops; target input is public RGB only. No physics/model calls."""
from pathlib import Path
import hashlib,json,time,itertools
import numpy as np,cv2
from camera_model import model_rgb_projection
H=Path('/out');B=Path('/history');W=B/'fastrsi_object_frame_recovery_20261017_v1';D=B/'fastrsi_placement_recovery_20261017_v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2)
def quat(q):
 w,x,y,z=np.asarray(q)/np.linalg.norm(q);return np.array([[1-2*(y*y+z*z),2*(x*y-w*z),2*(x*z+w*y)],[2*(x*y+w*z),1-2*(x*x+z*z),2*(y*z-w*x)],[2*(x*z-w*y),2*(y*z+w*x),1-2*(x*x+y*y)]])
def pose(r):
 t=np.eye(4);t[:3,3]=r['position'];t[:3,:3]=quat(r['quaternion']);return t
def vertices(g):
 out=[]
 for x in g['contact_geoms']:
  if x['type']!=6:continue
  for signs in itertools.product([-1,1],repeat=3):out.append(np.asarray(x['local_pos'])+quat(x['local_quat'])@(np.asarray(x['size'])*signs))
 assert out;return np.asarray(out)
def crop(image,points,t,cal):
 world=(t@np.c_[points,np.ones(len(points))].T).T[:,:3];uv=model_rgb_projection(world,cal)
 lo=np.maximum(0,np.floor(uv.min(0)).astype(int));hi=np.minimum(256,np.ceil(uv.max(0)).astype(int)+1);assert np.all(hi>lo)
 return image[lo[1]:hi[1],lo[0]:hi[0]].copy(),[*lo.tolist(),*hi.tolist()]
def main():
 started=time.time();images=[];templates={};truth=[]
 for src,task,subject in [('dev12',3,'bbq_sauce_1'),('dev13',5,'tomato_sauce_1'),('dev15',8,'chocolate_pudding_1')]:
  root=W/'cases/node187'/src;skill=json.loads((W/'skills'/(src+'.json')).read_text());geometry=json.loads((root/'OBJECT_GEOMETRY.json').read_text());assert subject in geometry and 'basket_1' in geometry
  events=[json.loads(x) for x in (root/'DIAGNOSTIC_EVENTS.jsonl').read_text().splitlines() if x];states={r['step']:r for r in events if r['event']=='diagnostic_state'}
  frames=json.loads((root/'FRAMES.json').read_text());held=min(x['step'] for x in frames if x['step']>=skill['source_segment_end_exclusive'] and states[x['step']]['objects'][subject]['two_finger_contact'])
  cal=json.loads((H/'calibration'/('CALIBRATION_environment_task'+str(task)+'.json')).read_text())['cameras']['agentview']
  for step in [0,held]:
   name=src+'_step'+str(step);p=root/'frames'/('%04d_front.png'%step);im=cv2.imread(str(p));dest=H/'images'/name;dest.mkdir(parents=True)
   (dest/'image.png').write_bytes(p.read_bytes());meta=dict(identity=name,task_id=task,variant='environment',phase='initial' if step==0 else 'held',subject=subject,source_public_image=str(p),image_sha256=sha(p));save(dest/'META.json',meta);images.append(meta)
   st=states[step]
   for label in [subject,'basket_1']:
    points=vertices(geometry[label]);patch,box=crop(im,points,pose(st['objects'][label]),cal);t=H/'templates'/label/(name+'.png');t.parent.mkdir(parents=True,exist_ok=True);assert cv2.imwrite(str(t),patch)
    templates.setdefault(label,[]).append(dict(path=str(t.relative_to(H)),sha256=sha(t),supervision='PERMITTED_SUCCESSFUL_DEVELOPMENT_SOURCE_ONLY',source_image_sha256=sha(p),source_crop=box))
   truth.append(dict(identity=name,subject=subject,subject_pose=pose(st['objects'][subject]).tolist(),basket_pose=pose(st['objects']['basket_1']).tolist(),held=st['objects'][subject]['two_finger_contact']))
 for id in ['initialskill00','initialskill03']:
  node='187' if id.endswith('00') else '188';r=D/'cases'/('node'+node)/id/'privileged_place';p=r/'cut_front.png';name=id+'_cut';dest=H/'images'/name;dest.mkdir(parents=True);(dest/'image.png').write_bytes(p.read_bytes())
  meta=dict(identity=name,task_id=3,variant='position',phase='held',subject='bbq_sauce_1',source_public_image=str(p),image_sha256=sha(p));save(dest/'META.json',meta);images.append(meta)
  gt=json.loads((r/'PRIVILEGED_INPUTS.json').read_text());truth.append(dict(identity=name,subject='bbq_sauce_1',subject_pose=gt['object_pose'],basket_pose=gt['basket_pose'],held=gt['held']))
 save(H/'IMAGE_MANIFEST.json',dict(time=time.time(),images=images,templates=templates,max_neural_images=8,max_object_pose_predictions=16,physical_attempts=0,no_target_truth_in_predictor=True))
 save(H/'DIAGNOSTIC_TRUTH_DO_NOT_MOUNT_TO_PREDICTOR.json',dict(rows=truth,scope='Prediction is written before this evidence may be compared; never a control input'))
 print(json.dumps(dict(images=len(images),seconds=time.time()-started)))
if __name__=='__main__':main()
