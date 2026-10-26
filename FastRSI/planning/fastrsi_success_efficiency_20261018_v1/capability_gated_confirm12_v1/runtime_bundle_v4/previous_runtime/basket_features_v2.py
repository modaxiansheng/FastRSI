"""CAD-linked visual features. Source annotations train the bank; prediction never opens them."""
from pathlib import Path
import json,hashlib,time,sys
import numpy as np,cv2
I=Path('/input');O=Path('/out')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def mesh(path):
 vertices=[];faces=[]
 for line in path.read_text().splitlines():
  parts=line.split()
  if not parts:continue
  if parts[0]=='v':vertices.append([float(x) for x in parts[1:4]])
  if parts[0]=='f':
   ids=[int(x.split('/')[0])-1 for x in parts[1:]];assert min(ids)>=0
   for k in range(1,len(ids)-1):faces.append([ids[0],ids[k],ids[k+1]])
 return np.asarray(vertices,float),np.asarray(faces,int)
def intersection(origin,direction,vertices,faces):
 a=vertices[faces[:,0]];e1=vertices[faces[:,1]]-a;e2=vertices[faces[:,2]]-a
 h=np.cross(np.broadcast_to(direction,e2.shape),e2);det=np.einsum('ij,ij->i',e1,h);valid=np.abs(det)>1e-10
 inv=np.zeros(len(det));inv[valid]=1/det[valid];s=origin-a;u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,e1);v=np.einsum('j,ij->i',direction,q)*inv;t=np.einsum('ij,ij->i',e2,q)*inv
 valid&=(u>=0)&(v>=0)&(u+v<=1)&(t>1e-6)
 if not valid.any():return None
 idx=np.where(valid)[0][np.argmin(t[valid])];return origin+t[idx]*direction

def train():
 begin=time.time();ann=json.loads((I/'SOURCE_ANNOTATIONS.json').read_text());assert {a['identity'] for a in ann['rows']}=={'dev12_step0','dev13_step0','dev15_step0','initialskill00_cut'}
 vertices,faces=mesh(Path('/geometry/basket.obj'));sift=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.02);xyz=[];desc=[];record=[]
 for a in ann['rows']:
  name=a['identity'];p=I/'images'/name/'image.png';assert sha(p)==a['image_sha256'];im=cv2.imread(str(p));cal=json.loads((I/'calibration'/a['calibration']).read_text())['cameras']['agentview'];mask=np.zeros((256,256),np.uint8)
  if a['source_bbox'] is None:
   T=np.linalg.inv(np.asarray(cal['T_world_camera_cv']))@np.asarray(a['T_world_basket']);xyz=(T@np.c_[vertices,np.ones(len(vertices))].T).T[:,:3];uv=(np.asarray(cal['K'])@xyz.T).T;uv=uv[:,:2]/uv[:,2:];uv[:,0]=255-uv[:,0];lo=np.maximum(0,np.floor(uv.min(0)).astype(int));hi=np.minimum(256,np.ceil(uv.max(0)).astype(int)+1);box=[*lo.tolist(),*hi.tolist()]
  else:box=a['source_bbox']
  x1,y1,x2,y2=box;mask[y1:y2,x1:x2]=255
  keypoints,descriptors=sift.detectAndCompute(im,mask);assert descriptors is not None
  world_obj=np.asarray(a['T_world_basket']);world_cam=np.asarray(cal['T_world_camera_cv']);obj_cam=np.linalg.inv(world_obj)@world_cam;Kinv=np.linalg.inv(np.asarray(cal['K']));count=0
  for k,d in zip(keypoints,descriptors):
   # Model image is mirrored in x relative to conventional CV projection.
   ray=Kinv@np.array([255-k.pt[0],k.pt[1],1.]);direction=obj_cam[:3,:3]@ray
   point=intersection(obj_cam[:3,3],direction,vertices,faces)
   if point is None:continue
   xyz.append(point);desc.append(d);count+=1;record.append(dict(source=name,source_xy=list(k.pt),point_basket=point.tolist()))
  print(json.dumps(dict(source=name,sift=len(keypoints),linked_3D=count)),flush=True)
 assert len(xyz)>=12,'Insufficient source visual 2D-3D correspondences'
 np.savez(str(O/'BANK.npz'),points=np.asarray(xyz,float),descriptors=np.asarray(desc,np.float32),source_ids=np.asarray([r['source'] for r in record]))
 write(O/'BANK_SOURCE_RECEIPT.json',dict(time=time.time(),source_rows=ann['rows'],mesh_sha256=sha(Path('/geometry/basket.obj')),bank_sha256=sha(O/'BANK.npz'),points=len(xyz),correspondences=record,seconds=time.time()-begin,source_GT_supervised=True,target_GT_read=False,physics=0,neural_calls=0))

def predict():
 begin=time.time();assert not (I/'SOURCE_ANNOTATIONS.json').exists() and not (I/'DIAGNOSTIC_TRUTH_DO_NOT_MOUNT_TO_PREDICTOR.json').exists()
 manifest=json.loads((I/'IMAGE_MANIFEST.json').read_text());bank=np.load('/bank/BANK.npz',allow_pickle=False);points=bank['points'];desc=bank['descriptors'];sift=cv2.SIFT_create(nfeatures=1000,contrastThreshold=.02);bf=cv2.BFMatcher(cv2.NORM_L2);rows=[]
 for meta in manifest['images']:
  name=meta['identity'];p=I/'images'/name/'image.png';assert sha(p)==meta['image_sha256'];im=cv2.imread(str(p));keys,des=sift.detectAndCompute(im,None);assert des is not None
  matches=[]
  for source in np.unique(bank['source_ids']):
   inds=np.where(bank['source_ids']==source)[0]
   for a,b in bf.knnMatch(des,desc[inds],k=2):
    if a.distance<.75*b.distance:a.trainIdx=int(inds[a.trainIdx]);matches.append(a)
  # A repeated source descriptor may not create repeated 3D evidence.
  used=set();used_query=set();unique=[]
  for m in sorted(matches,key=lambda x:x.distance):
   key=tuple(np.round(points[m.trainIdx],5))
   if key in used or m.queryIdx in used_query:continue
   used.add(key);used_query.add(m.queryIdx);unique.append(m)
  xyz=np.asarray([points[m.trainIdx] for m in unique],float);uv=np.asarray([[255-keys[m.queryIdx].pt[0],keys[m.queryIdx].pt[1]] for m in unique],float)
  cal=json.loads((I/'calibration'/('CALIBRATION_'+meta['variant']+'_task'+str(meta['task_id'])+'.json')).read_text())['cameras']['agentview'];K=np.asarray(cal['K'],float);rvec=None;tvec=None;inliers=None;ok=False;reason='INSUFFICIENT_VISUAL_CORRESPONDENCES'
  if len(unique)>=8:
   cv2.setRNGSeed(7);ok,rvec,tvec,inliers=cv2.solvePnPRansac(xyz,uv,K,None,iterationsCount=1000,reprojectionError=2.,confidence=.999,flags=cv2.SOLVEPNP_EPNP)
   ok=bool(ok and inliers is not None and len(inliers)>=8)
   reason=None if ok else 'PNP_INLIERS_BELOW_FIXED_8'
  pose=None;error=None
  if ok:
   ind=inliers.ravel();rvec,tvec=cv2.solvePnPRefineLM(xyz[ind],uv[ind],K,None,rvec,tvec);R,_=cv2.Rodrigues(rvec);T=np.eye(4);T[:3,:3]=R;T[:3,3]=tvec.ravel();projected,_=cv2.projectPoints(xyz[ind],rvec,tvec,K,None);error=float(np.sqrt(np.mean(np.sum((projected.reshape(-1,2)-uv[ind])**2,axis=1))))
   pose=np.asarray(cal['T_world_camera_cv'])@T
   if tvec[2,0]<=0 or error>2.:ok=False;reason='INVALID_DEPTH_OR_REPROJECTION';pose=None
  value=dict(time=time.time(),identity=name,status='VISUAL_PNP_BASKET_POSE' if ok else 'UNKNOWN',reason=reason,matches=len(unique),inliers=0 if inliers is None else len(inliers),reprojection_rmse_px=error,T_world_object=None if pose is None else pose.tolist(),
   input_rgb_sha256=sha(p),bank_sha256=sha(Path('/bank/BANK.npz')),GT_read=False,source_supervised_3D=True,method='SIFT ratio0.75 + CAD-linked 3D + RANSAC EPNP 2px/8inliers + LM; no target pose prior or label selection')
  write(O/name/'basket_1.json',value);rows.append(value);print(json.dumps({k:value[k] for k in ['identity','status','matches','inliers','reprojection_rmse_px']}),flush=True)
 write(O/'RESULT.json',dict(time=time.time(),seconds=time.time()-begin,rows=rows,physical_attempts=0,neural_calls=0,GT_read=False))
if __name__=='__main__':
 if sys.argv[1]=='train':train()
 elif sys.argv[1]=='predict':predict()
 else:raise ValueError(sys.argv[1])
