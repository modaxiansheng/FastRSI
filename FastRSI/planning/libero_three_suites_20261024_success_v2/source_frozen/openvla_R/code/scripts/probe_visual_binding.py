"""Execute saved-image correspondence without loading simulator annotations."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.visual_binding import match_template
p=argparse.ArgumentParser();p.add_argument('--plan',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];plan=json.loads(a.plan.read_text())
for name,digest in plan['sha256'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
start=time.perf_counter()
source=cv2.cvtColor(cv2.imread(str(root/plan['source_image'])),cv2.COLOR_BGR2RGB)
target=cv2.cvtColor(cv2.imread(str(root/plan['target_image'])),cv2.COLOR_BGR2RGB)
r=match_template(source,target,plan['source_roi_xyxy']);r['elapsed_seconds_including_image_decode']=time.perf_counter()-start
r['scope']=plan['scope'];r['opencv_version']=cv2.__version__;r['plan_sha256']=hashlib.sha256(a.plan.read_bytes()).hexdigest()
for i,item in enumerate(r['peaks']):
 x0,y0,x1,y1=item['box_xyxy'];cv2.rectangle(target,(x0,y0),(x1,y1),(0,255,0) if i==0 else (255,128,0),1);cv2.putText(target,str(i+1),(x0,y0-3),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,0),1)
cv2.imwrite(str(a.output.parent/'MATCHES.png'),cv2.cvtColor(target,cv2.COLOR_RGB2BGR))
with a.output.open('x') as f:json.dump(r,f,indent=2)
print(json.dumps(r))
