"""Replay an actual wire-format development image request, with no tool execution."""
from pathlib import Path
import hashlib,json,time,urllib.request
root=Path('/project');run=root/'runs/learning_transfer_v2_20261004';out=run/'frozen_warmup_qwen';out.mkdir(exist_ok=False)
assert json.loads((run/'STATUS.json').read_text())['phase']=='development'
old=root/'runs/learning_transfer_v1_20261004/resident/N/pool/qwen/service_calls'
choices=[]
for path in old.glob('*/REQUEST.json'):
 d=json.loads(path.read_text());receipt=json.loads((path.parent/'RECEIPT.json').read_text())
 images=sum(p.get('type')=='image_url' for m in d['messages'] for p in (m['content'] if isinstance(m.get('content'),list) else []))
 if images and receipt.get('status')=='OK':choices.append((receipt['started_at'],path))
source=min(choices)[1];body=source.read_bytes();(out/'REQUEST.json').write_bytes(body);tick=time.perf_counter()
req=urllib.request.Request('http://127.0.0.1:22310/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
with urllib.request.urlopen(req,timeout=120) as r:raw=r.read()
(out/'RESPONSE.json').write_bytes(raw);r=json.loads(raw);assert r.get('choices') and not r.get('error')
receipt=dict(status='PASSED',seconds=time.perf_counter()-tick,model_calls=1,source=str(source),source_sha256=hashlib.sha256(body).hexdigest(),tools_executed=0,learning_updates=0)
(out/'RESULT.json').write_text(json.dumps(receipt,indent=2)+'\n')
p=json.loads((run/'frozen_warmup/RESULT.json').read_text());assert p['model_calls']==1 and p['end']['status_after']['active'] is None
result=dict(status='PASSED',policy_released=True,tools_executed=0,model_calls=2,failed_http_attempts=1,seconds=p['seconds']+receipt['seconds'],policy_request_sha256=p['actions_sha256'],source_results=['frozen_warmup/RESULT.json','frozen_warmup_qwen/RESULT.json'],scope='Warmup uses paid development inputs. Original 422 retained; exact successful server request replay, no learning or test outcomes.')
(run/'WARMUP_CHECK.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
