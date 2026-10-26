"""Recorded warm-state parity for frozen R before any evaluation; no tools executed."""
import hashlib,json,sys,time,urllib.request
from pathlib import Path
sys.path.insert(0,'/project/src');sys.path.insert(0,'/project/scripts')
from prepare_resident_policy_inputs import reconstruct
from replay_resident_policy import Wire
from fastrsi.resident_client import ResidentClient
root=Path('/project');run=root/'runs/learning_transfer_v2_20261004';out=run/'frozen_warmup';out.mkdir(exist_ok=False)
source=root/'runs/learning_transfer_v1_20261004/development/pair0/campaign_N/N/stream';start=time.perf_counter();result={'status':'RUNNING','phase':'warmup_only','model_calls':0,'robot_actions':0,'learning_updates':0}
def save():(out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
try:
 state=json.loads((run/'STATUS.json').read_text());assert state['phase'] in ('preparation','development')
 client=ResidentClient(22300,'learning-transfer-frozen-warmup',timeout_seconds=30)
 assert client.status()['active'] is None
 arrays,binding=reconstruct(source,'net_t3_i1',0);arrays['prompt']=str(arrays['prompt'].item());result['input_binding']=binding
 result['begin']=client.begin('warmup-begin',1);wire=Wire(22300)
 try:
  tick=time.perf_counter();response=wire.call(arrays);actions=response['actions'];assert actions.shape==(10,7)
  result.update(policy_seconds=time.perf_counter()-tick,actions_sha256=hashlib.sha256(actions.tobytes()).hexdigest(),model_calls=1)
 finally:wire.close();result['end']=client.end('warmup-end')
 body=(source/'net_t3_i1/agent_call_1/REQUEST.json').read_bytes();(out/'QWEN_REQUEST.json').write_bytes(body)
 result['qwen_request_sha256']=hashlib.sha256(body).hexdigest();tick=time.perf_counter()
 req=urllib.request.Request('http://127.0.0.1:22310/v1/chat/completions',data=body,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(req,timeout=120) as response:raw=response.read()
 (out/'QWEN_RESPONSE.json').write_bytes(raw);decoded=json.loads(raw);assert decoded.get('choices') and not decoded.get('error')
 result.update(qwen_seconds=time.perf_counter()-tick,model_calls=2,status='PASSED',tools_executed=0,policy_released=client.status()['active'] is None)
except BaseException:
 import traceback
 result.update(status='FAILED',error=traceback.format_exc());raise
finally:result['seconds']=time.perf_counter()-start;save();print(json.dumps(result),flush=True)
