"""Real frozen-pi service check on four saved initial requests; zero robot steps."""
import asyncio,hashlib,json,time
from pathlib import Path
import numpy as np
import websockets
from PhyAgentOS.runtime.policy.msgpack_numpy import packb,unpackb
from fastrsi.complete_flow_memory import CompleteFlowMemory,legal_key

async def main():
 p=Path('/project');o=Path('/check');spec=json.loads((o/'SPEC.json').read_text());registry=json.loads((p/spec['registry']).read_text());memory=CompleteFlowMemory(registry,spec['model_sha256'],registry['teacher_server_sha256']);rows=[];started=time.perf_counter()
 async with websockets.connect(spec['endpoint'],compression=None,max_size=None,ping_interval=None,ping_timeout=None) as ws:
  meta=unpackb(await ws.recv());assert meta['flow_mode']=='C' and meta['flow_registry_sha256']==spec['registry_sha256'] and meta['external_model_sha256']==spec['model_sha256']
  for case in spec['cases']:
   original=p/case['request'];raw=original.read_bytes();assert hashlib.sha256(raw).hexdigest()==case['request_sha256'];request=unpackb(raw);tag=request['_fastrsi_value_v1'];assert tag['step']==0 and tag['enabled'] is True
   decision=memory.choose(tag['session_id'],0,request,packb);assert decision['mode']==case['expected_mode']
   tick=time.perf_counter();await ws.send(packb(request));reply_raw=await ws.recv();assert not isinstance(reply_raw,str),reply_raw;reply=unpackb(reply_raw);elapsed=time.perf_counter()-tick
   expected_path=p/case['expected_response'];assert hashlib.sha256(expected_path.read_bytes()).hexdigest()==case['expected_response_sha256'];expected=unpackb(expected_path.read_bytes())
   assert np.array_equal(reply['actions'],expected['actions'])
   m=reply['policy_meta']['native_value_choice'];assert m['model_sha256']==spec['model_sha256'] and m['enabled']==(case['expected_mode']=='R')
   expected_count=expected['policy_meta']['native_value_choice']['candidate_count'];assert m['candidate_count']==expected_count
   (o/f"init{case['initial']}_response.msgpack").write_bytes(reply_raw);rows.append(dict(initial=case['initial'],mode=decision['mode'],actions_exact=True,actual_candidate_inferences=m['candidate_count'],rpc_seconds=elapsed,legal_input_sha256=legal_key(request,packb)))
 result=dict(status='PASSED_REAL_NATIVE_COMPLETE_FLOW_PACKET_BINDING',rows=rows,actual_pi_inferences=sum(r['actual_candidate_inferences'] for r in rows),robot_actions=0,pi_updates=0,registry_updates=0,seconds=time.perf_counter()-started,not_new_episode_performance=True)
 (o/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':asyncio.run(main())
