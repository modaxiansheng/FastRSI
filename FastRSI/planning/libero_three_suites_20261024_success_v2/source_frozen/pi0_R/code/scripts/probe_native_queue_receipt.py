"""Replay one saved post-acceptance failure with truthful queue feedback only."""
import json
import os
import time
from pathlib import Path
import httpx
from fastrsi.queue_receipt import accepted_queue_receipt

root=Path(os.environ['FASTRSI_NATIVE_RUN'])
source=Path('/project/runs/native_terminal_eval_v2_20261002/stream')
paths=[p for p in (source/'service_calls').glob('*/RECEIPT.json') if json.loads(p.read_text()).get('status')!='OK']
assert len(paths)==1
receipt=json.loads(paths[0].read_text());assert receipt['generation_finish_reason']=='length'
body=json.loads((paths[0].parent/'REQUEST.json').read_text())
assert body['messages'][-1]['role']=='tool' and 'Runtime Session History' in body['messages'][-1]['content']
events=[json.loads(s) for s in (source/'eval_terminal_t4_i23/AGENT_EVENTS.jsonl').read_text().splitlines()]
accepted=next(e for e in events if e.get('event')=='queue_accepted')
assert accepted['time']<receipt['started_at'],'The receipt must establish acceptance before the failed call'
body['messages'][-1]['content']+=accepted_queue_receipt(accepted['session_id'])
(root/'UPDATED_REQUEST.json').write_text(json.dumps(body,indent=2))
(root/'SOURCE_BINDING.json').write_text(json.dumps({'source':str(paths[0].parent),'old_receipt':receipt,
    'queue_acceptance_event':accepted,'change':'Only append current accepted-queue receipt to the actual final tool result. No omitted history, temperature or token-budget change.','controls_execution':False},indent=2))
started=time.time()
with httpx.Client(timeout=240) as client:
    response=client.post('http://127.0.0.1:49510/v1/chat/completions',json=body)
    (root/'HTTP_RESPONSE.json').write_text(json.dumps({'status':response.status_code,'body':response.json()},indent=2))
    response.raise_for_status();value=response.json()
choice=value['choices'][0];message=choice['message']
assert choice['finish_reason']=='stop' and not message.get('tool_calls'),'Queue handoff still attempted a new tool or failed to finish'
assert accepted['session_id'] in message['content'] and any(w in message['content'].lower() for w in ['queued','accepted'])
result={'status':'PASSED_ONE_SAVED_INPUT','seconds':time.time()-started,'response':message['content'],
    'usage':value.get('usage'),'finish_reason':choice['finish_reason'],'controls_execution':False,
    'scope':'Real-model protocol replay of an already accepted queue; not a new robot run, learned-policy benefit or clean full-run cost measurement'}
(root/'RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
