"""Replay duplicate reads against their exact saved visible model messages."""
import hashlib,json,os
from pathlib import Path
from fastrsi.visible_reads import VisibleReads
root=Path(__file__).resolve().parents[1]
# Campaign snapshots sit under runs/<name>/code; runtime project is mounted here.
root=Path('/project') if Path('/project/runs').exists() else root
folder=root/'runs/workflow_matched_v1_20261002/B/stream/matched_t6_i9'
requests=sorted(folder.glob('agent_call_*/REQUEST.json'),key=lambda p:int(p.parent.name.split('_')[-1]))
rows=[]
for request in requests:
    messages=json.loads(request.read_text())['messages']
    # Last tool batch is the fresh result being replayed. Prior messages are the
    # context the model actually saw before issuing that call (not later history).
    if not messages or messages[-1].get('role')!='tool':continue
    try: payload=json.loads(messages[-1]['content'])['fresh_reads']
    except (ValueError,KeyError,TypeError):continue
    paths=[x['path'] for x in payload]
    if not all(p.startswith('/project/') for p in paths):continue
    workspace=next(Path(p).parent for p in paths if p.endswith('/REQUEST.json'))
    v=VisibleReads(workspace);v.observe(messages[:-2])
    rendered=[{'path':x['path'],'result':v.render(x['path'],x['result'])} for x in payload]
    old=json.dumps({'fresh_reads':payload},ensure_ascii=False);new=json.dumps({'fresh_reads':rendered},ensure_ascii=False)
    rows.append({'request':str(request.relative_to(root)),'sha256':hashlib.sha256(request.read_bytes()).hexdigest(),
                 'files':len(payload),'references':sum(x['result']!=y['result'] for x,y in zip(payload,rendered)),
                 'before_chars':len(old),'after_chars':len(new),'changed_file_fallback':all(v.render(x['path'],x['result']+' changed')==x['result']+' changed' for x in payload)})
assert rows and any(x['references']>=5 for x in rows),rows
out={'scope':'Saved-input CPU replay only, no model latency or quality claim','cases':rows}
print(json.dumps(out,indent=2))
if os.environ.get('FASTRSI_REPLAY_OUTPUT'):
    with open(os.environ['FASTRSI_REPLAY_OUTPUT'],'x') as f:json.dump(out,f,indent=2)
