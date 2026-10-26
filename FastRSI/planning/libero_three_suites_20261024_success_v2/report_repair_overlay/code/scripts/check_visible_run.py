"""Validate every delivered reference against the actual model request history."""
import argparse,json,hashlib
from pathlib import Path
from fastrsi.visible_reads import VisibleReads
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--out',required=True);a=p.parse_args()
root=Path(__file__).resolve().parents[1];run=root/'runs'/a.run
folder=run/'B/stream/matched_t6_i9'
events=[json.loads(x) for x in (folder/'AGENT_EVENTS.jsonl').read_text().splitlines()]
refs=[e for e in events if e.get('event')=='visible_read_reused']
proofs={}
for req in sorted(folder.glob('agent_call_*/REQUEST.json')):
    messages=json.loads(req.read_text())['messages']
    for index,msg in enumerate(messages):
        if msg.get('role')!='tool':continue
        try:rows=json.loads(msg['content'])['fresh_reads']
        except (KeyError,ValueError,TypeError):continue
        for row in rows:
            if not row['result'].startswith('Fresh read matches the complete numbered contents already visible'):continue
            workspace=Path('/project/runs')/a.run/'B/stream/matched_t6_i9/workspace'
            visible=VisibleReads(workspace);visible.observe(messages[:index])
            matches=[(key,source) for key,source in visible.visible.items() if key[0]==visible.canonical(row['path']) and visible.render(row['path'],key[1])==row['result']]
            assert len(matches)==1,(req,row['path'],len(matches))
            (path,original),source=matches[0]
            assert any(e['source_tool_call_id']==source and e['target_path']==row['path'] and e['original_chars']==len(original) for e in refs)
            proofs[(msg['tool_call_id'],path)]={'path':path,'source_call':source,'first_checked_request':str(req.relative_to(root)),'original_sha256':hashlib.sha256(original.encode()).hexdigest()}
assert len(proofs)==len(refs),(len(proofs),len(refs))
restored_deliveries=0
for req in folder.glob('agent_call_*/REQUEST.json'):
    for msg in json.loads(req.read_text())['messages']:
        if msg.get('role')!='tool':continue
        try:rows=json.loads(msg['content'])['fresh_reads']
        except (KeyError,ValueError,TypeError):continue
        for row in rows:
            key=(msg['tool_call_id'],str(Path(row['path'])))
            if key not in proofs or row['result'].startswith('Fresh read matches'):continue
            assert hashlib.sha256(row['result'].encode()).hexdigest()==proofs[key]['original_sha256'],(req,key)
            restored_deliveries+=1
assert restored_deliveries==sum(e.get('event')=='visible_read_restored' for e in events)
value={'restored_deliveries_verified':restored_deliveries,'status':'PASSED','references':len(refs),'proofs':list(proofs.values()),'scope':'Complete original text visible in actual model requests; native raw reads preserved separately. Not proof of reasoning correctness.'}
with (root/a.out).open('x') as f:json.dump(value,f,indent=2)
print(json.dumps({'status':value['status'],'references':len(refs)}))
