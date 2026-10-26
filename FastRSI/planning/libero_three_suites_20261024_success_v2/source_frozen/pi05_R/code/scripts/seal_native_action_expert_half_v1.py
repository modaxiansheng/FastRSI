"""Immutable closed native-case copies; each host reads its own actual runs."""
import hashlib,json,shutil,sys
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI');slot=int(sys.argv[1]);assert slot in (0,1)
B=P/f'planning/native_action_expert_batch_host{slot}_v{1 if slot==0 else 2}_20261010'
assert json.loads((B/'STATUS.json').read_text())['status']=='COMPLETED'
assert json.loads((B/'AUDITS.json').read_text())['status']=='PASSED'
EXCLUDED={'tokenizer_cache','__pycache__'}
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
entries=[]
for version in range(16+slot*2,18+slot*2):
    name=f'native_value_phase_v{version}_20261010';source=P/'runs'/name;target=P/'planning'/name
    assert json.loads((source/'STATUS.json').read_text())['status']=='COMPLETED'
    for p in sorted(source.rglob('*')):
        rel=p.relative_to(source)
        if p.is_file() and not EXCLUDED.intersection(rel.parts):
            q=target/rel;q.parent.mkdir(parents=True,exist_ok=True)
            if q.exists():assert sha(q)==sha(p),str(q)
            else:shutil.copy2(p,q)
    for folder in [target,P/'planning'/(name+'_expert_audit')]:
        for p in sorted(folder.rglob('*')):
            if p.is_file() and not EXCLUDED.intersection(p.parts):entries.append(dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p)))
for p in sorted(B.rglob('*')):
    if p.is_file() and not EXCLUDED.intersection(p.parts) and p.name!='SEALED_HALF.json':entries.append(dict(path=str(p.relative_to(P)),bytes=p.stat().st_size,sha256=sha(p)))
out=B/'SEALED_HALF.json';assert not out.exists()
out.write_text(json.dumps(dict(status='SEALED_CLOSED_NATIVE_HALF',host_slot=slot,count=len(entries),bytes=sum(e['bytes'] for e in entries),entries=entries,default_sha256=sha(P/'CURRENT_NATIVE_ENTRY.json'),goal_complete=False),indent=2)+'\n')
print(json.dumps(dict(status='SEALED',slot=slot,files=len(entries))),flush=True)
