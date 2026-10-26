"""Idempotent report delivery with SHA readback and prior-byte preservation."""
import base64
import hashlib
import json
import subprocess
from pathlib import Path

root=Path(__file__).resolve().parents[1]
files=[Path(__file__),root/'scripts/analyze_native_stage1.py',root/'scripts/analyze_native_run.py',
       root/'docs/IMPLEMENTATION_PLAN.md',root/'docs/NATIVE_LIFECYCLE_PROGRESS_20261002.md',
       *[p for p in sorted(root.glob('planning/native_stage1_v1/*.json')) if p.name!='DELIVERY.json']]
payload={str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for p in files}
code='''from pathlib import Path
import base64,json,hashlib
root=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI');payload=json.loads(%r);manifest={}
def write(p,b):
 p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=b:
  old=p.read_bytes();q=root/'planning/revisions'/hashlib.sha256(old).hexdigest()/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True)
  if not q.exists():q.write_bytes(old)
 t=p.with_name(p.name+'.sync-tmp');t.write_bytes(b);t.replace(p);assert p.read_bytes()==b
for rel,s in payload.items():
 p=root/rel;b=base64.b64decode(s);write(p,b);manifest[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
value={'via_host':'188','shared_root':str(root),'verified_files':len(manifest),'files':manifest}
write(root/'planning/native_stage1_v1/DELIVERY.json',json.dumps(value,indent=2).encode())
print(json.dumps(value))
'''%json.dumps(payload)
argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
      '-o','ControlMaster=no','-o','ControlPath=none',
      '-o','ProxyCommand=ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=15 -W %h:%p flowwam-jump',
      'flowwam-gpu-188','python3','-']
r=subprocess.run(argv,input=code,text=True,capture_output=True,timeout=60)
if r.returncode:print(r.stderr);raise SystemExit(r.returncode)
value=json.loads(r.stdout)
expected={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
assert value['files']==expected
p=root/'planning/native_stage1_v1/DELIVERY.json';b=json.dumps(value,indent=2).encode()
if p.exists() and p.read_bytes()!=b:
    old=p.read_bytes();q=root/'planning/revisions'/hashlib.sha256(old).hexdigest()/p.relative_to(root);q.parent.mkdir(parents=True,exist_ok=True)
    if not q.exists():q.write_bytes(old)
p.write_bytes(b)
print(json.dumps({'status':'SHA_VERIFIED_SHARED_DELIVERY','files':value['verified_files'],'via_host':'188'}))
