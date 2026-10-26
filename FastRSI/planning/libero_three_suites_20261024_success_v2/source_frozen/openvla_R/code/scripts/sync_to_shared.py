"""Sync only our small source/docs files, retaining prior contents by hash."""
import argparse
import base64
import json
import subprocess
from pathlib import Path

parser=argparse.ArgumentParser()
parser.add_argument('--host',default='flowwam-gpu-187')
args=parser.parse_args()
root=Path(__file__).resolve().parents[1]
files=sorted([*root.glob('scripts/*.py'),*root.glob('docs/*.md'),*root.glob('configs/*.json'),
              *root.glob('src/fastrsi/*.py'),*root.glob('tests/*.py')])
payload={str(p.relative_to(root)):base64.b64encode(p.read_bytes()).decode() for p in files}
remote='''import base64,hashlib,json,time
from pathlib import Path
root=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
payload=json.loads(PAYLOAD)
rows=[]
for rel,value in payload.items():
 p=root/rel;data=base64.b64decode(value);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists() and p.read_bytes()!=data:
  old=p.read_bytes();archive=root/'planning/revisions'/hashlib.sha256(old).hexdigest()/rel
  archive.parent.mkdir(parents=True,exist_ok=True)
  if not archive.exists():archive.write_bytes(old)
 if not p.exists() or p.read_bytes()!=data:
  temp=p.with_suffix(p.suffix+'.sync-tmp');temp.write_bytes(data);temp.replace(p)
 assert p.read_bytes()==data
 rows.append({'path':str(p),'sha256':hashlib.sha256(data).hexdigest()})
print(json.dumps({'time':time.time(),'files':rows}))
'''.replace('PAYLOAD',repr(json.dumps(payload)))
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15',
    '-o','ServerAliveInterval=10','-o','ServerAliveCountMax=2',
    '-o','ControlPath=/tmp/fastrsi-run-%C',args.host,'python3','-'],
    input=remote,text=True,capture_output=True,timeout=90)
if result.returncode:raise SystemExit(result.stderr)
receipt=json.loads(result.stdout)
out=root/'runs/pilot_20261001'
out.mkdir(parents=True,exist_ok=True)
(out/'SOURCE_SYNC.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(f'Verified {len(receipt["files"])} source/doc files on {args.host}')
