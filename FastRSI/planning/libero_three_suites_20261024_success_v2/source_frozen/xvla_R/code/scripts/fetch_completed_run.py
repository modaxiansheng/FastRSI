"""Copy completed own run with file hashes; never replace differing local evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('run_id')
    parser.add_argument('--allow-failed',action='store_true')
    parser.add_argument('--host',default='flowwam-gpu-187',choices=['flowwam-gpu-187','flowwam-gpu-188'])
    parser.add_argument('--control-path',help='Reuse an already established SSH connection when the jump host rejects a fresh connection')
    args=parser.parse_args()
    assert re.fullmatch(r'[a-zA-Z0-9_-]+',args.run_id) or re.fullmatch(r'representative_native_[a-zA-Z0-9_]+/batch_[0-9]+_s[012]',args.run_id)
    root=Path(__file__).resolve().parents[1]
    remote_base='/mnt/data/task/yuchen/FastPhyagentos/FastRSI/runs'
    ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ControlMaster=no','-o','ControlPath=none',
         '-o','ProxyCommand=ssh -o BatchMode=yes -o ControlMaster=no -o ControlPath=none -o ConnectTimeout=15 -W %h:%p flowwam-jump',args.host]
    if args.control_path:
        ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=20','-o','ControlPath='+args.control_path,args.host]
    code='''import hashlib,json,sys
from pathlib import Path
root=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI/runs')/sys.argv[1]
status=json.loads((root/'STATUS.json').read_text())['status']
assert status=='COMPLETED' or (sys.argv[2]=='allow-failed' and status=='FAILED')
files={str(p.relative_to(root)):{'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':p.stat().st_size}
       for p in root.rglob('*') if p.is_file() and 'tokenizer_cache' not in p.parts and p.name!='DELIVERY_MANIFEST.json'}
receipt={'run_id':sys.argv[1],'excluded':['tokenizer_cache'],'files':files}
p=root/'DELIVERY_MANIFEST.json'
if p.exists():assert json.loads(p.read_text())==receipt, 'Remote run changed after delivery'
else:
 with p.open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''
    cmd='python3 - '+shlex.quote(args.run_id)+' '+('allow-failed' if args.allow_failed else 'completed-only')
    receipt=json.loads(subprocess.run(ssh+[cmd],input=code,text=True,capture_output=True,check=True,timeout=90).stdout)
    target=root/'runs'/args.run_id
    for rel,row in receipt['files'].items():
        p=target/rel
        if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'], 'Local evidence differs: '+rel
    with tempfile.TemporaryDirectory(prefix='fastrsi-delivery-') as tmp:
        archive=Path(tmp)/'run.tar.gz'
        cmd=shlex.join(['tar','--exclude=tokenizer_cache','-czf','-','-C',remote_base,args.run_id])
        with archive.open('wb') as out:subprocess.run(ssh+[cmd],stdout=out,check=True,timeout=120)
        with tarfile.open(archive) as tar:
            for member in tar.getmembers():
                path=(root/'runs'/member.name).resolve()
                assert path.is_relative_to(target.resolve()) and (member.isfile() or member.isdir())
            tar.extractall(root/'runs')
    for rel,row in receipt['files'].items():
        p=target/rel
        assert p.stat().st_size==row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==row['sha256'],rel
    print(json.dumps({'run_id':args.run_id,'verified_files':len(receipt['files']),
                     'verified_bytes':sum(r['bytes'] for r in receipt['files'].values())}))


if __name__=='__main__':main()
