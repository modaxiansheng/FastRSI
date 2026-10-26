"""Seal completed native runs and receipts without changing their evidence."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
O=P/'planning/native_value_stage_receipt_v1_20261010'
EXCLUDE={'tokenizer_cache','__pycache__'}

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    assert json.loads((P/'planning/native_value_batch_v2_20261010/STATUS.json').read_text())['status']=='COMPLETED'
    O.mkdir(exist_ok=False);copied=[]
    names=[f'native_value_phase_v{v}_20261010' for v in range(1,12)]
    for name in names:
        source=P/'runs'/name;dest=P/'planning'/name
        for f in sorted(source.rglob('*')):
            if not f.is_file() or EXCLUDE.intersection(f.relative_to(source).parts):continue
            target=dest/f.relative_to(source);target.parent.mkdir(parents=True,exist_ok=True)
            if target.exists():assert sha(target)==sha(f)
            else:shutil.copy2(f,target)
            copied.append(dict(source=str(f.relative_to(P)),snapshot=str(target.relative_to(P))))
    names+=['native_value_source_v1_20261010','native_value_binding_audit_v1_20261010','native_value_binding_audit_v2_20261010',
        'native_value_batch_v1_20261010','native_value_batch_v2_20261010','native_value_batch_report_v1_20261010',
        'native_value_feedback_v1_20261010']
    names += [f'native_value_evaluation_freeze_v{v}_20261010' for v in range(1,4)]
    names += [f'native_value_phase_v{v}_20261010_audit' for v in range(8,12)]
    files=[]
    for name in names:
        root=P/'planning'/name
        for f in sorted(root.rglob('*')):
            if f.is_file() and not EXCLUDE.intersection(f.relative_to(root).parts):
                files.append(dict(path=str(f.relative_to(P)),bytes=f.stat().st_size,sha256=sha(f)))
    default=P/'CURRENT_NATIVE_ENTRY.json';expected='a3b27c5762d82656a07c4a8d65a454a42f5a64295ba3ebd104cc7fbe04244837'
    assert sha(default)==expected
    containers=[]
    for name in subprocess.check_output(['docker','ps','-a','--format','{{.Names}}'],text=True).splitlines():
        if not name.startswith(('yuchen-fastrsi-native_value_phase_','yuchen-fastrsi-native-value-')):continue
        detail=json.loads(subprocess.check_output(['docker','inspect',name]))[0];state=detail['State']
        assert not state['Running'] and state['Pid']==0,name
        containers.append(dict(name=name,image=detail['Image'],state=state))
    (O/'EVIDENCE_MANIFEST.json').write_text(json.dumps(dict(status='SEALED_NATIVE_STAGE',entries=files,
        files=len(files),bytes=sum(f['bytes'] for f in files),copied=copied,containers=containers,default_sha256=expected,
        preserved_failed_attempts=True,full_recursive_update_gain_established=False,goal_complete=False),indent=2)+'\n')
    print(json.dumps(dict(files=len(files),bytes=sum(f['bytes'] for f in files),exited_containers=len(containers))),flush=True)

if __name__=='__main__':main()
