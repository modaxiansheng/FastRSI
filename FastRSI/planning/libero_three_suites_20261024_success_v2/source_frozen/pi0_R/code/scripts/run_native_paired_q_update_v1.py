"""Owned fixed CPU update using real paired feedback; preserves all old receipts."""
import fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path
from run_native_value_batch_v1 import gate
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI');B=P/'planning/native_paired_q_update_v1_20261010'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
    lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    B.mkdir(exist_ok=False);gate(B);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
    assert json.loads((P/'planning/native_value_branch_batch_v2_20261010/RESULT.json').read_text())['status']=='COMPLETED'
    (B/'code').mkdir()
    for name in ('update_native_paired_q_v1.py','phase_iql_feedback_update_v1.py','phase_iql_value_v1.py'):
        shutil.copy2(P/'scripts'/name,B/'code'/name)
    save(B/'REGISTRATION.json',dict(status='REGISTERED_BEFORE_PARAMETER_UPDATES',updates=300,learning_rate=5e-5,
        old_retention_coefficient=5,ranking_coefficient=1,seed=37,old_rows=1320,pairs=6,initial_groups=2,
        only_q_updated=True,v_pca_normalization_pi_frozen=True,held_out_labels_used=False,
        future_roster=[[i,10101000+i] for i in range(32,36)],future_arms=['default','frozen','paired_updated'],
        source_sha256={f.name:sha(f) for f in (B/'code').iterdir()},default_sha256=before))
    template=json.loads((P/'planning/phase_iql_feedback_v1_20261010/LAUNCH.json').read_text())
    old=str(P/'runs/phase_iql_feedback_v1_20261010')
    argv=[x.replace(old,str(B)).replace('phase_iql_feedback_update_v1.py','update_native_paired_q_v1.py') for x in template]
    name='yuchen-fastrsi-native-paired-q-update-v1-20261010';argv[argv.index('--name')+1]=name
    save(B/'LAUNCH.json',argv);tick=time.perf_counter()
    try:
        with (B/'TRAINING.log').open('x') as f:r=subprocess.run(argv,stdout=f,stderr=subprocess.STDOUT,timeout=310)
        assert r.returncode==0,r.returncode
        result=json.loads((B/'training/RESULT.json').read_text());assert result['updates']==300 and result['pi_updates']==result['v_updates']==0
        assert result['updated_sha256']==sha(B/'training/paired_router.npz')
    finally:
        record=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        if record['State']['Running']:subprocess.run(['docker','stop','-t','10',name],check=True,capture_output=True)
        record=json.loads(subprocess.check_output(['docker','inspect',name]))[0];assert not record['State']['Running'] and record['State']['Pid']==0
        save(B/'STOP_RECEIPT.json',dict(name=record['Name'],image=record['Image'],state=record['State']))
        save(B/'HOST_RESULT.json',dict(seconds=time.perf_counter()-tick,default_unchanged=before==sha(P/'CURRENT_NATIVE_ENTRY.json')))
    assert before==sha(P/'CURRENT_NATIVE_ENTRY.json');print(json.dumps({k:v for k,v in result.items() if k!='input_development_pairs'}),flush=True)
if __name__=='__main__':main()
