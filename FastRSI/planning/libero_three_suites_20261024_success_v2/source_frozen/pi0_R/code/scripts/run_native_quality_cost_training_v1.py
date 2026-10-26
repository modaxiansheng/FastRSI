"""Owned CPU fit/update, pre-registered sources and unchanged default entry."""
import fcntl,hashlib,json,shutil,subprocess,time
from pathlib import Path
from run_native_value_batch_v1 import gate
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI');B=P/'planning/native_quality_cost_training_v1_20261010'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def save(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
def main():
    lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    B.mkdir(exist_ok=False);gate(B);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
    registration=P/'planning/native_quality_cost_head_registration_v1_20261010/REGISTRATION.json'
    assert json.loads(registration.read_text())['status']=='REGISTERED_BEFORE_NEW_QUALITY_COST_HEAD_FITTING'
    (B/'code').mkdir()
    for name in ['train_native_quality_cost_v1.py','native_quality_cost_v1.py','phase_iql_value_v1.py','run_native_quality_cost_training_v1.py','run_native_value_batch_v1.py']:
        shutil.copy2(P/'scripts'/name,B/'code'/name)
    save(B/'REGISTRATION.json',dict(status='BOUND_TO_PRE_REGISTERED_QUALITY_COST_HEAD_PLAN',
        plan_sha256=sha(registration),training_device='cpu',default_sha256=before,
        source_sha256={f.name:sha(f) for f in (B/'code').iterdir()},pi_updates=0,llm_updates=0,goal_complete=False))
    previous=P/'planning/native_paired_q_update_v1_20261010'
    argv=[x.replace(str(previous),str(B)).replace('update_native_paired_q_v1.py','train_native_quality_cost_v1.py')
        for x in json.loads((previous/'LAUNCH.json').read_text())]
    name='yuchen-fastrsi-native-quality-cost-head-v1-g187-20261010';argv[argv.index('--name')+1]=name
    save(B/'LAUNCH.json',argv);tick=time.perf_counter()
    try:
        with (B/'TRAINING.log').open('x') as log:r=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=310)
        assert r.returncode==0,r.returncode
        result=json.loads((B/'training/RESULT.json').read_text())
        assert result['initial_updates']==1000 and result['feedback_updates']==500
        assert result['pi_updates']==result['llm_updates']==result['robot_executions']==0
        for arm,modelsha in result['model_sha256'].items():assert sha(B/'training'/f'quality_cost_{arm}.npz')==modelsha
    finally:
        record=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        if record['State']['Running']:subprocess.run(['docker','stop','-t','10',name],check=True,capture_output=True)
        record=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        assert not record['State']['Running'] and record['State']['Pid']==0
        save(B/'STOP_RECEIPT.json',dict(name=record['Name'],image=record['Image'],state=record['State']))
        save(B/'HOST_RESULT.json',dict(seconds=time.perf_counter()-tick,default_unchanged=before==sha(P/'CURRENT_NATIVE_ENTRY.json')))
    assert before==sha(P/'CURRENT_NATIVE_ENTRY.json')
    print(json.dumps({k:v for k,v in result.items() if k not in ('development_pair_selections','diagnostics')}),flush=True)
if __name__=='__main__':main()
