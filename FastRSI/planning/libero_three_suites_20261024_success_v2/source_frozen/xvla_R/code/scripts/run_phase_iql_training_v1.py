"""Owned CPU training of external semi-MDP Q/V; no base model loaded."""
import fcntl,hashlib,json,shutil,subprocess,time,re
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
A=P.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    root=P/'runs/phase_iql_training_v1_20261010';root.mkdir(exist_ok=False)
    def save(name,value):(root/name).write_text(json.dumps(value,indent=2)+'\n')
    assert json.loads((P/'planning/phase_value_features_v1_20261010/RESULT.json').read_text())['status']=='COMPLETED'
    before=sha(P/'CURRENT_NATIVE_ENTRY.json');(root/'code').mkdir()
    def cap(argv):
        r=subprocess.run(argv,capture_output=True,text=True,timeout=40)
        return dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
    pre=dict(hcu=cap(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showmemuse']),
        containers=cap(['docker','ps','--format','{{.Names}}']),holders=cap(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*']))
    save('PREFLIGHT.json',pre)
    mem=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',pre['hcu']['stdout'])
    assert len(mem)==8 and all(float(x)==0 for _,x in mem) and not pre['containers']['stdout'].strip()
    for pid in set(map(int,pre['holders']['stdout'].split())):
        assert (Path('/proc')/str(pid)/'comm').read_text().strip()=='hymgr'
        assert str((Path('/proc')/str(pid)/'exe').resolve())=='/usr/local/hyhal/bin/hymgr'
    for name in ('phase_iql_value_v1.py','run_phase_iql_training_v1.py'):
        shutil.copy2(P/'scripts'/name,root/'code'/name)
    save('PROTOCOL.json',dict(default_sha256=before,updates=3000,twin_q_heads=2,device='cpu',driver_visible_device=2,reason='CPU tensor training; HCU torch runtime requires visible driver device',
        input_data_sha256=sha(P/'planning/phase_value_dataset_v1_20261010/TRANSITIONS.jsonl'),
        new_completed_data_sha256=sha(P/'planning/phase_value_features_v1_20261010/RESULT.json'),
        native_policy_updates=0,validation_groups=['demo_25','init_37','init_40'],
        code_sha256={f.name:sha(f) for f in (root/'code').iterdir()}))
    name='yuchen-fastrsi-phase-iql-training-v1-20261010'
    argv=['docker','run','--name',name,'--network','none','--cpus','4','--memory','12g','--pids-limit','512',
        '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--group-add','render',
        '-e','HIP_VISIBLE_DEVICES=2','-e','CUDA_VISIBLE_DEVICES=2','-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
        '-e','PYTHONDONTWRITEBYTECODE=1','-e','PYTHONPATH=/check/code:/core',
        '-v',str(P)+':/project:ro','-v',str(root)+':/check',
        '-v',str(A/'dependencies/PhyAgentOS-core')+':/core:ro','-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
        'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81',
        'timeout','300','bash','-lc','set -e; source /opt/dtk/env.sh; source /opt/hyhal/env.sh; exec python /check/code/phase_iql_value_v1.py --project /project --output /check/training']
    save('LAUNCH.json',argv);tick=time.perf_counter();code=-1
    try:
        with (root/'TRAINING.log').open('x') as log:
            result=subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,timeout=350);code=result.returncode
    finally:
        inspection=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        if inspection['State']['Running']:
            subprocess.run(['docker','stop','-t','10',name],capture_output=True)
            inspection=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
        save('CONTAINER.json',inspection)
        save('HOST_RESULT.json',dict(exit_code=code,full_wall_seconds=time.perf_counter()-tick,default_unchanged=before==sha(P/'CURRENT_NATIVE_ENTRY.json')))
    assert code==0 and not inspection['State']['Running'] and before==sha(P/'CURRENT_NATIVE_ENTRY.json')
    report=json.loads((root/'training/RESULT.json').read_text())
    assert report['native_pi_parameter_updates']==0 and report['model_sha256']==sha(root/'training/value_model.npz')
    print(json.dumps(dict(status=report['status'],training_rows=report['train_rows'],validation_rows=report['validation_rows'],
        full_wall_seconds=time.perf_counter()-tick,model_sha256=report['model_sha256'])),flush=True)
if __name__=='__main__':main()
