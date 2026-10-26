"""Bounded two-GPU sampler probe and eight normal-start runs; no model training."""
from concurrent.futures import ThreadPoolExecutor
import fcntl,hashlib,json,os,re,shutil,subprocess,time
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
B=P/'planning/bowl_value_selection_v1_20261009'
def capture(argv):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=40)
    return dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,data):(B/name).write_text(json.dumps(data,indent=2)+'\n')
def main():
    lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    B.mkdir(exist_ok=False);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
    pre=dict(hcu=capture(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse']),
        containers=capture(['docker','ps','--format','{{.ID}} {{.Names}}']),holders=capture(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*']))
    save('PREFLIGHT.json',pre)
    memory=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',pre['hcu']['stdout'])
    assert pre['hcu']['returncode']==0 and len(memory)==8 and all(float(v)==0 for _,v in memory)
    assert pre['containers']['returncode']==0 and not pre['containers']['stdout'].strip()
    identities=[]
    for pid in [int(x) for x in pre['holders']['stdout'].split()]:
        comm=(Path('/proc')/str(pid)/'comm').read_text().strip();exe=str((Path('/proc')/str(pid)/'exe').resolve(strict=True))
        identities.append(dict(pid=pid,comm=comm,exe=exe));assert comm=='hymgr' and exe=='/usr/local/hyhal/bin/hymgr'
    save('HOLDER_IDENTITIES.json',identities)
    worker=P/'scripts/run_bowl_value_selection_worker_v1.py'
    (B/'code').mkdir()
    for name in ('run_bowl_value_selection_v1.py','run_bowl_value_selection_worker_v1.py','serve_bowl_candidates_v2.py','check_bowl_candidates_rollout_v2.py'):
        shutil.copy2(P/'scripts'/name,B/'code'/name)
    model=P/'runs/bowl_mc_value_v2_20261009/training/value_model.npz'
    save('PARENT_GATE.json',dict(pid=os.getpid(),worker_sha256=sha(worker),value_model_sha256=sha(model),default_sha256=before,
        gpus=[0,1,2],initials=[45,46,47],arms=['single','random2','value2'],task_id=64,suite='libero_90',
        total_runs=9,max_task_actions=3600,maximum_rollout_candidate_inferences=600,
        offline_probe_inferences=78,training_steps=0,complete_agent_comparison=False,
        scope='Frozen MC outcome selector actual consequences; weights never updated during comparison'))
    started=time.perf_counter()
    def run(slot):
        tick=time.perf_counter()
        with (B/('gpu%d.log'%slot)).open('x') as log:
            try:r=subprocess.run(['python3',str(worker),str(slot)],stdout=log,stderr=subprocess.STDOUT,timeout=1300,
                env=dict(os.environ,FASTRSI_VALUE_PARENT_PID=str(os.getpid())));code=r.returncode
            except subprocess.TimeoutExpired:code=124
        result=dict(slot=slot,exit_code=code,host_seconds=time.perf_counter()-tick);save('GPU%d_HOST.json'%slot,result);return result
    with ThreadPoolExecutor(max_workers=3) as pool:results=list(pool.map(run,[0,1,2]))
    save('EXECUTION.json',dict(results=results,parallel_host_seconds=time.perf_counter()-started))
    cleanup=[]
    for slot in (0,1,2):
        names=['yuchen-fastrsi-bowl-value-selection-policy-gpu%d-v1-20261009'%slot]
        names+=['yuchen-fastrsi-bowl-value-selection-gpu%d-init%d_%s-v1-20261009'%(slot,initial,
            ('single','random2','value2')[(i+slot)%3]) for i,initial in enumerate((45,46,47))]
        for name in names:
            info=capture(['docker','inspect',name])
            if info['returncode']!=0:continue
            state=json.loads(info['stdout'])[0]['State']
            if state['Running']:capture(['docker','stop','-t','10',name])
            final=json.loads(subprocess.check_output(['docker','inspect',name]))[0]['State']
            cleanup.append(dict(name=name,state=final))
    save('PARENT_CLEANUP.json',cleanup)
    assert all(not c['state']['Running'] and c['state']['Pid']==0 for c in cleanup)
    assert all(r['exit_code']==0 for r in results),results
    cases=[];containers=[];probes=[]
    for slot in (0,1,2):
        root=P/'runs'/('bowl_value_selection_gpu%d_v1_20261009'%slot)
        cases+=json.loads((root/'AGGREGATE.json').read_text())['cases']
        final=json.loads((root/'FINAL_STATE.json').read_text());assert final['default_unchanged'];containers+=final['containers']
        probes.append(json.loads((root/'SAMPLER_PROBE.json').read_text()))
    pairs=[]
    for initial in (45,46,47):
        pair={r['arm']:r for r in cases if r['initial']==initial}
        assert len(pair)==3 and pair['single']['initial_input_sha256']==pair['random2']['initial_input_sha256']==pair['value2']['initial_input_sha256']
        pairs.append(dict(initial=initial,initial_inputs_exact=True,**pair))
    arms={}
    for arm in ('single','random2','value2'):
        subset=[r for r in cases if r['arm']==arm]
        arms[arm]=dict(successes=sum(r['native_success'] for r in subset),runs=len(subset),actions=sum(r['actions'] for r in subset),
            control_requests=sum(r['control_requests'] for r in subset),candidate_inferences=sum(r['candidate_inferences'] for r in subset),
            full_seconds=sum(r['full_seconds'] for r in subset))
    assert before==sha(P/'CURRENT_NATIVE_ENTRY.json')
    save('AGGREGATE.json',dict(status='COMPLETED',arms=arms,pairs=pairs,probes=probes,training_steps=0,
        default_unchanged=True,learning_gain_proven=False,complete_agent_comparison=False))
    save('FINAL_STATE.json',dict(containers=containers,default_sha256=before))
    print(json.dumps(dict(status='COMPLETED',arms=arms)),flush=True)
if __name__=='__main__':main()
