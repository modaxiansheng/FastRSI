"""Three predeclared source starts; one unchanged policy service, no training."""
import hashlib, json, os, shutil, socket, subprocess, sys, time, urllib.request
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
A=P.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
ARM=sys.argv[1]
assert ARM in ('seed7','seed19')
SEED=7 if ARM=='seed7' else 19
SLOT=0 if ARM=='seed7' else 1
PORT=56980+2*SLOT
R=P/'runs'/('bowl_correction_'+ARM+'_v1_20261009')
def save(name,obj): (R/name).write_text(json.dumps(obj,indent=2)+'\n')
def capture(argv):
    p=subprocess.run(argv,capture_output=True,text=True,timeout=40)
    return dict(argv=argv,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def main():
    gate=json.loads((P/'planning/bowl_correction_acquisition_v1_20261009/PARENT_GATE.json').read_text())
    assert int(os.environ['FASTRSI_CORRECTION_PARENT_PID'])==gate['pid']
    os.kill(gate['pid'],0)
    assert gate['worker_sha256']==hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    R.mkdir(exist_ok=False)
    before=hashlib.sha256((P/'CURRENT_NATIVE_ENTRY.json').read_bytes()).hexdigest()
    s=socket.socket();s.bind(('127.0.0.1',PORT));s.close()
    (R/'code').mkdir()
    for f in ['qualify_bowl_correction_v1.py','serve_bowl_correction_policy_v1.py','replay_stack_progress.py','stage_trace.py','run_bowl_correction_worker_v1.py']:
        shutil.copy2(P/'scripts'/f,R/'code'/f)
    shutil.copytree(P/'models/openpi_data',R/'tokenizer_cache')
    save('PROTOCOL.json',dict(arm=ARM,slot=SLOT,case='demo_0',start_steps=[120,250,270],cap=400,execution_window=10,sampling_steps=10,
        policy_seed=SEED,initial_restore_once=True,all_cases_development=True,training_steps=0,
        complete_agent_comparison=False,default_before_sha256=before,
        code_sha256={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in (R/'code').iterdir()},
        timing_scope='Policy service startup shared; source replay wall not comparable to policy online wall'))
    policy_name='yuchen-fastrsi-correction-policy-'+ARM+'-v1-20261009';owned=[];results=[];started=time.perf_counter()
    argv=['docker','run','-d','--name',policy_name,'--network','host','--cpus','8','--memory','64g','--pids-limit','1024','--shm-size','2g',
        '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--group-add','render',
        '-e','HIP_VISIBLE_DEVICES='+str(SLOT),'-e','CUDA_VISIBLE_DEVICES='+str(SLOT),'-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
        '-e','OPENPI_DATA_HOME=/workspace/openpi_data','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','PYTHONPATH=/check/code:/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/workspace/PhyAgentOS-core',
        '-v',str(R)+':/check','-v',str(A/'dependencies/openpi')+':/workspace/openpi:ro',
        '-v',str(A/'dependencies/PhyAgentOS-core')+':/workspace/PhyAgentOS-core:ro',
        '-v',str(A/'weights/pi05_libero_pytorch_fp32')+':/model:ro','-v',str(R/'tokenizer_cache')+':/workspace/openpi_data',
        '-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
        'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81',
        'timeout','1200','bash','-lc','set -e; source /opt/dtk/env.sh; source /opt/hyhal/env.sh; exec python /check/code/serve_bowl_correction_policy_v1.py --checkpoint-dir /model --policy-config pi05_libero --pytorch-device cuda --disable-torch-compile --torch-seed '+str(SEED)+' --host 127.0.0.1 --port '+str(PORT)]
    image_index=next(i for i,value in enumerate(argv) if value.startswith('sha256:'))
    extras=['-v',str(P)+':/project:ro']
    argv[image_index:image_index]=extras
    save('POLICY_LAUNCH.json',argv)
    try:
        r=capture(argv);save('POLICY_DISPATCH.json',r);assert r['returncode']==0,r;owned.append(policy_name)
        deadline=time.monotonic()+220
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:'+str(PORT)+'/healthz',timeout=3) as response: assert response.status==200
                break
            except Exception:
                info=json.loads(subprocess.check_output(['docker','inspect',policy_name]))[0]
                assert info['State']['Running'],info['State']
                if time.monotonic()>deadline:raise TimeoutError('Native policy startup')
                time.sleep(2)
        save('STARTUP.json',dict(seconds=time.perf_counter()-started))
        for index in [120,250,270]:
            case='step_'+str(index);name='yuchen-fastrsi-correction-'+ARM+'-'+case+'-v1-20261009';owned.append(name)
            cmd=['docker','run','--name',name,'--network','host','--cpus','4','--memory','12g','--pids-limit','1024','--shm-size','2g','--entrypoint','python',
                '-e','PYTHONPATH=/workspace/openpi/third_party/libero:/core:/check/code',
                '-e','LIBERO_CONFIG_PATH=/libero-config','-e','MUJOCO_GL=egl','-e','LIBGL_ALWAYS_SOFTWARE=1',
                '-v',str(P)+':/project:ro','-v',str(R)+':/check',
                '-v',str(A/'dependencies/openpi')+':/workspace/openpi:ro',
                '-v',str(A/'dependencies/PhyAgentOS-core')+':/core:ro','-v',str(P/'configs/libero')+':/libero-config:ro',
                'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95',
                '/check/code/qualify_bowl_correction_v1.py','--demo','/project/planning/bowl_behavior_adaptation_v1_20261009/exports/demo_0',
                '--output','/check/'+case,'--endpoint','ws://127.0.0.1:'+str(PORT),'--start-step',str(index),'--policy-seed',str(SEED)]
            save(case+'_LAUNCH.json',cmd);tick=time.perf_counter()
            with (R/(case+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            save(case+'_HOST.json',dict(exit_code=r.returncode,seconds=time.perf_counter()-tick))
            assert r.returncode==0,case
            result=json.loads((R/case/'RESULT.json').read_text());assert result['status']=='COMPLETED'
            row=dict(case=case,policy_seed=SEED,
                intervention_inputs_exact=all(result['intervention_inputs_exact'].values()),
                policy_success=result['native_success'],prefix_actions=index,suffix_actions=result['suffix_actions'],
                policy_vla_calls=result['vla_calls'],suffix_seconds=result['suffix_seconds'],
                full_seconds=result['seconds_including_close'])
            results.append(row);save('PARTIAL.json',results);print(json.dumps(row),flush=True)
        save('AGGREGATE.json',dict(status='COMPLETED',cases=results,
            policy_successes=sum(r['policy_success'] for r in results),
            suffix_actions=sum(r['suffix_actions'] for r in results),
            scope='Three exposed difficult states, unchanged policy sampler candidates; not deployment or learning gain'))
    finally:
        cleanup=[]
        for name in owned:
            info=capture(['docker','inspect',name])
            if info['returncode']!=0:continue
            state=json.loads(info['stdout'])[0]['State']
            if state['Running']:capture(['docker','stop','-t','10',name])
            final=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            save(name+'_CONTAINER.json',final);cleanup.append(dict(name=name,state=final['State']))
            if name==policy_name:
                logs=capture(['docker','logs',name]);(R/'POLICY.log').write_text(logs['stdout']+logs['stderr'])
        after=hashlib.sha256((P/'CURRENT_NATIVE_ENTRY.json').read_bytes()).hexdigest()
        save('FINAL_STATE.json',dict(containers=cleanup,default_unchanged=before==after,default_after_sha256=after,
            host_wall_seconds=time.perf_counter()-started))
        assert before==after and all(not c['state']['Running'] and c['state']['Pid']==0 for c in cleanup)
if __name__=='__main__':main()
