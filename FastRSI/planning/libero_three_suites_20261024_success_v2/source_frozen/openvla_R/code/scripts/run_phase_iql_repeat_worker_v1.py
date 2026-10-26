"""One owned policy service; four-arm bounded adaptive sampling pilot."""
import hashlib,json,os,shutil,socket,subprocess,sys,time,urllib.request
from pathlib import Path
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
A=P.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
SLOT=int(sys.argv[1]);assert SLOT in range(8)
PORT=57700+SLOT;R=P/'runs'/('phase_iql_repeat_gpu%d_v1_20261010'%SLOT)
def save(name,value):(R/name).write_text(json.dumps(value,indent=2)+'\n')
def capture(argv):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=40)
    return dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    gate_path=Path(os.environ.get('FASTRSI_PHASE_REPEAT_PARENT_GATE',str(P/'planning/phase_iql_repeat_v1_20261010/PARENT_GATE.json')))
    assert gate_path.parent.parent == P/'planning'
    gate=json.loads(gate_path.read_text())
    assert gate['pid']==int(os.environ['FASTRSI_PHASE_REPEAT_PARENT_PID']);os.kill(gate['pid'],0)
    assert gate['worker_sha256']==sha(Path(__file__))
    R.mkdir(exist_ok=False);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
    sock=socket.socket();sock.bind(('127.0.0.1',PORT));sock.close()
    (R/'code').mkdir()
    for filename in ('serve_phase_iql_repeat_v1.py','check_phase_iql_repeat_v1.py','run_phase_iql_repeat_worker_v1.py','phase_iql_value_v1.py','phase_iql_feedback_update_v1.py','prefix_feature_capture_v1.py','replay_stack_progress.py','stage_trace.py'):
        shutil.copy2(P/'scripts'/filename,R/'code'/filename)
    shutil.copytree(P/'models/openpi_data',R/'tokenizer_cache')
    for name in ('frozen','updated'):
        shutil.copy2(P/f'runs/phase_iql_feedback_v1_20261010/training/{name}_router.npz',R/f'{name}_router.npz')
        assert gate[name+'_model_sha256']==sha(R/f'{name}_router.npz')
    i=SLOT%4;arm='single' if (SLOT<4)==(i%2==0) else 'frozen'
    cases=[(20+i,arm)]
    save('PROTOCOL.json',dict(slot=SLOT,cases=cases,task_id=64,suite='libero_90',max_steps=400,execution_horizon=10,
        same_first_candidate_noise_stream=True,sampling_noise_bases={i:8101000+i for i,arm in cases},training_steps=0,complete_agent_comparison=False,
        evaluation_result_persistent_updates=False,default_sha256=before,
        code_sha256={f.name:sha(f) for f in (R/'code').iterdir()}))
    policy_name='yuchen-fastrsi-phase-iql-repeat-policy-gpu%d-v1-20261010'%SLOT
    owned=[];results=[];started=time.perf_counter()
    argv=['docker','run','-d','--name',policy_name,'--network','host','--cpus','6','--memory','48g','--pids-limit','1024','--shm-size','2g',
        '--device','/dev/kfd','--device','/dev/mkfd','--device','/dev/dri','--group-add','video','--group-add','render',
        '-e','HIP_VISIBLE_DEVICES='+str(SLOT),'-e','CUDA_VISIBLE_DEVICES='+str(SLOT),'-e','OMP_NUM_THREADS=4','-e','MKL_NUM_THREADS=4',
        '-e','OPENPI_DATA_HOME=/workspace/openpi_data','-e','PYTHONDONTWRITEBYTECODE=1',
        '-e','PYTHONPATH=/check/code:/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/workspace/PhyAgentOS-core',
        '-v',str(P)+':/project:ro','-v',str(R)+':/check',
        '-v',str(A/'dependencies/openpi')+':/workspace/openpi:ro','-v',str(A/'dependencies/PhyAgentOS-core')+':/workspace/PhyAgentOS-core:ro',
        '-v',str(A/'weights/pi05_libero_pytorch_fp32')+':/model:ro','-v',str(R/'tokenizer_cache')+':/workspace/openpi_data',
        '-v','/opt/hyhal:/opt/hyhal:ro','-v','/etc/hfm:/etc/hfm:ro',
        'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81',
        'timeout','1400','bash','-lc','set -e; source /opt/dtk/env.sh; source /opt/hyhal/env.sh; exec python /check/code/serve_phase_iql_repeat_v1.py --checkpoint-dir /model --policy-config pi05_libero --pytorch-device cuda --disable-torch-compile --torch-seed 7 --host 127.0.0.1 --port '+str(PORT)]
    save('POLICY_LAUNCH.json',argv)
    try:
        dispatch=capture(argv);save('POLICY_DISPATCH.json',dispatch);assert dispatch['returncode']==0,dispatch;owned.append(policy_name)
        deadline=time.monotonic()+260
        while True:
            try:
                with urllib.request.urlopen('http://127.0.0.1:'+str(PORT)+'/healthz',timeout=3) as reply:assert reply.status==200
                break
            except Exception:
                info=json.loads(subprocess.check_output(['docker','inspect',policy_name]))[0]
                assert info['State']['Running'],info['State']
                if time.monotonic()>deadline:raise TimeoutError('Candidate policy startup/probe')
                time.sleep(2)
        probe=json.loads((R/'SAMPLER_PROBE.json').read_text());assert probe['status']=='PASSED'
        save('STARTUP.json',dict(seconds=time.perf_counter()-started,includes_offline_probe=True))
        for initial,arm in cases:
            case='init%d_%s'%(initial,arm);name='yuchen-fastrsi-phase-iql-repeat-gpu%d-%s-v1-20261010'%(SLOT,case);owned.append(name)
            cmd=['docker','run','--name',name,'--network','host','--cpus','4','--memory','12g','--pids-limit','1024','--shm-size','2g','--entrypoint','python',
                '-e','PYTHONPATH=/workspace/openpi/third_party/libero:/core:/check/code','-e','PYTHONDONTWRITEBYTECODE=1',
                '-e','LIBERO_CONFIG_PATH=/libero-config','-e','MUJOCO_GL=egl','-e','LIBGL_ALWAYS_SOFTWARE=1',
                '-v',str(P)+':/project:ro','-v',str(R)+':/check','-v',str(A/'dependencies/openpi')+':/workspace/openpi:ro',
                '-v',str(A/'dependencies/PhyAgentOS-core')+':/core:ro','-v',str(P/'configs/libero')+':/libero-config:ro',
                'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95',
                '/check/code/check_phase_iql_repeat_v1.py','--initial',str(initial),'--arm',arm,'--output','/check/'+case,
                '--endpoint','ws://127.0.0.1:'+str(PORT)]
            save(case+'_LAUNCH.json',cmd);tick=time.perf_counter()
            with (R/(case+'.log')).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=300)
            save(case+'_HOST.json',dict(exit_code=r.returncode,seconds=time.perf_counter()-tick));assert r.returncode==0,case
            result=json.loads((R/case/'RESULT.json').read_text());assert result['status']=='COMPLETED'
            results.append(dict(case=case,initial=initial,arm=arm,native_success=result['native_success'],actions=result['executed_actions'],
                control_requests=result['control_requests'],candidate_inferences=result['candidate_inferences'],
                full_seconds=result['seconds_including_close'],initial_input_sha256=result['initial_input_sha256']))
            save('PARTIAL.json',results);print(json.dumps(results[-1]),flush=True)
        save('AGGREGATE.json',dict(status='COMPLETED',cases=results))
    finally:
        cleanup=[]
        for name in owned:
            inspection=capture(['docker','inspect',name])
            if inspection['returncode']!=0:continue
            state=json.loads(inspection['stdout'])[0]['State']
            if state['Running']:capture(['docker','stop','-t','10',name])
            final=json.loads(subprocess.check_output(['docker','inspect',name]))[0]
            save(name+'_CONTAINER.json',final);cleanup.append(dict(name=name,state=final['State']))
            if name==policy_name:
                logs=capture(['docker','logs',name]);(R/'POLICY.log').write_text(logs['stdout']+logs['stderr'])
        after=sha(P/'CURRENT_NATIVE_ENTRY.json')
        save('FINAL_STATE.json',dict(containers=cleanup,default_unchanged=before==after,default_after_sha256=after,
            host_wall_seconds=time.perf_counter()-started))
        assert before==after and all(not c['state']['Running'] and c['state']['Pid']==0 for c in cleanup)
if __name__=='__main__':main()
