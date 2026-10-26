"""Paid source, full candidate, and reused-prefix candidate on one resident pool."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time,traceback
import orchestrate_full_os as life
import orchestrate_structural_window as pools
from orchestrate_resident_services import configure_context,check_ports
from orchestrate_resident_full_os import BINDINGS,read,tree,freeze
NAME='trial_fork_v7_20261004'

def main():
    p=life.PROJECT;run=p/'runs'/NAME;lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read(p/'runs/trial_prefix_cpu_v4_20261004/STATUS.json')['status']=='COMPLETED'
    assert not (run/'STATUS.json').exists()
    for rel,h in read(run/'MANIFEST.json').items():assert hashlib.sha256((run/rel).read_bytes()).hexdigest()==h
    started=time.time();deadline=started+2300;configure_context(run,deadline)
    state=dict(status='RUNNING',started_at=started,groups=[]);life.save(run/'STATUS.json',state)
    worker=None
    def interrupted(*args):raise KeyboardInterrupt('Trial interrupted')
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    try:
        life.gate(run);check_ports([22100,22102,22110,22120])
        probe=run/'target_import';probe.mkdir()
        life.save(run/'IMPORT_TRIAL.json',dict(mode='source'))
        old=read(p/'runs/trial_fork_v4_20261004/source/N/stream/LAUNCH_target.json')['argv']
        argv=[x.replace('trial_fork_v4_20261004/source/N/code','trial_fork_v7_20261004/code') for x in old]
        argv[argv.index('--name')+1]='yuchen-fastrsi-'+NAME+'-target-import'
        argv=[('FASTRSI_TRIAL_SPEC='+life.BASE+'/IMPORT_TRIAL.json') if x.startswith('FASTRSI_TRIAL_SPEC=') else x for x in argv]
        cid=life.launch('target_import',argv+['--help'],probe);life.wait_exit(cid,30);life.save(probe/'STOP_RECEIPT.json',life.stop_all())
        source=p/'runs/shared_flow_net_parallel_v2_20261002'
        for rel in BINDINGS:shutil.copyfile(source/rel,run/rel)
        shutil.copytree(source/'initial_experience',run/'initial_experience')
        original_launch=life.launch
        def launch(role,argv,folder):
            if role=='pi05':
                argv=[x.replace('/resident_openpi_server.py','/trial_fork_server.py') for x in argv]
                pos=argv.index(life.IMAGE);argv[pos:pos]=['-e','FASTRSI_TRIAL_SOURCE_SESSION='+NAME+'-source','-e','FASTRSI_TRIAL_PREFIX_INFERENCES=18']
            return original_launch(role,argv,folder)
        pools.NAME=NAME;life.launch=launch
        try:services,state['pool_startup_seconds']=pools.pool(run,run/'resident','N',deadline)
        finally:life.launch=original_launch
        configure_context(run,deadline)
        case=dict(mode='development',suite='libero_10',task_id=0,init_state_id=6,max_steps=520,
                  task_description='put both the alphabet soup and the tomato sauce in the basket',native_experience_maintenance=True)
        for mode in ('source','full','reuse'):
            pair=run/mode;pair.mkdir();group=pair/'N'
            for rel in BINDINGS:shutil.copyfile(run/rel,pair/rel)
            shutil.copytree(run/'initial_experience',pair/'initial_experience');assert tree(pair/'initial_experience')==read(run/'EXPERIENCE_BINDING.json')
            trial=dict(mode=mode,boundary=180,suffix_commit_steps=5,keep_prefix_rendering=True,cache_loop=True,source_session=NAME+'-source',
                scope='Same-instance development candidate validation; not cross-instance learning efficacy')
            if mode=='reuse':
                stream=run/'source/N/stream';trace=stream/'stage_trace/net_t0_i6/TRACE.jsonl'
                trial.update(source_trace=str(trace).replace(str(p),'/project'),source_trace_sha256=hashlib.sha256(trace.read_bytes()).hexdigest())
                policy_path=stream/'TRIAL_POLICY.jsonl'
                trial.update(source_policy=str(policy_path).replace(str(p),'/project'),source_policy_sha256=hashlib.sha256(policy_path.read_bytes()).hexdigest())
                rows=[json.loads(line) for line in policy_path.read_text().splitlines()]
                trial['policy_prefix_sha256']=rows[17]['policy_meta']['trial_checkpoint']['prefix_sha256']
            life.save(pair/'TRIAL.json',trial);life.save(pair/'PLAN.json',{'cases':[case]})
            protocol=dict(slot=0,absolute_deadline=deadline,diagnostic_trace=True,control_kind='feedback_window',arm_modes={'N':'fixed10'},
                port_offset=-27400,gpu_bases={'N':0},terminal_evidence_view=True,
                trial_prefix_probe={'sha256':hashlib.sha256((pair/'TRIAL.json').read_bytes()).hexdigest()},
                resident_services=dict(policy_port=22100,qwen_port=22110,policy_container=services['pi05'],qwen_container=services['qwen'],
                                       session_id=NAME+'-'+mode,max_inferences=156))
            life.save(pair/'PROTOCOL.json',protocol);freeze(run/'code',group/'code')
            for rel in ['scripts/run_native_baseline.py','scripts/qwen_agent_server.py']:
                path=group/'code'/rel;s=path.read_text()
                for port in (49500,49502,49510,49520):s=s.replace(str(port),str(port-27400))
                path.write_text(s)
            life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
            with (group/'controller.log').open('x') as log:
                worker=subprocess.Popen(['python3',str(run/'code/scripts/run_placement_worker.py'),str(pair.relative_to(p/'runs')),'N'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            life.save(group/'DISPATCH.json',dict(pid=worker.pid,started_at=time.time()))
            while worker.poll() is None:
                if time.time()>deadline-30:raise TimeoutError('Trial deadline')
                time.sleep(2)
            result=read(group/'STATUS.json');state['groups'].append(dict(mode=mode,**result));life.save(run/'STATUS.json',state)
            assert worker.returncode==0 and result['status']=='COMPLETED',result
        def rows(mode,name):return [json.loads(x) for x in (run/mode/'N/stream'/name).read_text().splitlines()]
        src,full,reuse=[rows(m,'TRIAL_POLICY.jsonl') for m in ('source','full','reuse')]
        keys=('step','prompt','inputs','actions')
        assert [{k:r[k] for k in keys} for r in src if r['step']<180]==[{k:r[k] for k in keys} for r in full if r['step']<180]
        assert [{k:r[k] for k in keys} for r in full if r['step']>=180]==[{k:r[k] for k in keys} for r in reuse if r['step']>=180]
        assert [r for r in rows('full','TRIAL_PHYSICS.jsonl') if r['step']>=180]==[r for r in rows('reuse','TRIAL_PHYSICS.jsonl') if r['step']>=180]
        state.update(status='COMPLETED',exact_suffix_equivalence=True,
                     actual_policy_calls={m:sum(r.get('origin','policy')=='policy' for r in rows(m,'TRIAL_POLICY.jsonl')) for m in ('source','full','reuse')})
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        if worker is not None and worker.poll() is None:
            worker.send_signal(signal.SIGTERM)
            try:worker.wait(timeout=90)
            except subprocess.TimeoutExpired:state['cleanup_attention']='Worker still alive; inspect exact ID'
        life.save(run/'POOL_STOP_RECEIPT.json',life.stop_all())
        state.update(finished_at=time.time(),full_wall_seconds=time.time()-started);life.save(run/'STATUS.json',state)
        print(json.dumps(state),flush=True)
    if state['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
