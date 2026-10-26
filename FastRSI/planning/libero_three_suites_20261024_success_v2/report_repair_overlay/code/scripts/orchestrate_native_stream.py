"""Bounded native Agent development stream, followed by read-only evaluation.

Uses the existing run_native_baseline.main and unchanged native memory triggers.
No custom Agent, forced reflection, /new, or automatic retry of this campaign.
"""
import argparse,fcntl,hashlib,json,shutil,signal,socket,time,traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env


def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def validate(protocol):
    cases=protocol['cases']
    if len(cases)!=3 or [c['mode'] for c in cases]!=['development','development','evaluation']:
        raise ValueError('Exactly two development cases followed by one fixed evaluation required')
    if not all(c['suite']=='libero_10' and c['max_steps']==520 and c['session_intent'] is True
               and c['native_experience_maintenance'] is True and c['feedback_window_mode']=='fixed10'
               and not c.get('session_flow_artifact') for c in cases):
        raise ValueError('Unbound native stream configuration')
    if len({c['id'] for c in cases})!=3 or cases[0]['conversation_key']!=cases[1]['conversation_key']:
        raise ValueError('Development must share one explicit conversation identity')
    if cases[2]['conversation_key']==cases[0]['conversation_key']:
        raise ValueError('Evaluation must use independent task-local conversation')
    if protocol['max_campaign_seconds']!=1800 or protocol['max_attempts_per_case']!=3:
        raise ValueError('Changed declared execution bound')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--name',default='native_stream_v1_20261003');args=parser.parse_args()
    name=args.name
    if not name.startswith('native_stream_') or not all(c.isalnum() or c=='_' for c in name):raise ValueError('Invalid run name')
    p=life.PROJECT;protocol=read(p/'planning'/name/'PROTOCOL.json');validate(protocol)
    lock=open(p/'runs/full_os_v1.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=p/'runs'/name;run.mkdir(exist_ok=False);stream=run/'stream';stream.mkdir()
    life.CAMPAIGN=run;life.BASE='/project/runs/'+name
    started=time.time();life.deadline=started+protocol['max_campaign_seconds']
    state={'status':'PREFLIGHT','started_at':started,'deadline':life.deadline}
    def interrupt(n,f):raise KeyboardInterrupt('Interrupted native stream '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    life.save(run/'STATUS.json',state)
    try:
        for rel in ('scripts','src/fastrsi','tests','configs'):
            shutil.copytree(p/rel,run/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
        life.save(run/'CODE_MANIFEST.json',{str(f.relative_to(run/'code')):sha(f) for f in (run/'code').rglob('*') if f.is_file()})
        source=p/'runs/shared_flow_net_parallel_v2_20261002'
        for filename in ('MODEL_CONFIG.json','MODEL_BINDING.json'):shutil.copyfile(source/filename,run/filename)
        cfg=read(run/'MODEL_CONFIG.json')
        core_names=('agent/loop.py','agent/context.py','agent/memory.py','agent/skills.py','session/manager.py',
                    'agent/session_verifier.py','runtime/watchdog/result_writer.py','skills/benchmarking/SKILL.md')
        life.save(run/'CORE_SOURCE_BINDING.json',{rel:sha(life.CORE/'PhyAgentOS'/rel) for rel in core_names})
        life.save(run/'PROTOCOL.json',protocol);life.save(run/'ROSTER.json',protocol['cases'])
        life.gate(stream)
        tests=run/'cpu';tests.mkdir()
        modules=('test_native_queue_schema test_native_maintenance test_native_stream_history test_native_workspace test_terminal_results '
                 'test_native_isolation test_experience_admission test_native_memory_budget test_terminal_evidence_view '
                 'test_feedback_window test_native_window_attempt')
        command='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest '+modules+' -v'
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-cpu',tests)+['-e','FASTRSI_NATIVE_RUN=/tmp/native-stream-cpu',
            life.IMAGE,'timeout','150','bash','-lc',command]
        cid=life.launch('cpu',argv,tests);life.wait_exit(cid,160)
        life.save(tests/'STOP_RECEIPT.json',life.stop_all());state['cpu']='PASSED';life.save(run/'STATUS.json',state)
        shutil.copytree(p/'models/openpi_data',stream/'tokenizer_cache')
        containers={}
        roles=[('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]
        if protocol.get('planning_repair_probe'):roles=[roles[1],roles[0],roles[2]]
        for role,receipt in roles:
            argv=read(life.OLD/receipt)['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+name+'-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',life.BASE+'/stream') for x in argv]
            if role=='pi05':argv=[x.replace(str(p/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv]
            if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            if role=='qwen':argv=configure_qwen(argv,cfg,life)
            containers[role]=life.launch(role,argv,stream)
            if role=='qwen' and protocol.get('planning_repair_probe'):
                life.health(49510)
                probe=stream/'planning_repair_probe';probe.mkdir()
                argv=life.cpu_args('yuchen-fastrsi-'+name+'-repair-probe',probe)+model_env(cfg)+[
                    '-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream/planning_repair_probe',life.IMAGE,'timeout','250','bash','-lc',
                    'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/probe_native_planning_repair.py']
                cid=life.launch('repair_probe',argv,probe);life.wait_exit(cid,255)
                assert read(probe/'RESULT.json')['status']=='PASSED'
                state['planning_repair_probe']='PASSED';life.save(run/'STATUS.json',state)
        life.health(49500);life.health(49510)
        until=min(time.time()+120,life.deadline)
        while True:
            try:
                with socket.create_connection(('127.0.0.1',49502),timeout=1):break
            except OSError:
                if time.time()>until:raise
                time.sleep(1)
        task_ids=sorted({c['task_id'] for c in protocol['cases']})
        code="import json; from libero.libero import benchmark; s=benchmark.get_benchmark_dict()['libero_10'](); print('TASKS='+json.dumps({i:s.get_task(i).language for i in "+repr(task_ids)+"}))"
        receipt=life.command(['docker','exec',containers['target'],'python','-c',code],timeout=45)
        life.save(run/'LIVE_TASK_BINDING.json',receipt);assert receipt['returncode']==0
        languages=json.loads(next(line[6:] for line in receipt['stdout'].splitlines() if line.startswith('TASKS=')))
        assert all(c['task_description']==languages[str(c['task_id'])] for c in protocol['cases'])
        remaining=int(life.deadline-time.time()-30)
        if remaining<180:raise TimeoutError('Insufficient stream budget after startup')
        state.update(status='RUNNING',startup_seconds=time.time()-started);life.save(run/'STATUS.json',state)
        argv=life.cpu_args('yuchen-fastrsi-'+name+'-runner',stream)+model_env(cfg)+[
            '-e','FASTRSI_NATIVE_RUN='+life.BASE+'/stream','-e','FASTRSI_CONSUME_TERMINAL=1',
            '-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1',life.IMAGE,'timeout',str(remaining),'bash','-lc',
            'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/run_native_baseline.py']
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,remaining+5)
        report=read(stream/'RESULT.json');assert report['status']=='COMPLETED' and len(report['cases'])==3
        assert all(c['chain_status']=='PASSED' for c in report['cases'])
        state.update(status='COMPLETED',controller_seconds=report['seconds'])
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc())
    finally:
        life.save(run/'STOP_RECEIPT.json',life.stop_all());state.update(finished_at=time.time(),full_wall_seconds=time.time()-started)
        life.save(run/'STATUS.json',state);print(json.dumps(state),flush=True)
    if state['status']!='COMPLETED':raise SystemExit(1)

if __name__=='__main__':main()
