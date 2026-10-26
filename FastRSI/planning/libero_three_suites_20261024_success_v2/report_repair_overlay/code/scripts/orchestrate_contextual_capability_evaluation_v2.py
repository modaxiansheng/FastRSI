"""Dependency-only retry using the exact completed init8 runtime snapshot.

N native; F same-data fixed nearest; R frozen seed; S feedback-updated ridge.
Tests read accepted experience but never persist new learning, including late callbacks.
"""
import argparse
import os
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
import traceback
import orchestrate_full_os as life

NAME=None
INIT=None
HOST=None
RUNTIME='planning/context_conditioned_capability_v1_20261006/evaluation_runtime_i8_v1'
REGISTRATION='planning/context_conditioned_capability_v1_20261006/NEXT_EVALUATION_REGISTRATION_v1.json'
SOURCE='runs/native_stage_continuation_probe_v1_20261006'
FIT='planning/context_conditioned_capability_v1_20261006/fit_v1'
GOAL='put the white mug on the left plate and put the yellow and white mug on the right plate'
ARMS=('N','F','R','S')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
def freeze(a,b):shutil.copytree(a,b,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def replace(text,old,new):
    if text.count(old)!=1:raise ValueError('Worker binding changed: '+old[:90])
    return text.replace(old,new)


def main():
    project=life.PROJECT;run=project/'runs'/NAME
    lock=(project/('runs/contextual_evaluation_'+HOST+'.lock')).open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);start=time.time();life.deadline=start+1200
    registered=json.loads((project/REGISTRATION).read_text())
    assert INIT in registered['initial_states'] and HOST in ('187','188')
    assert GOAL==registered['goal']
    shutil.copyfile(project/REGISTRATION,run/'EVALUATION_REGISTRATION.json')
    workers=[];state=dict(status='PREPARING',started_at=start)
    def stop(sig,frame):raise RuntimeError('Owned contextual development interrupted '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        cpu=json.loads((project/'planning/context_conditioned_capability_v1_20261006/evaluation_cpu_v1/RESULT.json').read_text())
        assert cpu['status']=='PASSED' and cpu['source_unchanged']
        scope_name='REMOTE_RESERVED_SCOPE_188_v1.json' if HOST=='188' else 'REMOTE_RESERVED_SCOPE_v1.json'
        scope=json.loads((project/'planning/context_conditioned_capability_v1_20261006'/scope_name).read_text())
        assert scope['complete'] and not scope['matches'][str(INIT)]
        life.save(run/'PRECONDITIONS.json',dict(cpu=cpu,scope=scope))
        check=json.loads((project/'runs/contextual_policy_real_check_v1_20261006/job/RESULT.json').read_text())
        assert check['status']=='COMPLETED' and check['unmarked_output_rng_equal']
        assert len(check['rows'])==6 and all(r['rng_equal'] and r['output_equal_to_expected'] for r in check['rows'])
        life.gate(run)
        for offset in (0,100,200,300):
            for port in (49500,49502,49510,49520):
                with socket.socket() as sock:
                    assert sock.connect_ex(('127.0.0.1',port+offset))!=0,'Port occupied'
        source=project/SOURCE
        for rel in ('MODEL_CONFIG.json','FLOW.json'):shutil.copyfile(source/rel,run/rel)
        freeze(source/'N/stream/net_t4_i26/admission/accepted_experience',run/'initial_experience')
        life.save(run/'EXPERIENCE_BINDING.json',tree(run/'initial_experience'))
        versions=json.loads((project/FIT/'VERSIONS.json').read_text())['versions']
        (run/'components').mkdir();components={}
        for arm,kind in [('F','fixed_nearest'),('R','seed_frozen'),('S','feedback_updated')]:
            version=versions[kind];p=project/FIT/version['path'];assert sha(p)==version['sha256']
            shutil.copyfile(p,run/'components'/p.name)
            components[arm]=dict(authority='CONTEXTUAL_CAPABILITY_DEVELOPMENT_VALUE',kind=kind,
                artifact='/project/runs/'+NAME+'/components/'+p.name,sha256=version['sha256'],
                evaluation_registration='/project/runs/'+NAME+'/EVALUATION_REGISTRATION.json',
                evaluation_registration_sha256=sha(run/'EVALUATION_REGISTRATION.json'))
            assert version['sha256']==registered['versions'][kind]['sha256']
        # Every arm's read-only project exposes all three components and identical
        # source feedback. Only the selected component is executed/updated offline.
        life.save(run/'COMMON_EXPERIENCE_ACCESS.json',dict(
            native_text=tree(run/'initial_experience'),versions=versions,
            feedback_registration=json.loads((project/'runs/contextual_capability_features_v1_20261006/inputs/REGISTRATION.json').read_text()),
            all_arms_same_read_access=True,task_updates_to_component=False))
        case=dict(suite='libero_10',task_id=4,init_state_id=INIT,mode='evaluation',persistent_updates=False,
            max_steps=520,native_experience_maintenance=False,task_description=GOAL)
        life.save(run/'PLAN.json',dict(cases=[case]))
        life.save(run/'PROTOCOL.json',dict(slot=0,absolute_deadline=life.deadline,
            diagnostic_trace=True,control_kind='feedback_window',
            arm_modes={a:'fixed10' for a in ARMS},terminal_evidence_view=True,
            planning_entries={a:'public_session_flow' for a in ARMS},
            contextual_components=components,operation_cap=2,physical_feedback_window=10,
            original_goal=True,original_memory_triggers=True,evaluation_persistent_updates=False,
            trigger='First fully consumed closed chunk after mixed closing, plus at most the next boundary; reject stops attempts',
            scope='Pre-registered related context, fixed component and readable accepted experience; no persistent evaluation feedback. New current RGB/proprioception each attempt; no hidden state input.',
            resources='Two cards per arm, eight cards independent parallel; same Qwen/pi05 weights, seed and runtime caps',
            accounting='Separate RPC, prefix encoder, full teacher decoder, local predictor, physical actions, retries, Qwen and full worker; charge rejected double encoding',
            stopping='One registered four-arm fixed evaluation per context, no refit on these results. Preserve all outcomes and immutable source components; no automatic adoption.'))
        runtime=project/RUNTIME
        manifest=json.loads((runtime/'MANIFEST.json').read_text())
        assert tree(runtime/'code')==manifest['files'], 'Bound init8 runtime changed'
        freeze(runtime/'code',run/'code')
        worker=run/'code/scripts/run_placement_worker.py'
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        life.save(run/'RUNTIME_SOURCE.json',manifest)
        for index,arm in enumerate(ARMS):
            group=run/arm;freeze(run/'code',group/'code')
            if index:
                for rel in ('scripts/run_native_baseline.py','scripts/qwen_agent_server.py'):
                    p=group/'code'/rel;t=p.read_text()
                    for port in (49500,49502,49510,49520):t=t.replace(str(port),str(port+index*100))
                    p.write_text(t)
            life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
        state.update(status='REGISTERED',registration='PROTOCOL.json');life.save(run/'STATUS.json',state)
        for arm in ARMS:
            with (run/arm/'controller.log').open('x') as log:
                proc=subprocess.Popen(['python3',str(worker),NAME,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((arm,proc));life.save(run/arm/'DISPATCH.json',dict(pid=proc.pid,started_at=time.time()))
        state['status']='RUNNING';life.save(run/'STATUS.json',state)
        while any(p.poll() is None for _,p in workers):
            if time.time()>life.deadline:raise TimeoutError('Registered four-arm time bound')
            if any(p.poll() not in (None,0) for _,p in workers):raise RuntimeError('Worker failure; preserve all arms')
            time.sleep(2)
        results={a:json.loads((run/a/'STATUS.json').read_text()) for a,_ in workers}
        assert all(p.returncode==0 for _,p in workers) and all(v['status']=='COMPLETED' for v in results.values())
        state.update(status='COMPLETED',arms=results)
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        for _,p in workers:
            if p.poll() is None:p.send_signal(signal.SIGTERM)
        for _,p in workers:
            try:p.wait(timeout=45)
            except subprocess.TimeoutExpired:state['cleanup_attention']='Check exact owned worker PIDs'
        state.update(finished_at=time.time(),seconds=time.time()-start);life.save(run/'STATUS.json',state)
        print(json.dumps(state),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--init',type=int,choices=(8,9),required=True)
    parser.add_argument('--host',choices=('187','188'),required=True)
    args=parser.parse_args();INIT=args.init;HOST=args.host
    NAME='contextual_capability_evaluation_i'+str(INIT)+'_'+HOST+'_v2_20261006'
    assert INIT==9, 'This retry is only the unexecuted registered init9'
    main()
