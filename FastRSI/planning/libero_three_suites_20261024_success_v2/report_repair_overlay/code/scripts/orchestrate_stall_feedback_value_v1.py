"""Three fixed full-Agent development policies on known init31.

N fixed10; F fixed5; R fixed10 then short feedback at an unchanged stall trigger.
This is recovery operation-value evidence, not learning or unseen evaluation.
"""
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

NAME='stall_feedback_value_v1_20261006'
SOURCE='runs/native_stage_continuation_probe_v1_20261006'
GOAL='put the white mug on the left plate and put the yellow and white mug on the right plate'
ARMS=('N','F','R')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p):return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
def freeze(a,b):shutil.copytree(a,b,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
def replace(text,old,new):
    if text.count(old)!=1:raise ValueError('Worker binding changed: '+old[:90])
    return text.replace(old,new)


def main():
    project=life.PROJECT;run=project/'runs'/NAME
    lock=(project/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run.mkdir(exist_ok=False);start=time.time();life.deadline=start+1200
    workers=[];state=dict(status='PREPARING',started_at=start)
    def stop(sig,frame):raise RuntimeError('Owned stall feedback development interrupted '+str(sig))
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    try:
        registration=project/'planning/stall_feedback_value_v1_20261006/REGISTRATION.json'
        declared=json.loads(registration.read_text())
        assert declared['initial_state']==31 and declared['test_data_used'] is False
        for rel,digest in declared['source_hashes'].items():assert sha(project/rel)==digest
        shutil.copyfile(registration,run/'REGISTRATION.json')
        life.gate(run)
        for offset in (0,100,200):
            for port in (49500,49502,49510,49520):
                with socket.socket() as sock:
                    assert sock.connect_ex(('127.0.0.1',port+offset))!=0,'Port occupied'
        source=project/SOURCE
        for rel in ('MODEL_CONFIG.json','FLOW.json'):shutil.copyfile(source/rel,run/rel)
        freeze(source/'N/stream/net_t4_i26/admission/accepted_experience',run/'initial_experience')
        life.save(run/'EXPERIENCE_BINDING.json',tree(run/'initial_experience'))
        case=dict(suite='libero_10',task_id=4,init_state_id=31,mode='development',
            max_steps=520,native_experience_maintenance=True,task_description=GOAL)
        life.save(run/'PLAN.json',dict(cases=[case]))
        life.save(run/'PROTOCOL.json',dict(slot=0,absolute_deadline=life.deadline,
            diagnostic_trace=True,control_kind='feedback_window',
            arm_modes=dict(N='fixed10',F='fixed5',R='fixed10'),terminal_evidence_view=True,
            planning_entries={a:'public_session_flow' for a in ARMS},
            stall_window_probe=dict(N='observe_only',R='switch_once'),
            original_goal=True,original_memory_triggers=True,
            scope='Known development init31 only. Existing unchanged stall trigger, native short feedback after trigger; no replacement actions. Three fixed policies, no learning claim.',
            resources='Two cards per arm, six independent cards; same weights, seed, data permissions and physical budget',
            stopping='One three-arm development comparison. No trigger tuning or amplitude search. Defaults remain unchanged. Actual recovery must precede applicability learning.'))
        for rel in ('scripts','src/fastrsi'):freeze(project/rel,run/'code'/rel)
        worker=run/'code/scripts/run_placement_worker.py';text=worker.read_text()
        text=replace(text,"        case['terminal_prefetch']=False", "        case['terminal_prefetch']=False\n        if arm in protocol['stall_window_probe']:\n            case['stall_window_probe']=protocol['stall_window_probe'][arm]")
        text=replace(text,"runner_script='run_native_baseline.py'","runner_script='run_stall_window_value_v1.py'")
        worker.write_text(text);life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
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
            if time.time()>life.deadline:raise TimeoutError('Registered three-arm time bound')
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


if __name__=='__main__':main()
