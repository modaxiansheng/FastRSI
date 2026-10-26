"""Two different native tasks for capability diagnosis, not an A/B comparison."""
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

NAME='cabinet_composition_baseline_v1_20261006'
SOURCE='runs/native_stage_continuation_probe_v1_20261006'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p): return {str(f.relative_to(p)):sha(f) for f in p.rglob('*') if f.is_file()}
def freeze(a,b): shutil.copytree(a,b,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))


def main():
    project=life.PROJECT;run=project/'runs'/NAME
    lock=(project/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (run/'STATUS.json').exists()
    start=time.time();life.deadline=start+1200;workers=[];state=dict(status='PREPARING',started_at=start)
    life.CAMPAIGN=run;life.BASE='/project/runs/'+NAME
    def interrupt(sig,frame): raise RuntimeError('Owned cabinet diagnosis interrupted '+str(sig))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    try:
        reg=json.loads((project/'planning/cabinet_composition_v1_20261006/REGISTRATION.json').read_text())
        for rel,digest in reg['code_bindings'].items(): assert sha(project/rel)==digest,rel
        life.gate(run)
        for shift in (0,100):
            for port in (49500,49502,49510,49520):
                with socket.socket() as s: assert s.connect_ex(('127.0.0.1',port+shift))!=0
        for rel in ('scripts','src/fastrsi','tests'): freeze(project/rel,run/'code'/rel)
        source=project/SOURCE
        for rel in ('MODEL_CONFIG.json','FLOW.json','EXPERIENCE_BINDING.json'):shutil.copyfile(source/rel,run/rel)
        freeze(source/'initial_experience',run/'initial_experience')
        assert tree(run/'initial_experience')==json.loads((run/'EXPERIENCE_BINDING.json').read_text())
        cpu=run/'cpu';cpu.mkdir()
        cmd=('cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest '
             'test_development_suite test_session_flow test_native_isolation -v')
        cid=life.launch('cpu',life.cpu_args('yuchen-fastrsi-cabinet-cpu-v1',cpu)+[
            '-e','FASTRSI_NATIVE_RUN=/tmp/cabinet-composition-cpu',life.IMAGE,
            'timeout','150','bash','-lc',cmd],cpu)
        life.wait_exit(cid,160);life.save(cpu/'STOP_RECEIPT.json',life.stop_all())
        state['cpu']='PASSED';life.save(run/'STATUS.json',state)
        old=json.loads((project/'runs/short_exit_v1_20261003/pair1/R/stream/LAUNCH_target.json').read_text())['argv']
        ix=next(i for i,v in enumerate(old) if v.startswith('sha256:'))
        argv=old[:ix];argv[argv.index('--name')+1]='yuchen-fastrsi-cabinet-asset-v1'
        clean=[];i=0
        while i<len(argv):
            if argv[i]=='-e' and argv[i+1].startswith('FASTRSI_'):i+=2;continue
            if argv[i].startswith('PYTHONPATH='):argv[i]+=':/project/runs/'+NAME+'/code/src'
            clean.append(argv[i]);i+=1
        base='/project/runs/'+NAME
        cid=life.launch('asset',clean+[old[ix],'timeout','150','python',base+'/code/scripts/probe_cabinet_composition_assets_v1.py'],run)
        life.wait_exit(cid,160);life.save(run/'ASSET_STOP_RECEIPT.json',life.stop_all())
        asset=json.loads((run/'ASSET.json').read_text());assert asset['status']=='PASSED'
        cases=[r['case'] for r in asset['cases']]
        assert cases==reg['cases'] and len(cases)==2
        # Initial success is reported, not replaced with a harder initial state.
        life.save(run/'PLAN.json',dict(cases=cases))
        life.save(run/'PROTOCOL.json',dict(slot=0,absolute_deadline=life.deadline,
            diagnostic_trace=True,control_kind='feedback_window',
            arm_modes=dict(N='fixed10',F='fixed10'),terminal_evidence_view=True,
            planning_entries=dict(N='public_session_flow',F='public_session_flow'),
            cases_by_arm=dict(N=[cases[0]],F=[cases[1]]),
            scope='Two different original tasks, primitive and composition; not causal A/B or learning evidence.',
            resources='two cards per original task; four independent cards',
            maximum_native_attempts_per_task=3,maximum_steps_per_attempt=400,
            retries='Original native verifier max_replans=2; preserve all attempts and full costs',
            stopping='One registered development initial state per task; no dynamic task replacement or sweep'))
        worker=run/'code/scripts/run_placement_worker.py';text=worker.read_text()
        old="    for item in plan['cases']:"
        assert text.count(old)==1
        worker.write_text(text.replace(old,"    for item in protocol['cases_by_arm'][arm]:"))
        life.save(run/'CODE_MANIFEST.json',tree(run/'code'))
        for index,arm in enumerate(('N','F')):
            group=run/arm;freeze(run/'code',group/'code')
            if index:
                for rel in ('scripts/run_native_baseline.py','scripts/qwen_agent_server.py'):
                    p=group/'code'/rel;t=p.read_text()
                    for port in (49500,49502,49510,49520):t=t.replace(str(port),str(port+100))
                    p.write_text(t)
            life.save(group/'CODE_MANIFEST.json',tree(group/'code'))
        life.gate(run)
        for arm in ('N','F'):
            with (run/arm/'controller.log').open('x') as log:
                proc=subprocess.Popen(['python3',str(worker),NAME,arm],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            workers.append((arm,proc));life.save(run/arm/'DISPATCH.json',dict(pid=proc.pid,started_at=time.time()))
        state['status']='RUNNING';life.save(run/'STATUS.json',state)
        while any(p.poll() is None for _,p in workers):
            if time.time()>life.deadline:raise TimeoutError('Registered cabinet diagnosis deadline')
            if any(p.poll() not in (None,0) for _,p in workers):raise RuntimeError('Worker failure')
            time.sleep(2)
        state.update(status='COMPLETED',arms={a:json.loads((run/a/'STATUS.json').read_text()) for a,_ in workers})
        assert all(s['status']=='COMPLETED' for s in state['arms'].values())
    except BaseException:
        state.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        for _,p in workers:
            if p.poll() is None:p.send_signal(signal.SIGTERM)
        for _,p in workers:
            try:p.wait(timeout=45)
            except subprocess.TimeoutExpired:state['cleanup_attention']='Inspect exact owned worker PID'
        life.save(run/'STOP_RECEIPT.json',life.stop_all())
        state.update(seconds=time.time()-start,finished_at=time.time());life.save(run/'STATUS.json',state)
        print(json.dumps(state),flush=True)


if __name__=='__main__':main()
