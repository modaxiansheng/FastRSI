"""Four preregistered native-Agent pairs, two pairs per eight-HCU wave."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
from pathlib import Path
import re
import shutil
import signal
import socket
import subprocess
import time
from run_native_value_phase_v1 import P,S,IMAGE,sha,save,capture

B=P/'planning/native_value_batch_v2_20261010'
MODEL=P/'planning/phase_iql_feedback_v1_20261010/training/frozen_router.npz'
GOAL='stack the right bowl on the left bowl and place them in the tray'
ROSTER=[(16,9101016),(17,9101017),(18,9101018),(19,9101019)]

def gate(folder):
    hcu=capture(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse'])
    containers=capture(['docker','ps','--format','{{.ID}} {{.Names}}'])
    holders=capture(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*'])
    save(folder/'PREFLIGHT.json',dict(hcu=hcu,containers=containers,holders=holders))
    mem=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',hcu['stdout'])
    assert hcu['returncode']==0 and len(mem)==8 and all(float(v)==0 for _,v in mem)
    assert containers['returncode']==0 and not containers['stdout'].strip()
    for pid in map(int,holders['stdout'].split()):
        proc=Path('/proc')/str(pid)
        argv=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        exe=str((proc/'exe').resolve())
        manager=exe=='/usr/local/hyhal/bin/hymgr' and argv==['/usr/local/hyhal/bin/hymgr']
        collector=exe=='/opt/gridview/agent/python3.12.12/bin/python3.12' and argv==[
            '/opt/gridview/agent/python3.12.12/bin/python3','/opt/gridview/agent/collector-agent/collector_agent.py']
        assert manager or collector, (pid,exe,argv)
    for shift in (0,1000):
        for port in [57800,57802,57810,57820,57900,57902,57910,57920]:
            with socket.socket() as sock:
                # Match service reuse semantics: closed TCP connections in
                # TIME_WAIT are not live port holders. LISTEN still fails.
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
                sock.bind(('127.0.0.1',port+shift))

def prepare(index,initial,noise,before,expected,deadline):
    name=f'native_value_phase_v{8+index}_20261010';c=P/'runs'/name;c.mkdir(exist_ok=False)
    for arm in ('N','F'):shutil.copytree(B/'code',c/arm/'code')
    shutil.copytree(B/'code',c/'code')
    for n in ('MODEL_CONFIG.json','FLOW.json'):shutil.copy2(S/n,c/n)
    shutil.copytree(S/'initial_experience',c/'initial_experience');(c/'templates').mkdir()
    for f in (S/'templates').glob('*.json'):
        value=json.loads(f.read_text());value['argv']=[a.replace(str(P/'runs/pilot_20261001'),str(c/'templates')) for a in value['argv']]
        save(c/'templates'/f.name,value)
    shutil.copy2(MODEL,c/'VALUE_MODEL.npz')
    case=dict(suite='libero_90',task_id=64,init_state_id=initial,seed=7,mode='evaluation',max_steps=400,
        development_suite_extension='libero90_stack_flow_v1',native_experience_maintenance=False,task_description=GOAL)
    save(c/'PLAN.json',dict(cases=[case]))
    slot=index%2
    save(c/'PROTOCOL.json',dict(slot=slot,absolute_deadline=deadline,diagnostic_trace=True,
        control_kind='feedback_window',arm_modes={'N':'fixed10','F':'fixed10'},port_offset=8300,
        gpu_bases={'N':slot*4,'F':slot*4+2},terminal_evidence_view=True))
    for arm in ('N','F'):
        save(c/('VALUE_CONFIG_'+arm+'.json'),dict(version='native_value_phase_binding_v1',enabled=arm=='F',
            model_path='/project/runs/'+name+'/VALUE_MODEL.npz',model_sha256=expected,noise_bases={str(initial):noise},
            source_feedback=dict(initial_training='planning/phase_iql_training_v1_20261010/training/RESULT.json',
                original_transitions_sha256='8731fb0a5bebd2c28e8bcf8b76da40ebaa2380664aca902a0fa725f602e130e7',
                qualification='planning/phase_goal_v1_20261010/CANDIDATE_ADMISSION.json',
                not_new_unseen_evidence=False,unseen_combination_not_unseen_initial_claim=True)))
    save(c/'REGISTRATION.json',dict(status='REGISTERED_NEW_COMBINATION_CONFIRMATION',batch=str(B.relative_to(P)),
        arms=['N','F'],initial=initial,noise_base=noise,original_agent_loop=True,no_eval_updates=True,pi_parameter_updates=0,
        maximum_sessions_per_arm=3,maximum_task_actions=2400,selected_model_sha256=expected,default_sha256=before,
        pair_resource=dict(policy_hcu=1,qwen_hcu=1,target_cpu_limit=8,runner_cpu_limit=8),
        source_sha256=json.loads((B/'REGISTRATION.json').read_text())['source_sha256']))
    return c

def run(c,arm):
    tick=time.perf_counter()
    with (c/(arm+'_controller.log')).open('x') as log:
        child=subprocess.Popen(['python3',str(c/'code/scripts/run_native_value_worker_v1.py'),c.name,arm],stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=1800)
        except BaseException:
            child.send_signal(signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    return dict(campaign=c.name,arm=arm,exit_code=code,host_seconds=time.perf_counter()-tick)

def main():
    lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    B.mkdir(exist_ok=False);before=sha(P/'CURRENT_NATIVE_ENTRY.json');deadline=time.time()+3800
    prior=json.loads((P/'planning/native_value_binding_audit_v2_20261010/report/AUDIT.json').read_text())
    assert prior['status']=='PASSED_NATIVE_BINDING_AUDIT'
    expected=prior['model_sha256'];assert sha(MODEL)==expected
    for rel,digest in json.loads((S/'SOURCE_MANIFEST.json').read_text()).items():assert sha(S/rel)==digest
    for rel in ('scripts','src/fastrsi','tests','configs'):
        shutil.copytree(P/rel,B/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    save(B/'REGISTRATION.json',dict(status='REGISTERED_BEFORE_NEW_OUTCOMES',roster=ROSTER,
        fixed_candidate_sha256=expected,thresholds=dict(native_success_count_not_lower=True,control_request_reduction=.10,episode_time_reduction=.05),
        default_sha256=before,paired_seed=7,new_noise_combinations=True,unseen_initials_not_claimed=True,
        model_and_rules_frozen_for_entire_batch=True,no_outcome_based_roster_changes=True,parallel_pairs_per_wave=2,
        maximum_total_task_actions=9600,full_cold_start_gain_not_assumed=True,
        source_sha256={str(p.relative_to(B/'code')):sha(p) for p in (B/'code').rglob('*') if p.is_file()}))
    campaigns=[prepare(i,*r,before,expected,deadline) for i,r in enumerate(ROSTER)]
    started=time.perf_counter();allrows=[]
    for wave in range(2):
        w=B/f'wave{wave}';w.mkdir();gate(w)
        c=campaigns[wave*2]
        if wave==0:
            import orchestrate_full_os as life
            life.CAMPAIGN=c;life.BASE='/project/runs/'+c.name;life.deadline=deadline
            cpu=B/'cpu';cpu.mkdir()
            cmd='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_native_value_binding_v1 test_session_flow test_terminal_evidence_view -v'
            argv=life.cpu_args('yuchen-fastrsi-native-value-batch-v1-cpu',cpu)+[IMAGE,'timeout','180','bash','-lc',cmd]
            cid=life.launch('cpu',argv,cpu)
            try:life.wait_exit(cid,190)
            finally:save(cpu/'STOP_RECEIPT.json',life.stop_all())
            save(cpu/'RESULT.json',dict(status='PASSED'))
        jobs=[(c,a) for c in campaigns[wave*2:wave*2+2] for a in ('N','F')]
        tick=time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:rows=list(pool.map(lambda x:run(*x),jobs))
        save(w/'EXECUTION.json',dict(results=rows,parallel_host_seconds=time.perf_counter()-tick));allrows+=rows
        for c in campaigns[wave*2:wave*2+2]:
            subset=[r for r in rows if r['campaign']==c.name]
            save(c/'EXECUTION.json',dict(results=subset))
            save(c/'STATUS.json',dict(status='COMPLETED' if all(r['exit_code']==0 for r in subset) else 'FAILED_NATIVE_PAIR',default_unchanged=sha(P/'CURRENT_NATIVE_ENTRY.json')==before))
        assert all(r['exit_code']==0 for r in rows)
        print(json.dumps(dict(wave=wave,completed=rows)),flush=True)
    assert sha(P/'CURRENT_NATIVE_ENTRY.json')==before
    save(B/'EXECUTION.json',dict(results=allrows,parallel_host_seconds=time.perf_counter()-started))
    save(B/'STATUS.json',dict(status='COMPLETED',default_unchanged=True))

if __name__=='__main__':main()
