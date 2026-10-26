"""First full native-Agent pair for the unchanged qualified value candidate."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import time

P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
NAME=os.environ.get('FASTRSI_NATIVE_VALUE_PHASE_NAME','native_value_phase_v1_20261010')
assert re.fullmatch(r'native_value_phase_v\d+_20261010',NAME)
C=P/'runs'/NAME
S=P/'planning/native_value_source_v1_20261010'
IMAGE='sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def capture(argv):
    r=subprocess.run(argv,capture_output=True,text=True,timeout=45)
    return dict(argv=argv,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)


def main():
    lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    C.mkdir(exist_ok=False);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
    hcu=capture(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse'])
    containers=capture(['docker','ps','--format','{{.ID}} {{.Names}}'])
    holders=capture(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*'])
    save(C/'PREFLIGHT.json',dict(hcu=hcu,containers=containers,holders=holders))
    mem=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',hcu['stdout'])
    assert hcu['returncode']==0 and len(mem)==8 and all(float(v)==0 for _,v in mem)
    assert containers['returncode']==0 and not containers['stdout'].strip()
    for pid in map(int,holders['stdout'].split()):
        assert (Path('/proc')/str(pid)/'comm').read_text().strip()=='hymgr'
        assert str((Path('/proc')/str(pid)/'exe').resolve())=='/usr/local/hyhal/bin/hymgr'
    for port in [57800,57802,57810,57820,57900,57902,57910,57920]:
        with socket.socket() as sock:sock.bind(('127.0.0.1',port))
    for rel,digest in json.loads((S/'SOURCE_MANIFEST.json').read_text()).items():assert sha(S/rel)==digest
    for rel in ['scripts','src/fastrsi','tests','configs']:
        shutil.copytree(P/rel,C/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    for arm in ['N','F']:shutil.copytree(C/'code',C/arm/'code')
    for n in ['MODEL_CONFIG.json','FLOW.json']:shutil.copy2(S/n,C/n)
    shutil.copytree(S/'initial_experience',C/'initial_experience')
    (C/'templates').mkdir()
    for f in (S/'templates').glob('*.json'):
        value=json.loads(f.read_text());value['argv']=[a.replace(str(P/'runs/pilot_20261001'),str(C/'templates')) for a in value['argv']]
        save(C/'templates'/f.name,value)
    model=P/'planning/phase_iql_feedback_v1_20261010/training/frozen_router.npz'
    expected=json.loads((P/'planning/phase_goal_v1_20261010/CANDIDATE_ADMISSION.json').read_text())['accepted_frozen_candidate_sha256']
    assert sha(model)==expected;shutil.copy2(model,C/'VALUE_MODEL.npz')
    goal='stack the right bowl on the left bowl and place them in the tray'
    case=dict(suite='libero_90',task_id=64,init_state_id=20,seed=7,mode='evaluation',max_steps=400,
        development_suite_extension='libero90_stack_flow_v1',native_experience_maintenance=False,task_description=goal)
    save(C/'PLAN.json',dict(cases=[case]))
    deadline=time.time()+2100
    save(C/'PROTOCOL.json',dict(slot=0,absolute_deadline=deadline,diagnostic_trace=True,
        control_kind='feedback_window',arm_modes={'N':'fixed10','F':'fixed10'},port_offset=8300,
        gpu_bases={'N':0,'F':2},terminal_evidence_view=True))
    for arm in ['N','F']:
        save(C/('VALUE_CONFIG_'+arm+'.json'),dict(version='native_value_phase_binding_v1',
            enabled=arm=='F',model_path='/project/runs/'+NAME+'/VALUE_MODEL.npz',model_sha256=expected,
            noise_bases={'20':8101020},source_feedback=dict(initial_training='planning/phase_iql_training_v1_20261010/training/RESULT.json',
                original_transitions_sha256='8731fb0a5bebd2c28e8bcf8b76da40ebaa2380664aca902a0fa725f602e130e7',
                qualification='planning/phase_goal_v1_20261010/CANDIDATE_ADMISSION.json',
                not_new_unseen_evidence=True)))
    save(C/'REGISTRATION.json',dict(status='REGISTERED_NATIVE_BINDING_CHECK',arms=['N','F'],
        old_case_used_for_integration_not_generalization=True,original_agent_loop=True,
        no_eval_updates=True,pi_parameter_updates=0,maximum_sessions_per_arm=3,maximum_task_actions=2400,
        selected_model_sha256=expected,default_sha256=before,
        next_action='Run new registered pairs only after native binding/verification/isolation receipt',
        source_sha256={str(p.relative_to(C/'code')):sha(p) for p in (C/'code').rglob('*') if p.is_file()}))
    # CPU checks use the already installed native OS dependencies, no HCU.
    import orchestrate_full_os as life
    life.CAMPAIGN=C;life.BASE='/project/runs/'+NAME;life.deadline=deadline
    cpu=C/'cpu';cpu.mkdir()
    command='source /opt/dtk/env.sh && cd '+life.BASE+'/code/tests && /project/envs/os-v1/bin/python -m unittest test_native_value_binding_v1 test_session_flow test_terminal_evidence_view -v'
    argv=life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu',cpu)+[IMAGE,'timeout','180','bash','-lc',command]
    cid=life.launch('cpu',argv,cpu)
    try:life.wait_exit(cid,190)
    finally:save(cpu/'STOP_RECEIPT.json',life.stop_all())
    save(cpu/'RESULT.json',dict(status='PASSED'))
    started=time.perf_counter()
    def run(arm):
        tick=time.perf_counter()
        with (C/(arm+'_controller.log')).open('x') as log:
            r=subprocess.run(['python3',str(C/'code/scripts/run_native_value_worker_v1.py'),NAME,arm],
                             stdout=log,stderr=subprocess.STDOUT,timeout=2050)
        return dict(arm=arm,exit_code=r.returncode,host_seconds=time.perf_counter()-tick)
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,['N','F']))
    save(C/'EXECUTION.json',dict(results=results,parallel_host_seconds=time.perf_counter()-started))
    assert before==sha(P/'CURRENT_NATIVE_ENTRY.json')
    save(C/'STATUS.json',dict(status='COMPLETED' if all(r['exit_code']==0 for r in results) else 'FAILED_NATIVE_PAIR',
                              results=results,default_unchanged=True))
    print(json.dumps(results),flush=True)


if __name__=='__main__':main()
