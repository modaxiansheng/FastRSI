"""Four registered later-state alternatives; exact factual default outcomes reused."""
from concurrent.futures import ThreadPoolExecutor
import fcntl,hashlib,json,shutil,socket,subprocess,time
from pathlib import Path
from urllib.request import urlopen
from run_native_value_batch_v1 import gate
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI');B=P/'planning/native_value_branch_batch_v2_20261010'

def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def run(job):
    slot,context=job;initial=context["initial"];fork=context["step"];root=P/'runs'/f'native_value_branch_gpu{slot}_v2_20261010';root.mkdir(exist_ok=False)
    shutil.copytree(B/'code',root/'code');shutil.copytree(P/'models/openpi_data',root/'tokenizer_cache')
    port=57300+slot
    with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
    source=pref=P/'runs'/f'phase_iql_rollout_gpu{slot}_v1_20261010'
    original=str(source);base=P/'runs/phase_iql_rollout_gpu0_v1_20261010'
    policy=json.loads((source/'POLICY_LAUNCH.json').read_text())
    policy=[x.replace(original,str(root)).replace('/check/code/serve_phase_iql_candidates_v1.py','/check/code/serve_native_branch_candidates_v2.py') for x in policy]
    policy[policy.index('--name')+1]=f'yuchen-fastrsi-native-branch-policy-gpu{slot}-v2-20261010'
    case=json.loads((base/'init28_single_LAUNCH.json').read_text())
    case=[str(root)+':/check' if x==str(base)+':/check' else x for x in case]
    case[case.index('--name')+1]=f'yuchen-fastrsi-native-branch-case-gpu{slot}-v2-20261010'
    index=case.index('/check/code/check_phase_iql_rollout_v1.py');case=case[:index]+['/check/code/run_native_value_branch_v2.py']
    save(root/'SPEC.json',dict(context=context,endpoint=f'ws://127.0.0.1:{port}'))
    save(root/'POLICY_LAUNCH.json',policy);save(root/'CASE_LAUNCH.json',case)
    owned=[];start=time.perf_counter()
    try:
        cid=subprocess.check_output(policy,text=True).strip();owned.append(cid);deadline=time.monotonic()+280
        while True:
            try:
                with urlopen(f'http://127.0.0.1:{port}/healthz',timeout=2) as r:assert r.status==200
                break
            except Exception:
                assert json.loads(subprocess.check_output(['docker','inspect',cid]))[0]['State']['Running']
                if time.monotonic()>deadline:raise TimeoutError('Policy readiness')
                time.sleep(2)
        save(root/'STARTUP.json',dict(seconds=time.perf_counter()-start))
        owned.append(case[case.index('--name')+1])
        with (root/'CASE.log').open('x') as f:r=subprocess.run(case,stdout=f,stderr=subprocess.STDOUT,timeout=360)
        assert r.returncode==0,r.returncode
        result=json.loads((root/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        return dict(slot=slot,initial=initial,fork_step=fork,native_success=result['native_success'],actions=result['physical_steps'],
            requests=result['remaining_control_requests'],seconds=result['seconds_including_close'])
    finally:
        records=[]
        for cid in owned:
            record=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
            if record['State']['Running']:subprocess.run(['docker','stop','-t','10',cid],check=True,capture_output=True)
            record=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert not record['State']['Running'] and record['State']['Pid']==0
            records.append(dict(name=record['Name'],image=record['Image'],state=record['State']))
        save(root/'STOP_RECEIPT.json',records);save(root/'HOST.json',dict(seconds=time.perf_counter()-start))

def main():
    lock=(P/'runs/full_os_v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    B.mkdir(exist_ok=False);gate(B);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
    audit=json.loads((P/'planning/native_value_branch_audit_v1_20261010/RESULT.json').read_text())
    assert audit['status']=='AUDITED_REAL_BRANCH_SUPERVISION' and audit['default_branch_reproduced_native_rgb_robot_and_actions']
    plan_path=P/'planning/native_value_branch_next_v1_20261010/PLAN.json'
    plan=json.loads(plan_path.read_text());assert plan['status']=='REGISTERED_DEVELOPMENT_PROBES_NOT_YET_EXECUTED'
    (B/'code').mkdir()
    for n in ['serve_native_branch_candidates_v2.py','run_native_value_branch_v2.py','prefix_feature_capture_v1.py','replay_stack_progress.py','stage_trace.py']:
        shutil.copy2(P/'scripts'/n,B/'code'/n)
    jobs=list(enumerate(plan['contexts']));assert len(jobs)==4
    save(B/'REGISTRATION.json',dict(status='REGISTERED_BEFORE_BRANCH_OUTCOMES',jobs=jobs,source_registration_sha256=sha(plan_path),
        goal='Execute registered candidate1 alternatives; reuse exactly bound factual candidate0 default continuations',not_independent_efficiency_evidence=True,
        original_full_task_goal_and_success_unchanged=True,max_actions_per_episode=400,max_total_actions=1600,
        learned_selector=False,actor_parameter_updates=0,persistent_updates=0,reference_reuse_costs_include_prior_reproductions=True,
        source_sha256={x.name:sha(x) for x in (B/'code').iterdir()},default_sha256=before))
    tick=time.perf_counter()
    with ThreadPoolExecutor(max_workers=4) as pool:results=list(pool.map(run,jobs))
    assert before==sha(P/'CURRENT_NATIVE_ENTRY.json')
    save(B/'RESULT.json',dict(status='COMPLETED',cases=results,parallel_seconds=time.perf_counter()-tick,default_unchanged=True,
        learned_improvement_claimed=False,full_native_agent_comparison=False));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
