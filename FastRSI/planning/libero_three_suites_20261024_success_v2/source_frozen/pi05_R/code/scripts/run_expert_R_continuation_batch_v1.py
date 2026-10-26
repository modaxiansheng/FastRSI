"""Four real first-change diagnostic branches on authorized idle node136."""
from concurrent.futures import ThreadPoolExecutor
import fcntl,hashlib,json,shutil,socket,subprocess,time
from pathlib import Path
from urllib.request import urlopen
import re
from run_native_value_phase_v1 import capture
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI');B=P/'planning/expert_R_continuation_batch_h136_v1_20261010'

def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def gate(folder):
    hcu=capture(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse'])
    holders=capture(['bash','-lc','fuser -v /dev/kfd /dev/mkfd /dev/dri/render*'])
    ids=subprocess.check_output(['docker','ps','-q'],text=True).split()
    containers=json.loads(subprocess.check_output(['docker','inspect']+ids)) if ids else []
    save(folder/'PREFLIGHT.json',dict(hcu=hcu,holders=holders,containers=containers,
        authorization='User explicitly authorized idle GPUs on node136 irrespective of sleeping containers; no other workload stopped',
        parallel_other_project_trials_on187_188_not_modified=True))
    mem=re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',hcu['stdout'])
    assert hcu['returncode']==0 and len(mem)==8 and all(float(v)==0 for _,v in mem)
    for c in containers:
        assert c['Name'] in ['/exo2world-virtualrgb-full8-two-stage-v2-20261006','/exo2world-virtualocc-pilot2-20260922']
        assert c['Config']['Cmd']==['-lc','exec sleep infinity']
        rows=subprocess.check_output(['docker','top',c['Id'],'-eo','pid,args'],text=True).splitlines()
        assert len(rows)==2 and rows[1].split(maxsplit=1)[1]=='sleep infinity',rows
    for pid in map(int,holders['stdout'].split()):
        proc=Path('/proc')/str(pid);argv=[s.decode() for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
        assert str((proc/'exe').resolve())=='/usr/local/hyhal/bin/hymgr' and argv==['/usr/local/hyhal/bin/hymgr']
    for port in range(57300,57304):
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))

def run(job):
    slot,initial=job;root=P/'runs'/f'expert_R_continuation_h136_slot{slot}_v1_20261010';root.mkdir(exist_ok=False)
    shutil.copytree(B/'code',root/'code');shutil.copytree(P/'models/openpi_data',root/'tokenizer_cache')
    port=57300+slot
    with socket.socket() as sock:sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
    source=P/'planning'/f'phase_iql_rollout_gpu{slot}_v1_20261010'
    original=str(P/'runs'/f'phase_iql_rollout_gpu{slot}_v1_20261010');base=P/'planning/phase_iql_rollout_gpu0_v1_20261010'
    policy=json.loads((source/'POLICY_LAUNCH.json').read_text())
    policy=[x.replace(original,str(root)).replace('/check/code/serve_phase_iql_candidates_v1.py','/check/code/serve_expert_R_continuation_v1.py') for x in policy]
    policy[policy.index('--name')+1]=f'yuchen-fastrsi-expert-R-continuation-policy-gpu{slot}-v1-g136-20261010'
    case=json.loads((base/'init28_single_LAUNCH.json').read_text())
    case=[str(root)+':/check' if x==str(P/'runs/phase_iql_rollout_gpu0_v1_20261010')+':/check' else x for x in case]
    case[case.index('--name')+1]=f'yuchen-fastrsi-expert-R-continuation-case-gpu{slot}-v1-g136-20261010'
    index=case.index('/check/code/check_phase_iql_rollout_v1.py');case=case[:index]+['/check/code/run_expert_R_continuation_v1.py']
    save(root/'SPEC.json',dict(source=f'planning/expert_first_change_inputs_v1_20261010/init{initial}',endpoint=f'ws://127.0.0.1:{port}'))
    save(root/'POLICY_LAUNCH.json',policy);save(root/'CASE_LAUNCH.json',case)
    assert policy[-1].endswith('--port '+str(port))
    owned=[];start=time.perf_counter()
    try:
        cid=subprocess.check_output(policy,text=True).strip();owned.append(cid);deadline=time.monotonic()+420
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
        with (root/'CASE.log').open('x') as f:r=subprocess.run(case,stdout=f,stderr=subprocess.STDOUT,timeout=600)
        assert r.returncode==0,r.returncode
        result=json.loads((root/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        return dict(slot=slot,initial=initial,fork_step=result["fork_step"],native_success=result['native_success'],actions=result['physical_steps'],
            requests=result['remaining_control_requests'],seconds=result['seconds_including_close'])
    finally:
        records=[]
        for cid in owned:
            record=json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
            (root/('LOG_'+cid.replace('/','_')+'.txt')).write_text(subprocess.check_output(['docker','logs',cid],stderr=subprocess.STDOUT,text=True))
            if record['State']['Running']:subprocess.run(['docker','stop','-t','10',cid],check=True,capture_output=True)
            record=json.loads(subprocess.check_output(['docker','inspect',cid]))[0];assert not record['State']['Running'] and record['State']['Pid']==0
            records.append(dict(name=record['Name'],image=record['Image'],state=record['State']))
        save(root/'STOP_RECEIPT.json',records);save(root/'HOST.json',dict(seconds=time.perf_counter()-start))

def main():
    lock=Path('/var/tmp/yuchen-fastrsi-expert-R-continuation-h136-v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    B.mkdir(exist_ok=False);gate(B);before=sha(P/'CURRENT_NATIVE_ENTRY.json')
    plan_path=P/'planning/expert_first_change_inputs_v1_20261010/RESULT.json'
    plan=json.loads(plan_path.read_text());assert plan['status']=='PREPARED_EXPERT_FIRST_CHANGE_DIAGNOSTICS'
    (B/'code').mkdir()
    for n in ['serve_expert_R_continuation_v1.py','run_expert_R_continuation_v1.py','prefix_feature_capture_v1.py','action_expert_capture_v1.py','action_expert_ranker_v1.py','replay_stack_progress.py','stage_trace.py','phase_iql_feedback_update_v1.py','phase_iql_value_v1.py']:
        shutil.copy2(P/'scripts'/n,B/'code'/n)
    jobs=[(0,38),(1,39)]
    save(B/'REGISTRATION.json',dict(status='REGISTERED_BEFORE_BRANCH_OUTCOMES',jobs=jobs,source_registration_sha256=sha(plan_path),
        host='gpu136',goal='Compare same first action under frozen R continuation, not default continuation; only two previously audited contexts',not_independent_efficiency_evidence=True,same_cross_host_candidates_required=True,
        original_full_task_goal_and_success_unchanged=True,max_actions_per_episode=400,max_total_actions=800,
        learned_selector=False,actor_parameter_updates=0,persistent_updates=0,reference_reuse_costs_include_prior_reproductions=True,
        source_sha256={x.name:sha(x) for x in (B/'code').iterdir()},default_sha256=before))
    tick=time.perf_counter()
    with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(run,jobs))
    assert before==sha(P/'CURRENT_NATIVE_ENTRY.json')
    save(B/'RESULT.json',dict(status='COMPLETED',cases=results,parallel_seconds=time.perf_counter()-tick,default_unchanged=True,
        learned_improvement_claimed=False,full_native_agent_comparison=False));print(json.dumps(results),flush=True)

if __name__=='__main__':main()
