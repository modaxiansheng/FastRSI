"""Preregistered native default / frozen / paired-update comparison, six HCU per wave."""
from concurrent.futures import ThreadPoolExecutor
import fcntl,json,shutil,signal,socket,subprocess,time,sys
from pathlib import Path
from run_expert_first_change_batch_v2 import gate
from run_native_value_phase_v1 import P,S,sha,save
HOST_SLOT=0
B=P/'planning/native_complete_flow_batch_h136_v1_20261010'
ALL_ROSTER=[(i,10301000+i) for i in range(32,36)]
ROSTER=ALL_ROSTER
ARMS=('N','F','R')

def prepare(index,initial,noise,models,before,deadline):
    c=P/'runs'/f'native_value_phase_v{20+index}_20261010';c.mkdir(exist_ok=False)
    shutil.copytree(B/'code',c/'code')
    for arm in ARMS:shutil.copytree(B/'code',c/arm/'code')
    for name in ('MODEL_CONFIG.json','FLOW.json'):shutil.copy2(S/name,c/name)
    shutil.copytree(S/'initial_experience',c/'initial_experience');(c/'templates').mkdir()
    for f in (S/'templates').glob('*.json'):
        value=json.loads(f.read_text());value['argv']=[x.replace(str(P/'runs/pilot_20261001'),str(c/'templates')) for x in value['argv']]
        save(c/'templates'/f.name,value)
    goal='stack the right bowl on the left bowl and place them in the tray'
    save(c/'PLAN.json',dict(cases=[dict(suite='libero_90',task_id=64,init_state_id=initial,seed=7,mode='evaluation',
        max_steps=400,development_suite_extension='libero90_stack_flow_v1',native_experience_maintenance=False,task_description=goal)]))
    save(c/'PROTOCOL.json',dict(slot=0,absolute_deadline=deadline,diagnostic_trace=True,control_kind='feedback_window',
        arm_modes={arm:'fixed10' for arm in ARMS},port_offset=8300,gpu_bases={'N':0,'F':2,'R':4},terminal_evidence_view=True))
    for arm,model in models.items():
        dest=c/('VALUE_MODEL_'+arm+'.npz');shutil.copy2(model,dest);digest=sha(dest)
        save(c/('VALUE_CONFIG_'+arm+'.json'),dict(version='native_value_phase_binding_v1',enabled=arm!='N',
            model_path='/project/'+str(dest.relative_to(P)),model_sha256=digest,noise_bases={str(initial):noise},
            source_feedback=dict(initial_training='planning/native_paired_q_update_v1_20261010/training/RESULT.json',
                paired_update='planning/complete_flow_memory_v1_20261010/RESULT.json',
                qualification='DEVELOPMENT_NATIVE_THREE_ARM_COMPARISON',not_new_unseen_evidence=False)))
    shutil.copy2(P/'planning/complete_flow_memory_v1_20261010/REGISTRY.json',c/'FLOW_REGISTRY.json')
    save(c/'FLOW_REGISTRY_SPEC.json',dict(sha256=sha(c/'FLOW_REGISTRY.json')))
    save(c/'REGISTRATION.json',dict(status='REGISTERED_NEW_COMBINATION_CONFIRMATION',arms=list(ARMS),initial=initial,noise_base=noise,
        original_agent_loop=True,no_eval_updates=True,pi_parameter_updates=0,maximum_sessions_per_arm=3,maximum_task_actions=3600,
        model_sha256_by_arm={a:sha(c/('VALUE_MODEL_'+a+'.npz')) for a in ARMS},default_sha256=before,
        pair_resource=dict(policy_hcu=1,qwen_hcu=1,target_cpu_limit=8,runner_cpu_limit=8),
        source_sha256=json.loads((B/'REGISTRATION.json').read_text())['source_sha256']))
    return c

def run(c,arm):
    tick=time.perf_counter()
    with (c/(arm+'_controller.log')).open('x') as log:
        child=subprocess.Popen(['python3',str(c/'code/scripts/run_native_action_expert_worker_v1.py'),c.name,arm],stdout=log,stderr=subprocess.STDOUT)
        try:code=child.wait(timeout=1800)
        except BaseException:
            child.send_signal(signal.SIGTERM)
            try:child.wait(timeout=60)
            except subprocess.TimeoutExpired:child.kill();child.wait()
            raise
    return dict(campaign=c.name,arm=arm,exit_code=code,host_seconds=time.perf_counter()-tick)

def main():
    assert Path('/var/tmp/yuchen-fastrsi-models/qwen3-vl-8b-instruct-0c351dd01ed8/config.json').exists()
    # Other owned campaigns on187/188 continue; this lease and resource gate are local136.
    # Every case/model/experience is isolated and frozen for the complete roster.
    lock=Path('/var/tmp/yuchen-fastrsi-native-complete-flow-host136-v1.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    B.mkdir(exist_ok=False);before=sha(P/'CURRENT_NATIVE_ENTRY.json');deadline=time.time()+8000
    registry=P/'planning/complete_flow_memory_v1_20261010/REGISTRY.json'
    check=json.loads((P/'planning/native_complete_flow_check_h136_v1_20261010/RESULT.json').read_text())
    assert check['status']=='PASSED_REAL_NATIVE_COMPLETE_FLOW_PACKET_BINDING'
    cache=json.loads((P/'planning/qwen3_h136_cache_binding_v1_20261010/RESULT.json').read_text());assert cache['status']=='VERIFIED_OWNED_LOCAL_CACHE'
    frozen=updated=P/'planning/native_paired_q_update_v1_20261010/training/paired_router.npz'
    assert sha(frozen)==json.loads(registry.read_text())['teacher_model_sha256']
    for rel,digest in json.loads((S/'SOURCE_MANIFEST.json').read_text()).items():assert sha(S/rel)==digest
    for rel in ('scripts','src/fastrsi','tests','configs'):
        shutil.copytree(P/rel,B/'code'/rel,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    save(B/'REGISTRATION.json',dict(status='REGISTERED_BEFORE_NEW_OUTCOMES',roster=ROSTER,all_hosts_roster=ALL_ROSTER,host_slot=HOST_SLOT,arms=list(ARMS),
        fixed_model_sha256=sha(frozen),updated_model_sha256=sha(updated),whole_flow_registry_sha256=sha(registry),
        whole_flow_modes=dict(N='default_pi',F='always_existing_learned_flow',R='experience_selected_whole_flow'),
        known_initial_scenes_new_future_noises=True,unseen_scene_generalization_not_claimed=True,default_sha256=before,
        thresholds=dict(native_success_count_not_lower=True,control_request_reduction=.10,episode_time_reduction=.05),
        paired_seed=7,new_noise_combinations=True,unseen_initials_not_claimed=True,no_outcome_based_roster_changes=True,
        model_and_rules_frozen_for_entire_batch=True,shared_source_experience_and_skill_permissions=True,
        full_cold_start_gain_not_assumed=True,learning_cost_source='planning/complete_flow_memory_v1_20261010; teacher and historical collection costs retained',
        common_sampling_protocol='Same frozen teacher sampling/ranking when whole-flow R is selected; N unchanged pi; maximum20 extras',
        ranking_rule='Memory chooses a whole versioned flow once from initial legal observation; no mid-task memory update',
        prior_quality_probe='Three complete-flow memories from four older actual paired cases; exact scenes only; no learned neural classifier claim',
        source_sha256={str(f.relative_to(B/'code')):sha(f) for f in (B/'code').rglob('*') if f.is_file()}))
    cases=[prepare(i,*r,dict(N=frozen,F=frozen,R=updated),before,deadline) for i,r in enumerate(ROSTER)]
    started=time.perf_counter();allrows=[]
    for wave,c in enumerate(cases):
        w=B/f'wave{wave}';w.mkdir();gate(w)
        for port in (58000,58002,58010,58020):
            with socket.socket() as sock:
                sock.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);sock.bind(('127.0.0.1',port))
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(lambda arm:run(c,arm),ARMS))
        allrows+=rows;save(c/'EXECUTION.json',dict(results=rows))
        status='COMPLETED' if all(r['exit_code']==0 for r in rows) else 'FAILED_NATIVE_THREE_ARM'
        save(c/'STATUS.json',dict(status=status,results=rows,default_unchanged=before==sha(P/'CURRENT_NATIVE_ENTRY.json')))
        save(B/'EXECUTION.json',dict(results=allrows,completed_waves=wave+1,parallel_host_seconds=time.perf_counter()-started))
        assert status=='COMPLETED',rows
    assert before==sha(P/'CURRENT_NATIVE_ENTRY.json')
    save(B/'STATUS.json',dict(status='COMPLETED',default_unchanged=True));print(json.dumps(allrows),flush=True)
if __name__=='__main__':main()
