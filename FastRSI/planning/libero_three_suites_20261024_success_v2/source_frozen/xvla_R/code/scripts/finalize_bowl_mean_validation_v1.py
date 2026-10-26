"""Read back completed sampling trials, accounting and immutable raw-copy manifests."""
from pathlib import Path
import hashlib,json,subprocess
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    assert not p.exists(),str(p);p.write_text(json.dumps(value,indent=2)+'\n')
def main():
    parent=P/'planning/bowl_mean_validation_v1_20261010'
    agg=json.loads((parent/'AGGREGATE.json').read_text());assert agg['status']=='COMPLETED'
    out=P/'planning/bowl_mean_validation_report_v1_20261010';out.mkdir(exist_ok=True)
    assert json.loads((out/'COMMAND_CHECK.json').read_text())['status']=='VERIFIED_ACTUAL_SAVED_COMBINATIONS'
    gate=json.loads((parent/'PARENT_GATE.json').read_text())
    assert sha(P/'CURRENT_NATIVE_ENTRY.json')==gate['default_sha256']
    containers=[]
    for item in json.loads((parent/'PARENT_CLEANUP.json').read_text()):
        state=json.loads(subprocess.check_output(['docker','inspect',item['name']]))[0]['State']
        assert not state['Running'] and state['Pid']==0
        containers.append(dict(name=item['name'],state=state))
    costs={arm:dict(full_episode_seconds=0.,rpc_seconds=0.,environment_seconds=0.,evidence_seconds=0.,
        candidate_generation_seconds=0.,combination_seconds=0.,actions=0,control_requests=0,candidate_inferences=0,successes=0,runs=0)
        for arm in ('single','mean2')}
    cases=[]
    for slot in (0,1,2,3):
        root=P/'runs'/('bowl_mean_validation_gpu%d_v1_20261010'%slot)
        probe=json.loads((root/'SAMPLER_PROBE.json').read_text());assert probe['status']=='PASSED'
        for case in json.loads((root/'AGGREGATE.json').read_text())['cases']:
            result=json.loads((root/case['case']/'RESULT.json').read_text());assert result['status']=='COMPLETED'
            assert result['persistent_updates']==0 and result['training_steps']==0
            cost=costs[case['arm']]
            cost['runs']+=1;cost['successes']+=int(result['native_success'])
            for dst,src in (('full_episode_seconds','seconds_including_close'),('actions','executed_actions'),
                            ('control_requests','control_requests'),('candidate_inferences','candidate_inferences')):cost[dst]+=result[src]
            previous=0
            for row in result['rows']:
                assert row['step_before']==previous and previous<row['step_after']<=400
                previous=row['step_after'];selected=row['selected_index']
                cost['combination_seconds']+=row['combination_seconds']
                assert selected==0 if case['arm']=='single' else selected is None
                for dst,src in (('rpc_seconds','rpc_seconds'),('environment_seconds','environment_seconds'),
                                ('evidence_seconds','evidence_seconds'),('candidate_generation_seconds','candidate_seconds')):cost[dst]+=row[src]
            assert previous==result['executed_actions']
            cases.append(dict(slot=slot,**case))
    pairs=agg['pairs'];oracle=[]
    for pair in pairs:
        a,b=pair['single'],pair['mean2']
        assert a['initial_input_sha256']==b['initial_input_sha256']
        # Whole-flow retrospective upper bound, never per-action training labels.
        if a['native_success']!=b['native_success']:chosen=a if a['native_success'] else b
        elif a['native_success']:chosen=min((a,b),key=lambda x:x['full_seconds'])
        else:chosen=a
        oracle.append(dict(initial=pair['initial'],selected_arm=chosen['arm'],success=chosen['native_success'],
            full_seconds=chosen['full_seconds'],actions=chosen['actions'],candidate_inferences=chosen['candidate_inferences']))
    save(out/'COST_BREAKDOWN.json',dict(arms=costs,
        parallel_host_seconds=json.loads((parent/'EXECUTION.json').read_text())['parallel_host_seconds'],
        service_hosts=[json.loads((P/'runs'/('bowl_mean_validation_gpu%d_v1_20261010'%slot)/'FINAL_STATE.json').read_text())['host_wall_seconds'] for slot in (0,1,2,3)],
        startup_and_probes=[json.loads((P/'runs'/('bowl_mean_validation_gpu%d_v1_20261010'%slot)/'STARTUP.json').read_text()) for slot in (0,1,2,3)],
        timing_rule='RPC contains candidate generation; do not add overlapping fields. Host parallel wall differs from sums.',
        historical_learning_cost_not_included_in_method_gain=True,training_steps=0))
    save(out/'PAIR_ANALYSIS.json',dict(cases=cases,pairs=pairs,
        rescued=sum(not p['single']['native_success'] and p['mean2']['native_success'] for p in pairs),
        lost=sum(p['single']['native_success'] and not p['mean2']['native_success'] for p in pairs),
        both_success=sum(p['single']['native_success'] and p['mean2']['native_success'] for p in pairs),
        both_fail=sum(not p['single']['native_success'] and not p['mean2']['native_success'] for p in pairs),
        retrospective_whole_flow_oracle=oracle,
        oracle_scope='Optimistic diagnostic on observed whole flows only; not deployed score or same-state optimal-action supervision',
        learning_gain_proven=False,complete_agent_comparison=False))
    save(out/'FINAL_READBACK.json',dict(status='COMPLETED_NOT_PROMOTED',owned_containers=containers,
        default_unchanged=True,default_sha256=gate['default_sha256'],training_steps=0,learned_selection=False))
    roots=[parent]+[P/'runs'/('bowl_mean_validation_gpu%d_v1_20261010'%i) for i in range(4)]+[out]
    for root in roots:
        files=[]
        for f in sorted(root.rglob('*')):
            rel=f.relative_to(root)
            if f.is_file() and f.name!='COPY_MANIFEST.json' and not any(x in ('tokenizer_cache','__pycache__') for x in rel.parts):
                files.append(dict(path=str(rel),bytes=f.stat().st_size,sha256=sha(f)))
        save(root/'COPY_MANIFEST.json',dict(root=str(root),files=files,count=len(files),bytes=sum(r['bytes'] for r in files)))
    print(json.dumps(dict(arms=costs,owned_containers=len(containers),oracle=oracle)),flush=True)
if __name__=='__main__':main()
