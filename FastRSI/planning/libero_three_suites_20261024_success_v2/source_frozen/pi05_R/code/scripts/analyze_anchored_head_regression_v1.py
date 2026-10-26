"""Matched complete-Agent results, including failures and learning overhead."""
import json
from pathlib import Path
from summarize_local_bowl_transition_v1 import summarize,lines,digest
P=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
B=P/'planning/bowl_anchored_head_regression_v1_20261009'

def main():
    roster=json.loads((B/'FROZEN_ROSTER.json').read_text());pairs=[]
    for seed in sorted({r['seed'] for r in roster}):
        group={r['arm']:Path(r['campaign']) for r in roster if r['seed']==seed};arms={};receipts={}
        for arm,c in group.items():
            s,_=summarize(c);raw=json.loads((c/'stream/RESULT.json').read_text());case=raw['cases'][0]
            s.update(experience_fixed=case['experience_before']==case['experience_after'],learned_action_parameters=arm=='adapted',heldout_initial_draw=False)
            assert s['experience_fixed'] and s['default_unchanged']
            load=json.loads((c/'stream/POLICY_LOAD.json').read_text());assert load['adapter_enabled']==(arm=='adapted')
            s['policy_load']=load;arms[arm]=s
            receipts[arm]=lines(c/'stream/NATIVE_SAMPLER_RESTORE.jsonl')[0]['restore']
        for field in ('public_rgb','public_robot8','post_settle_state'):
            assert receipts['original'][field]==receipts['adapted'][field],field
        for camera in ('agentview_image','robot0_eye_in_hand_image'):
            paths=[group[arm]/'stream/stage_trace'/arms[arm]['root_session']/('0000_'+camera+'.png') for arm in ('original','adapted')]
            assert digest(paths[0])==digest(paths[1])
        a,b=arms['original'],arms['adapted'];pairs.append(dict(seed=seed,arms=arms,initial_inputs_exact=True,
            new_success=b['environment_success'] is True and a['environment_success'] is False,
            new_failure=a['environment_success'] is True and b['environment_success'] is False,
            full_wall_delta=b['full_wall_seconds']-a['full_wall_seconds']))
    keys=['actual_actions','total_actual_vla_calls','total_control_requests','Qwen_requests','full_wall_seconds','startup_seconds']
    totals={arm:{k:sum(p['arms'][arm].get(k) or 0 for p in pairs) for k in keys} for arm in ('original','adapted')}
    for arm,g in totals.items():
        g['native_successes']=sum(p['arms'][arm]['environment_success'] is True for p in pairs)
        g['chain_passed']=sum(p['arms'][arm]['chain_status']=='PASSED' for p in pairs)
        g['os_execution_seconds']=sum(p['arms'][arm]['os_cost']['inclusive_seconds_do_not_sum_nested'].get('os_execution',0) for p in pairs)
    collection=json.loads((P/'runs/bowl_demo_collection_v1_20261009/DATASET.json').read_text())
    trainhost=json.loads((P/'runs/bowl_anchored_head50_v1_20261009/HOST_RESULT.json').read_text())
    previous_project_training=json.loads((P/'runs/bowl_action_adaptation_v2_20261009/HOST_RESULT.json').read_text())
    learning=dict(demo_collection_parallel_wall_seconds=collection['acquisition_wall_seconds'],demo_collection_worker_wall_sum=collection['worker_wall_seconds_sum'],
        training_host_seconds=trainhost['full_wall_seconds'],prior_failed_large_update_training_host_seconds=previous_project_training['full_wall_seconds'],
        known_serial_method_setup_seconds=collection['acquisition_wall_seconds']+trainhost['full_wall_seconds'],
        excludes_previous_download_and_one_step_development=True,
        scope='Acquisition and training kept separate from model runtime; no worker sum added to wall')
    out=dict(pairs=pairs,totals=totals,learning_costs=learning,pairs_count=len(pairs),
        new_successes=sum(p['new_success'] for p in pairs),new_failures=sum(p['new_failure'] for p in pairs),
        sum_full_wall_reduction_fraction=1-totals['adapted']['full_wall_seconds']/totals['original']['full_wall_seconds'],
        scope='Two previously exposed developmental cases; frozen base versus anchored output-row50 on full native OS. Not independent generalization, official LIBERO10 or cyclic FastRSI proof',
        retention_evaluated=False,default_promoted=False)
    (B/'AGGREGATE.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k!='pairs'},indent=2))
if __name__=='__main__':main()
