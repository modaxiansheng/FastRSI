"""Read-only transition references from real outcomes, including failures.

No imitation labels for failed commands and no labels for unexecuted branches.
Does not decode/modify observations, train a model, or write persistent OS memory.
"""
import argparse, hashlib, json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();p=args.project;out=args.output
    assert not out.exists(), 'Preserve prior exports'
    sources=[]
    for arm in ('original','adapted'):
        root=p/'runs'/('bowl_lora_source_'+arm+'_v1_20261009')
        for case in ('demo_0','demo_7','demo_25'):
            sources.append((root/case,case,arm))
    root=p/'runs/bowl_phase_lora_source_v1_20261009'
    for case in ('demo_0','demo_7','demo_25'):
        sources.append((root/case,case,'phase_lora'))
    for arm in ('seed7','seed19'):
        root=p/'runs'/('bowl_correction_'+arm+'_v1_20261009')
        for case in ('step_120','step_250','step_270'):
            sources.append((root/case,'demo_0','correction_'+arm))
    episodes=[];records=[];excluded=[]
    def ref(f):
        return dict(path=str(f.relative_to(p)),sha256=sha(f),bytes=f.stat().st_size)
    for root,group,arm in sources:
        result_path=root/'RESULT.json'
        if not result_path.exists():
            excluded.append(dict(path=str(result_path.relative_to(p)),reason='Result missing'));continue
        result=json.loads(result_path.read_text())
        if result['status']!='COMPLETED':
            excluded.append(dict(path=str(result_path.relative_to(p)),reason='No completed outcome'));continue
        rows=result['rows'];assert rows
        success=bool(result['native_success']);end=int(result['executed_actions'])
        assert success or end==400, 'Do not invent failure/absorbing labels for incomplete episodes'
        episode_id=str(root.relative_to(p));start=rows[0]['step_before']
        episode=dict(id=episode_id,initial_state_group=group,arm=arm,source_result=ref(result_path),
            native_success=success,start_step=start,end_step=end,transition_count=len(rows),
            repeated_prefix_excluded=start,training_or_test_role='EXPOSED_DEVELOPMENT_ONLY')
        episodes.append(episode)
        for j,row in enumerate(rows):
            assert row['chunk']==j and row['step_after']>row['step_before']
            if j:assert rows[j-1]['step_after']==row['step_before']
            terminal=j==len(rows)-1
            request=root/('%03d_request.msgpack'%j);response=root/('%03d_response.msgpack'%j)
            next_request=root/('%03d_request.msgpack'%(j+1)) if not terminal else None
            records.append(dict(episode_id=episode_id,initial_state_group=group,chunk=j,
                observation=ref(request),allowed_input_keys=['observation/image','observation/wrist_image','observation/state','prompt'],
                excluded_input_keys=['_source_case','_source_chunk','_correction_case','_correction_chunk'],
                action_proposal=ref(response),executed_action_prefix_length=row['step_after']-row['step_before'],
                step_before=row['step_before'],remaining_execution_budget=400-row['step_before'],
                next_observation=ref(next_request) if next_request else None,terminal=terminal,
                terminal_reason=('native_success' if success else 'finite_budget_exhausted') if terminal else None,
                native_terminal_reward=int(success and terminal),observed_episode_return=int(success),
                observed_remaining_actions=end-row['step_before'],
                observed_remaining_rpc_seconds=sum(x['model_rpc_seconds'] for x in rows[j:]),
                observed_remaining_environment_seconds=sum(x['environment_seconds'] for x in rows[j:]),
                behavior_cloning_eligible=False,observed_successful_episode=success,
                imitation_admission='REQUIRES_SEPARATE_BEHAVIOR_AND_COST_VALIDATION',optimal_action_proven=False,
                counterfactual_branch_label=None))
    out.mkdir(parents=True)
    (out/'TRANSITIONS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    (out/'EPISODES.json').write_text(json.dumps(episodes,indent=2)+'\n')
    summary=dict(schema_version='bowl_value_records.v2',status='REAL_OUTCOME_REFERENCE_EXPORT_NO_TRAINING',episodes=len(episodes),
        successes=sum(x['native_success'] for x in episodes),failures=sum(not x['native_success'] for x in episodes),
        transitions=len(records),initial_state_groups=sorted(set(x['initial_state_group'] for x in episodes)),
        duplicated_initial_state_branches_are_not_independent=True,repeated_prefixes_not_new_training_rows=True,
        split_rule='Group by original source initial state and complete episode; no random frame split for generalization',
        costs_are_observed_labels_not_controller_inputs=True,
        terminal_next_observation_null_is_absorbing_not_a_fabricated_sensor_frame=True,
        failed_actions_not_imitation_supervision=True,cyclic_learning=False,default_promoted=False,
        excluded=excluded,files={f.name:ref(f) for f in out.iterdir() if f.is_file()},
        caveat='Native success is benchmark label, not semantic release/stability or per-action optimality. This small exposed task64 corpus cannot establish generalization.')
    (out/'MANIFEST.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:summary[k] for k in ('status','episodes','successes','failures','transitions','initial_state_groups')}))
if __name__=='__main__':main()
