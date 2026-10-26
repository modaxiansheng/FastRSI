"""Audit actual learned candidate budgets, choices and feedback versions."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from phase_iql_feedback_update_v1 import AdaptivePredictor
P=Path('/project');O=Path('/check');rows=[];pairs=[];firsts={};checked=0;start=time.perf_counter()
model=P/'runs/phase_iql_feedback_v1_20261010/training/updated_router.npz'
selectors={key:AdaptivePredictor(model.parent/(key+'_router.npz')) for key in ('frozen','updated')}
for slot in range(8):
    root=P/'runs'/f'phase_iql_repeat_gpu{slot}_v1_20261010'
    agg=json.loads((root/'AGGREGATE.json').read_text());assert agg['status']=='COMPLETED'
    first={}
    for case in agg['cases']:
        folder=root/case['case'];result=json.loads((folder/'RESULT.json').read_text());assert result['status']=='COMPLETED'
        assert result['persistent_updates']==0 and not result['complete_agent_comparison']
        predictor=selectors.get(case['arm']);extras=0
        if predictor:assert result['value_model_sha256']==predictor.sha256
        for r in result['rows']:
            i=r['chunk'];request=unpackb((folder/f'{i:03d}_request.msgpack').read_bytes())
            response=unpackb((folder/f'{i:03d}_response.msgpack').read_bytes());candidates=response['candidates']
            command=np.load(folder/f'{i:03d}_command.npz',allow_pickle=False)
            assert np.array_equal(command['planned_actions'],candidates[r['selected_index']])
            assert np.array_equal(command['submitted_actions'],command['planned_actions'][:min(10,400-r['step_before'])])
            assert 1<=r['step_after']-r['step_before']<=len(command['submitted_actions'])
            for j,digest in enumerate(r['candidate_noise_sha256']):
                noise=np.random.default_rng(np.random.SeedSequence([8101000+case['initial'],i,j])).standard_normal((10,32)).astype(np.float32)
                assert hashlib.sha256(noise.tobytes()).hexdigest()==digest
            if predictor:
                selected,info=predictor.score(response['prefix_feature'],request['observation/state'],candidates,400-r['step_before'])
                assert selected==r['selected_index'] and np.allclose(info['q_scores'],r['choice_meta']['q_scores'],atol=1e-6,rtol=0)
                _,one=predictor.score(response['prefix_feature'],request['observation/state'],candidates[:1],400-r['step_before'])
                value=predictor.value(response['prefix_feature'],request['observation/state'],400-r['step_before'])
                assert abs(value-r['choice_meta']['state_value'])<1e-5
                needed=one['q_scores'][0]<value and extras<20
                assert needed==(len(candidates)==2) and r['choice_meta']['extras_before']==extras
                extras+=int(needed)
            elif case['arm']=='mix':assert len(candidates)==(2 if i%2==0 else 1)
            if i==0:first[case['arm']]=candidates[0]
            checked+=1
        assert extras<=20
        if predictor:assert extras==sum(r['candidate_count']-1 for r in result['rows'])
        rows.append(dict(initial=case['initial'],arm=case['arm'],success=result['native_success'],actions=result['executed_actions'],
            requests=result['control_requests'],candidate_inferences=result['candidate_inferences'],seconds=result['seconds_including_close'],
            rpc_seconds=sum(r['rpc_seconds'] for r in result['rows']),environment_seconds=sum(r['environment_seconds'] for r in result['rows']),
            evidence_seconds=sum(r['evidence_seconds'] for r in result['rows']),
            selection_seconds=sum(r['choice_meta'].get('selection_seconds',0) for r in result['rows']),
            selected_second=sum(r['selected_index']==1 for r in result['rows'])))
    firsts.setdefault(case['initial'],{}).update(first)
for initial,first in firsts.items():
    assert len(first)==2 and np.array_equal(first['single'],first['frozen'])
    pairs.append(dict(initial=initial,first_candidate_exact=True))
summary={}
for arm in ('single','frozen'):
    group=[r for r in rows if r['arm']==arm]
    summary[arm]={key:sum(r[key] for r in group) for key in ('success','actions','requests','candidate_inferences','seconds','rpc_seconds','environment_seconds','evidence_seconds','selection_seconds','selected_second')}
    summary[arm]['runs']=len(group)
data=json.loads((P/'planning/phase_value_dataset_v1_20261010/RESULT.json').read_text())
historical=[]
for episode in json.loads((P/'planning/phase_value_dataset_v1_20261010/EPISODES.json').read_text()):
    ref=episode['source_result'];raw=(P/ref['path']).read_bytes()
    assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
    result=json.loads(raw);assert result['status']=='COMPLETED'
    historical.append(dict(episode=episode['id'],seconds=result['seconds_including_close']))
for r in map(json.loads,(P/'planning/phase_value_dataset_v1_20261010/TRANSITIONS.jsonl').read_text().splitlines()):
    for name in ('observation','next_observation','source_response','source_command'):
        ref=r.get(name)
        if ref is not None:
            raw=(P/ref['path']).read_bytes()
            assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256']
feat=json.loads((P/'planning/phase_value_features_v1_20261010/HOST_RESULT.json').read_text())
training=json.loads((model.parent/'RESULT.json').read_text());trainhost=json.loads((model.parent.parent/'HOST_RESULT.json').read_text())
parent=json.loads((P/'planning/phase_iql_repeat_v1_20261010/EXECUTION.json').read_text())
startups=[json.loads((P/'runs'/f'phase_iql_repeat_gpu{i}_v1_20261010/STARTUP.json').read_text()) for i in range(8)]
benefit=json.loads((P/'planning/phase_iql_repeat_v1_20261010/BENEFIT_CHECK.json').read_text())
cost=dict(feature_replay_native_inferences=data['unique_requests'],feature_replay_parallel_host_seconds=feat['parallel_seconds'],
    feature_replay_sum_shard_host_seconds=sum(r['seconds'] for r in feat['shards']),feedback_training_seconds=training['seconds'],
    initial_training_seconds=json.loads((P/'runs/phase_iql_training_v1_20261010/training/RESULT.json').read_text())['seconds'],
    training_host_seconds=trainhost['full_wall_seconds'],pilot_parallel_host_seconds=parent['parallel_host_seconds'],
    policy_startup_including_probe_sum_seconds=sum(r['seconds'] for r in startups),
    reuse_of_previous_development_data_is_not_free=True,historical_source_episodes=len(historical),
    historical_source_episode_seconds=sum(r['seconds'] for r in historical),
    historical_policy_startup_and_old_learning_costs_not_included_in_this_subtotal=True,
    full_cold_start_net_gain_established=False,
    warning='Different wall-time/compute scopes cannot be added or treated as complete improvement-cost saving. Historical collection costs remain additional.')
result=dict(status='AUDITED_COMPLETED_PILOT',commands=checked,cases=rows,arms=summary,first_candidates=pairs,benefit=benefit,
    costs=cost,models={k:v.sha256 for k,v in selectors.items()},complete_agent_comparison=False,cross_round_update_compared=False,
    phase_goal_complete=False,audit_seconds=time.perf_counter()-start)
(O/'AUDIT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(status=result['status'],arms=summary,benefit=benefit)),flush=True)
