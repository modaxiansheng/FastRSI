"""Bind four new diagnostic forks to the actual R choice and unchanged N prefix."""
import hashlib,json,time
from pathlib import Path
import importlib.util
import numpy as np
from action_expert_ranker_v1 import ExpertPairRanker
P=Path(__file__).resolve().parents[1]
O=P/'planning/expert_first_change_inputs_v1_20261010';O.mkdir(exist_ok=False)
s=importlib.util.spec_from_file_location('codec',P/'third_party/PhyAgentOS-core/PhyAgentOS/runtime/policy/msgpack_numpy.py')
codec=importlib.util.module_from_spec(s);s.loader.exec_module(codec)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return codec.unpackb(p.read_bytes())
cases=[]
for version in range(16,20):
    base=P/'planning'/f'native_value_phase_v{version}_20261010'
    assert json.loads((base/'STATUS.json').read_text())['status']=='COMPLETED'
    reg=json.loads((base/'REGISTRATION.json').read_text());initial=reg['initial']
    streams={a:base/a/'stream/value_service' for a in ['N','R']}
    calls={a:[json.loads(x) for x in (d/'VALUE_POLICY_CALLS.jsonl').read_text().splitlines()] for a,d in streams.items()}
    changed=next(r for r in calls['R'] if r['selected_index']==1)
    index=changed['index'];assert changed['step']==index*10
    prefix=[]
    for idx in range(index+1):
        n,r=calls['N'][idx],calls['R'][idx];assert n['index']==r['index']==idx
        assert n['step']==r['step']==idx*10
        requests={a:read(streams[a]/(calls[a][idx]['evidence_name']+'_request.msgpack')) for a in ['N','R']}
        replies={a:read(streams[a]/(calls[a][idx]['evidence_name']+'_response.msgpack')) for a in ['N','R']}
        assert all(np.array_equal(requests['N'][k],requests['R'][k]) for k in ['observation/image','observation/wrist_image','observation/state'])
        assert requests['N']['prompt']==requests['R']['prompt']
        assert np.array_equal(replies['N']['actions'],replies['R']['candidates'][0])
        if idx<index:
            assert r['selected_index']==0
            prefix.append(replies['N']['actions'].tolist())
        else:
            model=base/'VALUE_MODEL_R.npz';pred=ExpertPairRanker(model)
            selected,details=pred.choose(replies['R']['action_expert_tokens']);assert selected==1
            assert pred.sha256==r['model_sha256']
            assert abs(details['pair_preference_score']-r['choice_meta']['pair_preference_score'])<1e-9
            candidates=replies['R']['candidates']
    paths=dict(source_request=streams['R']/(changed['evidence_name']+'_request.msgpack'),
        source_response=streams['R']/(changed['evidence_name']+'_response.msgpack'),
        source_default_response=streams['N']/(calls['N'][index]['evidence_name']+'_response.msgpack'),
        default_outcome=base/'N/stream/RESULT.json',full_learned_outcome=base/'R/stream/RESULT.json',model=model)
    plan=dict(initial=initial,fork_step=changed['step'],noise_base=changed['noise_base'],prefix=prefix,
        original_R_selected_index=1,all_prior_R_actions_equal_default=True,matched_current_inputs=True,
        model_scores=details,scope='One actual R action choice then unchanged pi; diagnostic only, no global one-intervention policy adoption.')
    for k,p in paths.items():plan[k]=str(p.relative_to(P));plan[k+'_sha256']=sha(p)
    folder=O/f'init{initial}';folder.mkdir();np.savez_compressed(folder/'CANDIDATES.npz',candidates=candidates)
    (folder/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n')
    cases.append(dict(initial=initial,fork_step=changed['step'],model_sha256=pred.sha256,source_plan_sha256=sha(folder/'PLAN.json')))
(O/'RESULT.json').write_text(json.dumps(dict(status='PREPARED_EXPERT_FIRST_CHANGE_DIAGNOSTICS',cases=cases,
    maximum_followup_actions=1600,maximum_prefix_replay_actions=sum(c['fork_step'] for c in cases),
    new_robot_executions=0,online_ids_not_used=True,stage_goal_unchanged=True),indent=2)+'\n')
print(json.dumps(cases),flush=True)
