"""Immutable real semi-MDP data; no invented alternatives or imitation labels."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb

LEGAL=('observation/image','observation/wrist_image','observation/state','prompt')
PROMPT='stack the right bowl on the left bowl and place them in the tray'
def export(project, out):
    out.mkdir(exist_ok=False)
    rows=[];episodes=[];requests={}
    def ref(path):
        raw=path.read_bytes()
        return dict(path=str(path.relative_to(project)),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    def register(r):
        raw=(project/r['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==r['sha256']
        obs=unpackb(raw)
        assert obs['prompt']==PROMPT
        for key in LEGAL[:2]:assert np.asarray(obs[key]).shape==(256,256,3) and np.asarray(obs[key]).dtype==np.uint8
        assert np.asarray(obs[LEGAL[2]]).shape==(8,) and np.isfinite(obs[LEGAL[2]]).all()
        # Digest the original request; case tags are excluded from model input.
        requests.setdefault(r['sha256'],r)
    old=project/'planning/bowl_value_records_v2_20261009'
    for record in map(json.loads,(old/'TRANSITIONS.jsonl').read_text().splitlines()):
        response=unpackb((project/record['action_proposal']['path']).read_bytes())
        action=np.asarray(response['actions'],np.float32)[:10]
        assert action.shape==(10,7)
        rows.append(dict(episode=record['episode_id'],group=record['initial_state_group'],chunk=record['chunk'],
            observation=record['observation'],next_observation=record['next_observation'],planned_actions=action.tolist(),
            n=record['executed_action_prefix_length'],budget=record['remaining_execution_budget'],
            terminal=record['terminal'],terminal_reason=record['terminal_reason'],reward=record['native_terminal_reward'],
            observed_episode_success=record['observed_episode_return'],source_response=record['action_proposal'],
            counterfactual_label=None))
    episodes.extend(json.loads((old/'EPISODES.json').read_text()))
    roots=[project/'runs'/f'bowl_candidates_gpu{i}_v1_20261009' for i in range(2)]
    roots += [project/'runs'/f'bowl_value_selection_gpu{i}_v1_20261009' for i in range(3)]
    roots += [project/'runs'/f'bowl_candidate_mean_gpu{i}_v1_20261009' for i in range(2)]
    roots += [project/'runs'/f'bowl_mean_validation_gpu{i}_v1_20261010' for i in range(4)]
    for root in roots:
        agg=json.loads((root/'AGGREGATE.json').read_text());assert agg['status']=='COMPLETED'
        for case in agg['cases']:
            folder=root/case['case'];result=json.loads((folder/'RESULT.json').read_text())
            assert result['status']=='COMPLETED'
            trajectory=result['rows'];assert trajectory and (result['native_success'] or result['executed_actions']==400)
            eid=str(folder.relative_to(project));group='init_'+str(result['init_state_id'])
            episodes.append(dict(id=eid,initial_state_group=group,native_success=bool(result['native_success']),
                transition_count=len(trajectory),source_result=ref(folder/'RESULT.json')))
            for j,r in enumerate(trajectory):
                assert r['chunk']==j and (not j or trajectory[j-1]['step_after']==r['step_before'])
                response=ref(folder/f'{j:03d}_response.msgpack');data=unpackb((project/response['path']).read_bytes())
                if (folder/f'{j:03d}_command.npz').exists():
                    action=np.load(folder/f'{j:03d}_command.npz',allow_pickle=False)['planned_actions']
                    command=ref(folder/f'{j:03d}_command.npz')
                else:action=np.asarray(data['candidates'])[r['selected_index']];command=None
                action=np.asarray(action,np.float32);assert action.shape==(10,7) and np.isfinite(action).all()
                terminal=j==len(trajectory)-1;n=r['step_after']-r['step_before'];assert 1<=n<=10
                rows.append(dict(episode=eid,group=group,chunk=j,observation=ref(folder/f'{j:03d}_request.msgpack'),
                    next_observation=ref(folder/f'{j+1:03d}_request.msgpack') if not terminal else None,
                    planned_actions=action.tolist(),n=n,budget=400-r['step_before'],terminal=terminal,
                    terminal_reason=('native_success' if result['native_success'] else 'finite_budget_exhausted') if terminal else None,
                    reward=int(terminal and result['native_success']),observed_episode_success=int(result['native_success']),
                    source_response=response,source_command=command,counterfactual_label=None))
    for r in rows:
        register(r['observation'])
        if r['next_observation']:register(r['next_observation'])
        assert (r['reward']==0 or r['terminal']) and r['n']<=r['budget']
        r['role']='diagnostic_validation' if r['group'] in ('demo_25','init_37','init_40') else 'train'
    assert len({(r['episode'],r['chunk']) for r in rows})==len(rows)
    assert sum(r['terminal'] for r in rows)==len(episodes)
    (out/'TRANSITIONS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    (out/'REQUESTS.json').write_text(json.dumps(list(requests.values()),indent=2)+'\n')
    (out/'EPISODES.json').write_text(json.dumps(episodes,indent=2)+'\n')
    report=dict(status='REAL_TRANSITIONS_EXPORTED',episodes=len(episodes),rows=len(rows),
        successes=sum(r['reward'] for r in rows),unique_requests=len(requests),
        partial_terminal_rows=sum(r['terminal'] and r['n']<10 for r in rows),
        roles={role:sum(r['role']==role for r in rows) for role in ('train','diagnostic_validation')},
        online_inputs=list(LEGAL)+['remaining_budget','planned_candidate_actions'],
        post_execution_n_is_label_only=True,terminal_tails_not_marked_executed=True,
        all_counterfactual_labels_null=True,hidden_simulator_state_inputs=False,
        validation_groups=['demo_25','init_37','init_40'],test_groups_not_yet_executed=[28,29,30,31],
        caveat='Previously exposed task and starts; future new start/noise combinations are pilot evidence, not an untouched benchmark.')
    (out/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();export(a.project,a.output)
