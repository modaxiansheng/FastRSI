"""Verified observation/action transitions for value learning; no automatic imitation."""
import hashlib,json
from pathlib import Path
LEGAL_KEYS=('observation/image','observation/wrist_image','observation/state','prompt')

def read_transition(project, record):
    import numpy as np
    from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
    project=Path(project).resolve()
    def payload(ref):
        f=(project/ref['path']).resolve()
        assert project in f.parents and f.is_file()
        b=f.read_bytes();assert len(b)==ref['bytes'] and hashlib.sha256(b).hexdigest()==ref['sha256']
        return unpackb(b)
    def observation(ref):
        data=payload(ref);current={key:data[key] for key in LEGAL_KEYS}
        for key in LEGAL_KEYS[:2]:
            a=np.asarray(current[key]);assert a.shape==(256,256,3) and a.dtype==np.uint8
        state=np.asarray(current['observation/state']);assert state.shape==(8,) and np.isfinite(state).all()
        assert current['prompt']=='stack the right bowl on the left bowl and place them in the tray'
        return current
    current=observation(record['observation'])
    proposed=np.asarray(payload(record['action_proposal'])['actions'])
    n=record['executed_action_prefix_length'];assert 1<=n<=10 and proposed.ndim==2 and proposed.shape[1]==7 and len(proposed)>=n
    executed=proposed[:n].copy();assert np.isfinite(executed).all()
    terminal=record['terminal'];reward=record['native_terminal_reward']
    assert reward in (0,1) and (terminal or reward==0)
    assert not record['behavior_cloning_eligible'] and record['counterfactual_branch_label'] is None
    if terminal:
        assert record['next_observation'] is None
        assert record['terminal_reason'] in ('native_success','finite_budget_exhausted')
        next_current=None
    else:
        assert record['next_observation'] is not None;next_current=observation(record['next_observation'])
    # Future outcomes and costs are returned only as labels, never inside either observation.
    labels={key:record[key] for key in ('terminal','native_terminal_reward','observed_episode_return',
        'observed_remaining_actions','observed_remaining_rpc_seconds','observed_remaining_environment_seconds')}
    return dict(observation=current,executed_actions=executed,next_observation=next_current,
        labels=labels,remaining_execution_budget=record['remaining_execution_budget'],
        imitation_eligible=False)

def main():
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--dataset',type=Path,required=True);args=ap.parse_args()
    rows=[json.loads(s) for s in (args.dataset/'TRANSITIONS.jsonl').read_text().splitlines()]
    episodes=json.loads((args.dataset/'EPISODES.json').read_text());assert len(episodes)==15
    terminals=0;successes=0
    for row in rows:
        value=read_transition(args.project,row);assert set(value['observation'])==set(LEGAL_KEYS)
        terminals+=int(value['labels']['terminal']);successes+=value['labels']['native_terminal_reward']
    assert terminals==len(episodes) and successes==sum(x['native_success'] for x in episodes)
    print(json.dumps(dict(status='VERIFIED_ACTUAL_TRANSITIONS',transitions=len(rows),episodes=len(episodes),
        success_terminal_rewards=successes,failed_episodes=terminals-successes,
        all_observations_legal=True,all_imitation_admissions_disabled=True,training_steps=0)))
if __name__=='__main__':main()
