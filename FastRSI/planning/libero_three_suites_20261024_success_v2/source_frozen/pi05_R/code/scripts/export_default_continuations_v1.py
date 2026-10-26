"""Use recorded single-candidate pi trajectories for factual quality/cost targets.

The completion outcome labels observed return under the recorded continuation;
it does not declare every action in a successful episode correct. No actor or
critic updates are performed. Non-single, adapted and unbound sources excluded.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from phase_iql_value_v1 import Predictor


def main(project, output):
    source=project/'planning/phase_value_dataset_v1_20261010'
    rows=[json.loads(x) for x in (source/'TRANSITIONS.jsonl').read_text().splitlines()]
    episodes=json.loads((source/'EPISODES.json').read_text())
    by_episode={}
    for row in rows:
        by_episode.setdefault(row['episode'],[]).append(row)
    verified=[];excluded=[];records=[];vectors=[];labels=[]
    featurebank={}
    for slot in range(4):
        for f in (project/f'planning/phase_value_features_v1_20261010/gpu{slot}/features').glob('*.npz'):
            featurebank[f.stem]=f
    model=Predictor(project/'planning/native_paired_q_update_v1_20261010/training/paired_router.npz')
    def resolve(relative):
        f=project/relative
        if not f.exists() and relative.startswith('runs/'):
            f=project/'planning'/relative[len('runs/'):]
        return f
    for episode in episodes:
        eid=episode['id'];result_path=resolve(episode['source_result']['path'])
        if not result_path.is_file():
            excluded.append(dict(episode=eid,reason='Source terminal receipt not locally available'))
            continue
        result=json.loads(result_path.read_text())
        if result.get('arm')!='single':
            excluded.append(dict(episode=eid,reason='Not verified single-candidate default continuation'))
            continue
        assert hashlib.sha256(result_path.read_bytes()).hexdigest()==episode['source_result']['sha256']
        assert result['status']=='COMPLETED'
        assert result['training_steps']==result['persistent_updates']==0
        meta=result['policy_metadata']
        assert meta['backend']=='openpi_native' and meta['policy_config']=='pi05_libero'
        assert meta['checkpoint_format']=='pytorch_safetensors' and meta['checkpoint_dir']=='/model'
        assert meta['action_horizon']==10 and meta['torch_seed']==7
        assert all(x==0 for x in result['selected_indices'])
        assert all(x['candidate_count']==1 and x['selected_index']==0 for x in result['rows'])
        erows=by_episode[eid]
        assert len(erows)==len(result['rows'])==episode['transition_count']
        assert len({r['role'] for r in erows})==1
        assert result['native_success']==episode['native_success']
        assert sum(r['n'] for r in erows)==result['executed_actions']
        parent=result_path.parent.parent
        launch=json.loads((parent/'POLICY_LAUNCH.json').read_text())
        mount=[x for x in launch if x.endswith(':/model:ro')]
        assert len(mount)==1 and mount[0].endswith('/weights/pi05_libero_pytorch_fp32:/model:ro')
        for i,(row,actual) in enumerate(zip(erows,result['rows'])):
            assert actual['chunk']==row['chunk']==i
            assert actual['step_after']-actual['step_before']==row['n']
            assert row['terminal']==(i==len(erows)-1)
            observation=resolve(row['observation']['path'])
            assert hashlib.sha256(observation.read_bytes()).hexdigest()==row['observation']['sha256']
            z=dict(np.load(featurebank[row['observation']['sha256']],allow_pickle=False))
            state=model.state(z['feature'],z['state'],row['budget'])
            action=(np.asarray(row['planned_actions'],np.float32).ravel()-model.a['action_mean'])/model.a['action_std']
            vectors.append(np.concatenate([state,action]))
            remaining=sum(x['n'] for x in erows[i:])
            success=int(result['native_success'])
            labels.append([success,remaining/400.,float(success)])
            records.append(dict(episode=eid,group=row['group'],role=row['role'],chunk=i,
                observation_sha256=row['observation']['sha256'],budget=row['budget'],
                observed_native_success=success,observed_remaining_actions=remaining,
                cost_target_used_only_when_success=bool(success),
                completion_label_is_return_not_action_correctness=True))
        verified.append(dict(episode=eid,rows=len(erows),group=erows[0]['group'],role=erows[0]['role'],
            native_success=result['native_success'],source_result_sha256=episode['source_result']['sha256'],
            actor_mount=mount[0],policy_launch_sha256=hashlib.sha256((parent/'POLICY_LAUNCH.json').read_bytes()).hexdigest()))
    assert verified and len(vectors)==len(records)
    output.mkdir(exist_ok=False)
    np.savez_compressed(output/'SUPERVISION.npz',inputs=np.stack(vectors).astype(np.float32),
        targets=np.asarray(labels,np.float32),train=np.array([r['role']=='train' for r in records]))
    (output/'RECORDS.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
    result=dict(status='VERIFIED_DEFAULT_PI_CONTINUATIONS_EXPORTED',episodes=len(verified),rows=len(records),
        train_rows=sum(r['role']=='train' for r in records),
        diagnostic_validation_rows=sum(r['role']!='train' for r in records),
        successes=sum(e['native_success'] for e in verified),
        groups=sorted({r['group'] for r in records}),sources=verified,excluded_sources=excluded,
        representation_model_sha256=model.sha256,new_feature_encodings=0,robot_executions=0,model_updates=0,
        no_discount_in_completion_target=True,no_failure_speed_label=True,
        action_count_is_cost_proxy_not_wall_time=True,goal_complete=False,
        limitation='Single sampled continuation outcomes, not calibrated action values or paired alternative labels. Trajectory groups remain isolated; no performance claim.')
    (output/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','excluded_sources')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    main(args.project,args.output)
