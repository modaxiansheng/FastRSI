"""Read only pre-existing development IDs in our own copied archive."""
import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args()
    project=args.project.resolve()
    archive=project.parent/'evaluation/table4-full-20260929-v1'
    out=project/'planning/archive_development_v1_20261004';out.mkdir(exist_ok=False)
    split=project/'planning/representative_coverage_v1/RESERVED_SPLIT.json'
    pools=json.loads(split.read_text())['tasks'];rows=[];sources={};started=time.time()
    sources[str(split)]=hashlib.sha256(split.read_bytes()).hexdigest()
    for pool in pools:
        tid=pool['task_id']
        for iid in pool['held_development_init_ids']:
            assert iid not in pool['reserved_final_method_test_init_ids']
            # Resolve IDs before reading any RESULT. Never enumerate test outcomes.
            path=archive/'output'/f'pi05-libero_10-t{tid:02d}'/f'libero_10-t{tid}-i{iid}'/'RESULT.json'
            data=path.read_bytes();x=json.loads(data)
            assert x['model']=='pi05' and x['task']['suite']=='libero_10'
            assert (x['task']['task_id'],x['task']['init_state_id'])==(tid,iid)
            assert x['execution_status']=='completed' and not x.get('diagnostic_only')
            assert x['success_authority']=='libero_native',x['success_authority']
            attempts=[]
            for a in x['attempts']:
                summary=a['episode_summary'];native=summary['success']
                assert type(native) is bool and native==a['runtime_success']==a['success']
                attempts.append(dict(index=a['attempt_index'],native_success=native,steps=a['num_steps'],
                    policy_calls=summary['num_action_chunks'],instruction=a['task_description']))
            assert x['first_success']==attempts[0]['native_success'] and x['final_success']==attempts[-1]['native_success']
            sources[str(path)]=hashlib.sha256(data).hexdigest()
            rows.append(dict(task=tid,init=iid,source=str(path),sha256=sources[str(path)],attempts=attempts,
                first_success=x['first_success'],final_success=x['final_success'],environment_resets=x['environment_resets'],
                recovery_attempts=x['recovery_attempts'],verifier_calls=x['verifier_calls'],archive_wall_seconds=x['elapsed_seconds']))
    result=dict(status='COMPLETED',seconds=time.time()-started,rows=rows,
        summary=dict(cases=len(rows),first_failures=sum(not r['first_success'] for r in rows),
            recovered=sum(not r['first_success'] and r['final_success'] for r in rows)),
        scope='Historical copied execution/recovery archive; development discovery only. Not a current full-Agent result or transferable-skill label.',
        final_test_results_read=0,new_model_calls=0,new_robot_episodes=0)
    (out/'SOURCE_MANIFEST.json').write_text(json.dumps(sources,indent=2))
    (out/'RESULT.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result['summary']))


if __name__=='__main__':main()
