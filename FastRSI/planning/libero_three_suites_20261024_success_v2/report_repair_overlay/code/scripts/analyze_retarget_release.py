"""Audit the preregistered local release endpoint; never infer whole-task success."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def analyze(root, candidate=None):
    folder=root/'planning/retarget_release_20261005'
    plan=json.loads((folder/'PLAN.json').read_text())
    for rel,digest in plan['source_sha256'].items():
        assert sha(root/rel)==digest, rel
    result_path=folder/'cached/RESULT.json'
    result=json.loads(result_path.read_text())
    assert result['status']=='COMPLETED' and result['steps']==plan['steps']==130
    assert [x['step'] for x in result['rows']]==list(range(1,131))
    row=result['rows'][-1];bowl=row['bowls']['akita_black_bowl_1']
    checks={
        'bottom_in_tray':row['native_bottom_in_tray'],
        'no_grasp':not bowl['native_grasp'],
        'no_contact':not bowl['gripper_contact'],
        'both_fingers_open':all(abs(x)>0.03 for x in row['gripper_qpos']),
        'open_command':row['action'][-1]<0,
    }
    images={name:sha(folder/'cached'/name)==digest for r in result['rows'] for name,digest in r.get('images',{}).items()}
    assert len(images)==6 and all(images.values())
    out=dict(result_sha256=sha(result_path), endpoint=130, acceptance=checks,
             local_stage_passed=all(checks.values()), full_task_success=all(x['value'] for x in row['goal_predicates']),
             privileged_geometry=True, online_admission=False, learning_benefit_established=False,
             steps=130, model_calls=result['model_calls'], learning_updates=result['persistent_learning_updates'],
             result_seconds=result['seconds'], image_hashes_checked=len(images),
             original_exit=json.loads((folder/'STOP.json').read_text())[0]['exit_code'])
    if candidate is not None:
        other=json.loads(candidate.read_text())
        assert other['status']=='COMPLETED' and other['steps']==130
        for r in other['rows']:
            for name, digest in r.get('images',{}).items():
                assert sha(candidate.parent/name)==digest, name
        out['loader_comparison']=dict(candidate_sha256=sha(candidate),
            initial_identical=result['initial']==other['initial'],
            rows_identical=result['rows']==other['rows'],
            update_source_identical=result['update_source']==other['update_source'],
            seconds=other['seconds'])
    return out

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--candidate',type=Path)
    a=p.parse_args();print(json.dumps(analyze(a.root,a.candidate),indent=2))
