"""One preregistered grouped diagnostic, no sweep or execution authority."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from fastrsi.contextual_capability_v1 import fit, predict, command_is_in_scope, save_model

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT/'runs/capability_corpus_v4_20261006'


def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    with p.open('x') as f: json.dump(value, f, indent=2, allow_nan=False)
def dist(a, b):
    return np.maximum(0, (a*a).sum(1)[:,None]+(b*b).sum(1)[None,:]-2*a@b.T)


def main():
    recipe_path = ROOT/'planning/capability_corpus_v2_20261006/REGISTRATION_v2.json'
    recipe, reg, result = read(recipe_path), read(RUN/'inputs/REGISTRATION.json'), read(RUN/'job/RESULT.json')
    assert result['status']=='COMPLETED' and result['all_rng_unchanged']
    assert result['decoder_calls']==result['robot_steps']==0
    assert result['encoder_calls']==len(reg['samples'])
    assert sha(RUN/'job/FEATURES.npy')==result['feature_sha256']
    assert reg['recipe_sha256']==sha(recipe_path)
    rows = reg['samples']; x = np.load(RUN/'job/FEATURES.npy').astype(np.float64)
    assert x.shape==(len(rows),65544)
    train = read(RUN/'TRAIN_LABELS.json')
    query_indices = [i for i,r in enumerate(rows) if r['group'] in recipe['diagnostic_groups']]
    assert all(rows[i]['id'] not in train for i in query_indices)
    out = ROOT/'planning/capability_corpus_v2_20261006/fit_v1'
    out.mkdir(exist_ok=False)
    start=time.perf_counter(); versions, predictions = {}, {}
    for kind, groups in [('original_three', [26,27,29]),
                         ('expanded_eight', recipe['expanded_train_groups']),
                         ('fixed_nearest', recipe['expanded_train_groups'])]:
        indices = [i for i,r in enumerate(rows) if r['group'] in groups and train[r['id']]['closed_only']]
        negatives = [i for i,r in enumerate(rows) if r['group'] in groups and not train[r['id']]['closed_only']]
        y = np.asarray([train[rows[i]['id']]['actions'] for i in indices])
        assert all(command_is_in_scope(a) for a in y)
        tick=time.perf_counter()
        if kind=='fixed_nearest':
            model=dict(prototype_features=x[indices],prototype_actions=y)
            prediction=y[dist(x[query_indices],x[indices]).argmin(1)].astype(np.float32)
        else:
            model=fit(x[indices],y,.01); prediction=predict(model,x[query_indices])
        fit_seconds=time.perf_counter()-tick
        model.update(positive_scope_features=x[indices],negative_scope_features=x[negatives])
        artifact=out/(kind+'.npz'); digest=save_model(artifact,model)
        dp=dist(x[query_indices],x[indices]).min(1)
        dn=dist(x[query_indices],x[negatives]).min(1)
        predictions[kind]=[dict(id=rows[i]['id'], group=rows[i]['group'],actions=a.tolist(),
            proposed=bool(dp[j]<dn[j] and command_is_in_scope(a)))
            for j,(i,a) in enumerate(zip(query_indices,prediction))]
        versions[kind]=dict(sha256=digest,artifact=artifact.name,groups=groups,samples=len(indices),
            fit_seconds=fit_seconds,negative_samples=len(negatives),authority='OFFLINE_UNVALIDATED_COMPONENT')
    save(out/'VERSIONS.json',versions)
    save(out/'PREDICTIONS.json',predictions)
    # Persist every prediction before opening any diagnostic target.
    prediction_digest=sha(out/'PREDICTIONS.json')
    targets=read(RUN/'DIAGNOSTIC_LABELS.json')
    metrics={}
    for kind, ps in predictions.items():
        per_group=[]
        for group in recipe['diagnostic_groups']:
            gr=[r for r in ps if r['group']==group]
            closed=[r for r in gr if targets[r['id']]['closed_only']]
            residual=np.asarray([np.asarray(r['actions'])-np.asarray(targets[r['id']]['actions']) for r in closed])
            per_group.append(dict(group=group,closed_samples=len(closed),
                translation_rmse=float(np.sqrt(np.mean(residual[:,:,:3]**2))),
                rotation_rmse=float(np.sqrt(np.mean(residual[:,:,3:6]**2))),
                command_rmse=float(np.sqrt(np.mean(residual**2))),
                release_proposed=sum(r['proposed'] and not targets[r['id']]['closed_only'] for r in gr),
                closed_rejected=sum(not r['proposed'] for r in closed)))
        metrics[kind]=dict(per_group=per_group,
            macro_translation_rmse=float(np.mean([g['translation_rmse'] for g in per_group])),
            macro_command_rmse=float(np.mean([g['command_rmse'] for g in per_group])),
            release_proposed=sum(g['release_proposed'] for g in per_group))
    expanded=metrics['expanded_eight']
    favorable=all(expanded['macro_translation_rmse']<metrics[k]['macro_translation_rmse'] and
                  expanded['macro_command_rmse']<metrics[k]['macro_command_rmse'] and
                  expanded['release_proposed']<=metrics[k]['release_proposed']
                  for k in ['original_three','fixed_nearest'])
    report=dict(status='OFFLINE_GROUPED_DIAGNOSTIC_COMPLETE',versions=versions,metrics=metrics,
        prediction_sha256_before_targets=prediction_digest,seconds=time.perf_counter()-start,
        preliminary_prediction_signal=favorable,new_live_authority=False,new_robot_episodes=0,
        learned_operation_value=False,economic_goal_achieved=False,
        interpretation='Grouped known-development behavior matching only. One observed action is not the only correct action; lower error is not evidence of fewer decisions, equal success or net savings.',
        next_action='Inspect group errors before any new bounded real-operation registration' if favorable else
                    'Do not run expanded ridge robot candidate; the preregistered data-only change lacks even predictive support.')
    save(out/'RESULT.json',report)
    print(json.dumps(report))


if __name__=='__main__': main()
