"""Pre-registered group-held-out pair-ranking probe; no online permissions."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np

P = Path(__file__).resolve().parents[1]
B = P/'planning/action_expert_pair_probe_v1_20261010'
registration = P/'planning/action_expert_pair_probe_registration_v1_20261010/REGISTRATION.json'
assert json.loads(registration.read_text())['status'] == 'REGISTERED_BEFORE_EXPERT_FEATURE_RESULTS'
B.mkdir(exist_ok=False)
started = time.perf_counter()
rows = json.loads((P/'planning/native_quality_cost_pairs_v2_20261010/PAIRS.json').read_text())
features_root = P/'planning/action_expert_feature_check_v1_20261010'
assert json.loads((features_root/'RESULT.json').read_text())['status'] == 'PASSED_ACTION_IDENTITY_AND_CANDIDATE_FEATURE_AUDIT'
inputs = np.load(P/'planning/native_quality_cost_pairs_v2_20261010/INPUTS.npz')
old = np.load(P/'planning/native_paired_q_update_v1_20261010/training/paired_router.npz')
print('frozen_keys', old.files, flush=True)
selected = []
expert = []; action = []; y = []; groups = []
for row in rows:
    pref = row['quality_preference']
    if pref is None:
        pref = row['cost_preference_given_both_success']
    if pref is None:
        continue
    i = row['pair']; tokens = np.load(features_root/f'pair_{i:02d}.npz')['action_tokens'].astype(np.float64)
    # Per-token standardization preserves position/order. No future statistics.
    tokens = (tokens-tokens.mean(-1,keepdims=True))/np.maximum(tokens.std(-1,keepdims=True),1e-6)
    expert.append((tokens[1]-tokens[0]).reshape(-1))
    candidates = inputs[f'candidates_{i}'].astype(np.float64)
    # Exactly the previous training action scale, frozen before new feedback.
    scale = old['action_std'].reshape(10,7)
    action.append(((candidates[1]-candidates[0])/scale).reshape(-1))
    y.append(1. if pref==1 else -1.)
    groups.append(row['development_group']);selected.append(i)
y = np.asarray(y); groups = np.asarray(groups)
out = []
for name, values in [('raw_action_difference',action),('final_action_expert_difference',expert)]:
    x = np.stack(values)
    norms = np.linalg.norm(x,axis=1)
    assert np.isfinite(x).all() and (norms>0).all()
    x = x/norms[:,None]
    results = []
    for group in sorted(set(groups)):
        train = groups != group; test = ~train
        train_groups = groups[train]
        weights = np.array([1./np.sum(train_groups==g) for g in train_groups])
        root = np.sqrt(weights)
        design = x[train]*root[:,None]
        alpha = np.linalg.solve(design@design.T + np.eye(len(design)),y[train]*root)
        scores = x[test]@design.T@alpha
        for index,score in zip(np.flatnonzero(test),scores):
            results.append(dict(pair=selected[index],held_out_group=int(group),score=float(score),
                selected=int(score>0),observed_preference=int(y[index]>0),correct=bool((score>0)==(y[index]>0))))
    group_accuracy = [np.mean([r['correct'] for r in results if r['held_out_group']==g]) for g in sorted(set(groups))]
    # In-sample fit can show memorization; never deployment evidence.
    weights = np.array([1./np.sum(groups==g) for g in groups]);root=np.sqrt(weights);design=x*root[:,None]
    alpha = np.linalg.solve(design@design.T+np.eye(len(design)),y*root)
    np.savez_compressed(B/(name+'.npz'),training_normalized_differences=x,alpha=alpha,root_weights=root)
    out.append(dict(representation=name,dimension=x.shape[1],held_out_rows=results,
        pair_correct=sum(r['correct'] for r in results),pairs=len(results),
        group_macro_accuracy=float(np.mean(group_accuracy)),
        fitted_training_correct=int(np.sum((x@design.T@alpha>0)==(y>0))),
        group_identifiers_used_only_for_split_and_balance=True))
result=dict(status='ACTUAL_FIXED_LINEAR_PAIR_PROBE',results=out,informative_pairs=len(y),
    informative_groups=len(set(groups)),lambda_ridge=1,zero_score_selects_default=True,
    training_seconds=time.perf_counter()-started,robot_executions=0,pi_updates=0,
    calibration_or_success_probability=False,independent_test=False,goal_complete=False,
    registration_sha256=hashlib.sha256(registration.read_bytes()).hexdigest())
(B/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
