"""Initial five actual preferences then two new-context feedback preferences."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from action_expert_ranker_v1 import ExpertPairRanker,difference
P=Path(__file__).resolve().parents[1]
B=P/'planning/action_expert_ranker_training_v1_20261010'
B.mkdir(exist_ok=False)
tick=time.perf_counter()
old=P/'planning/phase_iql_feedback_v1_20261010/training/frozen_router.npz'
oldsha=hashlib.sha256(old.read_bytes()).hexdigest()
assert oldsha=='428fb35dc588458178f06865ddc9d2963f33880126ddd70f7f84ba8ec2ea44c0'
rows=json.loads((P/'planning/native_quality_cost_pairs_v2_20261010/PAIRS.json').read_text())
source=P/'planning/action_expert_feature_check_v1_20261010'
assert json.loads((source/'RESULT.json').read_text())['status']=='PASSED_ACTION_IDENTITY_AND_CANDIDATE_FEATURE_AUDIT'
# Frozen old Q/V are retained only as a common sampling trigger, never updated
# by the new ranking heads. Neither arm compares new scores to that V.
registration=dict(status='REGISTERED_BEFORE_F_AND_R_FITS',F_source_pairs=list(range(6)),
    R_source_pairs=list(range(10)),algorithm='Group-balanced dual ridge lambda1; existing action-expert pair differences',
    interpretation='Lexicographic actual native completion / successful remaining-action preference',
    common_sampling_trigger_sha256=oldsha,threshold=0.,no_hyperparameter_search=True,
    no_pi_or_llm_updates=True,default_unchanged=True)
(B/'REGISTRATION.json').write_text(json.dumps(registration,indent=2)+'\n')
out=[]
for arm,limit in [('F',6),('R',10)]:
    indices=[];groups=[];y=[];features=[]
    for r in rows[:limit]:
        pref=r['quality_preference']
        if pref is None:pref=r['cost_preference_given_both_success']
        if pref is None:continue
        features.append(difference(np.load(source/f"pair_{r['pair']:02d}.npz")['action_tokens']))
        indices.append(r['pair']);groups.append(r['development_group']);y.append(1. if pref else -1.)
    x=np.stack(features);groups=np.asarray(groups);y=np.asarray(y)
    weights=np.array([1./np.sum(groups==g) for g in groups]);root=np.sqrt(weights)
    alpha=np.linalg.solve((x*root[:,None])@(x*root[:,None]).T+np.eye(len(x)),y*root)
    with np.load(old,allow_pickle=False) as f:bundle={k:f[k].copy() for k in f.files}
    bundle.update(expert_training_differences=x,expert_kernel_coefficients=root*alpha)
    model=B/f'expert_ranker_{arm}.npz';np.savez_compressed(model,**bundle)
    ranker=ExpertPairRanker(model)
    decisions=[]
    for r in rows:
        selected,meta=ranker.choose(np.load(source/f"pair_{r['pair']:02d}.npz")['action_tokens'])
        decisions.append(dict(pair=r['pair'],selected=selected,**meta))
    assert all(np.array_equal(bundle[k],np.load(old)[k]) for k in np.load(old).files)
    out.append(dict(arm=arm,informative_training_pairs=indices,informative_groups=len(set(groups)),
        model_sha256=ranker.sha256,development_choices=decisions))
result=dict(status='ACTUAL_FROZEN_AND_FEEDBACK_UPDATED_EXPERT_PAIR_FITS',arms=out,
    seconds=time.perf_counter()-tick,pi_updates=0,llm_updates=0,robot_executions=0,
    sampling_q_and_v_exactly_frozen=True,independent_gain=False,goal_complete=False)
(B/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
