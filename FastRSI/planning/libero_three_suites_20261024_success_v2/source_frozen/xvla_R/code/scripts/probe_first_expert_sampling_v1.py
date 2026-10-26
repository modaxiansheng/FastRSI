"""Cheap prospective-input preference probe. No online gate or net-value labels."""
import hashlib,json,time
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parents[1];O=P/'planning/first_expert_sampling_probe_v1_20261010';O.mkdir(exist_ok=False)
reg=P/'planning/first_expert_sampling_probe_registration_v1_20261010/REGISTRATION.json'
assert json.loads(reg.read_text())['status']=='REGISTERED_BEFORE_FIRST_EXPERT_SAMPLING_FIT'
rows=json.loads((P/'planning/native_quality_cost_pairs_v2_20261010/PAIRS.json').read_text())
bindings=json.loads((P/'planning/action_expert_feature_inputs_v1_20261010/BINDINGS.json').read_text())
x=[];noise=[];y=[];group=[];indices=[];tick=time.perf_counter()
for r in rows:
    pref=r['quality_preference']
    if pref is None:pref=r['cost_preference_given_both_success']
    if pref is None:continue
    i=r['pair'];b=bindings[i]
    t=np.load(P/'planning/action_expert_feature_check_v1_20261010'/f'pair_{i:02d}.npz')['action_tokens'][0].astype(np.float64)
    t=((t-t.mean(-1,keepdims=True))/np.maximum(t.std(-1,keepdims=True),1e-6)).reshape(-1)
    x.append(t/np.linalg.norm(t))
    n=np.random.default_rng(np.random.SeedSequence([b['noise_base'],b['step']//10,1])).standard_normal((10,32)).astype(np.float32)
    assert hashlib.sha256(n.tobytes()).hexdigest()==b['noise_sha256'][1]
    n=n.astype(np.float64).reshape(-1);noise.append(n/np.linalg.norm(n))
    y.append(1. if pref else -1.);group.append(r['development_group']);indices.append(i)
x=np.stack(x);noise=np.stack(noise);y=np.array(y);group=np.array(group);results=[]
for view,design in [('first_expert_only',x),('first_expert_plus_known_noise',np.concatenate([x,noise],axis=1)/np.sqrt(2))]:
    decisions=[]
    for g in sorted(set(group)):
        train=group!=g;test=~train;weight=np.array([1./np.sum(group[train]==h) for h in group[train]]);root=np.sqrt(weight)
        z=design[train]*root[:,None]
        alpha=np.linalg.solve(z@z.T+np.eye(len(z)),y[train]*root)
        score=design[test]@z.T@alpha
        for j,s in zip(np.flatnonzero(test),score):decisions.append(dict(pair=indices[j],group=int(g),score=float(s),predicted_preference=int(s>0),observed_preference=int(y[j]>0),correct=bool((s>0)==(y[j]>0))))
    macro=float(np.mean([np.mean([d['correct'] for d in decisions if d['group']==g]) for g in sorted(set(group))]))
    results.append(dict(view=view,rows=decisions,pair_correct=sum(d['correct'] for d in decisions),pairs=len(decisions),group_macro_accuracy=macro))
r=dict(status='COMPLETED_PROSPECTIVE_INPUT_PREFERENCE_PROBE',results=results,seconds=time.perf_counter()-tick,
    groups=len(set(group)),always_default_pair_correct=int((y<0).sum()),always_default_macro_accuracy=.5,
    no_candidate1_features_used=True,known_noise_is_not_initial_id=True,
    labels_are_preference_not_net_sampling_utility=True,not_independent_test=True,online_gate_enabled=False,
    goal_complete=False,registration_sha256=hashlib.sha256(reg.read_bytes()).hexdigest())
(O/'RESULT.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
