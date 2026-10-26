"""Use four audited actual branches to update the same external head, no robot rollout."""
import hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
from action_expert_ranker_v1 import ExpertPairRanker,difference
P=Path(__file__).resolve().parents[1];O=P/'planning/expert_verified_feedback_v2_20261010'
O.mkdir(exist_ok=False);tick=time.perf_counter()
spec=importlib.util.spec_from_file_location('codec',P/'third_party/PhyAgentOS-core/PhyAgentOS/runtime/policy/msgpack_numpy.py');codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
audit=P/'planning/expert_first_change_audit_v1_20261010/RESULT.json';a=json.loads(audit.read_text())
assert a['status']=='AUDITED_EXPERT_FIRST_CHANGE_DIAGNOSTICS' and a['robot_executions']==4
rows=json.loads((P/'planning/native_quality_cost_pairs_v2_20261010/PAIRS.json').read_text());tokens=[];records=[]
for r in rows:
 tokens.append(np.load(P/f"planning/action_expert_feature_check_v1_20261010/pair_{r['pair']:02d}.npz")['action_tokens'])
 records.append(dict(pair=r['pair'],group=r['development_group'],preference=r['quality_preference'] if r['quality_preference'] is not None else r['cost_preference_given_both_success'],source='previous ten real causal pairs',old_source=r))
for c in a['cases']:
 initial=c['initial'];plan_path=P/f'planning/expert_first_change_inputs_v1_20261010/init{initial}/PLAN.json';plan=json.loads(plan_path.read_text())
 reply=codec.unpackb((P/plan['source_response']).read_bytes());tokens.append(reply['action_expert_tokens'])
 records.append(dict(pair=len(records),group=initial,preference=c['local_preference'],source='audited one real change then default continuation',source_plan_sha256=sha(plan_path),audit_sha256=sha(audit),outcome=c))
registration=dict(status='REGISTERED_BEFORE_FEEDBACK_REFIT',pair_count=len(records),actual_new_pairs=4,new_informative_pairs=2,fit='same group-balanced dual ridge, lambda1, sign threshold0',continuation='all local supervision assumes unchanged default pi after one candidate; NOT whole reranking-policy labels',no_search=True,not_sampling_value_labels=True,no_pi_or_llm_updates=True,no_new_inferences_or_robot_runs=True)
(O/'REGISTRATION.json').write_text(json.dumps(registration,indent=2)+'\n')
(O/'PAIRS.json').write_text(json.dumps(records,indent=2)+'\n');np.savez_compressed(O/'INPUTS.npz',action_tokens=np.stack(tokens))
idx=[i for i,r in enumerate(records) if r['preference'] is not None];x=np.stack([difference(tokens[i]) for i in idx]);y=np.array([1. if records[i]['preference'] else -1. for i in idx]);g=np.array([records[i]['group'] for i in idx])
def fit(x,y,g):
 w=np.array([1./np.sum(g==z) for z in g]);root=np.sqrt(w)
 return root*np.linalg.solve((x*root[:,None])@(x*root[:,None]).T+np.eye(len(x)),y*root)
old=P/'planning/action_expert_ranker_training_v1_20261010/expert_ranker_R.npz'
with np.load(old,allow_pickle=False) as f:bundle={k:f[k].copy() for k in f.files}
bundle.update(expert_training_differences=x,expert_kernel_coefficients=fit(x,y,g));model=O/'expert_ranker_R2.npz';np.savez_compressed(model,**bundle)
before=ExpertPairRanker(old);after=ExpertPairRanker(model);changes=[]
for i in idx:
 s0,m0=before.choose(tokens[i]);s1,m1=after.choose(tokens[i]);changes.append(dict(pair=i,group=records[i]['group'],actual_preference=records[i]['preference'],before_choice=s0,after_choice=s1,before_score=m0['pair_preference_score'],after_score=m1['pair_preference_score']))
held=[]
for group in sorted(set(g)):
 train=g!=group;test=g==group;coef=fit(x[train],y[train],g[train]);scores=x[test]@x[train].T@coef
 for k,score in zip(np.where(test)[0],scores):held.append(dict(pair=idx[k],group=int(group),selected=int(score>0),actual=int(y[k]>0),correct=bool((score>0)==(y[k]>0))))
result=dict(status='REFIT_FROM_AUDITED_CAUSAL_FEEDBACK',model_sha256=after.sha256,source_model_sha256=before.sha256,informative_pairs=len(idx),groups=len(set(g)),development_changes=changes,group_held_out=held,group_held_out_correct=sum(r['correct'] for r in held),group_held_out_total=len(held),seconds=time.perf_counter()-tick,robot_executions=0,pi_inferences=0,llm_calls=0,default_unchanged=True,independent_quality_or_cost_gain=False,online_sampling_gate_not_qualified=True,goal_complete=False)
(O/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
