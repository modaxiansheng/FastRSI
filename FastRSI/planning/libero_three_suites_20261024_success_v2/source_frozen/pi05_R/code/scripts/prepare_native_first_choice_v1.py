"""Bind the first learned action change to an exactly shared factual prefix."""
import hashlib,json
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from phase_iql_feedback_update_v1 import AdaptivePredictor
P=Path('/project');O=Path('/check')
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
def load(f):return json.loads(f.read_text())
def main():
    model=P/'planning/native_paired_q_update_v1_20261010/training/paired_router.npz';pred=AdaptivePredictor(model);cases=[]
    for version in (13,14):
        root=P/'planning'/f'native_value_phase_v{version}_20261010';initial=load(root/'REGISTRATION.json')['initial']
        assert load(root/'STATUS.json')['status']=='COMPLETED'
        n=root/'N/stream';r=root/'R/stream'
        nc=list(map(json.loads,(n/'value_service/VALUE_POLICY_CALLS.jsonl').read_text().splitlines()))
        rc=list(map(json.loads,(r/'value_service/VALUE_POLICY_CALLS.jsonl').read_text().splitlines()))
        choice=next(x for x in rc if x['selected_index']==1);index=choice['index'];step=choice['step']
        for i in range(index+1):
            a=unpackb((n/'value_service'/(nc[i]['evidence_name']+'_request.msgpack')).read_bytes())
            b=unpackb((r/'value_service'/(rc[i]['evidence_name']+'_request.msgpack')).read_bytes())
            for key in ('observation/image','observation/wrist_image','observation/state'):assert np.array_equal(a[key],b[key])
            assert a['prompt']==b['prompt']
            if i<index:assert rc[i]['selected_index']==0
        request=r/'value_service'/(choice['evidence_name']+'_request.msgpack')
        response=r/'value_service'/(choice['evidence_name']+'_response.msgpack')
        obs=unpackb(request.read_bytes());resp=unpackb(response.read_bytes())
        nr=n/'value_service'/(nc[index]['evidence_name']+'_response.msgpack');default=unpackb(nr.read_bytes())
        assert np.array_equal(resp['candidates'][0],default['actions'])
        assert choice['model_sha256']==pred.sha256 and len(resp['candidates'])==2
        selected,rank=pred.score(resp['prefix_feature'],obs['observation/state'],resp['candidates'],400-step)
        assert selected==1 and np.allclose(rank['q_scores'],choice['choice_meta']['q_scores'],atol=1e-6,rtol=0)
        trace=list(map(json.loads,(r/'stage_trace'/choice['session_id']/'TRACE.jsonl').read_text().splitlines()))
        prefix=[x['actions'] for x in trace if x['method']=='action_chunk' and x['step']<=step]
        out=O/f'init{initial}';out.mkdir(exist_ok=False);np.savez_compressed(out/'CANDIDATES.npz',candidates=resp['candidates'])
        plan=dict(status='REGISTERED_FIRST_LEARNED_CHANGE_DIAGNOSTIC',initial=initial,fork_step=step,noise_base=choice['noise_base'],prefix=prefix,
            model=str(model.relative_to(P)),model_sha256=pred.sha256,source_request=str(request.relative_to(P)),source_request_sha256=sha(request),
            source_response=str(response.relative_to(P)),source_response_sha256=sha(response),source_default_response=str(nr.relative_to(P)),
            source_default_response_sha256=sha(nr),default_outcome=str((n/'RESULT.json').relative_to(P)),default_outcome_sha256=sha(n/'RESULT.json'),
            full_learned_outcome=str((r/'RESULT.json').relative_to(P)),full_learned_outcome_sha256=sha(r/'RESULT.json'),
            max_steps=400,post_choice_policy='unchanged pi, candidate0, same explicit future noise',
            held_out_result=False,operator_learning_or_speedup_claimed=False,new_outcome=None,
            extra_source_analyses_cost_kept=True)
        (out/'PLAN.json').write_text(json.dumps(plan,indent=2)+'\n');cases.append({k:v for k,v in plan.items() if k!='prefix'})
    (O/'RESULT.json').write_text(json.dumps(dict(status='PREPARED_REAL_FIRST_CHOICE_DIAGNOSTICS',cases=cases,robot_actions=0,parameter_updates=0),indent=2)+'\n')
    print(json.dumps(cases),flush=True)
if __name__=='__main__':main()
