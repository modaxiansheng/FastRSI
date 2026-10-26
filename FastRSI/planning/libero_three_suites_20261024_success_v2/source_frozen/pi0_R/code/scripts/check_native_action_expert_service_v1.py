"""Real native-model protocol check, not robot performance or new supervision."""
import hashlib,json,os,time
from pathlib import Path
import numpy as np
R=Path('/check');(R/'service').mkdir(exist_ok=False)
os.environ['FASTRSI_NATIVE_VALUE_SERVICE']=str(R/'service')
os.environ['FASTRSI_NATIVE_VALUE_MODEL']=str(R/'expert_ranker_F.npz')
import serve_native_action_expert_v1 as service
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from phase_iql_feedback_update_v1 import AdaptivePredictor
from action_expert_ranker_v1 import ExpertPairRanker
tick=time.perf_counter()
policy=service.native.NativeOpenPIPolicy(policy_config='pi05_libero',checkpoint_dir='/model',
    default_prompt=None,pytorch_device='cuda:0',disable_torch_compile=True,torch_seed=7)
versions={n:(id(p),p._version) for n,p in policy.policy._model.named_parameters()}
bindings=json.loads((R/'inputs/BINDINGS.json').read_text());rows=[]
for arm in ('F','R','N'):
    model=R/f"expert_ranker_{'F' if arm=='N' else arm}.npz"
    policy._value=AdaptivePredictor(model);policy._ranker=ExpertPairRanker(model)
    for i in ([0] if arm=='N' else [0,1,6]):
        b=bindings[i];raw=(R/'inputs'/b['request_file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==b['request_sha256']
        original=unpackb(raw);obs={k:original[k] for k in service.LEGAL}
        sid=f'service_check_{arm}_{i}';index=b['step']//10
        policy._sessions[sid]=dict(index=index,extras=0,last_step=-1)
        obs[service.KEY]=dict(session_id=sid,step=b['step'],remaining_budget=400-b['step'],
            noise_base=b['noise_base'],model_sha256=policy._value.sha256,
            manifest_sha256='isolated_saved_request_protocol_check',enabled=arm!='N')
        response=policy.infer(obs);meta=response['policy_meta']['native_value_choice']
        expected=np.load(R/'inputs'/b['candidate_file'])['candidates']
        evidence=list((R/'service').glob(hashlib.sha256(sid.encode()).hexdigest()[:20]+'_*_response.msgpack'))
        assert len(evidence)==1;stored=unpackb(evidence[0].read_bytes())
        count=meta['candidate_count'];assert count in (1,2)
        assert np.array_equal(stored['candidates'],expected[:count])
        assert np.array_equal(response['actions'],expected[meta['selected_index']])
        assert meta['candidate_noise_sha256']==b['noise_sha256'][:count]
        if count==2:
            selected,_=policy._ranker.choose(stored['action_expert_tokens'])
            assert selected==meta['selected_index']
        if i in (0,1) and arm!='N':assert count==2
        if arm=='N':assert count==1 and meta['selected_index']==0 and 'action_expert_tokens' not in stored
        rows.append(dict(arm=arm,pair=i,candidate_count=count,selected=meta['selected_index'],
            exact_source_candidates=True,actual_selected_action_returned=True,original_noise_bound=True,
            model_sha256=policy._value.sha256,choice_meta=meta['choice_meta']))
assert versions=={n:(id(p),p._version) for n,p in policy.policy._model.named_parameters()}
result=dict(status='PASSED_REAL_NATIVE_ACTION_EXPERT_PROTOCOL',rows=rows,
    native_inferences=sum(r['candidate_count'] for r in rows),robot_executions=0,pi_updates=0,
    seconds=time.perf_counter()-tick,production_default_unchanged=True,goal_complete=False)
(R/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
