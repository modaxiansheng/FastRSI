"""Frozen pi: explicit indexed noises, two candidates only at the registered fork."""
import hashlib,json,time
from phase_iql_feedback_update_v1 import AdaptivePredictor
from action_expert_ranker_v1 import ExpertPairRanker
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import packb
from prefix_feature_capture_v1 import capture_prefix
from action_expert_capture_v1 import capture_action_expert
ROOT=Path('/check');KEY='_branch_baseline_v2'
old_init=native.NativeOpenPIPolicy.__init__;old_infer=native.NativeOpenPIPolicy.infer

def initialize(self,**kwargs):
    old_init(self,**kwargs);self._branches={}
    self._extras={}
    spec=json.loads((ROOT/'SPEC.json').read_text());plan=json.loads((Path('/project')/spec['source']/'PLAN.json').read_text())
    model=Path('/project')/plan['model'];self._value=AdaptivePredictor(model);self._ranker=ExpertPairRanker(model)
    assert self._ranker.sha256==plan['model_sha256']
    assert self.policy._model.config.action_horizon==10
    assert self.policy._sample_kwargs.get('num_steps',10)==10
    self.metadata.update(branch_baseline_protocol='frozen_R_continuation_v1')
    (ROOT/'POLICY_LOAD.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

def infer(self,observation):
    original=dict(observation);obs=dict(original);tag=obs.pop(KEY)
    assert set(tag)=={'session_id','step','noise_base','fork_step'}
    assert set(obs)=={'observation/image','observation/wrist_image','observation/state','prompt'}
    step=tag['step'];fork=tag['fork_step'];sid=tag['session_id']
    assert type(step) is int and 0<=fork<=step<400 and step%10==fork%10==0
    assert step>self._branches.get(sid,-1)
    index=step//10;tick=time.perf_counter();candidates=[];noises=[];expert_tokens=[]
    assert fork==0
    def candidate(j):
        noise=np.random.default_rng(np.random.SeedSequence([tag['noise_base'],index,j])).standard_normal((10,32)).astype(np.float32)
        saved=self.policy.infer;self.policy.infer=lambda value:saved(value,noise=noise)
        try:
            with capture_prefix(self.policy._model) as stats, capture_action_expert(self.policy._model) as expert:
                response=old_infer(self,obs)
        finally:self.policy.infer=saved
        candidates.append(response['actions']);expert_tokens.append(expert['action_tokens']);noises.append(hashlib.sha256(noise.tobytes()).hexdigest())
        return response,stats['features'][0]
    first,feature=candidate(0)
    _,rank=self._value.score(feature,obs['observation/state'],[first['actions']],400-step)
    v=self._value.value(feature,obs['observation/state'],400-step)
    extra=self._extras.setdefault(sid,0)
    second=rank['q_scores'][0]<v and extra<20
    if second:candidate(1);self._extras[sid]+=1
    assert step!=fork or second
    proposed,details=self._ranker.choose(np.stack(expert_tokens)) if second else (0,dict(rule='one candidate unchanged'))
    selected=0 if step==fork else proposed
    response=dict(first);response['candidates']=np.stack(candidates);response['prefix_feature']=feature
    response['action_expert_tokens']=np.stack(expert_tokens);response['actions']=candidates[selected]
    response['branch_baseline']=dict(**tag,index=index,noise_sha256=noises,seconds=time.perf_counter()-tick,
        model_sha256=self._ranker.sha256,extras_before=extra,first_q=rank['q_scores'][0],state_value=v,
        proposed_index=proposed,selected_index=selected,choice_meta=details,
        continuation='same frozen R sampling and ranking; only first selection forced0')
    assert response['candidates'].shape==(len(noises),10,7) and np.isfinite(response['candidates']).all()
    key=hashlib.sha256(sid.encode()).hexdigest()[:16]+'_'+str(index).zfill(3)
    (ROOT/(key+'_request.msgpack')).write_bytes(packb(original));(ROOT/(key+'_response.msgpack')).write_bytes(packb(response))
    self._branches[sid]=step
    return response

native.NativeOpenPIPolicy.__init__=initialize;native.NativeOpenPIPolicy.infer=infer
if __name__=='__main__':native.main()
