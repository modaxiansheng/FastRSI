"""Native OpenPI protocol; return the chosen unmodified candidate as actions."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import packb
from phase_iql_feedback_update_v1 import AdaptivePredictor
from prefix_feature_capture_v1 import capture_prefix
from fastrsi.native_value_binding_v1 import KEY, GOAL

ROOT=Path(os.environ['FASTRSI_NATIVE_VALUE_SERVICE'])
MODEL=Path(os.environ['FASTRSI_NATIVE_VALUE_MODEL'])
LEGAL={'observation/image','observation/wrist_image','observation/state','prompt'}
old_init=native.NativeOpenPIPolicy.__init__
old_infer=native.NativeOpenPIPolicy.infer


def explicit(self, observation, noise):
    infer=self.policy.infer
    self.policy.infer=lambda value:infer(value, noise=noise)
    try:return old_infer(self,observation)
    finally:self.policy.infer=infer


def initialize(self, **kwargs):
    old_init(self,**kwargs)
    self._value=AdaptivePredictor(MODEL)
    self._sessions={}
    assert self.policy._model.config.action_horizon==10
    assert self.policy._sample_kwargs.get('num_steps',10)==10
    self.metadata.update(native_external_value_protocol='native_external_value_v1',
                         external_model_sha256=self._value.sha256,extra_candidate_cap=20)
    (ROOT/'VALUE_POLICY_LOAD.json').write_text(json.dumps(self.metadata,indent=2)+'\n')


def choose(self, observation):
    obs=dict(observation)
    tag=obs.pop(KEY)
    assert set(obs)==LEGAL
    assert set(tag)=={'session_id','step','remaining_budget','noise_base','model_sha256','manifest_sha256','enabled'}
    assert tag['model_sha256']==self._value.sha256
    assert type(tag['enabled']) is bool and type(tag['step']) is int
    assert type(tag['noise_base']) is int and type(tag['remaining_budget']) is int
    assert tag['remaining_budget']==400-tag['step'] and 0<=tag['step']<400
    assert not tag['enabled'] or obs['prompt']==GOAL
    sid=tag['session_id'];assert isinstance(sid,str) and sid
    state=self._sessions.setdefault(sid,dict(index=0,extras=0,last_step=-1))
    assert tag['step']>state['last_step'] and state['index']<40
    index=state['index'];tick=time.perf_counter()
    noises=[];candidates=[];choice={};feature=None
    noise=np.random.default_rng(np.random.SeedSequence([tag['noise_base'],index,0])).standard_normal((10,32)).astype(np.float32)
    if tag['enabled']:
        with capture_prefix(self.policy._model) as stats:first=explicit(self,obs,noise)
        feature=stats['features'][0]
        t=time.perf_counter();_,rank=self._value.score(feature,obs['observation/state'],[first['actions']],tag['remaining_budget'])
        v=self._value.value(feature,obs['observation/state'],tag['remaining_budget'])
        score_seconds=time.perf_counter()-t
        second=rank['q_scores'][0]<v and state['extras']<20
        choice.update(first_q=rank['q_scores'][0],state_value=v,extras_before=state['extras'],
                      rule='minQ(first)<V and extras<20; experimental heuristic')
    else:first=explicit(self,obs,noise);second=False;score_seconds=0.
    candidates.append(first['actions']);noises.append(hashlib.sha256(noise.tobytes()).hexdigest())
    if second:
        noise=np.random.default_rng(np.random.SeedSequence([tag['noise_base'],index,1])).standard_normal((10,32)).astype(np.float32)
        candidates.append(explicit(self,obs,noise)['actions'])
        noises.append(hashlib.sha256(noise.tobytes()).hexdigest())
        state['extras']+=1
    candidates=np.stack(candidates)
    assert candidates.shape==(len(noises),10,7) and np.isfinite(candidates).all()
    if tag['enabled']:
        selected,rank=self._value.score(feature,obs['observation/state'],candidates,tag['remaining_budget'])
        score_seconds+=rank['selection_seconds'];choice.update(rank)
    else:selected=0
    # The original OS consumes actions. Merely reporting selected_index would
    # leave the original first candidate in place and would not change execution.
    response=dict(first);response['actions']=candidates[selected].copy()
    response['policy_meta']={**response['policy_meta'],'native_value_choice':dict(
        **tag,index=index,selected_index=selected,candidate_count=len(candidates),
        candidate_noise_sha256=noises,selection_seconds=score_seconds,choice_meta=choice)}
    name=hashlib.sha256(sid.encode()).hexdigest()[:20]+'_'+str(index).zfill(3)
    (ROOT/(name+'_request.msgpack')).write_bytes(packb(observation))
    evidence=dict(response);evidence.update(candidates=candidates)
    if feature is not None:evidence['prefix_feature']=feature
    (ROOT/(name+'_response.msgpack')).write_bytes(packb(evidence))
    state.update(index=index+1,last_step=tag['step'])
    with (ROOT/'VALUE_POLICY_CALLS.jsonl').open('a') as f:
        f.write(json.dumps(dict(**response['policy_meta']['native_value_choice'],
                               evidence_name=name,seconds=time.perf_counter()-tick))+'\n')
    return response


native.NativeOpenPIPolicy.__init__=initialize
native.NativeOpenPIPolicy.infer=choose
if __name__=='__main__':native.main()
