"""Frozen pi: explicit indexed noises, two candidates only at the registered fork."""
import hashlib,json,time
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
    assert self.policy._model.config.action_horizon==10
    assert self.policy._sample_kwargs.get('num_steps',10)==10
    self.metadata.update(branch_baseline_protocol='same_future_noise_v2')
    (ROOT/'POLICY_LOAD.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

def infer(self,observation):
    original=dict(observation);obs=dict(original);tag=obs.pop(KEY)
    assert set(tag)=={'session_id','step','noise_base','fork_step'}
    assert set(obs)=={'observation/image','observation/wrist_image','observation/state','prompt'}
    step=tag['step'];fork=tag['fork_step'];sid=tag['session_id']
    assert type(step) is int and 0<=fork<=step<400 and step%10==fork%10==0
    assert step>self._branches.get(sid,-1)
    index=step//10;tick=time.perf_counter();candidates=[];noises=[];feature=None;first=None;expert_tokens=[]
    for j in range(2 if step==fork else 1):
        noise=np.random.default_rng(np.random.SeedSequence([tag['noise_base'],index,j])).standard_normal((10,32)).astype(np.float32)
        saved=self.policy.infer;self.policy.infer=lambda value:saved(value,noise=noise)
        try:
            with capture_prefix(self.policy._model) as stats, capture_action_expert(self.policy._model, extract=step==fork) as expert:
                response=old_infer(self,obs)
        finally:self.policy.infer=saved
        if step==fork:expert_tokens.append(expert['action_tokens'])
        if j==0:first=response;feature=stats['features'][0]
        candidates.append(response['actions']);noises.append(hashlib.sha256(noise.tobytes()).hexdigest())
    response=dict(first);response['candidates']=np.stack(candidates);response['prefix_feature']=feature
    assert response['candidates'].shape==(len(noises),10,7) and np.isfinite(response['candidates']).all()
    if step==fork:response['action_expert_tokens']=np.stack(expert_tokens)
    response['branch_baseline']=dict(**tag,index=index,noise_sha256=noises,seconds=time.perf_counter()-tick)
    key=hashlib.sha256(sid.encode()).hexdigest()[:16]+'_'+str(index).zfill(3)
    (ROOT/(key+'_request.msgpack')).write_bytes(packb(original));(ROOT/(key+'_response.msgpack')).write_bytes(packb(response))
    self._branches[sid]=step
    return response

native.NativeOpenPIPolicy.__init__=initialize;native.NativeOpenPIPolicy.infer=infer
if __name__=='__main__':native.main()
