"""Unchanged pi continuation with the original explicit future noise indices."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import packb
from prefix_feature_capture_v1 import capture_prefix
ROOT=Path('/check');KEY='_branch_baseline_v1'
old_init=native.NativeOpenPIPolicy.__init__;old_infer=native.NativeOpenPIPolicy.infer

def initialize(self,**kwargs):
    old_init(self,**kwargs);self._branches={}
    assert self.policy._model.config.action_horizon==10
    assert self.policy._sample_kwargs.get('num_steps',10)==10
    self.metadata.update(branch_baseline_protocol='same_future_noise_v1',candidate_count=1)
    (ROOT/'POLICY_LOAD.json').write_text(json.dumps(self.metadata,indent=2)+'\n')

def infer(self,observation):
    original=dict(observation);obs=dict(original);tag=obs.pop(KEY)
    assert set(tag)=={'session_id','step','noise_base'}
    assert set(obs)=={'observation/image','observation/wrist_image','observation/state','prompt'}
    step=tag['step'];assert type(step) is int and 20<=step<400 and step%10==0
    sid=tag['session_id'];assert step>self._branches.get(sid,-1)
    index=step//10;noise=np.random.default_rng(np.random.SeedSequence([tag['noise_base'],index,0])).standard_normal((10,32)).astype(np.float32)
    saved=self.policy.infer;tick=time.perf_counter()
    self.policy.infer=lambda value:saved(value,noise=noise)
    try:
        with capture_prefix(self.policy._model) as stats:response=old_infer(self,obs)
    finally:self.policy.infer=saved
    response['prefix_feature']=stats['features'][0]
    response['branch_baseline']=dict(**tag,index=index,noise_sha256=hashlib.sha256(noise.tobytes()).hexdigest(),seconds=time.perf_counter()-tick)
    key=hashlib.sha256(sid.encode()).hexdigest()[:16]+'_'+str(index).zfill(3)
    (ROOT/(key+'_request.msgpack')).write_bytes(packb(original));(ROOT/(key+'_response.msgpack')).write_bytes(packb(response))
    self._branches[sid]=step
    return response

native.NativeOpenPIPolicy.__init__=initialize;native.NativeOpenPIPolicy.infer=infer
if __name__=='__main__':native.main()
