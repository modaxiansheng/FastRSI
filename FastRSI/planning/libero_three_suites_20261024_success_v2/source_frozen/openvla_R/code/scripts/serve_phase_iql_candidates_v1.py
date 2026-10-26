"""Original native candidates plus read-only existing prefix representation."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from prefix_feature_capture_v1 import capture_prefix
ROOT=Path('/check');P=Path('/project')
LEGAL=('observation/image','observation/wrist_image','observation/state','prompt')
old_init=native.NativeOpenPIPolicy.__init__;old_infer=native.NativeOpenPIPolicy.infer
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def explicit(self,obs,noise):
    official=self.policy.infer;self.policy.infer=lambda value:official(value,noise=noise)
    try:return old_infer(self,obs)
    finally:self.policy.infer=official
def initialize(self,**kwargs):
    old_init(self,**kwargs);self._candidate_sequences={}
    assert self.policy._model.config.action_horizon==10 and self.policy._sample_kwargs.get('num_steps',10)==10
    ref=json.loads((P/'planning/phase_value_dataset_v1_20261010/REQUESTS.json').read_text())[0]
    raw=(P/ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256']
    data=unpackb(raw);obs={k:data[k] for k in LEGAL}
    noise=np.random.default_rng(6101000).standard_normal((10,32)).astype(np.float32)
    a=explicit(self,obs,noise)
    with capture_prefix(self.policy._model) as stats:b=explicit(self,obs,noise)
    assert np.array_equal(a['actions'],b['actions']) and stats['features'][0].shape==(4096,)
    (ROOT/'SAMPLER_PROBE.json').write_text(json.dumps(dict(status='PASSED',inferences=2,actions_exact=True,prefix_feature_dimension=4096))+'\n')
    self.metadata.update(candidate_protocol='phase_iql.v1',candidate_limit=2,selection_model=False)
    (ROOT/'POLICY_LOAD.json').write_text(json.dumps(dict(status='LOADED',metadata=self.metadata),indent=2)+'\n')
def infer(self,observation):
    obs=dict(observation);case=obs.pop('_candidate_case');idx=obs.pop('_candidate_chunk');k=obs.pop('_candidate_count')
    assert case in tuple(f'init{i}_{a}' for i in (28,29,30,31) for a in ('single','random2','iql2'))
    assert set(obs)==set(LEGAL) and idx==self._candidate_sequences.get(case,0) and 0<=idx<40
    assert k in (1,2) and (k==1)==case.endswith('_single')
    initial=int(case.split('_')[0][4:]);results=[];hashes=[];feature=None;tick=time.perf_counter()
    for j in range(k):
        noise=np.random.default_rng(np.random.SeedSequence([6101000+initial,idx,j])).standard_normal((10,32)).astype(np.float32)
        if j==0 and case.endswith('_iql2'):
            with capture_prefix(self.policy._model) as stats:result=explicit(self,obs,noise)
            feature=stats['features'][0]
        else:result=explicit(self,obs,noise)
        assert np.asarray(result['actions']).shape==(10,7) and np.isfinite(result['actions']).all()
        results.append(result);hashes.append(digest(noise))
    self._candidate_sequences[case]=idx+1
    response=dict(results[0]);response.update(candidates=np.stack([r['actions'] for r in results]),
        candidate_noise_sha256=hashes,candidate_count=k,candidate_seconds=time.perf_counter()-tick)
    if feature is not None:response['prefix_feature']=feature
    with (ROOT/'POLICY_CALLS.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,chunk=idx,candidate_count=k,
        candidate_seconds=response['candidate_seconds'],noise_sha256=hashes,feature_extracted=feature is not None))+'\n')
    return response
native.NativeOpenPIPolicy.__init__=initialize;native.NativeOpenPIPolicy.infer=infer
if __name__=='__main__':native.main()
