"""Isolated K1/K2 native samples with matched explicit noise. No learned selection."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb

ROOT=Path('/check')
PROJECT=Path('/project')
LEGAL=('observation/image','observation/wrist_image','observation/state','prompt')
old_init=native.NativeOpenPIPolicy.__init__
old_infer=native.NativeOpenPIPolicy.infer

def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def explicit(self,obs,noise):
    official=self.policy.infer
    self.policy.infer=lambda value:official(value,noise=noise)
    try:return old_infer(self,obs)
    finally:self.policy.infer=official

def audit(self):
    dataset=PROJECT/'planning/bowl_value_records_v2_20261009/TRANSITIONS.jsonl'
    rows=[json.loads(x) for x in dataset.read_text().splitlines()]
    selected=[]
    for group in ('demo_0','demo_7','demo_25'):
        subset=[r for r in rows if r['initial_state_group']==group and '/bowl_lora_source_original_' in r['episode_id']]
        assert len(subset)>=2
        selected += [subset[0],subset[len(subset)//2]]
    def obs(row):
        ref=row['observation'];raw=(PROJECT/ref['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==ref['sha256']
        data=unpackb(raw);return {key:data[key] for key in LEGAL}
    model=self.policy._model
    cpu=torch.get_rng_state().clone();gpu=[x.clone() for x in torch.cuda.get_rng_state_all()]
    original_noise=model.sample_noise;capture={}
    def remember(shape,device):
        noise=original_noise(shape,device);capture['noise']=noise.detach().cpu().numpy().copy();return noise
    model.sample_noise=remember
    try:natural=old_infer(self,obs(selected[0]))
    finally:model.sample_noise=original_noise
    assert capture['noise'].shape==(1,10,32)
    same=explicit(self,obs(selected[0]),capture['noise'][0])
    assert np.array_equal(natural['actions'],same['actions']), 'Explicit-noise native compatibility failed'
    torch.set_rng_state(cpu);torch.cuda.set_rng_state_all(gpu)
    report=dict(status='RUNNING',native_vs_explicit_exact=True,cold_compatibility_calls=2,
        native_noise_sha256=digest(capture['noise']),sampling_steps=10,execution_horizon=10,
        base_weight_updates=0,training_steps=0,probe_rows=[])
    for i,row in enumerate(selected):
        data=obs(row)
        noises=[np.random.default_rng(np.random.SeedSequence([20261009,i,j])).standard_normal((10,32)).astype(np.float32) for j in (0,1)]
        def run(k):
            started=time.perf_counter();results=[explicit(self,data,noises[j]) for j in range(k)]
            return results,time.perf_counter()-started
        if i%2==0:single,t1=run(1);dual,t2=run(2)
        else:dual,t2=run(2);single,t1=run(1)
        repeat=explicit(self,data,noises[1])
        assert np.array_equal(single[0]['actions'],dual[0]['actions'])
        assert np.array_equal(repeat['actions'],dual[1]['actions'])
        arrays=[np.asarray(r['actions']) for r in dual]
        assert all(a.shape==(10,7) and np.isfinite(a).all() for a in arrays)
        difference=arrays[0]-arrays[1]
        np.savez_compressed(ROOT/('probe_%02d.npz'%i),noise0=noises[0],noise1=noises[1],actions0=arrays[0],actions1=arrays[1])
        report['probe_rows'].append(dict(group=row['initial_state_group'],chunk=row['chunk'],
            request_sha256=row['observation']['sha256'],single_seconds=t1,dual_seconds=t2,
            first_candidate_exact=True,second_repeat_exact=True,max_abs_difference=float(np.max(np.abs(difference))),
            rms_difference=float(np.sqrt(np.mean(difference**2)))))
        (ROOT/'SAMPLER_PROBE.json').write_text(json.dumps(report,indent=2)+'\n')
    report.update(status='PASSED',measured_single_seconds=sum(r['single_seconds'] for r in report['probe_rows']),
                  measured_dual_seconds=sum(r['dual_seconds'] for r in report['probe_rows']),total_probe_candidate_inferences=26)
    (ROOT/'SAMPLER_PROBE.json').write_text(json.dumps(report,indent=2)+'\n')

def initialize(self,**kwargs):
    old_init(self,**kwargs)
    assert self.policy._model.config.action_horizon==10 and self.policy._model.config.action_dim==32
    assert self.policy._sample_kwargs.get('num_steps',10)==10
    self._candidate_sequences={}
    audit(self)
    self.metadata.update(candidate_protocol='matched_explicit_noise.v1',candidate_limit=2,selection_model=False)
    (ROOT/'POLICY_LOAD.json').write_text(json.dumps(dict(status='LOADED',metadata=self.metadata),indent=2)+'\n')

def infer(self,observation):
    value=dict(observation)
    case=value.pop('_candidate_case');index=value.pop('_candidate_chunk');k=value.pop('_candidate_count')
    assert case in tuple('init%d_%s'%(i,a) for i in (45,46,47) for a in ('single','random2','value2'))
    assert type(index) is int and 0<=index<40 and index==self._candidate_sequences.get(case,0)
    assert type(k) is int and k in (1,2)
    assert (k==1)==case.endswith('_single') and set(value)==set(LEGAL)
    started=time.perf_counter();results=[];noise_hashes=[]
    for j in range(k):
        noise=np.random.default_rng(np.random.SeedSequence([20261009,index,j])).standard_normal((10,32)).astype(np.float32)
        result=explicit(self,value,noise)
        actions=np.asarray(result['actions'])
        assert actions.shape==(10,7) and np.isfinite(actions).all()
        results.append(result);noise_hashes.append(digest(noise))
    self._candidate_sequences[case]=index+1
    response=dict(results[0]);response.update(candidates=np.stack([r['actions'] for r in results]),
        candidate_noise_sha256=noise_hashes,candidate_count=k,candidate_seconds=time.perf_counter()-started)
    with (ROOT/'POLICY_CALLS.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,chunk=index,
        count=k,seconds=response['candidate_seconds'],noise_sha256=noise_hashes))+'\n')
    return response

native.NativeOpenPIPolicy.__init__=initialize
native.NativeOpenPIPolicy.infer=infer
if __name__=='__main__':native.main()
