"""Test learned Q/V compute allocation; at most20 extra candidates, no action mix."""
import hashlib,json,time
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from prefix_feature_capture_v1 import capture_prefix
from phase_iql_feedback_update_v1 import AdaptivePredictor
ROOT=Path('/check');P=Path('/project')
LEGAL=('observation/image','observation/wrist_image','observation/state','prompt')
old_init=native.NativeOpenPIPolicy.__init__;old_infer=native.NativeOpenPIPolicy.infer
def digest(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def explicit(self,obs,noise):
    original=self.policy.infer;self.policy.infer=lambda value:original(value,noise=noise)
    try:return old_infer(self,obs)
    finally:self.policy.infer=original
def initialize(self,**kwargs):
    old_init(self,**kwargs);self._candidate_sequences={};self._extras={}
    self._selectors={key:AdaptivePredictor(ROOT/(key+'_router.npz')) for key in ('frozen','updated')}
    assert self.policy._model.config.action_horizon==10 and self.policy._sample_kwargs.get('num_steps',10)==10
    ref=json.loads((P/'planning/phase_value_dataset_v1_20261010/REQUESTS.json').read_text())[0]
    raw=(P/ref['path']).read_bytes();assert hashlib.sha256(raw).hexdigest()==ref['sha256']
    data=unpackb(raw);obs={k:data[k] for k in LEGAL};noise=np.random.default_rng(7101000).standard_normal((10,32)).astype(np.float32)
    a=explicit(self,obs,noise)
    with capture_prefix(self.policy._model) as stats:b=explicit(self,obs,noise)
    assert np.array_equal(a['actions'],b['actions']) and stats['features'][0].shape==(4096,)
    (ROOT/'SAMPLER_PROBE.json').write_text(json.dumps(dict(status='PASSED',inferences=2,actions_exact=True,prefix_feature_dimension=4096))+'\n')
    self.metadata.update(candidate_protocol='phase_iql_adaptive.v1',candidate_limit=2,extra_candidates_limit=20,
        **{key+'_model_sha256':value.sha256 for key,value in self._selectors.items()})
    (ROOT/'POLICY_LOAD.json').write_text(json.dumps(dict(status='LOADED',metadata=self.metadata),indent=2)+'\n')
def infer(self,observation):
    obs=dict(observation);case=obs.pop('_candidate_case');idx=obs.pop('_candidate_chunk');requested=obs.pop('_candidate_count');budget=obs.pop('_remaining_budget')
    assert case in tuple(f'init{i}_{a}' for i in (24,25,26,27) for a in ('single','mix','frozen','updated'))
    assert set(obs)==set(LEGAL) and idx==self._candidate_sequences.get(case,0) and 0<=idx<40
    assert type(budget) is int and 0<budget<=400
    initial=int(case.split('_')[0][4:]);arm=case.split('_')[1];adaptive=arm in self._selectors
    tick=time.perf_counter();noise=np.random.default_rng(np.random.SeedSequence([7101000+initial,idx,0])).standard_normal((10,32)).astype(np.float32)
    feature=None;choice={};score_seconds=0.
    if adaptive:
        assert requested==-1
        with capture_prefix(self.policy._model) as stats:first=explicit(self,obs,noise)
        feature=stats['features'][0];predictor=self._selectors[arm]
        start=time.perf_counter();_,first_meta=predictor.score(feature,obs['observation/state'],[first['actions']],budget)
        v=predictor.value(feature,obs['observation/state'],budget);score_seconds+=time.perf_counter()-start
        extras=self._extras.get(case,0);second=first_meta['q_scores'][0]<v and extras<20
        choice.update(first_q=first_meta['q_scores'][0],state_value=v,extras_before=extras,
            second_requested=second,rule='minQ(first)<learnedV and extras<20; heuristic, not safety confidence',model_sha256=predictor.sha256)
    else:
        first=explicit(self,obs,noise);second=arm=='mix' and idx%2==0
        assert requested==(2 if second else 1)
    results=[first];hashes=[digest(noise)]
    if second:
        noise=np.random.default_rng(np.random.SeedSequence([7101000+initial,idx,1])).standard_normal((10,32)).astype(np.float32)
        results.append(explicit(self,obs,noise));hashes.append(digest(noise));self._extras[case]=self._extras.get(case,0)+1
    candidates=np.stack([r['actions'] for r in results]);assert candidates.shape==(len(results),10,7) and np.isfinite(candidates).all()
    if adaptive:
        selected,rank=predictor.score(feature,obs['observation/state'],candidates,budget);score_seconds+=rank['selection_seconds'];choice.update(rank)
        choice['selection_seconds']=score_seconds
    else:selected=int(np.random.default_rng(np.random.SeedSequence([9102026,idx])).integers(0,2)) if second else 0
    self._candidate_sequences[case]=idx+1
    response=dict(first);response.update(candidates=candidates,candidate_noise_sha256=hashes,candidate_count=len(results),
        candidate_seconds=time.perf_counter()-tick,selected_index=selected,choice_meta=choice)
    if feature is not None:response['prefix_feature']=feature
    with (ROOT/'POLICY_CALLS.jsonl').open('a') as f:f.write(json.dumps(dict(case=case,chunk=idx,count=len(results),
        seconds=response['candidate_seconds'],selected_index=selected,choice_meta=choice))+'\n')
    return response
native.NativeOpenPIPolicy.__init__=initialize;native.NativeOpenPIPolicy.infer=infer
if __name__=='__main__':native.main()
