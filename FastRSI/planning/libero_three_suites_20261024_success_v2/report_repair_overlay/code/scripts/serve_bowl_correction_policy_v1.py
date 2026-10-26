"""Native unchanged policy; reset the sampler once per corrective-state search.

The case/chunk keys are diagnostic metadata, removed before native inference.
No warm-up inference, training, extra state features or action edits.
"""
import hashlib
import json
from pathlib import Path
import time
import torch
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native

ROOT = Path('/check')
original_init = native.NativeOpenPIPolicy.__init__
original_infer = native.NativeOpenPIPolicy.infer

def initialize(self, **kwargs):
    original_init(self, **kwargs)
    self._source_cpu_rng = torch.get_rng_state().clone()
    self._source_gpu_rng = [s.clone() for s in torch.cuda.get_rng_state_all()]
    self._source_chunks = {}
    receipt = dict(status='LOADED', metadata=self.metadata,
        cpu_rng_sha256=hashlib.sha256(self._source_cpu_rng.cpu().numpy().tobytes()).hexdigest(),
        gpu_rng_sha256=[hashlib.sha256(s.cpu().numpy().tobytes()).hexdigest() for s in self._source_gpu_rng],
        restore_rule='Post-load RNG once before each source first inference; no probing/warm-up')
    (ROOT/'POLICY_LOAD.json').write_text(json.dumps(receipt,indent=2)+'\n')

def infer(self, observation):
    observation=dict(observation)
    case=observation.pop('_correction_case')
    index=observation.pop('_correction_chunk')
    assert case in ('step_120','step_250','step_270') and type(index) is int
    assert index == self._source_chunks.get(case,0)
    if index==0:
        assert case not in self._source_chunks
        torch.set_rng_state(self._source_cpu_rng)
        torch.cuda.set_rng_state_all(self._source_gpu_rng)
    started=time.perf_counter()
    result=original_infer(self,observation)
    self._source_chunks[case]=index+1
    with (ROOT/'POLICY_CALLS.jsonl').open('a') as f:
        f.write(json.dumps(dict(case=case,chunk=index,seconds=time.perf_counter()-started,
                               rng_restored=index==0))+'\n')
    return result

native.NativeOpenPIPolicy.__init__=initialize
native.NativeOpenPIPolicy.infer=infer
if __name__=='__main__':native.main()
