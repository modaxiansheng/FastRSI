"""Choose a whole frozen flow once, then reuse the original native Q/pi service.

A narrow exact-observation memory baseline, not a distilled neural model.
The immutable registry is read-only throughout an evaluation Session.
"""
import hashlib,json,os,time
from pathlib import Path
import serve_native_value_v1 as base
from fastrsi.complete_flow_memory import CompleteFlowMemory
from PhyAgentOS.runtime.policy.msgpack_numpy import packb

REGISTRY=Path(os.environ['FASTRSI_COMPLETE_FLOW_REGISTRY'])
EXPECTED=os.environ['FASTRSI_COMPLETE_FLOW_REGISTRY_SHA256']
MODE=os.environ['FASTRSI_COMPLETE_FLOW_MODE'];assert MODE in ('N','F','C')


def initialize(self,**kwargs):
    base.initialize(self,**kwargs)
    assert hashlib.sha256(REGISTRY.read_bytes()).hexdigest()==EXPECTED
    registry=json.loads(REGISTRY.read_text())
    self._flow_memory=CompleteFlowMemory(registry,self._value.sha256,
        hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest())
    self.metadata.update(complete_flow_protocol='whole_flow_memory_v1',flow_mode=MODE,flow_registry_sha256=EXPECTED)
    (base.ROOT/'VALUE_POLICY_LOAD.json').write_text(json.dumps(self.metadata,indent=2)+'\n')
    (base.ROOT/'FLOW_REGISTRY_BINDING.json').write_text(json.dumps(dict(mode=MODE,registry_sha256=EXPECTED,
        model_sha256=self._value.sha256,exact_scene_only=True,pi_parameters_unchanged=True),indent=2)+'\n')


def choose(self,observation):
    original=dict(observation);client_tag=dict(original[base.KEY]);sid=client_tag['session_id'];step=client_tag['step']
    legal={k:original[k] for k in base.LEGAL}
    started=time.perf_counter()
    if MODE=='C':decision=self._flow_memory.choose(sid,step,legal,packb)
    else:decision=dict(mode='N' if MODE=='N' else 'R',evidence_read=False,scope='fixed flow comparator')
    tag={**client_tag,'enabled':client_tag['enabled'] and decision['mode']=='R'}
    routed={**original,base.KEY:tag};routing_seconds=time.perf_counter()-started
    response=base.choose(self,routed)
    row=dict(session_id=sid,step=step,client_tag=client_tag,execution_tag=tag,decision=decision,
        routing_seconds=routing_seconds,registry_sha256=EXPECTED,policy_meta=response['policy_meta']['native_value_choice'])
    with (base.ROOT/'FLOW_ROUTING.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
    return response

base.native.NativeOpenPIPolicy.__init__=initialize
base.native.NativeOpenPIPolicy.infer=choose
if __name__=='__main__':base.native.main()
