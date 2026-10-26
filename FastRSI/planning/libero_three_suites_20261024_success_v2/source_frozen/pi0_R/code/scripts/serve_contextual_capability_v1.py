"""Explicit experimental server wrapper; original frozen server is unchanged."""
import json
import os
import time
from pathlib import Path

from PhyAgentOS.runtime.policy.openpi import native_openpi_server as server
from fastrsi.contextual_policy_probe_v1 import ContextualPolicyProbe, load_artifact

spec=json.loads(Path(os.environ['FASTRSI_CONTEXTUAL_SPEC']).read_text())
if spec['authority'] != 'CONTEXTUAL_CAPABILITY_DEVELOPMENT_VALUE':
    raise ValueError('Not a registered development service')
original_init=server.NativeOpenPIPolicy.__init__
original_infer=server.NativeOpenPIPolicy.infer

def emit(row):
    with Path('/pilot/CONTEXTUAL_POLICY_EVENTS.jsonl').open('a') as f:
        f.write(json.dumps(dict(time=time.time(),**row))+'\n')

def initialize(self,*args,**kwargs):
    original_init(self,*args,**kwargs)
    model=load_artifact(spec['artifact'],spec['sha256'],spec['kind'])
    # Bind the original method on a thin facade to avoid wrapper recursion.
    class Original:
        policy=self.policy
        def infer(_, observation):return original_infer(self,observation)
    self.contextual=ContextualPolicyProbe(Original(),model,spec['kind'],
        session_id=spec['session_id'],goal=spec['goal'],artifact_sha256=spec['sha256'],emit=emit)
    emit(dict(event='contextual_bound',spec=spec,mode='immutable_development_component'))

server.NativeOpenPIPolicy.__init__=initialize
server.NativeOpenPIPolicy.infer=lambda self,obs:self.contextual.infer(obs)
server.main()
