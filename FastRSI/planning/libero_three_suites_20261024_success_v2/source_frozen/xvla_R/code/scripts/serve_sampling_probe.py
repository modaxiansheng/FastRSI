"""Opt-in wrapper over the copied native server. Frozen dependency stays read-only."""
import json,os,time
from pathlib import Path
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as server
from fastrsi.sampling_budget import bind

steps=int(os.environ['FASTRSI_DENOISING_STEPS'])
log=Path('/pilot/SAMPLING_EVENTS.jsonl')
def emit(value):
    with log.open('a') as f:f.write(json.dumps({'time':time.time(),**value})+'\n')
original=server.NativeOpenPIPolicy.__init__
def initialize(self,*args,**kwargs):
    original(self,*args,**kwargs)
    receipt=bind(self.policy,steps,emit)
    self.metadata['fastrsi_sampling_probe']=receipt
    emit({'event':'sampling_binding',**receipt})
server.NativeOpenPIPolicy.__init__=initialize
server.main()
