"""Native policy server with an optional strict, action-only parameter update."""
import hashlib
import json
import os
from pathlib import Path
import time
import torch
from safetensors.torch import load_file
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native

ALLOWED=('paligemma_with_expert.gemma_expert.', 'action_in_proj.', 'action_out_proj.', 'time_mlp_in.', 'time_mlp_out.')
original=native.NativeOpenPIPolicy.__init__

def initialize(self,*args,**kwargs):
    original(self,*args,**kwargs)
    started=time.perf_counter();path=os.environ.get('FASTRSI_ACTION_ADAPTER');receipt=dict(adapter_enabled=bool(path),native_loader_preserved=True)
    if path:
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
        assert h.hexdigest()==os.environ['FASTRSI_ACTION_ADAPTER_SHA256']
        model=self.policy._model;params={n:p for n,p in model.named_parameters() if n.startswith(ALLOWED)}
        weights=load_file(path,device='cpu');assert set(weights)==set(params)
        with torch.no_grad():
            for n,p in params.items():
                w=weights[n];assert w.shape==p.shape and w.dtype==p.dtype and torch.isfinite(w).all();p.copy_(w.to(p.device))
        model.eval();receipt.update(adapter_sha256=h.hexdigest(),tensors=len(weights),action_only=True)
    receipt.update(status='LOADED',seconds=time.perf_counter()-started)
    Path(os.environ['FASTRSI_POLICY_LOAD_RECEIPT']).write_text(json.dumps(receipt,indent=2)+'\n')
    print('[action-adapter] '+json.dumps(receipt),flush=True)

native.NativeOpenPIPolicy.__init__=initialize
if __name__=='__main__':native.main()
