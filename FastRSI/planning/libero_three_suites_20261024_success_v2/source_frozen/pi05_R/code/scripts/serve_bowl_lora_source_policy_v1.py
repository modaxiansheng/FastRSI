"""Source-check native server, optionally attaching the declared expert adapter."""
import hashlib
import json
import os
from pathlib import Path
import time
import serve_bowl_source_policy_v1 as source
from bowl_expert_lora_v1 import load_adapter, adapter_tensors

native = source.native
original = native.NativeOpenPIPolicy.__init__

def initialize(self, **kwargs):
    # Captures the original post-load RNG before any adapter initialization.
    original(self, **kwargs)
    path = os.environ.get('FASTRSI_EXPERT_LORA')
    started = time.perf_counter()
    receipt = dict(enabled=bool(path), native_base_preserved=True)
    if path:
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        assert digest == os.environ['FASTRSI_EXPERT_LORA_SHA256']
        modules = load_adapter(self.policy._model, path)
        receipt.update(sha256=digest, modules=len(modules),
            parameters=sum(p.numel() for p in adapter_tensors(modules).values()))
    receipt.update(status='LOADED', seconds=time.perf_counter()-started)
    (source.ROOT/'EXPERT_ADAPTER_LOAD.json').write_text(json.dumps(receipt,indent=2)+'\n')

native.NativeOpenPIPolicy.__init__ = initialize
if __name__=='__main__': native.main()
