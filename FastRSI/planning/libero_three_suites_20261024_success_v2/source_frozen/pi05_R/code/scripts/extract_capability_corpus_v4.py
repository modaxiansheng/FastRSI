"""Extract registered frozen current-prefix features without action sampling."""
import hashlib
import importlib
import inspect
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from fastrsi.contextual_capability_v1 import current_prefix, feature_from_prefix
from fastrsi.resident_policy import RNGSnapshot, OpenPIStateGuard


def save(path, obj): path.write_text(json.dumps(obj,indent=2)+'\n')


def main():
    output=Path('/output');inputs=Path('/audit/inputs');tick=time.time()
    registration=json.loads((inputs/'REGISTRATION.json').read_text())
    status=dict(status='LOADING',started_at=tick,encoder_calls=0,decoder_calls=0,robot_steps=0)
    save(output/'STATUS.json',status)
    for name,digest in registration['input_manifest'].items():
        assert hashlib.sha256((inputs/name).read_bytes()).hexdigest()==digest,name
    for module,digest in registration['bound_module_sha256'].items():
        assert hashlib.sha256(Path(inspect.getfile(importlib.import_module(module))).read_bytes()).hexdigest()==digest,module
    from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
    t=time.perf_counter()
    native=NativeOpenPIPolicy(policy_config='pi05_libero',checkpoint_dir='/model',default_prompt=None,
                            pytorch_device='cuda:0',disable_torch_compile=True,torch_seed=7)
    torch.cuda.synchronize();status['load_seconds']=time.perf_counter()-t
    guard=OpenPIStateGuard(native)
    vectors=[];measurements=[];references=[]
    status['status']='EXTRACTING';save(output/'STATUS.json',status)
    for row in registration['samples']:
        with np.load(inputs/row['file']) as data:
            obs={k:data[k].copy() for k in data.files}
        obs['prompt']=row['prompt']
        before=RNGSnapshot()
        raw_state=obs['observation/state'].copy()
        torch.cuda.synchronize();t=time.perf_counter()
        hidden,masks=current_prefix(native,obs)
        torch.cuda.synchronize();prefix_seconds=time.perf_counter()-t
        t=time.perf_counter();feature=feature_from_prefix(hidden,masks,raw_state)
        torch.cuda.synchronize();pool_transfer_seconds=time.perf_counter()-t
        assert np.array_equal(obs['observation/state'],raw_state)
        guard.reset();assert RNGSnapshot().hashes==before.hashes
        reference=Path('/reference/job')/row['id']/'prefix_hidden.npy'
        reference_seconds=0.0
        if reference.exists():
            t=time.perf_counter()
            assert np.array_equal(hidden.float().cpu().numpy(),np.load(reference)),row['id']
            assert np.array_equal(masks.float().cpu().numpy(),np.load(reference.parent/'prefix_mask.npy'))
            reference_seconds=time.perf_counter()-t
            references.append(row['id'])
        vectors.append(feature)
        measurements.append(dict(id=row['id'],prefix_seconds=prefix_seconds,
                                 pool_transfer_seconds=pool_transfer_seconds,
                                 reference_check_seconds=reference_seconds,rng_unchanged=True))
        status['encoder_calls']+=1
        save(output/'STATUS.json',status)
        del hidden,masks,feature
    assert set(references)==set(registration['reference_ids'])
    t=time.perf_counter();np.save(output/'FEATURES.npy',np.stack(vectors),allow_pickle=False)
    status.update(status='COMPLETED',finished_at=time.time(),seconds=time.time()-tick,
                  feature_write_seconds=time.perf_counter()-t,reference_full_prefix_bit_equal=references,
                  all_rng_unchanged=True,feature_shape=list(np.stack(vectors).shape))
    save(output/'MEASUREMENTS.json',measurements)
    save(output/'RESULT.json',dict(**status,feature_sha256=hashlib.sha256((output/'FEATURES.npy').read_bytes()).hexdigest(),
                                 authority='OFFLINE_CURRENT_PREFIX_FEATURES_NOT_CONTROLLER_APPROVAL'))
    save(output/'STATUS.json',status);print(json.dumps(status),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        p=Path('/output/STATUS.json');s=json.loads(p.read_text()) if p.exists() else {}
        s.update(status='FAILED',error=traceback.format_exc(),finished_at=time.time());save(p,s);raise
