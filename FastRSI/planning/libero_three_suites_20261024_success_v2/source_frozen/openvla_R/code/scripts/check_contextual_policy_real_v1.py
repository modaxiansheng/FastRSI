"""Real frozen model: accepted local path and rejected/original RNG regression."""
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import torch
from fastrsi.contextual_policy_probe_v1 import ContextualPolicyProbe,load_artifact,KEY,AUTHORITY
from fastrsi.resident_policy import RNGSnapshot,OpenPIStateGuard


def save(path,data):path.write_text(json.dumps(data,indent=2)+'\n')


def main():
    out=Path('/output');start=time.time();state=dict(status='LOADING',started_at=start)
    save(out/'STATUS.json',state)
    spec=json.loads(Path('/audit/inputs/REGISTRATION.json').read_text())
    from PhyAgentOS.runtime.policy.openpi.native_openpi_server import NativeOpenPIPolicy
    t=time.perf_counter()
    native=NativeOpenPIPolicy(policy_config='pi05_libero',checkpoint_dir='/model',default_prompt=None,
        pytorch_device='cuda:0',disable_torch_compile=True,torch_seed=7)
    torch.cuda.synchronize();state['load_seconds']=time.perf_counter()-t
    guard=OpenPIStateGuard(native);base=RNGSnapshot();rows=[];calls=0;events=[]
    # Each observed request is tested with the same post-load RNG. No hidden state.
    for entry in spec['observations']:
        p=Path('/audit/inputs')/entry['file']
        assert hashlib.sha256(p.read_bytes()).hexdigest()==entry['sha256']
        with np.load(p,allow_pickle=False) as z:obs={k:z[k].copy() for k in z.files}
        obs['prompt']=spec['goal']
        base.restore();guard.reset();expected=native.infer(obs);calls+=1
        expected_rng=RNGSnapshot().hashes
        for kind,item in spec['versions'].items():
            model=load_artifact(Path('/audit/inputs')/item['file'],item['sha256'],kind)
            probe=ContextualPolicyProbe(native,model,kind,session_id='real-check',goal=spec['goal'],
                artifact_sha256=item['sha256'],emit=events.append)
            request=dict(obs);request[KEY]=dict(authority=AUTHORITY,session_id='real-check',
                artifact_sha256=item['sha256'],attempt=1)
            base.restore();guard.reset();t=time.perf_counter();result=probe.infer(request)
            torch.cuda.synchronize();elapsed=time.perf_counter()-t
            receipt=result.pop(KEY);rng_equal=RNGSnapshot().hashes==expected_rng
            assert rng_equal,(entry['id'],kind,'rng')
            assert receipt['used'] == entry['expected_used'],(entry['id'],kind,'route')
            if not receipt['used']:
                calls+=1
                assert np.array_equal(result['actions'],expected['actions']),(entry['id'],kind,'fallback')
            else:
                saved=np.asarray(item['expected_predictions'][entry['id']],np.float32)
                assert np.max(np.abs(result['actions']-saved)) <= 1e-6,(entry['id'],kind,'prediction')
            guard.reset()
            rows.append(dict(id=entry['id'],kind=kind,used=receipt['used'],rng_equal=rng_equal,
                             output_equal_to_expected=True,seconds=elapsed))
            save(out/'STATUS.json',dict(**state,status_progress=rows))
    # Normal unmarked traffic must also be bit-identical through the wrapper.
    base.restore();guard.reset();plain=probe.infer(obs);calls+=1
    assert np.array_equal(plain['actions'],expected['actions'])
    assert RNGSnapshot().hashes==expected_rng;guard.reset()
    save(out/'EVENTS.json',events)
    state.update(status='COMPLETED',finished_at=time.time(),seconds=time.time()-start,
        rows=rows,unmarked_output_rng_equal=True,original_source_guard=True,
        actual_full_decoder_calls=calls,extra_prefix_calls=6,robot_episodes=0,
        scope='Protocol/output/RNG only; does not establish physical quality or full Agent evaluation isolation')
    save(out/'RESULT.json',state);save(out/'STATUS.json',state);print(json.dumps(state),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        p=Path('/output/STATUS.json');s=json.loads(p.read_text()) if p.exists() else {}
        s.update(status='FAILED',error=traceback.format_exc(),finished_at=time.time());save(p,s);raise
