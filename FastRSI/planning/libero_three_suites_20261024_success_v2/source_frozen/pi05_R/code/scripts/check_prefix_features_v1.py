"""Saved-request native action identity and actual feature extraction fee; no robot."""
from pathlib import Path
import hashlib
import json
import time
import numpy as np
import torch
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from prefix_feature_capture_v1 import capture_prefix

P = Path('/project')
R = Path('/check')
LEGAL = ('observation/image', 'observation/wrist_image', 'observation/state', 'prompt')
started = time.perf_counter()
dataset = P/'planning/bowl_value_records_v2_20261009/TRANSITIONS.jsonl'
records = [json.loads(s) for s in dataset.read_text().splitlines()]
chosen = []
for group in ('demo_0', 'demo_7', 'demo_25'):
    subset = [r for r in records if r['initial_state_group'] == group and '/bowl_lora_source_original_' in r['episode_id']]
    chosen += [subset[0], subset[len(subset)//2]]
policy = native.NativeOpenPIPolicy(policy_config='pi05_libero', checkpoint_dir='/model',
    default_prompt=None, pytorch_device='cuda:0', disable_torch_compile=True, torch_seed=7)
model = policy.policy._model
versions = {n: (id(p), p._version) for n, p in model.named_parameters()}
official = policy.policy.infer
def infer(obs, noise):
    policy.policy.infer = lambda value: official(value, noise=noise)
    try:
        return policy.infer(obs)
    finally:
        policy.policy.infer = official

report = dict(status='RUNNING', rows=[], robot_executions=0, training_steps=0,
    native_policy_parameter_updates=0, source_groups=['demo_0', 'demo_7', 'demo_25'],
    scope='Interface and fee check only; no feature quality, generalization or learned ranking evidence')
for i, row in enumerate(chosen):
    raw = (P/row['observation']['path']).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == row['observation']['sha256']
    value = unpackb(raw);obs = {k: value[k] for k in LEGAL}
    assert obs['prompt'] == 'stack the right bowl on the left bowl and place them in the tray'
    noise = np.random.default_rng(np.random.SeedSequence([20261010,i,0])).standard_normal((10,32)).astype(np.float32)
    # Counterbalance identity comparisons, retain a separate second-noise feature check.
    modes = (False, True) if i % 2 == 0 else (True, False)
    outputs = {};stats_by_mode = {};times = {}
    for extract in modes:
        tick = time.perf_counter()
        with capture_prefix(model, extract=extract) as stats:
            outputs[extract] = infer(obs, noise)['actions'].copy()
        times[extract] = time.perf_counter()-tick
        assert stats['prefix_passes'] == 1 and stats['image_passes'] > 0
        stats_by_mode[extract] = stats
    assert np.array_equal(outputs[False], outputs[True])
    assert stats_by_mode[False]['image_passes'] == stats_by_mode[True]['image_passes']
    feature = stats_by_mode[True]['features'][0]
    assert feature.ndim == 1 and np.isfinite(feature).all()
    second_noise = np.random.default_rng(np.random.SeedSequence([20261010,i,1])).standard_normal((10,32)).astype(np.float32)
    with capture_prefix(model) as repeat:
        infer(obs, second_noise)
    assert repeat['prefix_passes'] == 1 and np.array_equal(feature, repeat['features'][0])
    np.savez_compressed(R/('feature_%02d.npz'%i),feature=feature,state=obs['observation/state'],
                        matched_actions=outputs[False])
    report['rows'].append(dict(group=row['initial_state_group'],chunk=row['chunk'],
        request_sha256=row['observation']['sha256'],actions_exact=True,noise_independent_feature_exact=True,
        dimension=len(feature),prefix_passes=1,image_passes=repeat['image_passes'],
        baseline_seconds=times[False],feature_inference_seconds=times[True],
        extraction_seconds=stats_by_mode[True]['feature_seconds']))
    (R/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
assert versions == {n: (id(p), p._version) for n,p in model.named_parameters()}
report.update(status='PASSED',native_inferences=18,feature_extra_encoder_passes=0,
    feature_dimensions=sorted(set(r['dimension'] for r in report['rows'])),
    measured_feature_extraction_seconds=sum(r['extraction_seconds'] for r in report['rows']),
    total_wall_seconds=time.perf_counter()-started)
(R/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
