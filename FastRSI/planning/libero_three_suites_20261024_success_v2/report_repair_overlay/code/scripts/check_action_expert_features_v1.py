"""Bound to ten REAL paired feedback requests; feature-only identity/fee audit."""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from PhyAgentOS.runtime.policy.openpi import native_openpi_server as native
from PhyAgentOS.runtime.policy.msgpack_numpy import unpackb
from action_expert_capture_v1 import capture_action_expert

R = Path('/check')
LEGAL = ('observation/image', 'observation/wrist_image', 'observation/state', 'prompt')
sha = lambda b: hashlib.sha256(b).hexdigest()
rows = json.loads((R/'inputs/BINDINGS.json').read_text())
assert len(rows) == 10
started = time.perf_counter()
policy = native.NativeOpenPIPolicy(policy_config='pi05_libero', checkpoint_dir='/model',
    default_prompt=None, pytorch_device='cuda:0', disable_torch_compile=True, torch_seed=7)
model = policy.policy._model
assert model.config.action_horizon == 10 and policy.policy._sample_kwargs.get('num_steps', 10) == 10
versions = {n: (id(p), p._version) for n, p in model.named_parameters()}
official = policy.policy.infer

def infer(obs, noise):
    policy.policy.infer = lambda value: official(value, noise=noise)
    try:
        return policy.infer(obs)['actions'].copy()
    finally:
        policy.policy.infer = official

report = dict(status='RUNNING', rows=[], native_inferences=0, robot_executions=0,
    pi_updates=0, training_steps=0, extra_encoder_passes=0,
    scope='Candidate feature identity and cost only; no choice or efficiency claim')
for row in rows:
    i = row['pair']
    raw = (R/'inputs'/row['request_file']).read_bytes()
    assert sha(raw) == row['request_sha256']
    request = unpackb(raw)
    obs = {k: request[k] for k in LEGAL}
    assert obs['prompt'] == 'stack the right bowl on the left bowl and place them in the tray'
    expected = np.load(R/'inputs'/row['candidate_file'])['candidates']
    features = []
    measured = []
    for j in range(2):
        noise = np.random.default_rng(np.random.SeedSequence([row['noise_base'], row['step']//10, j])).standard_normal((10,32)).astype(np.float32)
        assert sha(noise.tobytes()) == row['noise_sha256'][j]
        outputs = {}; timings = {}; counters = {}
        for extract in ((False, True) if (i+j)%2 == 0 else (True, False)):
            tick = time.perf_counter()
            with capture_action_expert(model, extract=extract) as stats:
                outputs[extract] = infer(obs, noise)
            report['native_inferences'] += 1
            timings[extract] = time.perf_counter()-tick; counters[extract] = stats
        assert np.array_equal(outputs[False], outputs[True])
        assert np.array_equal(outputs[True], expected[j]), (i,j,'source candidate differs')
        assert counters[False]['image_passes'] == counters[True]['image_passes'] > 0
        assert counters[False]['projection_passes'] == counters[True]['projection_passes'] == 10
        feature = counters[True]['action_tokens']
        assert np.isfinite(feature).all()
        features.append(feature)
        measured.append(dict(candidate=j, exact_actions=True, exact_source_candidate=True,
            noise_sha256=row['noise_sha256'][j], baseline_seconds=timings[False],
            capture_seconds=timings[True], extraction_seconds=counters[True]['feature_seconds'],
            image_passes=counters[True]['image_passes'], projection_passes=10))
        if i == 0:
            with capture_action_expert(model) as repeat:
                assert np.array_equal(infer(obs, noise), expected[j])
            report['native_inferences'] += 1
            assert np.array_equal(feature, repeat['action_tokens'])
    features = np.stack(features)
    assert not np.array_equal(features[0], features[1])
    np.savez_compressed(R/f'pair_{i:02d}.npz',action_tokens=features,candidates=expected)
    report['rows'].append(dict(pair=i, request_sha256=row['request_sha256'], shape=list(features.shape),
        candidate_conditioned=True, source_response_sha256=row['response_sha256'], measurements=measured))
    (R/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
assert versions == {n: (id(p), p._version) for n,p in model.named_parameters()}
report.update(status='PASSED_ACTION_IDENTITY_AND_CANDIDATE_FEATURE_AUDIT',
    parameter_identity_and_versions_unchanged=True, total_wall_seconds=time.perf_counter()-started,
    final_pre_projection_not_final_action=True,
    extraction_seconds=sum(m['extraction_seconds'] for r in report['rows'] for m in r['measurements']))
(R/'RESULT.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)
