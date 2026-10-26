"""Opt-in full native OS experiment; one stored prefix versus fresh inference."""
import asyncio
import hashlib
import json
import os
import time
import numpy as np
import run_native_baseline as native
from fastrsi.prefix_memory import load, PrefixProbe
from cost_trace import emit


def main():
    cases = json.loads((native.CAMPAIGN / 'ROSTER.json').read_text())
    if len(cases) != 1:
        raise ValueError('Single-case prefix diagnostic only')
    case = cases[0]
    artifact = load(case['prefix_artifact'], case['prefix_sha256'])
    model_sha = hashlib.sha256((native.CAMPAIGN / 'MODEL_CONFIG.json').read_bytes()).hexdigest()
    probe = PrefixProbe(artifact, case, model_sha, case['prefix_probe_enabled'])
    original = native.OpenPIClientPolicyWrapper.infer

    def infer(client, observation):
        started = time.perf_counter()
        actions = probe.choose(observation.get('prompt'), os.environ.get('FASTRSI_CURRENT_CASE'))
        emit(dict(event='prefix_memory_choice', case=case['id'], request_index=probe.calls,
                  artifact_sha256=case['prefix_sha256'], reuse=actions is not None,
                  selection_seconds=time.perf_counter()-started,
                  scope='development intervention; applicability not learned'))
        if actions is not None:
            return dict(actions=np.asarray(actions, dtype=np.float32),
                        policy_meta=dict(prefix_source_sha256=case['prefix_sha256'],
                                         prediction_origin='stored_development_trajectory',
                                         policy_latency_ms=0.0))
        emit(dict(event='native_policy_request', case=case['id'], request_index=probe.calls))
        return original(client, observation)

    native.OpenPIClientPolicyWrapper.infer = infer
    try:
        asyncio.run(native.main())
    finally:
        native.OpenPIClientPolicyWrapper.infer = original
        (native.ROOT / 'PREFIX_RECEIPT.json').write_text(json.dumps(dict(
            requests=probe.calls, reused_prefixes=probe.reused,
            native_policy_requests=probe.calls-probe.reused,
            source_sha256=case['prefix_sha256'],
            caveat='Skipping inference changes later sampler RNG positions. This is a full-method probe, not isolated deterministic action causality.'), indent=2))


if __name__ == '__main__':
    main()
