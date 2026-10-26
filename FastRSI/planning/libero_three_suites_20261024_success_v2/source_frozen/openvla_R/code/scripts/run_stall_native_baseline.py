"""Full native Agent run with a declared, one-shot physical retreat probe."""
import asyncio
import hashlib
import json
import re
import time
import numpy as np
import run_native_baseline as native
from cost_trace import emit
from fastrsi.stall_recovery import StallRecovery
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter


def main():
    cases = json.loads((native.CAMPAIGN/'ROSTER.json').read_text())
    if len(cases) != 1:
        raise ValueError('One-case recovery probe only')
    case = cases[0]
    raw = open(case['stall_artifact'], 'rb').read()
    if hashlib.sha256(raw).hexdigest() != case['stall_sha256']:
        raise ValueError('Recovery artifact changed')
    probe = StallRecovery(case, case['stall_enabled'], json.loads(raw)['spec'])
    original_input = OpenPIPi05Adapter.to_policy_input
    original_infer = native.OpenPIClientPolicyWrapper.infer
    current = {}
    calls = replacements = 0

    def bind(adapter, observation, context):
        payload = original_input(adapter, observation, context)
        match = re.fullmatch(r'libero_obs_(\d+)', str(observation.get('observation_id', '')))
        if match is None:
            raise ValueError('No actual execution step in current observation')
        current.update(step=int(match.group(1)), session_id=context.get('session_id'))
        return payload

    def infer(client, observation):
        nonlocal calls, replacements
        # Always pay for and retain the original policy draw, including during
        # retreat. This probe tests recovery, not fictional saved inference.
        prediction = original_infer(client, observation)
        tick = time.perf_counter()
        actions, event = probe.choose(observation['observation/state'],
            current['step'], current['session_id'], observation.get('prompt'))
        calls += 1
        event.update(event='stall_recovery_choice', case=case['id'],
                     artifact_sha256=case['stall_sha256'], selector_seconds=time.perf_counter()-tick)
        if actions is not None:
            replacements += 1
            event['original_prediction_sha256'] = hashlib.sha256(
                np.asarray(prediction['actions']).tobytes()).hexdigest()
            prediction = {**prediction, 'actions':np.asarray(actions, dtype=np.float32)}
        emit(event)
        return prediction

    OpenPIPi05Adapter.to_policy_input = bind
    native.OpenPIClientPolicyWrapper.infer = infer
    try:
        asyncio.run(native.main())
    finally:
        OpenPIPi05Adapter.to_policy_input = original_input
        native.OpenPIClientPolicyWrapper.infer = original_infer
        (native.ROOT/'STALL_RECEIPT.json').write_text(json.dumps(dict(
            policy_requests=calls, replaced_chunks=replacements,
            trigger_step=probe.trigger_step, uses=probe.used,
            learned=False, native_inference_retained=True), indent=2))


if __name__ == '__main__':
    main()
