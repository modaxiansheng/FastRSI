"""Original native Agent plus explicitly fixed developmental input intervention."""
import asyncio
import hashlib
import json
from pathlib import Path
import run_native_baseline as native
from cost_trace import emit
from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
from fastrsi.remaining_goal_value_probe_v1 import RemainingGoalValueProbe

original_case = native.run_case


async def run_case(case, accepted):
    spec = case['remaining_goal_value_probe']
    raw = Path(spec['artifact']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != spec['sha256']: raise ValueError('Manual probe changed')
    cfg = (native.CAMPAIGN / 'MODEL_CONFIG.json').read_bytes()
    probe = RemainingGoalValueProbe(case, json.loads(raw), hashlib.sha256(cfg).hexdigest(), enabled=spec['enabled'])
    previous = OpenPIPi05Adapter.to_policy_input
    events = []
    def wrapped(adapter, observation, context):
        payload = previous(adapter, observation, context)
        result, receipt = probe.input(payload, observation, context)
        if receipt is not None:
            event = dict(event='remaining_goal_value_probe', case=case['id'], **receipt)
            events.append(event); emit(event)
            with (native.ROOT / 'REMAINING_GOAL_EVENTS.jsonl').open('a') as f: f.write(json.dumps(event) + '\n')
        return result
    OpenPIPi05Adapter.to_policy_input = wrapped
    try: return await original_case(case, accepted)
    finally:
        OpenPIPi05Adapter.to_policy_input = previous
        (native.ROOT / 'REMAINING_GOAL_RECEIPT.json').write_text(json.dumps(dict(
            authority=json.loads(raw)['authority'], enabled=spec['enabled'],
            applied_inference_boundaries=sum(e['applied'] for e in events),
            total_root_boundaries=len(events), actual_new_teacher_calls=0,
            learned_updates=0, benchmark_goal=case['task_description'], events=events), indent=2))


native.run_case = run_case
if __name__ == '__main__': asyncio.run(native.main())
