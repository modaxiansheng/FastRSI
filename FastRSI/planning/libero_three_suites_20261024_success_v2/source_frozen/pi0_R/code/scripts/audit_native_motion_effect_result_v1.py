"""Inventory and account one completed native motion-effect development pair."""
import argparse
import collections
import datetime
import hashlib
import json
from pathlib import Path
import struct


def read(path):
    return json.loads(path.read_text())


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def floats_sha(values):
    flat = [v for row in values for v in row]
    return hashlib.sha256(struct.pack('<%df' % len(flat), *flat)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    root = args.run
    manifest = read(root/'RAW_MANIFEST_v2.json')['files']
    mismatches = [name for name, digest in manifest.items()
                  if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest]
    assert not mismatches, mismatches
    cleanup = read(root/'FRESH_CLEANUP_CHECK_v2.json')
    assert len(cleanup['containers']) == 8
    assert all(not c['state']['Running'] and not c['state']['OOMKilled'] for c in cleanup['containers'])
    protocol = read(root/'PROTOCOL.json')
    status = read(root/'STATUS.json')
    skill = read(root/'MOTION_SKILL.json')
    traces, inputs, arms = {}, {}, {}
    for arm in ('N', 'F'):
        stream = root/arm/'stream'
        result = read(stream/'net_t4_i36/RESULT.json')
        trace = jsonl(stream/'stage_trace/net_t4_i36/TRACE.jsonl')
        traces[arm] = trace
        events = jsonl(stream/'OS_COST_EVENTS.jsonl')
        inputs[arm] = [e for e in events if e['event'] == 'actual_policy_input']
        ends = [e for e in events if e['event'] == 'end']
        times = collections.defaultdict(float)
        for e in ends:
            times[e['name']] += e['duration_seconds']
        receipts = [read(p) for p in sorted((stream/'service_calls').glob('call_*/RECEIPT.json'))]
        assert all(r['status'] == 'OK' for r in receipts)
        actual = [t for t in trace if t['method'] == 'action_chunk']
        choices = [e for e in events if e['event'] == 'feedback_window_choice']
        outputs = [e for e in events if e['event'] == 'native_motion_effect_skill_output']
        assert all(e['prompt'] == skill['goal'] for e in inputs[arm])
        assert all(e['requested_steps'] == 10 and e['original_steps'] == 10 for e in choices)
        assert sum(t['method'] == 'reset' for t in trace) == 1
        native = result['native_terminal_by_session']['net_t4_i36']
        consumed = [trace[i]['step']-trace[i-1]['step'] for i in range(1, len(trace))]
        assert sum(consumed) == native['steps']
        assert all(0 < n <= len(t['actions']) for n, t in zip(consumed, actual, strict=True))
        for e in outputs:
            committed = next(t for t in actual if t['step'] == e['physical_steps']+10)
            assert floats_sha(committed['actions']) == e['actions_float32_sha256']
            assert floats_sha(skill['chunks'][e['chunk_index']]) == e['actions_float32_sha256']
        stops = read(stream/'STOP_RECEIPT.json')
        allocated = sum((datetime.datetime.fromisoformat(s['finished'].replace('Z', '+00:00'))-
                         datetime.datetime.fromisoformat(s['started'].replace('Z', '+00:00'))).total_seconds()
                        for s in stops if s['role'] in ('pi05', 'qwen'))
        arms[arm] = dict(environment_success=result['environment_success'],
                         verification_status=result['verification_status'],
                         chain_status=result['chain_status'],
                         admission_status=result['experience_admission']['status'],
                         reset_count=1, physical_steps=native['steps'],
                         action_submissions=len(actual), actual_pi_calls=len(inputs[arm]),
                         skill_outputs=len(outputs), qwen_service_calls=len(receipts),
                         agent_model_calls=sum(e['name']=='agent_model_call' for e in ends),
                         verification_model_calls=sum(e['name']=='verification_model' for e in ends),
                         qwen_generation_seconds=sum(r['generation_seconds'] for r in receipts),
                         qwen_queue_seconds=sum(r['queue_seconds'] for r in receipts),
                         qwen_prompt_tokens=sum(r['prompt_tokens'] for r in receipts),
                         qwen_completion_tokens=sum(r['completion_tokens'] for r in receipts),
                         condition_seconds=sum(e['seconds'] for e in events if e['event']=='native_motion_effect_condition_cost'),
                         spans_seconds=dict(times),
                         startup_seconds=status['arms'][arm]['startup_seconds'],
                         controller_seconds=status['arms'][arm]['controller_seconds'],
                         worker_seconds=status['arms'][arm]['full_wall_seconds'],
                         allocated_model_card_seconds=allocated,
                         decisions=[e for e in events if e['event']=='native_motion_effect_skill_decision'])
    def without_session(t):
        return {k:v for k,v in t.items() if k != 'session_id'}
    pre = {a:[without_session(t) for t in traces[a] if t['step'] <= 60] for a in ('N','F')}
    assert pre['N'] == pre['F']
    assert [e['array_bindings'] for e in inputs['N'][:6]] == [e['array_bindings'] for e in inputs['F'][:6]]
    assert read(root/'N/stream/net_t4_i36/RESULT.json')['accepted_source'] == read(root/'F/stream/net_t4_i36/RESULT.json')['accepted_source']
    summary = dict(status='VERIFIED_DEVELOPMENT_RESULT', source_files_verified=len(manifest),
                   raw_manifest_sha256=hashlib.sha256((root/'RAW_MANIFEST_v2.json').read_bytes()).hexdigest(),
                   prefix_through_60_exact=True, original_goal_bound=True, effective_H10_W10=True,
                   shared_experience=True, max_steps=520, arms=arms,
                   parallel_parent_seconds=status['seconds'],
                   sum_worker_seconds=sum(a['worker_seconds'] for a in arms.values()),
                   allocated_model_card_seconds=sum(a['allocated_model_card_seconds'] for a in arms.values()),
                   cleanup={'owned_stopped':8,'oom':False},
                   interpretation={'actual_pi_call_reduction':1,'physical_step_increase':3,
                                   'action_submission_increase':1,'valid_net_benefit_demonstrated':False,
                                   'learned_vs_frozen_policy_advantage_demonstrated':False,
                                   'new_actor_evaluation_isolation_demonstrated':False,
                                   'neural_cyclic_distillation_demonstrated':False,
                                   'default_changed':False},
                   timing_scope='Spans are nested; never sum with worker. Card allocation is not busy GPU compute. Development pair is not independent held-out evaluation.',
                   protocol_sha256=hashlib.sha256((root/'PROTOCOL.json').read_bytes()).hexdigest())
    args.out.mkdir(parents=True, exist_ok=False)
    with (args.out/'SUMMARY.json').open('x') as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k!='arms'}))


if __name__ == '__main__':
    main()
