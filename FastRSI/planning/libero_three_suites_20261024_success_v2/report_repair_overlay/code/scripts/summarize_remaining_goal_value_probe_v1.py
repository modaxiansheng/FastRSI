"""Read real actions, native score, untrusted semantic status and full expenses."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path


def read(p): return json.loads(p.read_text())
def lines(p): return [json.loads(x) for x in p.read_text().splitlines()]
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def stamp(t): return datetime.fromisoformat(t.replace('Z', '+00:00')).timestamp()


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True); args = parser.parse_args()
    from fastrsi.object_stage_memory_v1 import GOAL, REMAINING
    project = args.project; run = project / 'runs/remaining_goal_value_probe_v1_20261006'
    audit = project / 'runs/remaining_goal_physical_audit_v1_20261006'
    parent = read(run / 'STATUS.json'); expected = read(run / 'MANUAL_REMAINING.json')['prefix_bindings']
    if parent['status'] != 'COMPLETED': raise ValueError('Actual pair incomplete')
    source = project / 'runs/task4_informative_pair_dev_187_v8_20261006/native10a/init26/F/stream/stage_trace/net_t4_i26/TRACE.jsonl'
    original = lines(source); arms = {}; actions = {}; manifest = {str(source.relative_to(project)): sha(source)}
    for arm in ('N', 'F'):
        group = run / arm; stream = group / 'stream'; status = read(group / 'STATUS.json')
        result = read(stream / 'RESULT.json'); row = result['cases'][0]
        if status['status'] != 'COMPLETED' or row['chain_status'] != 'PASSED':
            raise ValueError('Full native chain did not complete')
        if row['case']['task_description'] != GOAL or row['case']['feedback_window_mode'] != 'fixed10':
            raise ValueError('Original task or action window changed')
        if not row['terminal_report']['read_coverage']['complete']: raise ValueError('Native report incomplete')
        events = lines(stream / 'REMAINING_GOAL_EVENTS.jsonl')
        inputs = [x for x in lines(stream / 'OS_COST_EVENTS.jsonl') if x.get('event') == 'actual_policy_input']
        if len(inputs) != len(events): raise ValueError('Probe receipt and real model inputs differ')
        for event, actual in zip(events, inputs):
            active = arm == 'F' and 120 <= event['step'] < 220
            if event['applied'] != active or actual['prompt'] != (REMAINING['white_mug'] if active else GOAL):
                raise ValueError('Real intervention differs from fixed registration')
            if event['array_bindings'] != actual['array_bindings']: raise ValueError('Arrays modified after probe')
            if event['step'] <= 120 and actual['array_bindings'] != expected[str(event['step'])]:
                raise ValueError('Prefix differs from source')
        trace = lines(stream / 'stage_trace/net_t4_i26/TRACE.jsonl')
        for actual, reference in zip(trace, original):
            if actual['step'] > 120: break
            if actual['images'] != reference['images'] or actual.get('actions') != reference.get('actions'):
                raise ValueError('Actual pre-intervention control or RGB differs')
        episode = read(stream / 'net_t4_i26/workspace/artifacts/runtime/net_t4_i26/episode.json')
        native_success = episode['success']
        if native_success != row['environment_success'] or episode['num_steps'] != trace[-1]['step']:
            raise ValueError('Native result and actual steps disagree')
        actions[arm] = [a for entry in trace[1:] for a in entry['actions']][:episode['num_steps']]
        cost = lines(stream / 'OS_COST_EVENTS.jsonl'); spans = defaultdict(float)
        for event in cost:
            if event.get('event') == 'end': spans[event['name']] += event['duration_seconds']
        target = lines(stream / 'TARGET_COST_EVENTS.jsonl')
        target_spans = defaultdict(float)
        for event in target:
            if event.get('event') == 'end': target_spans[event['name']] += event['duration_seconds']
        receipts = [read(p) for p in sorted((stream / 'service_calls').glob('call_*/RECEIPT.json'))]
        if any(r['status'] != 'OK' for r in receipts): raise ValueError('Qwen protocol error')
        stop = read(stream / 'STOP_RECEIPT.json')
        if len(stop) != 4 or any(s['running'] for s in stop): raise ValueError('Owned services not stopped')
        model_seconds = {s['role']: stamp(s['finished']) - stamp(s['started'])
                         for s in stop if s['role'] in ('pi05', 'qwen')}
        physical = read(audit / arm / 'REPLAY.json')
        if (physical['status'] != 'EXACT_IMAGE_REPLAY' or physical['source_trace_sha256'] != sha(stream / 'stage_trace/net_t4_i26/TRACE.jsonl')
                or physical['source_result_sha256'] != sha(stream / 'RESULT.json')):
            raise ValueError('Physical diagnosis did not match actual source')
        arms[arm] = dict(native_success=native_success, semantic_verdicts=[x['verdict'] for x in row['outcomes']],
            chain_status=row['chain_status'], original_goal=GOAL, init=26,
            action_steps=episode['num_steps'], environment_attempts=len(row['outcomes']),
            environment_resets=sum(x.get('event') == 'end' and x.get('name') == 'environment_reset' for x in target),
            actual_control_submissions=sum(x.get('event') == 'end' and x.get('name') == 'environment_execution' for x in target),
            actual_vla_calls=len(inputs), actual_qwen_calls=len(receipts), new_decision_model_calls=0,
            applied_boundaries=sum(x['applied'] for x in events), exact_prefix_through120=True,
            worker_full_seconds=status['full_wall_seconds'], startup_seconds=status['startup_seconds'],
            controller_seconds=status['controller_seconds'], two_card_reservation_seconds=2 * status['full_wall_seconds'],
            actual_model_container_seconds=model_seconds, actual_model_allocation_card_seconds=sum(model_seconds.values()),
            nested_os_spans=dict(spans), nested_target_spans=dict(target_spans),
            qwen_generation_seconds=sum(x['generation_seconds'] for x in receipts),
            qwen_prompt_tokens=sum(x['prompt_tokens'] for x in receipts),
            qwen_completion_tokens=sum(x['completion_tokens'] for x in receipts),
            experience_admission=row['experience_admission'],
            terminal_offline_physical_facts=physical['rows'][-1]['offline_supervision'],
            white_proxy_persisted_after120=all(x['offline_supervision']['white_mug']['released_support_proxy']
                for x in physical['rows'] if x['step'] >= 120), physical_stability_proven=False,
            stopped_containers=[s['container'] for s in stop])
    first = next((i for i, (a, b) in enumerate(zip(actions['N'], actions['F']), 1) if a != b), None)
    summary = dict(status='COMPLETED_OPERATION_VALUE_AUDIT', arms=arms,
        first_actual_action_difference=first,
        different_action_steps=sum(a != b for a, b in zip(actions['N'], actions['F'])),
        worker_wall_difference_seconds=arms['F']['worker_full_seconds'] - arms['N']['worker_full_seconds'],
        worker_wall_difference_percent=100 * (arms['F']['worker_full_seconds'] / arms['N']['worker_full_seconds'] - 1),
        parallel_parent_seconds=parent['seconds'],
        experiment_two_card_reservation_seconds=sum(x['two_card_reservation_seconds'] for x in arms.values()),
        experiment_actual_model_allocation_card_seconds=sum(x['actual_model_allocation_card_seconds'] for x in arms.values()),
        physical_replay_batch_seconds=read(audit / 'STATUS.json')['seconds'],
        stage_decision='RETIRE_CURRENT_REMAINING_GOAL_CANDIDATE_NO_EXPANSION',
        learning_claim=False, cold_start_payback_established=False, default_changed=False,
        primary_score='Native episode success; Session.result.success may be overwritten by semantic validation.',
        accounting='Startup, execution, validation, original Agent report, admission, failures and teardown '
                   'are inside worker/allocated-container intervals. Nested spans are explanations, not extra charges. '
                   'Parallel parent time is experiment throughput; separate CPU diagnosis is project expense.',
        scope='One previously seen developer init with fixed retrospective intervention. No online physical authorization, '
              'unseen transfer, learning efficiency, noninferiority or speedup claim.')
    for root in (run, audit):
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts:
                manifest[str(p.relative_to(project))] = sha(p)
    args.out.mkdir(parents=True, exist_ok=False)
    (args.out / 'SUMMARY.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    (args.out / 'SOURCE_MANIFEST.json').write_text(json.dumps(manifest, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != 'arms'}, ensure_ascii=False))


if __name__ == '__main__': main()
