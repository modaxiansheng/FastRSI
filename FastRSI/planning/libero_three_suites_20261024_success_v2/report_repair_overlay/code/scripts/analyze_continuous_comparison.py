"""Compare two real, isolated streams without attributing static gains to learning."""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from analyze_native_stream import analyze, lines, read, sha


def reduction(a, b):
    return {'P': a, 'F': b, 'P_minus_F': a-b,
            'observed_reduction_percent': 100*(a-b)/a if a else None}


def disjoint_cost(events, controller_seconds):
    """Partition selected orchestration intervals; expose overlaps instead of adding them."""
    names = {'native_agent_planning', 'public_session_flow_planning', 'os_execution',
             'semantic_verification', 'native_terminal_report', 'experience_admission',
             'development_budget_checkpoint'}
    starts, endpoints = {}, []
    for event in events:
        if event.get('name') not in names:
            continue
        if event.get('event') == 'begin':
            starts[event['span_id']] = event
        elif event.get('event') == 'end':
            begin = starts.pop(event['span_id'])
            assert begin['process'] == event['process'] and begin['mono_ns'] <= event['mono_ns']
            endpoints.extend([(begin['mono_ns'], begin['name'], 1), (event['mono_ns'], begin['name'], -1)])
    assert not starts, 'Open phase spans cannot produce a final cost partition'
    active, buckets, previous = Counter(), defaultdict(float), None
    for timestamp, name, change in sorted(endpoints):
        if previous is not None and active:
            buckets[' + '.join(sorted(active))] += (timestamp-previous)/1e9
        active[name] += change
        if not active[name]:
            del active[name]
        previous = timestamp
    covered = sum(buckets.values())
    assert covered <= controller_seconds+0.01, (covered, controller_seconds)
    buckets['other_controller_setup_shutdown_and_gaps'] = controller_seconds-covered
    return dict(buckets)


def summarize(run):
    run = Path(run)
    groups = {arm: analyze(run/arm) for arm in ('P', 'F')}
    events = {arm: lines(run/arm/'stream/OS_COST_EVENTS.jsonl') for arm in groups}
    pairs = []
    keys = ('id', 'suite', 'task_id', 'init_state_id', 'task_description', 'max_steps',
            'mode', 'feedback_window_mode', 'native_experience_maintenance')
    for left, right in zip(groups['P']['cases'], groups['F']['cases']):
        case_id = left['case']['id']
        inputs = {arm: [e for e in events[arm] if e.get('event') == 'actual_policy_input'
                       and e.get('case') == case_id] for arm in groups}
        windows = {arm: [e for e in events[arm] if e.get('event') == 'feedback_window_choice'
                        and e.get('case') == case_id] for arm in groups}
        pairs.append({
            'id': case_id,
            'protocol_fields_equal': all(left['case'].get(k) == right['case'].get(k) for k in keys),
            'first_policy_arrays_equal': bool(inputs['P'] and inputs['F']) and
                inputs['P'][0]['array_bindings'] == inputs['F'][0]['array_bindings'],
            'actual_policy_prompts_match_goal': bool(inputs['P'] and inputs['F']) and all(
                e.get('prompt') == left['case']['task_description'] for values in inputs.values() for e in values),
            'all_policy_input_arrays_equal': [e['array_bindings'] for e in inputs['P']] ==
                [e['array_bindings'] for e in inputs['F']],
            'all_predicted_actions_equal': bool(windows['P'] and windows['F']) and
                [e.get('predicted_actions') for e in windows['P']] ==
                [e.get('predicted_actions') for e in windows['F']],
            'prediction_trace_counts_match_policy_calls': all(
                len(windows[arm]) == len(inputs[arm]) and bool(inputs[arm]) for arm in groups),
            'case_seconds': reduction(left['case_seconds'], right['case_seconds']),
            'environment_success': {'P': left['environment_success'], 'F': right['environment_success']},
            'semantic_verdicts': {'P': left['semantic_verdicts'], 'F': right['semantic_verdicts']},
            'vla_calls': reduction(left['vla_calls'], right['vla_calls']),
            'agent_calls': reduction(left['agent_model_calls'], right['agent_model_calls']),
            'planning_calls': reduction(left['agent_calls_by_phase'].get('native_agent_planning', 0),
                                        right['agent_calls_by_phase'].get('native_agent_planning', 0)),
        })
    totals = {}
    for arm, result in groups.items():
        result['disjoint_controller_cost_seconds'] = disjoint_cost(events[arm], result['controller_seconds'])
        cases = result['cases']
        totals[arm] = {'controller_seconds': result['controller_seconds'],
            'full_arm_lifecycle_seconds': result['full_wall_seconds'],
            'startup_seconds': result['run_status']['startup_seconds'],
            'vla_calls': sum(c['vla_calls'] for c in cases),
            'action_steps': sum(t['steps'] for c in cases for t in c['runtime_terminals'].values()),
            'planning_calls': sum(c['agent_calls_by_phase'].get('native_agent_planning', 0) for c in cases),
            'agent_calls': sum(c['agent_model_calls'] for c in cases),
            'verifier_calls': sum(c['verifier_model_calls'] for c in cases),
            'environment_successes': sum(c['environment_success'] is True for c in cases),
            'qwen_service_calls': result['qwen_service_requests_total']}
    deltas = {k: reduction(totals['P'][k], totals['F'][k]) for k in totals['P']}
    model_configs_equal = read(run/'P/MODEL_CONFIG.json') == read(run/'F/MODEL_CONFIG.json')
    checkpoints = [read(p) for p in (run/'F/stream').glob('budget_checkpoint_*/RECEIPT.json')]
    inputs_bound = len(pairs) == 3 and model_configs_equal and all(
        p['protocol_fields_equal'] and p['first_policy_arrays_equal'] and
        p['actual_policy_prompts_match_goal'] and p['prediction_trace_counts_match_policy_calls'] for p in pairs)
    closed = read(run/'CONTAINER_AUDIT.json')
    complete = read(run/'STATUS.json')['status'] == 'COMPLETED' and all(
        result['status'] == 'PASSED' for result in groups.values())
    source_records = [run/'STATUS.json', run/'PROTOCOL.json'] + [
        run/arm/rel for arm in groups for rel in ('STATUS.json', 'stream/RESULT.json', 'MODEL_CONFIG.json')]
    return {'status': 'PASSED' if complete and inputs_bound else 'INCOMPLETE_OR_FAILED',
        'scope': 'Same-stream public engineering optimization plus budget integration; no independent learning-effect arm',
        'source_sha256': {str(p.relative_to(run)): sha(p) for p in source_records},
        'groups': groups, 'paired_cases': pairs, 'model_configs_equal': model_configs_equal,
        'totals': totals, 'observed_differences': deltas, 'budget_checkpoints': checkpoints,
        'campaign_wall_seconds': read(run/'STATUS.json')['full_wall_seconds'],
        'owned_containers_stopped': bool(closed) and all(not c['running'] for c in closed),
        'learning_efficiency_gain_established': False,
        'limits': ['All three initial states were previously seen in development.',
                   'One parallel placement, no crossover or repeated timing interval.',
                   'Failure, reporting, validation, admission and budget overhead remain charged.',
                   'Parallel campaign wall is not per-method speedup.',
                   'Stage timers can nest or overlap and must not be added as an exclusive cost sum.',
                   'Reading inherited experience does not itself prove an improved decision.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = summarize(args.run)
    with Path(args.output).open('x') as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    print(json.dumps({'status': result['status'], 'totals': result['totals']}, ensure_ascii=False))
