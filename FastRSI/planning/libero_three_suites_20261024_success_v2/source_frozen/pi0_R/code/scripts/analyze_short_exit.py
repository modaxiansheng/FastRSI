"""Read-only analysis of the single short-window-exit intervention.

All output goes to planning, never back into immutable run directories. Missing
or interrupted evidence stays missing; post-intervention trajectories are not
aligned by step and are never spliced into a counterfactual rollout.
"""
import argparse
import collections
import hashlib
import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'short_exit_v1_20261003'
ARRAY_KEYS = ('observation/image', 'observation/state', 'observation/wrist_image')


def read_json(path, errors, sources):
    if not path.exists():
        errors.append({'path': str(path), 'reason': 'missing'})
        return {}
    data = path.read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    try:
        return json.loads(data)
    except (ValueError, UnicodeError) as exc:
        errors.append({'path': str(path), 'reason': 'invalid_json', 'error': str(exc)})
        return {}


def read_lines(path, errors, sources):
    if not path.exists():
        errors.append({'path': str(path), 'reason': 'missing'})
        return []
    data = path.read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    rows = []
    for number, line in enumerate(data.splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except (ValueError, UnicodeError) as exc:
            errors.append({'path': str(path), 'line': number,
                           'reason': 'invalid_jsonl_record', 'error': str(exc)})
    return rows


def span_costs(events):
    """Completed spans are subtotals, not an additive decomposition."""
    begins = {(e.get('process'), e.get('span_id')): e for e in events
              if e.get('event') == 'begin'}
    finished = set()
    totals = collections.Counter()
    counts = collections.Counter()
    issues = []
    for e in events:
        if e.get('event') != 'end':
            continue
        key = (e.get('process'), e.get('span_id'))
        duration = e.get('duration_seconds')
        if key in finished or key not in begins or not isinstance(duration, (int, float)) or not math.isfinite(duration) or duration < 0:
            issues.append({'span_id': e.get('span_id'), 'reason': 'unmatched_duplicate_or_invalid_end'})
            continue
        finished.add(key)
        name = begins[key].get('name', e.get('name', 'unknown'))
        totals[name] += duration
        counts[name] += 1
    return {'completed_seconds_by_name': dict(totals), 'completed_counts_by_name': dict(counts),
            'open_spans': [{'name': e.get('name'), 'span_id': e.get('span_id')}
                           for k, e in begins.items() if k not in finished], 'issues': issues}


def finite_actions(value):
    return (isinstance(value, list) and len(value) == 10 and
            all(isinstance(a, list) and len(a) == 7 and
                all(isinstance(v, (int, float)) and math.isfinite(v) for v in a) for a in value))


def valid_bindings(value):
    if not isinstance(value, dict) or any(k not in value for k in ARRAY_KEYS):
        return False
    return all(isinstance(value[k], dict) and isinstance(value[k].get('shape'), list)
               and isinstance(value[k].get('dtype'), str)
               and re.fullmatch(r'[0-9a-f]{64}', str(value[k].get('sha256', '')))
               for k in ARRAY_KEYS)


def compare_prefix(reference, candidate):
    """Strict evidence test through the intervention input, not after its output."""
    rc, sc = reference['choices'], candidate['choices']
    ri, si = reference['inputs'], candidate['inputs']
    rt, st = reference['submitted'], candidate['submitted']
    probes = [i for i, e in enumerate(sc)
              if e.get('short_exit_probe', {}).get('intervention_applied') is True]
    result = {'status': 'NOT_ESTABLISHED', 'comparison': 'R gripper_transition vs S short_exit_once',
              'interventions_recorded': len(probes), 'fork_step': None,
              'nodes_checked_through_fork': 0, 'issues': [],
              'post_fork_alignment': 'Not required or attempted; paths may immediately diverge.'}
    if len(probes) != 1:
        result['issues'].append('Expected exactly one recorded intervention; missing trigger is not a failed task.')
        return result
    fork = probes[0]
    result.update(fork_step=sc[fork].get('step'), fork_choice_index=fork)
    if any(len(rows) <= fork for rows in (rc, ri, si, rt, st)):
        result['issues'].append('Missing paired policy input, choice or actual submitted actions through fork.')
        return result
    if any(e.get('short_exit_probe', {}).get('intervention_applied') is True for e in rc):
        result['issues'].append('Reference records an intervention.')
    checks = []
    for i in range(fork + 1):
        a, b, ia, ib = rc[i], sc[i], ri[i], si[i]
        check = {'index': i, 'step_R': a.get('step'), 'step_S': b.get('step'), 'issues': []}
        failures = check['issues']
        if a.get('step') != b.get('step') or a.get('source_observation_id') != b.get('source_observation_id'):
            failures.append('observation_index_mismatch')
        if not isinstance(ia.get('prompt'), str) or ia.get('prompt') != ib.get('prompt'):
            failures.append('prompt_missing_or_mismatch')
        if not valid_bindings(ia.get('array_bindings')) or not valid_bindings(ib.get('array_bindings')):
            failures.append('complete_input_array_binding_missing')
        elif ia['array_bindings'] != ib['array_bindings']:
            failures.append('input_array_binding_mismatch')
        if not isinstance(ia.get('robot_state'), list) or not ia['robot_state'] or ia.get('robot_state') != ib.get('robot_state'):
            failures.append('robot_state_missing_or_mismatch')
        if not finite_actions(a.get('predicted_actions')) or not finite_actions(b.get('predicted_actions')):
            failures.append('complete_numeric_prediction_missing')
        elif a['predicted_actions'] != b['predicted_actions']:
            failures.append('full_prediction_mismatch')
        for arm, choice, actual in [('R', a, rt[i]), ('S', b, st[i])]:
            n = choice.get('submitted_steps')
            if not isinstance(n, int) or actual.get('actions') != choice.get('predicted_actions', [])[:n]:
                failures.append(arm + '_actual_submission_not_predicted_prefix')
        if i < fork:
            if a.get('submitted_steps') != b.get('submitted_steps') or rt[i].get('actions') != st[i].get('actions'):
                failures.append('pre_fork_submitted_actions_mismatch')
            if rt[i].get('step') != st[i].get('step'):
                failures.append('pre_fork_actual_execution_end_mismatch')
            if i > 0 and sc[i - 1].get('submitted_steps') == 5 and b.get('submitted_steps') == 5:
                failures.append('earlier_consecutive_short_was_not_intervened')
        else:
            if (a.get('requested_steps'), a.get('submitted_steps'), b.get('requested_steps'), b.get('submitted_steps')) != (5, 5, 10, 10):
                failures.append('fork_is_not_original_5_vs_forced_10')
            if i == 0 or rc[i - 1].get('submitted_steps') != 5 or sc[i - 1].get('submitted_steps') != 5:
                failures.append('previous_actual_submission_not_5')
            elif a.get('step') != rc[i - 1].get('step', -10) + 5 or b.get('step') != sc[i - 1].get('step', -10) + 5:
                failures.append('fork_not_immediately_after_five_executed_steps')
            if not (a.get('within_chunk_transition') or a.get('boundary_transition')):
                failures.append('original_gripper_rule_not_short')
        checks.append(check)
    result['checks'] = checks
    result['nodes_checked_through_fork'] = len(checks)
    result['issues'].extend({'index': c['index'], 'reason': reason} for c in checks for reason in c['issues'])
    result['fork'] = {'R_requested': rc[fork].get('requested_steps'),
                      'S_requested': sc[fork].get('requested_steps'),
                      'R_submitted': rc[fork].get('submitted_steps'),
                      'S_submitted': sc[fork].get('submitted_steps'),
                      'candidate_probe': sc[fork].get('short_exit_probe'),
                      'decision_diagnostics': sc[fork].get('decision_diagnostics')}
    result['status'] = 'PASSED' if not result['issues'] else 'NOT_ESTABLISHED'
    return result


def analyze_arm(group, arm):
    errors, sources = [], {}
    status = read_json(group / 'STATUS.json', errors, sources)
    roster = read_json(group / 'ROSTER.json', errors, sources)
    stream = group / 'stream'
    result = read_json(stream / 'RESULT.json', errors, sources)
    rows = result.get('cases', [])
    case = rows[0].get('case', {}) if rows else roster[0] if isinstance(roster, list) and roster else {}
    if not rows and case.get('id') and (stream / case['id'] / 'RESULT.json').exists():
        rows = [read_json(stream / case['id'] / 'RESULT.json', errors, sources)]
    row = rows[0] if rows else {}
    events = read_lines(stream / 'OS_COST_EVENTS.jsonl', errors, sources)
    target_events = read_lines(stream / 'TARGET_COST_EVENTS.jsonl', errors, sources)
    trace = read_lines(stream / 'stage_trace' / case.get('id', '_unknown') / 'TRACE.jsonl', errors, sources)
    choices = [e for e in events if e.get('event') == 'feedback_window_choice']
    inputs = [e for e in events if e.get('event') == 'actual_policy_input']
    submitted = [e for e in trace if e.get('method') == 'action_chunk']
    if len(choices) != len(inputs) or len(choices) != len(submitted):
        errors.append({'reason': 'incomplete_input_choice_submission_counts',
                       'inputs': len(inputs), 'choices': len(choices), 'submissions': len(submitted)})
    os_spans, target_spans = span_costs(events), span_costs(target_events)
    native = row.get('native_terminal_by_session', {})
    steps = sum(v['steps'] for v in native.values()) if native and all(isinstance(v.get('steps'), (int, float)) for v in native.values()) else None
    outcomes = row.get('outcomes', [])
    phases = os_spans['completed_seconds_by_name']
    qwen = []
    for p in sorted((stream / 'service_calls').glob('*/RECEIPT.json')):
        qwen.append(read_json(p, errors, sources))
    incomplete_qwen_fields = {k: sum(not isinstance(r.get(k), (int, float)) for r in qwen)
                              for k in ['prompt_tokens', 'completion_tokens', 'generation_seconds', 'seconds']}
    summary = {'arm': arm, 'worker_status': status.get('status', 'MISSING'),
               'result_status': result.get('status'), 'task': case.get('task_id'), 'init': case.get('init_state_id'),
               'mode': case.get('feedback_window_mode'), 'case_id': case.get('id'),
               'environment_success': row.get('environment_success'),
               'native_terminal_by_session': native, 'actual_steps': steps,
               'last_observed_steps_if_terminal_missing': trace[-1].get('step') if trace and steps is None else None,
               'verification_status': row.get('verification_status'),
               'semantic_verdicts': [o.get('verdict') for o in outcomes],
               'verification_records': [{k: o.get(k) for k in ('session_id', 'ok', 'code', 'status', 'verdict')} for o in outcomes],
               'chain_status': row.get('chain_status'), 'admission': row.get('experience_admission'),
               'vla_input_calls': len(inputs), 'recorded_window_choices': len(choices),
               'actual_action_submissions': len(submitted),
               'requested_window_counts': dict(collections.Counter(e.get('requested_steps') for e in choices)),
               'submitted_window_counts': dict(collections.Counter(e.get('submitted_steps') for e in choices)),
               'policy_client_completed_seconds': phases.get('policy_client'),
               'selector_seconds': sum(e.get('selector_seconds', 0) for e in choices),
               'full_wall_seconds': status.get('full_wall_seconds'), 'startup_seconds': status.get('startup_seconds'),
               'controller_seconds': status.get('controller_seconds', result.get('seconds')),
               'report_seconds': phases.get('native_terminal_report'),
               'verification_seconds': phases.get('semantic_verification'),
               'os_spans': os_spans, 'target_spans': target_spans,
               'qwen_receipts': {'calls': len(qwen),
                   'known_sums': {k: sum(r[k] for r in qwen if isinstance(r.get(k), (int, float))) for k in incomplete_qwen_fields},
                   'missing_counts': incomplete_qwen_fields},
               'recorded_interventions': sum(e.get('short_exit_probe', {}).get('intervention_applied') is True for e in choices),
               'errors_or_missing_evidence': errors}
    return summary, {'choices': choices, 'inputs': inputs, 'submitted': submitted}, sources


def known_sum(rows, key):
    values = [r.get(key) for r in rows]
    return {'known_sum': sum(v for v in values if isinstance(v, (int, float))),
            'missing_rows': sum(not isinstance(v, (int, float)) for v in values)}


def analyze_run(run, provisioning_receipt=None):
    errors, sources = [], {}
    status = read_json(run / 'STATUS.json', errors, sources)
    protocol = read_json(run / 'PROTOCOL.json', errors, sources)
    arms = protocol.get('active_arms', ['N', 'F', 'R', 'S'])
    pairs, rows = [], []
    for folder in sorted(run.glob('pair*')):
        if not folder.is_dir():
            continue
        pair_status = read_json(folder / 'STATUS.json', errors, sources)
        pair = {'pair': folder.name, 'status': pair_status.get('status'),
                'parallel_pair_wall_seconds': pair_status.get('full_wall_seconds'), 'rows': []}
        details = {}
        for arm in arms:
            summary, details[arm], bound = analyze_arm(folder / arm, arm)
            summary['pair'] = folder.name
            sources.update(bound)
            pair['rows'].append(summary)
            rows.append(summary)
        if 'R' in details and 'S' in details:
            pair['common_prefix'] = compare_prefix(details['R'], details['S'])
        pairs.append(pair)
    totals = {}
    for arm in arms:
        own = [r for r in rows if r['arm'] == arm]
        totals[arm] = {'episodes_present': len(own),
                       'completed_workers': sum(r['worker_status'] == 'COMPLETED' for r in own),
                       'native_successes': sum(r['environment_success'] is True for r in own),
                       'native_failures': sum(r['environment_success'] is False for r in own),
                       'native_unknown': sum(r['environment_success'] is None for r in own),
                       **{k: known_sum(own, k) for k in ['full_wall_seconds', 'controller_seconds', 'actual_steps',
                            'vla_input_calls', 'policy_client_completed_seconds', 'selector_seconds']}}
    provision = None
    if provisioning_receipt:
        provision = {'source': str(provisioning_receipt),
                     'receipt': read_json(provisioning_receipt, errors, sources),
                     'accounting': 'Provisioning may overlap experiment execution. Preserved separately, not added to campaign wall or arm lifecycles.'}
    return {'scope': 'Two-condition four-arm development intervention; no learned gain, representative-quality or noninferiority claim.',
            'run': str(run), 'campaign_status': status.get('status'), 'protocol': protocol,
            'pairs': pairs, 'totals_by_arm': totals,
            'cost_ledger': {'parallel_campaign_wall_seconds': status.get('full_wall_seconds'),
                'sum_worker_lifecycle_seconds': known_sum(rows, 'full_wall_seconds'),
                'lifecycle_unit': 'sum of per-arm two-GPU reservation durations; not parallel wall time or measured utilization',
                'nonadditive_components': 'Startup, controller, report, verification and nested OS/target spans may overlap. Never sum them into a new total; they are already inside worker lifecycle.',
                'provisioning': provision, 'human_and_unrecorded_preparation_seconds': None},
            'errors_or_missing_evidence': errors,
            'source_sha256': sources,
            'limitations': ['All selected initial conditions are previously used development information.',
                'Incomplete and failed arms are retained; absent costs remain unknown.',
                'Common-prefix PASS verifies recorded legal policy inputs and actions, not unobserved simulator state.',
                'Different post-fork step indices are expected and never matched as the same state.',
                'Only this protocol is analyzed; no pooling with old report or timing protocols.'],
            'default_promoted': False, 'learning_benefit_claim': None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default=NAME)
    ap.add_argument('--root', type=Path, default=ROOT)
    ap.add_argument('--output', type=Path)
    ap.add_argument('--provisioning-receipt', type=Path)
    args = ap.parse_args()
    if not re.fullmatch(r'[a-zA-Z0-9_]+', args.name):
        raise ValueError('Invalid run name')
    root = args.root.resolve()
    run = root / 'runs' / args.name
    out = (args.output or root / 'planning' / args.name / 'ANALYSIS.json').resolve()
    if out.is_relative_to((root / 'runs').resolve()):
        raise ValueError('Analysis must not mutate immutable run artifacts')
    value = analyze_run(run, args.provisioning_receipt)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)
    print(json.dumps({'output': str(out), 'campaign_status': value['campaign_status'],
                      'prefix': [{k: p['common_prefix'].get(k) for k in ('status', 'fork_step', 'interventions_recorded')}
                                 for p in value['pairs'] if 'common_prefix' in p],
                      'totals': value['totals_by_arm']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
