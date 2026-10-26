"""Read-only fixed-10/fixed-5 structural screening; no selector fitting.

Both arms share the resident deployment implementation. Initial inputs must
match, but subsequent trajectories are allowed to diverge because the submitted
actions differ. Every failure, disagreement and paid branch remains in the bill.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re
import struct

from analyze_resident_full_os import (ARRAY_KEYS, analyze_case, digest, known_sum,
                                    read_json, service_index)
from analyze_short_exit import finite_actions, read_lines, valid_bindings

ROOT = Path(__file__).resolve().parents[1]
NAME = 'structural_window_v1_20261003'
MODES = {'N': ('fixed10', 10), 'F': ('fixed5', 5)}


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def difference(short, long):
    return short - long if numeric(short) and numeric(long) else None


def input_issues(value):
    issues = []
    if value.get('keys') != sorted([*ARRAY_KEYS, 'prompt']) or not isinstance(value.get('prompt'), str):
        issues.append('input_schema_missing_or_unsupported')
    bindings = value.get('array_bindings')
    if not valid_bindings(bindings):
        issues.append('complete_input_array_binding_missing')
    else:
        for key in ARRAY_KEYS:
            shape, dtype = bindings[key]['shape'], bindings[key]['dtype']
            if key == 'observation/state':
                if shape != [8] or dtype != 'float32': issues.append('state_shape_or_dtype_wrong')
            elif (len(shape) != 3 or any(type(n) is not int or n <= 0 for n in shape)
                  or (shape[0] != 3 and shape[-1] != 3) or dtype != 'uint8'):
                issues.append('image_shape_or_dtype_wrong')
    state = value.get('robot_state')
    if not isinstance(state, list) or len(state) != 8 or not all(numeric(x) for x in state):
        issues.append('finite_robot_state_missing')
    elif isinstance(bindings, dict) and isinstance(bindings.get('observation/state'), dict):
        try:
            sha = hashlib.sha256(struct.pack('<8f', *state)).hexdigest()
            if sha != bindings['observation/state'].get('sha256'):
                issues.append('robot_state_does_not_match_recorded_float32_binding')
        except (OverflowError, struct.error):
            issues.append('robot_state_not_representable_as_float32')
    return issues


def audit_trajectory(details, arm, max_steps=520):
    mode, horizon = MODES[arm]
    counts = {key: len(details[key]) for key in ('inputs', 'choices', 'submitted')}
    issues, checks = [], []
    if len(set(counts.values())) != 1 or not counts['inputs']:
        issues.append({'reason': 'missing_or_unequal_records', 'counts': counts})
    for i in range(min(counts.values())):
        inp, choice, actual = details['inputs'][i], details['choices'][i], details['submitted'][i]
        failures = input_issues(inp)
        if 'session_goals' in details:
            goal = details['session_goals'].get(actual.get('session_id'))
            if not isinstance(goal, str) or inp.get('prompt') != goal:
                failures.append('policy_prompt_not_bound_to_actual_native_session')
        step, submitted = choice.get('step'), choice.get('submitted_steps')
        if choice.get('mode') != mode or choice.get('requested_steps') != horizon:
            failures.append('predeclared_fixed_window_not_executed')
        if type(step) is not int or not 0 <= step < max_steps:
            failures.append('invalid_predecision_step')
        elif submitted != min(horizon, max_steps - step):
            failures.append('wrong_submitted_window_length')
        if choice.get('source_observation_id') != 'libero_obs_' + str(step):
            failures.append('decision_not_bound_to_current_observation')
        if not finite_actions(choice.get('predicted_actions')):
            failures.append('complete_finite_10x7_prediction_missing')
        if type(submitted) is not int or not 1 <= submitted <= 10:
            failures.append('invalid_submitted_steps')
        elif actual.get('actions') != choice.get('predicted_actions', [])[:submitted]:
            failures.append('submission_not_exact_prediction_prefix')
        end = actual.get('step')
        if type(step) is not int or type(end) is not int or type(submitted) is not int or not step < end <= step + submitted:
            failures.append('executed_step_count_inconsistent')
        checks.append({'index': i, 'step': step, 'issues': failures})
        issues.extend({'index': i, 'reason': reason} for reason in failures)
    if not details.get('trace'):
        issues.append({'reason': 'target_trace_missing'})
    if details.get('source_errors'):
        issues.append({'reason': 'source_missing_or_invalid', 'count': len(details['source_errors'])})
    return {'status': 'PASSED' if not issues else 'NOT_ESTABLISHED', 'counts': counts,
            'checked_decisions': len(checks), 'checks': checks, 'issues': issues,
            'scope': 'Chronological records within this executed arm; no counterfactual trajectory reconstruction.'}


def initial_pairing(left, right):
    issues = []
    if not left.get('inputs') or not right.get('inputs'):
        return {'status': 'NOT_ESTABLISHED', 'issues': ['initial_policy_input_missing']}
    a, b = left['inputs'][0], right['inputs'][0]
    issues.extend('N_' + x for x in input_issues(a)); issues.extend('F_' + x for x in input_issues(b))
    for key in ('keys', 'prompt', 'array_bindings', 'robot_state'):
        if a.get(key) != b.get(key): issues.append('initial_' + key + '_mismatch')
    for arm, data in [('N', left), ('F', right)]:
        if not data.get('choices') or data['choices'][0].get('step') != 0:
            issues.append(arm + '_first_choice_not_initial_state')
    reset = [[r for r in d.get('trace', []) if r.get('method') == 'reset'] for d in (left, right)]
    if not all(reset):
        issues.append('initial_reset_trace_missing')
    elif any(reset[0][0].get(k) != reset[1][0].get(k) for k in ('step', 'observation_id', 'native_success', 'images')):
        issues.append('initial_reset_observation_mismatch')
    predictions = [d['choices'][0].get('predicted_actions') if d.get('choices') else None for d in (left, right)]
    first_prediction_equal = all(finite_actions(p) for p in predictions) and predictions[0] == predictions[1]
    if not first_prediction_equal: issues.append('same_initial_input_prediction_missing_or_mismatch')
    return {'status': 'PASSED' if not issues else 'NOT_ESTABLISHED', 'issues': issues,
            'initial_complete_prediction_equal': first_prediction_equal,
            'post_first_action_equivalence_required': False,
            'scope': 'Equal initial observations and stochastic policy start; different committed prefixes may legitimately change all later observations.'}


def quality_state(row):
    native = row.get('environment_success')
    outcomes = row.get('semantic_outcomes') or []
    final = outcomes[-1].get('verdict') if outcomes else None
    final_conflict = type(native) is bool and final in ('success', 'failure') and native != (final == 'success')
    admission = row.get('experience_admission') or {}
    attempts = (row.get('admission_receipt') or {}).get('attempts', [])
    conflicts = [a for a in attempts if a.get('rejection') == 'runtime_semantic_conflict'
                 or (type(a.get('runtime_success')) is bool and a.get('verdict') in ('success', 'failure')
                     and a['runtime_success'] != (a['verdict'] == 'success'))]
    conflict = final_conflict or bool(conflicts) or any('runtime_semantic_conflict' in str(r) for r in admission.get('reasons') or [])
    return {'native_success': native, 'final_semantic_verdict': final,
            'native_semantic_conflict': conflict, 'final_native_semantic_conflict': final_conflict,
            'conflicting_attempts': conflicts, 'chain_status': row.get('chain_status'),
            'admission_status': admission.get('status'), 'admission_reasons': admission.get('reasons'),
            'reliable_success_observation': native is True and final == 'success' and row.get('chain_status') == 'PASSED'
                and admission.get('status') == 'ADMITTED',
            'label_scope': 'Admission records agreed raw development feedback; it does not certify narrative accuracy or a transferable skill.'}


def begin_receipt(row):
    """ResidentClient wraps the wire receipt with request/status/latency evidence."""
    begin = (row.get('resident_session') or {}).get('begin') or {}
    receipt = begin.get('receipt') or {}
    if begin.get('operation') != 'begin' or receipt.get('op') != 'begin':
        return {}
    return receipt


def compare_screening(long, short, paired):
    issues = []
    fields = ('suite', 'task_id', 'init_state_id', 'task_description', 'max_steps', 'mode', 'terminal_evidence_view')
    if any(long.get('case', {}).get(k) != short.get('case', {}).get(k) for k in fields):
        issues.append('task_or_evaluation_protocol_mismatch')
    snapshots = [r.get('initial_experience', {}) for r in (long, short)]
    if (not all(s.get('actual_equals_manifest') for s in snapshots)
            or snapshots[0].get('manifest_sha256') != snapshots[1].get('manifest_sha256')):
        issues.append('initial_experience_not_matched')
    starts = [begin_receipt(r) for r in (long, short)]
    if not all(s.get('rng_hashes') for s in starts) or starts[0].get('rng_hashes') != starts[1].get('rng_hashes'):
        issues.append('postload_rng_start_not_matched')
    for arm, row in [('N', long), ('F', short)]:
        if row.get('case', {}).get('mode') != 'development': issues.append(arm + '_not_development')
        if row.get('trajectory_audit', {}).get('status') != 'PASSED': issues.append(arm + '_trajectory_incomplete')
        if row.get('chain_status') != 'PASSED': issues.append(arm + '_chain_incomplete')
        if row.get('resident_session', {}).get('release_confirmed') is not True: issues.append(arm + '_resident_not_released')
        if row.get('independent_initial_lineage') is not True: issues.append(arm + '_initial_lineage_not_independent')
    if paired.get('status') != 'PASSED': issues.append('initial_pairing_not_established')
    native = [r.get('environment_success') for r in (long, short)]
    if issues or not all(type(x) is bool for x in native):
        finding = 'INCONCLUSIVE_BINDING_OR_INCOMPLETE_CHAIN'
    elif native == [False, True]: finding = 'SHORT_WINDOW_NATIVE_CAPABILITY_ADVANTAGE_OBSERVED'
    elif native == [True, False]: finding = 'LONG_WINDOW_NATIVE_CAPABILITY_ADVANTAGE_OBSERVED'
    elif native == [False, False]: finding = 'BOTH_NATIVE_FAILED'
    else: finding = 'BOTH_NATIVE_SUCCEEDED_COMPARE_OBSERVED_COSTS'
    qn, qf = quality_state(long), quality_state(short)
    costs = ('worker_full_wall_seconds', 'controller_seconds', 'os_execution', 'policy_client',
             'semantic_verification', 'native_terminal_report', 'experience_admission')
    return {'finding': finding, 'issues': issues, 'N_quality': qn, 'F_quality': qf,
            'semantic_conflict_retained': qn['native_semantic_conflict'] or qf['native_semantic_conflict'],
            'observed_F_minus_N': {**{k: difference(short.get('costs', {}).get(k), long.get('costs', {}).get(k)) for k in costs},
                                   'VLA_calls': difference(short.get('VLA_calls'), long.get('VLA_calls')),
                                   'executed_steps': difference(short.get('executed_steps'), long.get('executed_steps'))},
            'interpretation': 'One paid paired development observation. No threshold fitting, stable winner, noninferiority, task-family generalization or learning-gain claim.'}


def analyze_development_case(group, services, arm):
    # Both N and F use resident services; reuse its validated source/image parser.
    summary, details, sources = analyze_case(group, services, 'R')
    summary['arm'] = arm
    errors = summary['errors_or_missing_evidence']
    raw = read_json(group / 'stream/RESULT.json', errors, sources)
    rows = raw.get('cases', [])
    row = rows[0] if rows else {}
    details['session_goals'] = {s['session_id']: s.get('task_description') for s in row.get('sessions', []) if s.get('session_id')}
    if not summary.get('case'):
        plan = read_json(group.parent / 'PLAN.json', errors, sources)
        if plan.get('cases'): summary['case'] = plan['cases'][0]
    case = summary.get('case', {})
    summary['experience_admission'] = row.get('experience_admission')
    summary['evidence_layers'] = row.get('evidence_layers')
    path = group / 'stream' / case.get('id', '_unknown') / 'admission/ADMISSION.json'
    summary['admission_receipt'] = read_json(path, errors, sources) if path.exists() else None
    if summary['admission_receipt'] is None:
        errors.append({'reason': 'admission_missing_not_assumed_success', 'path': str(path)})
    accepted = row.get('accepted_source')
    expected = (group.parent / 'initial_experience').resolve().as_posix()
    # Local delivery and container roots differ; compare their complete run-relative path.
    expected_suffix = expected.split('/runs/', 1)[-1]
    accepted_suffix = str(accepted).split('/runs/', 1)[-1]
    summary['accepted_source'] = accepted
    summary['independent_initial_lineage'] = (isinstance(accepted, str) and accepted_suffix == expected_suffix
        and (summary['admission_receipt'] or {}).get('parent') == accepted)
    before, after = summary.get('experience_before') or {}, summary.get('experience_after') or {}
    summary['development_changed_persistent_paths'] = [p for p in sorted(set(before) | set(after)) if before.get(p) != after.get(p)]
    summary['experience_hash_scope'] = 'Development changes are allowed inside this case and fully disclosed. The next case is independently copied from the common initial snapshot.'
    summary.pop('evaluation_persistence_unchanged', None)
    phases = summary['costs']['os_spans']['completed_seconds_by_name']
    summary['costs']['experience_admission'] = phases.get('experience_admission')
    summary['costs']['environment_reset'] = summary['costs']['target_spans']['completed_seconds_by_name'].get('environment_reset')
    summary['costs']['resident_policy_reset_server_seconds'] = begin_receipt(summary).get('reset_seconds')
    summary['VLA_calls'] = len(details['inputs'])
    native = summary.get('native_terminal') or []
    summary['executed_steps'] = sum(n['steps'] for n in native) if native and all(type(n.get('steps')) is int for n in native) else None
    summary['trajectory_audit'] = audit_trajectory(details, arm, case.get('max_steps', 520))
    summary['quality'] = quality_state(summary)
    return summary, details, sources


def analyze_run(run):
    errors, sources = [], {}
    protocol = read_json(run / 'PROTOCOL.json', errors, sources)
    hosts, all_cases = {}, []
    for host_name in ('host187', 'host188'):
        host = run / host_name
        host_errors = []
        status = read_json(host / 'STATUS.json', host_errors, sources)
        branches, details = {}, {}
        for arm in ('N', 'F'):
            branch = host / arm
            state = read_json(branch / 'STATUS.json', host_errors, sources)
            services = service_index(branch / 'pool/qwen/service_calls', host_errors, sources)
            cases = []
            for pair in sorted(branch.glob('pair*')):
                if not pair.is_dir(): continue
                summary, trace, bound = analyze_development_case(pair / arm, services, arm)
                summary.update(host=host_name, pair=pair.name)
                cases.append(summary); all_cases.append(summary)
                details[arm, pair.name] = trace; sources.update(bound)
            assigned = {r['service_directory'] for c in cases for r in c['qwen_service_attribution']['rows'] if r['status'] == 'ATTRIBUTED'}
            receipts = [s['receipt'] for s in services]
            branches[arm] = {'status': state.get('status'), 'branch_full_wall_seconds': state.get('full_wall_seconds'),
                             'pool_startup_seconds_once': state.get('pool_startup_seconds'), 'cases': cases,
                             'serial_hot_worker_lifecycle_sum': known_sum([c['costs'] for c in cases], 'worker_full_wall_seconds'),
                             'scope': 'One resident two-GPU branch on this host. Startup is already inside branch wall; cases and nested spans are not additional bills.',
                             'qwen_service_ledger': {'all_recorded_calls': len(services),
                                'costs': {k: known_sum(receipts, k) for k in ('seconds', 'generation_seconds', 'queue_seconds', 'prompt_tokens', 'completion_tokens')},
                                'unattributed_service_calls': [s['directory'] for s in services if s['directory'] not in assigned],
                                'receipts': services}}
        comparisons = []
        for name in sorted({pair for _, pair in details}):
            a = next((c for c in branches['N']['cases'] if c['pair'] == name), None)
            b = next((c for c in branches['F']['cases'] if c['pair'] == name), None)
            if a is None or b is None:
                comparisons.append({'pair': name, 'finding': 'MISSING_ARM'}); continue
            paired = initial_pairing(details['N', name], details['F', name])
            comparisons.append({'pair': name, 'task_id': a['case'].get('task_id'), 'init_state_id': a['case'].get('init_state_id'),
                                'initial_pairing': paired, 'screening': compare_screening(a, b, paired)})
        hosts[host_name] = {'status': status.get('status'), 'host_parallel_wall_seconds': status.get('full_wall_seconds'),
                            'branches': branches, 'comparisons': comparisons,
                            'branch_wall_F_minus_N': difference(branches['F']['branch_full_wall_seconds'], branches['N']['branch_full_wall_seconds']),
                            'errors_or_missing_evidence': host_errors}
    expected = protocol.get('maximum_first_batch_executions', 8)
    completed = sum(c.get('worker_status') == 'COMPLETED' for c in all_cases)
    manifests = {c.get('initial_experience', {}).get('manifest_sha256') for c in all_cases}
    return {'run': str(run), 'protocol': protocol, 'hosts': hosts,
            'analyzer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'screening_status': 'ALL_PREDECLARED_WORKERS_COMPLETED' if len(all_cases) == expected and completed == expected else 'PARTIAL_KEEP_ALL_EVIDENCE',
            'counts': {'observed_workers': len(all_cases), 'expected_workers': expected, 'completed_workers': completed,
                       'native_successes': sum(c.get('environment_success') is True for c in all_cases),
                       'native_failures': sum(c.get('environment_success') is False for c in all_cases),
                       'native_unknown': sum(type(c.get('environment_success')) is not bool for c in all_cases),
                       'semantic_conflicts': sum(c['quality']['native_semantic_conflict'] for c in all_cases),
                       'admitted': sum(c['quality']['admission_status'] == 'ADMITTED' for c in all_cases),
                       'quarantined': sum(c['quality']['admission_status'] == 'QUARANTINED' for c in all_cases)},
            'all_observed_cases_start_from_same_independent_snapshot': bool(all_cases) and len(manifests) == 1
                and all(c.get('independent_initial_lineage') and c.get('initial_experience', {}).get('actual_equals_manifest') for c in all_cases),
            'cost_accounting': {'main': 'Host and branch wall clocks remain separate. No sum over hosts is labelled single-device elapsed time.',
                                'startup': 'Model pool startup is paid once per branch and remains inside its measured wall.',
                                'hot': 'Each fresh Target/runner case has its own lifecycle; model reset, verification, reporting and admission spans are disclosed without adding nested times.',
                                'failures': 'Failed, conflicted and incomplete executions are retained. Missing cost remains unknown.',
                                'qwen': 'Case spans and complete pool service totals are separate; identity-less service calls stay unattributed.',
                                'human_development_seconds': None},
            'source_sha256': sources, 'errors_or_missing_evidence': errors,
            'default_promoted': False, 'selector_fitted': False, 'learning_benefit_claim': None,
            'interpretation': 'Bounded structural development screening only. Windows are interventions, so matching post-intervention trajectories is not a criterion.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--name', default=NAME)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.name): raise ValueError('Invalid run name')
    root = args.root.resolve(); out = (args.output or root / 'planning' / args.name / 'ANALYSIS.json').resolve()
    if out.is_relative_to(root / 'runs'): raise ValueError('Never write analysis into immutable runs')
    result = analyze_run(root / 'runs' / args.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle: json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps({'output': str(out), 'status': result['screening_status'], 'counts': result['counts'],
                      'comparisons': {h: v['comparisons'] for h, v in result['hosts'].items()}}, indent=2))


if __name__ == '__main__': main()
