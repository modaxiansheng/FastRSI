"""Read-only fresh/resident OS regression and nonadditive cost ledger.

No approximate action matching, trajectory splicing, timestamp-based Qwen call
attribution, or writing into immutable runs. This evaluates common service reuse,
not a learning gain or held-out robot quality.
"""
import argparse
import collections
import hashlib
import json
import math
from pathlib import Path
import re

from analyze_short_exit import (ARRAY_KEYS, finite_actions, read_json, read_lines,
                                span_costs, valid_bindings)

ROOT = Path(__file__).resolve().parents[1]
NAME = 'resident_full_os_v1_20261003'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def known_sum(rows, key):
    values = [r.get(key) for r in rows]
    valid = lambda x: type(x) in (int, float) and math.isfinite(x) and x >= 0
    return {'known_sum': sum(x for x in values if valid(x)),
            'missing_or_invalid': sum(not valid(x) for x in values)}


def compare_trajectories(left, right):
    """Compare every chronological decision and target observation, without tolerance."""
    result = {'status': 'NOT_ESTABLISHED', 'issues': [], 'checks': [],
              'scope': 'All recorded legal inputs, complete 10x7 predictions, submitted commands and target image hashes.',
              'ignored_fields': ['process', 'wall_ns', 'span UUIDs', 'selector latency',
                                 'other timing metadata', 'Session labels mapped by execution order']}
    lengths = {arm: {key: len(data[key]) for key in ('inputs', 'choices', 'submitted')}
               for arm, data in [('N', left), ('R', right)]}
    result['counts'] = lengths
    if len({v for row in lengths.values() for v in row.values()}) != 1 or not lengths['N']['inputs']:
        result['issues'].append({'reason': 'missing_or_unequal_trajectory_counts', 'counts': lengths})
    count = min(v for row in lengths.values() for v in row.values())
    for i in range(count):
        issues = []
        ia, ib = left['inputs'][i], right['inputs'][i]
        a, b = left['choices'][i], right['choices'][i]
        aa, ab = left['submitted'][i], right['submitted'][i]
        if ia.get('keys') != sorted([*ARRAY_KEYS, 'prompt']) or ib.get('keys') != ia.get('keys'):
            issues.append('input_key_set_missing_or_unsupported')
        if not isinstance(ia.get('prompt'), str) or ia.get('prompt') != ib.get('prompt'):
            issues.append('task_prompt_missing_or_mismatch')
        if not valid_bindings(ia.get('array_bindings')) or not valid_bindings(ib.get('array_bindings')):
            issues.append('complete_input_binding_missing')
        elif ia['array_bindings'] != ib['array_bindings']:
            issues.append('input_array_binding_mismatch')
        for arm, value in [('N', ia), ('R', ib)]:
            state = value.get('robot_state')
            if not isinstance(state, list) or len(state) != 8 or not all(type(v) in (int, float) and math.isfinite(v) for v in state):
                issues.append(arm + '_complete_robot_state_missing')
        if ia.get('robot_state') != ib.get('robot_state'):
            issues.append('robot_state_mismatch')
        for key in ('step', 'source_observation_id', 'requested_steps', 'submitted_steps', 'mode'):
            if key not in a or key not in b or a[key] != b[key]:
                issues.append(key + '_missing_or_mismatch')
        for arm, choice, actual in [('N', a, aa), ('R', b, ab)]:
            if not finite_actions(choice.get('predicted_actions')):
                issues.append(arm + '_complete_prediction_missing')
            n = choice.get('submitted_steps')
            if type(n) is not int or not 1 <= n <= 10 or actual.get('actions') != choice.get('predicted_actions', [])[:n]:
                issues.append(arm + '_submission_not_exact_predicted_prefix')
            if type(choice.get('step')) is not int or type(actual.get('step')) is not int:
                issues.append(arm + '_execution_index_missing')
            elif type(n) is int and not choice['step'] < actual['step'] <= choice['step'] + n:
                issues.append(arm + '_execution_index_inconsistent')
        if a.get('predicted_actions') != b.get('predicted_actions'):
            issues.append('full_prediction_mismatch')
        if aa.get('actions') != ab.get('actions'):
            issues.append('submitted_action_mismatch')
        for key in ('step', 'observation_id', 'native_success', 'images'):
            if key not in aa or key not in ab or aa[key] != ab[key]:
                issues.append('target_' + key + '_missing_or_mismatch')
        result['checks'].append({'index': i, 'step_N': a.get('step'), 'step_R': b.get('step'), 'issues': issues})
        result['issues'].extend({'index': i, 'reason': issue} for issue in issues)
    if not left['trace'] or not right['trace']:
        result['issues'].append({'reason': 'target_trace_missing'})
    # Include reset and final observations; every extra or missing event matters.
    def trace_values(rows):
        return [{k: e.get(k) for k in ('method', 'step', 'observation_id', 'native_success', 'images', 'actions')}
                for e in rows]
    if trace_values(left['trace']) != trace_values(right['trace']):
        result['issues'].append({'reason': 'complete_target_trace_mismatch'})
    if left.get('source_errors') or right.get('source_errors'):
        result['issues'].append({'reason': 'missing_or_invalid_source_evidence'})
    result['checked_decisions'] = count
    result['status'] = 'PASSED' if not result['issues'] else 'NOT_ESTABLISHED'
    return result


def tool_ids(response):
    if isinstance(response.get('choices'), list) and response['choices']:
        response = response['choices'][0].get('message', {})
    return tuple(sorted(t.get('id') for t in response.get('tool_calls', []) or [] if isinstance(t.get('id'), str)))


def service_index(root, errors, sources):
    """Index entire service log including failed or unowned calls."""
    rows = []
    if not root.is_dir():
        errors.append({'reason': 'missing_service_ledger', 'path': str(root)})
        return rows
    for folder in sorted(root.glob('call_*')):
        if not folder.is_dir():
            continue
        receipt = read_json(folder / 'RECEIPT.json', errors, sources)
        response = read_json(folder / 'RESPONSE.json', errors, sources) if (folder / 'RESPONSE.json').exists() else {}
        rows.append({'directory': str(folder), 'response_id': response.get('id'),
                     'tool_call_ids': list(tool_ids(response)), 'receipt': receipt})
    return rows


def attribute_calls(case_folder, services, errors, sources):
    """Use saved response identities only. Pure text without saved ID stays unknown."""
    rows = []
    for path in sorted(case_folder.glob('*_call_*/RESPONSE.json')):
        response = read_json(path, errors, sources)
        ident = response.get('id') or response.get('response_id')
        ids = tool_ids(response)
        matches = [s for s in services if (ident and s['response_id'] == ident)
                   or (ids and tuple(s['tool_call_ids']) == ids)]
        row = {'case_response': str(path), 'role': path.parent.name.split('_call_', 1)[0],
               'saved_response_id': ident, 'saved_tool_call_ids': list(ids)}
        if len(matches) == 1:
            row.update(status='ATTRIBUTED', method='response_id' if ident else 'exact_tool_call_ids',
                       service_directory=matches[0]['directory'], service_response_id=matches[0]['response_id'])
        else:
            row.update(status='UNATTRIBUTED', reason='missing_saved_response_identity' if not (ident or ids) else 'ambiguous_or_missing_service_identity',
                       candidate_count=len(matches))
        rows.append(row)
    assigned = [r['service_directory'] for r in rows if r['status'] == 'ATTRIBUTED']
    duplicates = [path for path, n in collections.Counter(assigned).items() if n > 1]
    chosen = [s['receipt'] for s in services if s['directory'] in set(assigned)]
    return {'rows': rows, 'attributed_count': len(assigned), 'unattributed_count': len(rows) - len(assigned),
            'duplicate_attribution': duplicates, 'known_partial_service_costs': {
                key: known_sum(chosen, key) for key in ('seconds', 'generation_seconds', 'queue_seconds', 'prompt_tokens', 'completion_tokens')},
            'scope': 'Only identity-linked calls. No inference from clock proximity, prose similarity or case order.'}


def initial_binding(pair, errors, sources):
    manifest = read_json(pair / 'EXPERIENCE_BINDING.json', errors, sources)
    actual = {}
    for path in sorted((pair / 'initial_experience').rglob('*')):
        if path.is_file():
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            sources[str(path)] = sha
            actual[str(path.relative_to(pair / 'initial_experience'))] = sha
    return {'manifest_sha256': digest(manifest), 'files': len(actual),
            'actual_equals_manifest': bool(actual) and actual == manifest}


def analyze_case(group, services, arm):
    errors, sources = [], {}
    status = read_json(group / 'STATUS.json', errors, sources)
    result = read_json(group / 'stream/RESULT.json', errors, sources)
    rows = result.get('cases', [])
    if len(rows) != 1:
        errors.append({'reason': 'expected_exactly_one_case', 'actual': len(rows)})
    row = rows[0] if rows else {}
    case = row.get('case', {})
    stream = group / 'stream'
    events = read_lines(stream / 'OS_COST_EVENTS.jsonl', errors, sources)
    targets = read_lines(stream / 'TARGET_COST_EVENTS.jsonl', errors, sources)
    session_ids = [s['session_id'] for s in row.get('sessions', []) if s.get('session_id')]
    if not session_ids and case.get('id'):
        session_ids = [case['id']]
    trace = []
    for session in session_ids:
        directory = stream / 'stage_trace' / session
        part = read_lines(directory / 'TRACE.jsonl', errors, sources)
        for item in part:
            for name, expected in item.get('images', {}).items():
                image = directory / name
                if not image.is_file():
                    errors.append({'reason': 'missing_trace_image', 'path': str(image)})
                else:
                    sha = hashlib.sha256(image.read_bytes()).hexdigest()
                    sources[str(image)] = sha
                    if sha != expected:
                        errors.append({'reason': 'trace_image_hash_mismatch', 'path': str(image)})
        trace.extend(part)
    details = {'inputs': [e for e in events if e.get('event') == 'actual_policy_input'],
               'choices': [e for e in events if e.get('event') == 'feedback_window_choice'],
               'submitted': [e for e in trace if e.get('method') == 'action_chunk'], 'trace': trace,
               'source_errors': errors}
    os_spans, target_spans = span_costs(events), span_costs(targets)
    phases = os_spans['completed_seconds_by_name']
    before, after = row.get('experience_before'), row.get('experience_after')
    outcomes = [{key: o.get(key) for key in ('ok', 'code', 'status', 'verdict')} for o in row.get('outcomes', [])]
    native = [{key: record.get(key) for key in ('method', 'steps', 'native_success', 'runtime_success')}
              for _, record in row.get('native_terminal_by_session', {}).items()]
    summary = {'arm': arm, 'case': case, 'worker_status': status.get('status'),
               'result_status': result.get('status'), 'environment_success': row.get('environment_success'),
               'verification_status': row.get('verification_status'), 'semantic_outcomes': outcomes,
               'chain_status': row.get('chain_status'), 'native_terminal': native,
               'terminal_report_status': row.get('terminal_report', {}).get('status'),
               'terminal_report_complete': row.get('terminal_report', {}).get('read_coverage', {}).get('complete'),
               'experience_before': before, 'experience_after': after,
               'evaluation_persistence_unchanged': bool(before) and before == after,
               'initial_experience': initial_binding(group.parent, errors, sources),
               'trajectory_counts': {k: len(v) for k, v in details.items() if isinstance(v, list) and k != 'source_errors'},
               'costs': {'worker_full_wall_seconds': status.get('full_wall_seconds'),
                         'worker_startup_seconds': status.get('startup_seconds'),
                         'controller_seconds': status.get('controller_seconds', result.get('seconds')),
                         **{name: phases.get(name) for name in ('os_execution', 'policy_client', 'semantic_verification',
                               'native_terminal_report', 'agent_model_call', 'verifier_model_call',
                               'resident_policy_begin', 'resident_policy_release', 'resident_policy_handoff')},
                         'os_spans': os_spans, 'target_spans': target_spans,
                         'accounting': 'Components overlap and nest. Do not add them to worker/branch wall or service-internal totals.'},
               'qwen_service_attribution': attribute_calls(stream / case.get('id', '_unknown'), services, errors, sources),
               'errors_or_missing_evidence': errors}
    if arm == 'R':
        summary['resident_session'] = read_json(stream / 'RESIDENT_SESSION.json', errors, sources)
    return summary, details, sources


def compare_case_results(left, right):
    fields = ('environment_success', 'verification_status', 'semantic_outcomes', 'chain_status',
              'native_terminal', 'terminal_report_status', 'terminal_report_complete')
    differences = {key: {'N': left.get(key), 'R': right.get(key)} for key in fields if left.get(key) != right.get(key)}
    issues = []
    for arm, row in [('N', left), ('R', right)]:
        if (row.get('worker_status') != 'COMPLETED' or row.get('result_status') != 'COMPLETED'
                or row.get('chain_status') != 'PASSED' or row.get('terminal_report_status') != 'PASSED'
                or not row.get('terminal_report_complete')):
            issues.append(arm + '_full_chain_not_passed')
        if type(row.get('environment_success')) is not bool or not row.get('native_terminal'):
            issues.append(arm + '_native_terminal_evidence_missing')
        outcomes = row.get('semantic_outcomes')
        if (not isinstance(outcomes, list) or not outcomes
                or any(o.get('code') != 'verified' or not o.get('verdict') for o in outcomes)
                or not row.get('verification_status')):
            issues.append(arm + '_semantic_verification_evidence_missing')
        if row.get('case', {}).get('mode') != 'evaluation' or not row.get('evaluation_persistence_unchanged'):
            issues.append(arm + '_evaluation_persistence_not_proved')
        if not row.get('initial_experience', {}).get('actual_equals_manifest'):
            issues.append(arm + '_initial_snapshot_not_bound')
    if left.get('initial_experience', {}).get('manifest_sha256') != right.get('initial_experience', {}).get('manifest_sha256'):
        issues.append('initial_experience_snapshot_mismatch')
    a, b = left.get('experience_before') or {}, right.get('experience_before') or {}
    raw_diffs = [p for p in sorted(set(a) | set(b)) if a.get(p) != b.get(p)]
    resident = right.get('resident_session', {})
    if resident.get('status') != 'COMPLETED' or resident.get('release_confirmed') is not True:
        issues.append('resident_session_release_not_proved')
    return {'status': 'PASSED' if not issues and not differences else 'NOT_ESTABLISHED',
            'differences': differences, 'issues': issues,
            'raw_cross_arm_experience_before_equal': a == b and bool(a),
            'raw_cross_arm_experience_before_differing_paths': raw_diffs,
            'experience_comparison': 'Per-arm before/after hashes are exact. Initial accepted snapshot equality is separately bound; differing runtime registry/request hashes are disclosed, not silently normalized.'}


def analyze_run(run):
    errors, sources = [], {}
    status = read_json(run / 'STATUS.json', errors, sources)
    protocol = read_json(run / 'PROTOCOL.json', errors, sources)
    branches, pairs = {}, []
    raw = {}
    for arm in ('N', 'R'):
        branch = run / arm
        state = read_json(branch / 'STATUS.json', errors, sources)
        roots = [branch / 'pool/qwen/service_calls'] if arm == 'R' else [p / arm / 'stream/service_calls' for p in sorted(branch.glob('pair*')) if p.is_dir()]
        services = [s for root in roots for s in service_index(root, errors, sources)]
        cases = []
        for pair in sorted(branch.glob('pair*')):
            if not pair.is_dir():
                continue
            summary, details, bound = analyze_case(pair / arm, services, arm)
            sources.update(bound); summary['pair'] = pair.name; cases.append(summary)
            raw[(arm, pair.name)] = details
        assigned = {r['service_directory'] for c in cases for r in c['qwen_service_attribution']['rows'] if r['status'] == 'ATTRIBUTED'}
        receipts = [s['receipt'] for s in services]
        branches[arm] = {'status': state.get('status'), 'branch_full_wall_seconds': state.get('full_wall_seconds'),
                         'pool_startup_seconds': state.get('pool_startup_seconds'), 'cases': cases,
                         'qwen_service_ledger': {'all_recorded_calls': len(services),
                            'ledger_directories_present': all(root.is_dir() for root in roots) and bool(roots),
                            'costs': {key: known_sum(receipts, key) for key in ('seconds', 'generation_seconds', 'queue_seconds', 'prompt_tokens', 'completion_tokens')},
                            'unattributed_service_calls': [s['directory'] for s in services if s['directory'] not in assigned],
                            'receipts': services},
                         'handoff_probe': None}
        if arm == 'R' and (branch / 'handoff_probe/replay/RESULT.json').exists():
            branches[arm]['handoff_probe'] = read_json(branch / 'handoff_probe/replay/RESULT.json', errors, sources)
            branches[arm]['handoff_probe_accounting'] = 'A separate diagnostic after both OS cases, included in the recorded resident branch wall. Not robot task work or learning cost.'
    pair_names = sorted({p for _, p in raw})
    for name in pair_names:
        rows = {arm: next((c for c in branches[arm]['cases'] if c['pair'] == name), None) for arm in ('N', 'R')}
        if not all(rows.values()):
            pairs.append({'pair': name, 'status': 'MISSING_ARM'}); continue
        fields = ('suite', 'task_id', 'init_state_id', 'task_description', 'max_steps', 'mode', 'feedback_window_mode', 'terminal_evidence_view')
        config_match = all(rows['N']['case'].get(k) == rows['R']['case'].get(k) for k in fields)
        pairs.append({'pair': name, 'case_configuration_equal': config_match,
                      'trajectory': compare_trajectories(raw['N', name], raw['R', name]),
                      'quality_and_isolation': compare_case_results(rows['N'], rows['R'])})
    expected = len(protocol.get('cases', []))
    complete = bool(expected) and len(pairs) == expected and all(
        p.get('case_configuration_equal') and p.get('trajectory', {}).get('status') == 'PASSED'
        and p.get('quality_and_isolation', {}).get('status') == 'PASSED' for p in pairs)
    return {'run': str(run), 'campaign_status': status.get('status'), 'protocol': protocol,
            'full_os_regression': 'PASSED' if complete and status.get('status') == 'COMPLETED' else 'NOT_ESTABLISHED',
            'pairs': pairs, 'branches': branches,
            'cost_ledger': {'campaign_parallel_wall_seconds': status.get('full_wall_seconds'),
                'primary_comparison': 'Independent complete branch wall from branch start through cleanup; no waiting for the other branch. R includes separately disclosed diagnostic probe.',
                'nonadditive': 'Nested OS spans, target spans, service spans, pool lifetime and branch wall describe overlapping intervals. No summed synthetic total is computed.',
                'qwen_attribution': 'Exact saved response or tool-call identity only. Missing IDs remain unattributed, while branch service totals include every recorded call.',
                'human_development_seconds': None},
            'source_sha256': sources, 'errors_or_missing_evidence': errors,
            'scope': 'Common service residency on known development instances under evaluation isolation. No learned improvement, held-out success-rate or 30-percent claim.',
            'default_promoted': False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--name', default=NAME)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_]+', args.name): raise ValueError('Invalid run name')
    root = args.root.resolve()
    out = (args.output or root / 'planning' / args.name / 'ANALYSIS.json').resolve()
    if out.is_relative_to(root / 'runs'): raise ValueError('Never write analysis into immutable runs')
    result = analyze_run(root / 'runs' / args.name)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as handle: json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
    print(json.dumps({'output': str(out), 'full_os_regression': result['full_os_regression'],
                      'pairs': [{k: v for k, v in p.items() if k != 'trajectory'} |
                                {'trajectory_status': p.get('trajectory', {}).get('status')}
                                for p in result['pairs']]}, indent=2))


if __name__ == '__main__': main()
