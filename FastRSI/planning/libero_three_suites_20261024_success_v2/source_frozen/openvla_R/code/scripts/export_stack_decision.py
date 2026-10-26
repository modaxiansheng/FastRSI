"""Export the task63 step120 probe as observed decision evidence, not a win.

This is a narrow, read-only translation of the two actual runs. It reuses the
procedure-learning input adapter without adding version fields to native
RESULT.json or presenting a regenerated prefix as an original simulator state.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastrsi.decision_corpus import checked_image, digest
from fastrsi.procedure_learning import clef_request, comparison, make_decision
from analyze_stack_flow import GOALS, summarize_attempts
from analyze_stack_remaining import analyze as analyze_remaining

SID = 'net_t63_i0'
STEP = 120


def read(path):
    return json.loads(path.read_text())


def lines(path):
    return [json.loads(line) for line in path.read_bytes().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check(condition, reason):
    if not condition:
        raise ValueError(reason)


def bound_case(stream):
    result = read(stream / 'RESULT.json')
    check(result.get('status') == 'COMPLETED' and len(result.get('cases', [])) == 1,
          'Expected one completed native root')
    row = result['cases'][0]
    expected = dict(id=SID, suite='libero_90', task_id=63, init_state_id=0,
                    task_description=GOALS[63], mode='development', max_steps=400,
                    feedback_window_mode='fixed10', planning_entry='public_session_flow',
                    development_suite_extension='libero90_stack_flow_v1')
    for key, value in expected.items():
        check(type(row['case'].get(key)) is type(value) and row['case'][key] == value,
              'Case binding mismatch: ' + key)
    check(read(stream.parent / 'ROSTER.json') == [row['case']], 'ROSTER/result mismatch')
    return row


def boundary_input(stream, row):
    """Bind a pre-action input to its root trace without exposing future actions."""
    path = stream / 'OS_COST_EVENTS.jsonl'
    raw_lines = path.read_bytes().splitlines()
    numbered = [(i + 1, raw, json.loads(raw)) for i, raw in enumerate(raw_lines) if raw.strip()]
    inputs = [(i, raw, event) for i, raw, event in numbered if event.get('event') == 'actual_policy_input']
    windows = [event for _, _, event in numbered if event.get('event') == 'feedback_window_choice']
    check(len(inputs) == len(windows), 'Policy input/window count mismatch')
    root = [(source, window) for source, window in zip(inputs, windows)
            if window.get('runtime_session_id') == SID]
    trace = lines(stream / 'stage_trace' / SID / 'TRACE.jsonl')
    check(len(root) == len(trace) - 1, 'Root input/trace count mismatch')
    selected = None
    for (source, window), observed in zip(root, trace[:-1]):
        _, _, event = source
        check(event.get('case') == window.get('case') == SID, 'Policy input ownership mismatch')
        check(window['step'] == observed['step'] and window['source_observation_id'] == observed['observation_id'],
              'Policy input order differs from observation order')
        check(window['mode'] == 'fixed10' and window['native_attempt'] == 0, 'Changed root execution rule')
        if observed['step'] == STEP:
            selected = (source, observed)
    check(selected is not None, 'Root did not expose a pre-action input at step120')
    (line_number, raw, event), observed = selected
    images = {}
    for key, suffix in [('observation/image', 'agentview_image'),
                        ('observation/wrist_image', 'robot0_eye_in_hand_image')]:
        name = f'{STEP:04d}_{suffix}.png'
        image_path = stream / 'stage_trace' / SID / name
        pixels_sha = checked_image(image_path.read_bytes(), observed['images'][name], event['array_bindings'][key])
        images[key] = dict(path=str(image_path.resolve()), file_sha256=sha(image_path), pixels_sha256=pixels_sha)
    return event, dict(source_path=str(path.resolve()), source_file_sha256=sha(path),
                       line_number=line_number, line_sha256=hashlib.sha256(raw).hexdigest(),
                       line_hash_scope='UTF-8 JSONL line bytes excluding newline',
                       observation_id=observed['observation_id'], images=images)


def branch_facts(stream, row, procedure_key):
    events = lines(stream / 'OS_COST_EVENTS.jsonl')
    windows = [e for e in events if e.get('event') == 'feedback_window_choice']
    traces = {s['session_id']: lines(stream / 'stage_trace' / s['session_id'] / 'TRACE.jsonl') for s in row['sessions']}
    attempts = summarize_attempts(row, traces, windows, lines(stream / 'TARGET_COST_EVENTS.jsonl'))
    worker = read(stream.parent / 'STATUS.json')
    check(worker.get('status') == 'COMPLETED', 'Worker has not completed')
    spans = defaultdict(float)
    for event in events:
        if event.get('event') == 'end':
            spans[event['name']] += event['duration_seconds']
    requests = sorted((stream / 'service_calls').glob('*/REQUEST.json'))
    tokens = defaultdict(int)
    responses_with_usage = 0
    for request_path in requests:
        response_path = request_path.with_name('RESPONSE.json')
        if response_path.exists():
            usage = read(response_path).get('usage')
            if isinstance(usage, dict) and all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')):
                responses_with_usage += 1
                for key in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
                    tokens[key] += usage[key]
    first = attempts[0]
    known = (len(attempts) == 1 and row.get('status') == row.get('chain_status') == 'PASSED'
             and first['runtime_success'] == first['environment_success']
             and first['verification_code'] == 'verified'
             and first['semantic_verdict'] == ('success' if first['environment_success'] else 'failure'))
    label_status = ('OBSERVED_SUCCESS' if first['environment_success'] else 'OBSERVED_FAILURE') if known else 'UNKNOWN'
    return dict(procedure_key=procedure_key, label_status=label_status,
        first_attempt_environment_success=first['environment_success'],
        final_environment_success=row['environment_success'], attempts=attempts,
        semantic_verdicts=[v.get('verdict') for v in row['outcomes']], chain_status=row['chain_status'],
        admission_status=row.get('experience_admission', {}).get('status'),
        admission_scope='Native experience admission is not candidate success or procedure approval.',
        cost=dict(case_seconds=row['seconds'], worker_seconds=worker['full_wall_seconds'],
            controller_seconds=worker['controller_seconds'], startup_seconds=worker['startup_seconds'],
            nested_stage_seconds=dict(spans), all_attempt_action_steps=sum(a['steps'] for a in attempts),
            all_attempt_vla_calls=len(windows), physical_resets=sum(a['physical_resets'] for a in attempts),
            root_prefix_action_steps=STEP, root_prefix_vla_calls=STEP // 10,
            root_remaining_action_steps=first['steps'] - STEP,
            root_remaining_vla_calls=first['vla_calls'] - STEP // 10,
            all_qwen_requests=len(requests),
            agent_requests=len(list(stream.glob('*/agent_call_*/REQUEST.json'))),
            verifier_requests=sum(str(read(p).get('messages', [{}])[0].get('content', '')).startswith('Verify the Agent session') for p in requests),
            qwen_usage=dict(tokens), qwen_responses_with_usage=responses_with_usage,
            qwen_usage_complete=responses_with_usage == len(requests),
            method_training_performed=False, prefix_is_included_in_full_cost=True,
            unavailable_costs=['GPU device-seconds', 'manual candidate design time'],
            costs_outside_this_branch=['Earlier two-task discovery batch and offline replay are in project_diagnostic_costs']),
        source_result_sha256=sha(stream / 'RESULT.json'), selection_label=None)


def outcome_summary(branches):
    """Keep real success differences apart from a trainable/comparative target."""
    statuses = {key: branch['label_status'] for key, branch in branches.items()}
    known = all(v in ('OBSERVED_SUCCESS', 'OBSERVED_FAILURE') for v in statuses.values())
    successful = [key for key, value in statuses.items() if value == 'OBSERVED_SUCCESS']
    if not known:
        relation = 'unresolved_outcome_or_multi_attempt_path'
    elif not successful:
        relation = 'both_observed_failure_no_success_advantage'
    elif len(successful) == 2:
        relation = 'both_observed_success_no_cost_preference_inferred'
    else:
        relation = successful[0] + '_observed_success_only'
    return dict(outcome_relation=relation, observed_success_advantage=successful[0] if known and len(successful) == 1 else None,
                preferred_choice=None, preferred_choice_status='UNKNOWN', training_example=None,
                cost_used_as_preference=False, learning_gain_established=False,
                warning='A seen post-hoc operation consequence is not learned timing, transfer, or net learning benefit.')


def export(run, reference):
    run, reference = run.resolve(), reference.resolve()
    intervention = run / 'F' / 'stream'
    required = [run / 'STATUS.json', run / 'BOUNDARY.json', run / 'PROTOCOL.json',
                intervention / 'RESULT.json', intervention.parent / 'STATUS.json', reference / 'RESULT.json']
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        return dict(status='WAITING_FOR_ACTUAL_RESULTS', missing=missing, preferred_choice=None, training_example=None)
    analysis = analyze_remaining(run, reference)
    spec = read(run / 'BOUNDARY.json')
    rows = {'N': bound_case(reference), 'F': bound_case(intervention)}
    baseline, baseline_source = boundary_input(reference, rows['N'])
    changed, changed_source = boundary_input(intervention, rows['F'])
    check(baseline['prompt'] == spec['goal'] and changed['prompt'] == spec['remaining_goal'], 'Actual boundary operations differ from artifact')
    check(baseline['array_bindings'] == changed['array_bindings'] and baseline['robot_state'] == changed['robot_state'],
          'Boundary observations differ between executed branches')
    model_paths = [reference.parent.parent / 'MODEL_CONFIG.json', run / 'MODEL_CONFIG.json']
    flow_paths = [reference.parent.parent / 'FLOW.json', run / 'FLOW.json']
    check(sha(model_paths[0]) == sha(model_paths[1]), 'Model configurations differ')
    check(sha(flow_paths[0]) == sha(flow_paths[1]), 'Flow configurations differ')
    procedures = [dict(id='continue_original_goal', version='1', steps=[spec['goal']], status='active'),
                  dict(id='execute_remaining_stack', version='1', steps=[spec['remaining_goal']], status='active')]
    decision = make_decision(baseline, procedures, execution_rule='fixed10',
        stage='same_episode_existing_goal_continuation', provenance=dict(mode='development',
            task_group='libero90_dual_bowl_stack_tray', observation_id=baseline_source['observation_id'],
            source_event_sha256=baseline_source['line_sha256'], protocol_sha256=sha(run / 'PROTOCOL.json'),
            start_binding=dict(kind='observed_inputs')))
    branches = {key: branch_facts(stream, rows[key], procedures[i]['id'] + '@1')
                for i, (key, stream) in enumerate([('N', reference), ('F', intervention)])}
    sources = set(required + model_paths + flow_paths + [reference.parent.parent.parent / 'PROTOCOL.json'])
    for stream, row in [(reference, rows['N']), (intervention, rows['F'])]:
        sources.update([stream.parent / 'STATUS.json', stream.parent / 'ROSTER.json',
                        stream / 'OS_COST_EVENTS.jsonl', stream / 'TARGET_COST_EVENTS.jsonl'])
        for session in row['sessions']:
            sources.add(stream / 'stage_trace' / session['session_id'] / 'TRACE.jsonl')
        sources.update((stream / 'service_calls').glob('*/*.json'))
        sources.update(stream.glob('*/agent_call_*/*.json'))
    sources.add(intervention / f'BOUNDARY_{SID}.json')
    for source in (baseline_source, changed_source):
        sources.update(Path(image['path']) for image in source['images'].values())
    project = run.parents[1]
    project_costs = []
    for label, relative in [('original_two_task_batch', 'runs/stack_flow_v1_20261004/STATUS.json'),
                            ('offline_exact_replay', 'planning/stack_flow_v1_20261004/replay_v1/STATUS.json')]:
        path = project / relative
        if path.exists():
            value = read(path)
            check(value['status'] == 'COMPLETED', 'Incomplete diagnostic cost source')
            project_costs.append(dict(label=label, seconds=value['seconds'], source=relative, sha256=sha(path)))
            sources.add(path)
    strict = comparison(decision, [])
    return dict(status='OBSERVED_DECISION_FACTS_EXPORTED', decision=decision,
        decision_request=clef_request(decision, procedures), available_procedures=procedures,
        operation_binding=dict(N='continue_original_goal@1', F='execute_remaining_stack@1',
            timing='Manually selected after inspecting the original development trajectory.',
            original_goal=spec['goal'], boundary_step=STEP, whole_goal_cap=400,
            remaining_action_cap=400 - STEP, success_standard='Original native whole-goal evaluator unchanged.'),
        current_input_evidence=dict(N=baseline_source, F=changed_source,
            kind='regenerated_observed_prefix', original_full_simulator_snapshot_available=False,
            observed_prefix_exactly_matched=True, guard_before_action_submission=True,
            prefix_observations=13, prefix_action_chunks=12,
            excluded_from_model_input=['privileged replay predicates', 'future actions', 'future outcomes', 'teacher explanations', 'initial-state ID']),
        branches=branches, outcome=outcome_summary(branches),
        adapter=dict(input_adapter='procedure_learning_v2', native_import_status='NOT_IMPORTED',
            native_import_reason='Unmodified native RESULT cases have boundary_probe or plain execution, not procedure_probe version bindings.',
            strict_comparison=strict,
            interpretation='The original-snapshot requirement belongs to this existing strict comparative adapter only; observation-based and negative-evidence learning remain possible. No universal snapshot requirement is asserted.'),
        project_diagnostic_costs=project_costs,
        incremental_cost=dict(new_experiment='F', reference_cost_already_paid='N',
            new_worker_seconds=branches['F']['cost']['worker_seconds'],
            new_controller_seconds=branches['F']['cost']['controller_seconds'],
            new_batch_wall_seconds=analysis['batch_wall_seconds'],
            learning_update_performed=False,
            interpretation='F fully pays regeneration, all execution attempts, verification, reporting and recorded admission. Wall times are from different batches; no speed preference is inferred. Nested spans, case, controller and worker times overlap and must not be added. Earlier recorded discovery and replay batches are separately retained; unmeasured human development stays unknown.'),
        source_sha256={str(path): sha(path) for path in sorted(sources)},
        scope='One already-seen development boundary. No new evaluation, successful procedure approval, trained selector, or efficiency claim.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='stack_remaining run root')
    parser.add_argument('reference', type=Path, help='Original task63/N/stream directory')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        value = export(args.run, args.reference)
    except (AssertionError, KeyError, ValueError, OSError) as error:
        value = dict(status='NOT_EXPORTABLE', reason=str(error) or type(error).__name__,
                     preferred_choice=None, training_example=None)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
    print(json.dumps({key: value[key] for key in ('status', 'reason', 'outcome', 'missing') if key in value}, indent=2))
    if value['status'] != 'OBSERVED_DECISION_FACTS_EXPORTED':
        sys.exit(2)
