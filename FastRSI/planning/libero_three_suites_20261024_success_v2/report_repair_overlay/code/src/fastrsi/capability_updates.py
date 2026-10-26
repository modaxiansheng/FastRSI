"""Bounded capability applications at completed, independent Session boundaries.

Agent requests and automatic rules submit the same JSON operations. This module
never creates tasks, resets an environment, schedules an extra validation, calls
an LLM, or promotes a new skill into the accepted set. Its output binds the next
ordinary task to a frozen choice; a separate worker consumes that binding.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

from .experience_compiler import canonical_sha, validate_compiled

OPERATIONS = ('propose_current_success', 'trial_candidate', 'set_next_choice')
LIMITS = {'max_candidates': 2, 'max_probation_uses': 2}
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]{0,159}$')
LIFECYCLE_KIND = 'single_revalidation_v1'


def _policy(value):
    if (not isinstance(value, dict) or set(value) != {'kind', 'ordinary_sessions'} or
            value['kind'] != LIFECYCLE_KIND):
        raise ValueError('Unknown optional lifecycle policy')
    sessions = value['ordinary_sessions']
    if not isinstance(sessions, list) or not 4 <= len(sessions) <= 8 or len(set(sessions)) != len(sessions):
        raise ValueError('Preregister four to eight distinct ordinary Sessions')
    for sid in sessions:
        _identifier(sid)
    return value


def _lifecycle_record():
    return dict(suspension_session=None, cooldown_session=None, revalidation_session=None,
                revalidation_uses=0, revalidation_success=None, events=[])


def _lifecycle_event(candidate, sid, result_sha, event):
    candidate['lifecycle']['events'].append(dict(session_id=sid, result_sha256=result_sha, event=event))


def _validate_lifecycle(candidate, policy, completed=None):
    record = candidate['lifecycle']; expected = set(_lifecycle_record())
    if not isinstance(record, dict) or set(record) != expected:
        raise ValueError('Malformed lifecycle accounting')
    order = policy['ordinary_sessions']; suspension = record['suspension_session']
    if type(record['revalidation_uses']) is not int or record['revalidation_uses'] not in (0, 1):
        raise ValueError('One separate revalidation use is allowed')
    if record['revalidation_success'] is not None and type(record['revalidation_success']) is not bool:
        raise ValueError('Revalidation outcome must be explicit')
    if (record['revalidation_uses'] == 1) != (record['revalidation_success'] is not None):
        raise ValueError('Revalidation use/outcome accounting differs')
    if suspension is None:
        if any(record[k] is not None for k in ('cooldown_session', 'revalidation_session')) or record['revalidation_uses']:
            raise ValueError('Revalidation without prior suspension')
        if candidate['status'] in ('suspended', 'revalidation_pending'):
            raise ValueError('Suspended state without a source failure')
    else:
        if suspension not in order or candidate['probation_successes'] != 2:
            raise ValueError('Suspension requires two actual prior probation successes')
        index = order.index(suspension)
        cooldown = order[index+1] if index+1 < len(order) else None
        retry = order[index+2] if index+2 < len(order) else None
        if record['cooldown_session'] != cooldown or record['revalidation_session'] != retry:
            raise ValueError('Revalidation opportunity shifted from the registered ordinary task order')
        if completed is not None:
            if suspension not in completed:
                raise ValueError('Suspension lacks completed same-lane evidence')
            if candidate['status'] == 'revalidation_pending' and (cooldown not in completed or retry in completed):
                raise ValueError('Revalidation was not preceded by an actual completed cooldown task')
            if record['revalidation_uses'] and retry not in completed:
                raise ValueError('Revalidation result missing from completed ordinary tasks')
    if not isinstance(record['events'], list):
        raise ValueError('Lifecycle history must be retained')
    seen = set()
    allowed = {'SUSPENDED_AFTER_FAILURE', 'COOLDOWN_COMPLETED', 'REVALIDATION_RESERVED',
               'REVALIDATION_CANCELLED', 'REVALIDATION_SUCCESS', 'REVALIDATION_FAILURE', 'PERMANENTLY_RETIRED',
               'PROBATION_SUCCESS', 'PROBATION_FAILURE'}
    for event in record['events']:
        if (set(event) != {'session_id', 'result_sha256', 'event'} or event['session_id'] not in order or
                event['event'] not in allowed or (event['session_id'], event['event']) in seen):
            raise ValueError('Invalid or duplicate lifecycle history')
        seen.add((event['session_id'], event['event'])); _hash(event['result_sha256'])
        if completed is not None and completed.get(event['session_id']) != event['result_sha256']:
            raise ValueError('Lifecycle history was not produced by this lane')
    successes = sum(event['event'] == 'PROBATION_SUCCESS' for event in record['events'])
    trials = sum(event['event'] in ('PROBATION_SUCCESS', 'PROBATION_FAILURE') for event in record['events'])
    if successes != candidate['probation_successes'] or trials != candidate['probation_uses']:
        raise ValueError('Original probation evidence was removed or recounted')
    if suspension is not None and (suspension, 'SUSPENDED_AFTER_FAILURE') not in seen:
        raise ValueError('Original failure history was removed')
    if candidate['status'] == 'revalidation_pending' and (record['cooldown_session'], 'REVALIDATION_RESERVED') not in seen:
        raise ValueError('Revalidation reservation history missing')
    if record['revalidation_uses']:
        outcome = 'REVALIDATION_SUCCESS' if record['revalidation_success'] else 'REVALIDATION_FAILURE'
        if (record['revalidation_session'], outcome) not in seen:
            raise ValueError('Revalidation outcome history missing')


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _hash(value):
    if not isinstance(value, str) or not re.fullmatch('[0-9a-f]{64}', value):
        raise ValueError('Expected a SHA256 digest')
    return value


def _identifier(value):
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise ValueError('Unsafe or empty identifier')
    return value


def _pairs(items):
    result = {}
    for key, value in items:
        if key in result:
            raise ValueError('Duplicate JSON key')
        result[key] = value
    return result


def _read(path):
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Artifact symlinks are not permitted')
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=_pairs,
                       parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    return value, {'path': str(path), 'sha256': _sha(raw)}


def _read_trace(path):
    """Runtime TRACE is JSONL; a JSON array is also accepted for CPU fixtures."""
    path = Path(path).absolute()
    if path.is_symlink() or any(p.is_symlink() for p in path.parents):
        raise ValueError('Artifact symlinks are not permitted')
    raw = path.read_bytes()
    parse = lambda text: json.loads(text, object_pairs_hook=_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Nonfinite JSON')))
    value = parse(raw) if raw.lstrip().startswith(b'[') else [parse(line) for line in raw.splitlines() if line.strip()]
    if not isinstance(value, list):
        raise ValueError('Expected root trace rows')
    return value, {'path': str(path), 'sha256': _sha(raw)}


def _dump(path, value):
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as handle:
        handle.write(raw)
    return _sha(raw)


def _artifact(ref):
    if not isinstance(ref, dict) or set(ref) != {'path', 'sha256'}:
        raise ValueError('Expected an immutable path and raw SHA256 binding')
    value, actual = _read(ref['path'])
    if actual['sha256'] != _hash(ref['sha256']):
        raise ValueError('Bound artifact changed')
    return value


def _scope(value):
    if (not isinstance(value, dict) or set(value) != {'suite', 'task_id', 'goal'} or
            not isinstance(value['suite'], str) or not value['suite'] or
            type(value['task_id']) is not int or value['task_id'] < 0 or
            not isinstance(value['goal'], str) or not value['goal'].strip()):
        raise ValueError('Invalid original task scope')
    return value


def case_scope(case):
    return _scope({'suite': case['suite'], 'task_id': case['task_id'], 'goal': case['task_description']})


def initialize_state(scope, *, accepted_choices=None, default_choice='native', lifecycle_policy=None):
    """Bind existing accepted skills; never rewrite or demote their artifacts.

    Omission preserves the v1 one-failure retirement policy byte schema. The
    opt-in v2 policy preregisters a finite task order before its first source
    task; it is not a migration or reinterpretation of already-run histories.
    """
    accepted = {'native': None}
    for choice, ref in (accepted_choices or {}).items():
        if choice in ('native', 'current'):
            raise ValueError('Reserved accepted choice')
        _identifier(choice)
        accepted[choice] = copy.deepcopy(ref)
    state = dict(version=1, kind='capability_update_state', scope=copy.deepcopy(_scope(scope)),
                 limits=dict(LIMITS), default_choice=default_choice, accepted=accepted,
                 candidates={}, next_choice=default_choice, completed_sessions={})
    if lifecycle_policy is not None:
        state.update(version=2, lifecycle_policy=copy.deepcopy(_policy(lifecycle_policy)))
    return validate_state(state)


def validate_state(state):
    fields = {'version', 'kind', 'scope', 'limits', 'default_choice', 'accepted',
              'candidates', 'next_choice', 'completed_sessions'}
    policy = _policy(state.get('lifecycle_policy')) if state.get('version') == 2 else None
    if policy is not None:
        fields.add('lifecycle_policy')
    if (set(state) != fields or state['version'] not in (1, 2) or state['kind'] != 'capability_update_state' or
            state['limits'] != LIMITS):
        raise ValueError('Unknown capability state or enlarged budget')
    _scope(state['scope'])
    accepted = state['accepted']; candidates = state['candidates']
    if not isinstance(accepted, dict) or accepted.get('native', 'absent') is not None:
        raise ValueError('Native fallback missing')
    if not isinstance(candidates, dict) or len(candidates) > LIMITS['max_candidates']:
        raise ValueError('Lifetime candidate budget exceeded')
    if state['default_choice'] not in accepted:
        raise ValueError('Default must be previously accepted')
    for choice, ref in accepted.items():
        _identifier(choice)
        if choice == 'current':
            raise ValueError('Reserved current choice')
        if choice != 'native':
            skill = validate_compiled(_artifact(ref))
            if skill['source_goal'] != state['scope']['goal']:
                raise ValueError('Accepted skill has a different goal')
    fields = {'artifact', 'waypoints_sha256', 'source_session', 'status', 'probation_uses',
              'probation_successes', 'restore_choice'}
    if policy is not None:
        fields.add('lifecycle')
    for choice, candidate in candidates.items():
        _identifier(choice)
        if choice in accepted or choice == 'current' or set(candidate) != fields:
            raise ValueError('Invalid candidate record')
        skill = validate_compiled(_artifact(candidate['artifact']))
        if (skill['source_goal'] != state['scope']['goal'] or
                candidate['waypoints_sha256'] != skill['source_evidence']['waypoints_sha256'] or
                candidate['source_session'] != skill['source_evidence']['session_id']):
            raise ValueError('Candidate content binding differs')
        statuses = ('proposed', 'probation', 'provisional', 'retired') + (('suspended', 'revalidation_pending') if policy else ())
        if candidate['status'] not in statuses:
            raise ValueError('Unknown candidate status')
        uses = candidate['probation_uses']; successes = candidate['probation_successes']
        if (type(uses) is not int or not 0 <= uses <= 2 or type(successes) is not int or
                not 0 <= successes <= uses or candidate['restore_choice'] not in accepted):
            raise ValueError('Invalid probation accounting')
        if policy:
            _validate_lifecycle(candidate, policy, state['completed_sessions'])
            if state['completed_sessions'].get(candidate['source_session']) != skill['source_evidence']['result_sha256']:
                raise ValueError('Candidate source is not this lane\'s completed ordinary task')
    choice = state['next_choice']
    if choice not in accepted:
        candidate = candidates.get(choice)
        if not candidate or candidate['status'] in ('retired', 'suspended'):
            raise ValueError('Next task cannot use an unavailable candidate')
    sessions = state['completed_sessions']
    if not isinstance(sessions, dict):
        raise ValueError('Expected immutable Session result bindings')
    for sid, digest in sessions.items():
        _identifier(sid); _hash(digest)
    if policy and set(sessions) != set(policy['ordinary_sessions'][:len(sessions)]):
        raise ValueError('Completed tasks are not this preregistered lane prefix')
    return state


def make_catalog(state, session_id):
    validate_state(state); _identifier(session_id)
    selected = state['candidates'].get(state['next_choice'])
    catalog = dict(version=state['version'], kind='capability_update_catalog', session_id=session_id,
                scope=copy.deepcopy(state['scope']), state_sha256=canonical_sha(state),
                default_choice=state['default_choice'], accepted_choices=sorted(state['accepted']),
                current_choice=state['next_choice'],
                incumbent_choice=selected['restore_choice'] if selected else state['next_choice'],
                candidate_choices=sorted(k for k, c in state['candidates'].items()
                                         if c['status'] not in ('retired', 'suspended')),
                candidate_budget_used=len(state['candidates']),
                candidate_statuses={k: {field: c[field] for field in
                                        ('status', 'probation_uses', 'probation_successes')}
                                    for k, c in sorted(state['candidates'].items())},
                limits=dict(LIMITS), allowed_operations=list(OPERATIONS))
    if state['version'] == 2:
        catalog['lifecycle_policy'] = copy.deepcopy(state['lifecycle_policy'])
        for key, value in catalog['candidate_statuses'].items():
            value['lifecycle'] = copy.deepcopy(state['candidates'][key]['lifecycle'])
    return catalog


def validate_catalog(catalog):
    fields = {'version', 'kind', 'session_id', 'scope', 'state_sha256', 'default_choice',
              'accepted_choices', 'candidate_choices', 'candidate_budget_used', 'candidate_statuses',
              'current_choice', 'incumbent_choice',
              'limits', 'allowed_operations'}
    policy = _policy(catalog.get('lifecycle_policy')) if catalog.get('version') == 2 else None
    if policy:
        fields.add('lifecycle_policy')
    if (set(catalog) != fields or catalog['version'] not in (1, 2) or
            catalog['kind'] != 'capability_update_catalog' or catalog['limits'] != LIMITS or
            catalog['allowed_operations'] != list(OPERATIONS)):
        raise ValueError('Unknown capability catalog or permissions')
    _identifier(catalog['session_id']); _scope(catalog['scope']); _hash(catalog['state_sha256'])
    accepted = catalog['accepted_choices']; candidates = catalog['candidate_choices']
    if (not isinstance(accepted, list) or not isinstance(candidates, list) or
            len(set(accepted + candidates)) != len(accepted + candidates) or
            'native' not in accepted or catalog['default_choice'] not in accepted or len(candidates) > 2):
        raise ValueError('Invalid catalog choices')
    if catalog['current_choice'] not in accepted + candidates or catalog['incumbent_choice'] not in accepted:
        raise ValueError('Catalog current/incumbent choice differs')
    for choice in accepted + candidates:
        _identifier(choice)
        if choice == 'current':
            raise ValueError('Reserved symbolic choice')
    statuses = catalog['candidate_statuses']
    if (not isinstance(statuses, dict) or type(catalog['candidate_budget_used']) is not int or
            catalog['candidate_budget_used'] != len(statuses) or len(statuses) > 2 or
            set(candidates) != {k for k, c in statuses.items() if c.get('status') not in ('retired', 'suspended')}):
        raise ValueError('Catalog lifetime budget or available statuses differ')
    for choice, value in statuses.items():
        _identifier(choice)
        fields = {'status', 'probation_uses', 'probation_successes'} | ({'lifecycle'} if policy else set())
        legal = ('proposed', 'probation', 'provisional', 'retired') + (('suspended', 'revalidation_pending') if policy else ())
        if (choice in accepted or set(value) != fields or value['status'] not in legal or
                type(value['probation_uses']) is not int or not 0 <= value['probation_uses'] <= 2 or
                type(value['probation_successes']) is not int or
                not 0 <= value['probation_successes'] <= value['probation_uses']):
            raise ValueError('Malformed catalog candidate status')
        if policy:
            _validate_lifecycle(value, policy)
    return catalog


def make_request(catalog, catalog_sha256, operations):
    """Use this exact wrapper for a rule's application; Agent JSON is identical."""
    request = dict(version=1, session_id=catalog['session_id'], catalog_sha256=catalog_sha256,
                   operations=copy.deepcopy(operations))
    return validate_request(request, catalog, catalog_sha256)


def validate_request(request, catalog, catalog_sha256):
    validate_catalog(catalog); _hash(catalog_sha256)
    if (not isinstance(request, dict) or set(request) != {'version', 'session_id', 'catalog_sha256', 'operations'} or
            request['version'] != 1 or request['session_id'] != catalog['session_id'] or
            request['catalog_sha256'] != catalog_sha256):
        raise ValueError('Request is not bound to this Session and catalog')
    operations = request['operations']
    if not isinstance(operations, list) or not 1 <= len(operations) <= 3:
        raise ValueError('One to three operations required')
    seen = set()
    for item in operations:
        op = item.get('op') if isinstance(item, dict) else None
        if op not in OPERATIONS or op in seen:
            raise ValueError('Unknown or repeated operation')
        if op == 'propose_current_success':
            if set(item) != {'op', 'source'} or item['source'] != 'current':
                raise ValueError('Only the current real Session can be proposed')
        else:
            choices = catalog['accepted_choices'] + catalog['candidate_choices']
            if 'propose_current_success' in seen:
                choices += ['current']
            if set(item) != {'op', 'choice'} or item['choice'] not in choices:
                raise ValueError('Unknown or unbound requested choice')
            if op == 'trial_candidate' and item['choice'] in catalog['accepted_choices']:
                raise ValueError('An accepted choice is not a new probation candidate')
        seen.add(op)
    return request


def bind_session(state, case, output_dir):
    """Choose only a supplied normal task; create no initial-state selection."""
    validate_state(state)
    if case_scope(case) != state['scope'] or case.get('mode') not in ('development', 'evaluation'):
        raise ValueError('Task scope or mode differs')
    sid = _identifier(case['id'])
    if type(case.get('init_state_id')) is not int or case['init_state_id'] < 0:
        raise ValueError('A supplied initial state is required')
    if sid in state['completed_sessions']:
        raise ValueError('Session has already been consumed')
    policy = state.get('lifecycle_policy')
    if policy and case['mode'] == 'development':
        remaining = policy['ordinary_sessions'][len(state['completed_sessions']):]
        if not remaining or sid != remaining[0]:
            raise ValueError('Only the next preregistered ordinary task may be bound')
    choice = state['next_choice']
    candidate = state['candidates'].get(choice)
    artifact = candidate['artifact'] if candidate else state['accepted'][choice]
    binding = dict(version=1, kind='capability_session_binding', session_id=sid,
                   mode=case['mode'], scope=copy.deepcopy(state['scope']),
                   init_state_id=case['init_state_id'],
                   state_sha256=canonical_sha(state), choice=choice, skill=copy.deepcopy(artifact),
                   execution_format='experience_bank',
                   probation=bool(candidate and candidate['probation_uses'] < 2 and case['mode'] == 'development'),
                   probation_use=(candidate['probation_uses'] + 1) if candidate and candidate['probation_uses'] < 2 and case['mode'] == 'development' else None,
                   instruction='Use this frozen choice in the supplied ordinary task; retain native fallback and original limits.')
    if policy:
        revalidation = bool(candidate and candidate['status'] == 'revalidation_pending' and case['mode'] == 'development')
        if revalidation and candidate['lifecycle']['revalidation_session'] != sid:
            raise ValueError('Revalidation cannot move to another task')
        binding.update(lifecycle_policy_sha256=canonical_sha(policy), revalidation=revalidation,
                       revalidation_use=1 if revalidation else None,
                       evidence_root=str(Path(output_dir).absolute()))
    path = Path(output_dir) / 'BINDING.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        previous, _ = _read(path)
        if previous != binding:
            raise ValueError('Immutable Session binding collision')
    else:
        _dump(path, binding)
    return binding


def _root_success(result):
    sid = result['case']['id']
    terminal = result.get('native_terminal_by_session', {}).get(sid, {})
    return (result.get('chain_status') == 'PASSED' and result.get('environment_success') is True and
            terminal.get('session_id') == sid and terminal.get('native_success') is True and
            type(terminal.get('steps')) is int and terminal['steps'] > 0)


def _ordinary_completion(result, trace, trace_ref, live):
    """A cooldown is a real completed task, including failures, not elapsed time."""
    from .live_controller_trace import action_f32
    sid = result['case']['id']; case = result['case']
    terminal = result.get('native_terminal_by_session', {}).get(sid, {})
    steps = terminal.get('steps'); controls = (live or {}).get('controls', [])
    config = (live or {}).get('runtime_config', {})
    if (result.get('chain_status') != 'PASSED' or terminal.get('session_id') != sid or
            type(terminal.get('native_success')) is not bool or type(steps) is not int or not 0 < steps <= 400 or
            not isinstance(live, dict) or not trace or not trace_ref or
            live.get('kind') != 'live_controller_trace' or live.get('recording_role') != 'online_execution' or
            live.get('status') != 'LIVE_RECORDED' or live.get('session_id') != sid or
            len(controls) != steps or live.get('controls_count') != steps or
            live.get('controls_sha256') != canonical_sha(controls) or
            live.get('source_trace_sha256') != trace_ref['sha256'] or
            live.get('source_trace_rows_sha256') != canonical_sha(trace) or
            trace[-1].get('native_success') is not terminal.get('native_success') or
            live.get('terminal_native_success') is not terminal.get('native_success') or
            config.get('benchmark_name') != case['suite'] or config.get('task_id') != case['task_id'] or
            config.get('init_state_id') != case['init_state_id'] or config.get('max_steps') != 400 or
            config.get('control_mode') != 'relative'):
        raise ValueError('Lifecycle requires complete actual ordinary-task evidence from this lane')
    actual = []
    for index, control in enumerate(controls):
        if (control.get('session_id') != sid or control.get('step_before') != index or
                control.get('step_after') != index+1):
            raise ValueError('Ordinary task controller extent changed')
        actual.append(action_f32(control.get('actual_action')))
    if trace[0].get('method') != 'reset' or trace[0].get('step') != 0:
        raise ValueError('Ordinary task reset trace missing')
    previous = 0; recorded = []
    for row in trace:
        if row.get('session_id') != sid:
            raise ValueError('Ordinary task trace from another Session')
    for row in trace[1:]:
        step = row.get('step')
        if type(step) is not int or step <= previous or row.get('method') != 'action_chunk':
            raise ValueError('Ordinary task trace extent changed')
        actions = row.get('actions', [])
        if len(actions) < step-previous:
            raise ValueError('Ordinary task executed actions missing')
        recorded.extend(action_f32(a) for a in actions[:step-previous]); previous = step
    if previous != steps or canonical_sha(actual) != canonical_sha(recorded):
        raise ValueError('Ordinary task trace does not match actually executed actions')


def _actual_use(result, binding, trace, trace_ref, live):
    """Verify target-recorded bank actions, not an applicant's success claim."""
    from .live_controller_trace import action_f32
    if not isinstance(live, dict) or not trace:
        raise ValueError('Actual candidate-use evidence missing')
    sid = binding['session_id']; controls = live.get('controls', [])
    terminal = result.get('native_terminal_by_session', {}).get(sid, {})
    steps = terminal.get('steps')
    config = live.get('runtime_config', {})
    if (live.get('session_id') != sid or live.get('kind') != 'live_controller_trace' or
            live.get('recording_role') != 'online_execution' or live.get('status') != 'LIVE_RECORDED' or
            type(steps) is not int or not 0 < steps <= 400 or len(controls) != steps or
            live.get('controls_count') != steps or live.get('controls_sha256') != canonical_sha(controls) or
            live.get('source_trace_sha256') != trace_ref['sha256'] or
            live.get('source_trace_rows_sha256') != canonical_sha(trace) or
            trace[-1].get('step') != steps or any(row.get('session_id') != sid for row in trace)):
        raise ValueError('Actual candidate use is not bound to the complete root execution')
    case = result['case']
    if (config.get('benchmark_name') != case['suite'] or config.get('task_id') != case['task_id'] or
            config.get('init_state_id') != case['init_state_id'] or
            config.get('control_mode') != 'relative' or config.get('max_steps') != 400):
        raise ValueError('Actual execution changed the bound task or execution budget')
    if binding.get('execution_format') != 'experience_bank':
        raise ValueError('This executor requires the declared bank wrapper')
    skill_sha = canonical_sha(_artifact(binding['skill']))
    count = native_tail = fallback = 0
    actual = []
    for index, control in enumerate(controls):
        if (control.get('session_id') != sid or control.get('step_before') != index or
                control.get('step_after') != index + 1):
            raise ValueError('Actual control extent differs')
        origin = control.get('execution_origin')
        if origin == 'goal_skill_action':
            if control.get('source_skill_sha256') != skill_sha:
                raise ValueError('A different candidate was actually executed')
            count += 1
        elif origin == 'native_after_validity_exit':
            fallback += 1
        elif origin == 'native_action':
            native_tail += 1
        else:
            raise ValueError('Unknown actual execution origin')
        actual.append(action_f32(control.get('actual_action')))
    recorded = []; previous = 0
    if trace[0].get('method') != 'reset' or trace[0].get('step') != 0:
        raise ValueError('Missing root reset trace')
    for row in trace[1:]:
        step = row.get('step')
        if type(step) is not int or step <= previous or row.get('method') != 'action_chunk':
            raise ValueError('Invalid root action extent')
        actions = row.get('actions', [])
        if len(actions) < step - previous:
            raise ValueError('Missing actually executed actions')
        recorded.extend(action_f32(a) for a in actions[:step - previous]); previous = step
    if canonical_sha(actual) != canonical_sha(recorded) or count == 0:
        raise ValueError('No matching actual candidate actions or changed trace actions')
    return dict(candidate_sha256=skill_sha, actual_skill_steps=count,
                native_tail_steps=native_tail, native_after_validity_exit_steps=fallback,
                validity_exit=fallback > 0, actual_use_verified=True)


def _existing(output, operation_id):
    receipt, _ = _read(output / 'RECEIPT.json')
    if receipt.get('operation_id') != operation_id:
        raise ValueError('Immutable completion directory belongs to different inputs')
    for relative, digest in receipt['output_sha256'].items():
        if relative not in ('STATE.json', 'CANDIDATE.json') or _sha((output / relative).read_bytes()) != digest:
            raise ValueError('Committed completion artifact changed')
    return receipt


def finish_session(state, *, binding_path, catalog_path, request_path, result_path,
                   trace_path=None, live_receipt_path=None, output_dir, compiler=None, ledger_dir=None):
    """Apply one completion transaction and return its immutable receipt.

    All file provenance is computed here, never supplied by the applicant.
    Malformed requests are recorded and rejected without suppressing a required
    probation failure/retirement. Failed or rejected updates keep accepted data.
    """
    tick = time.perf_counter(); compile_seconds = 0.0
    validate_state(state)
    binding, binding_ref = _read(binding_path)
    if state.get('lifecycle_policy'):
        evidence_root = Path(binding_ref['path']).parent
        if binding.get('evidence_root') != str(evidence_root):
            raise ValueError('Lifecycle evidence root differs from the bound ordinary task')
        for path in (catalog_path, request_path, result_path, trace_path, live_receipt_path):
            if path is not None and evidence_root not in Path(path).absolute().parents:
                raise ValueError('Lifecycle cannot consume another task or lane\'s evidence')
    catalog, catalog_ref = _read(catalog_path)
    result, result_ref = _read(result_path)
    if 'cases' in result:
        if len(result['cases']) != 1:
            raise ValueError('Completion must contain exactly one ordinary task')
        result = result['cases'][0]
    validate_catalog(catalog)
    sid = _identifier(result['case']['id'])
    expected_catalog = make_catalog(state, sid)
    if catalog != expected_catalog or case_scope(result['case']) != state['scope']:
        raise ValueError('Current result/catalog differs from the frozen state and scope')
    if (binding.get('kind') != 'capability_session_binding' or binding.get('version') != 1 or
            binding.get('session_id') != sid or binding.get('state_sha256') != canonical_sha(state) or
            binding.get('scope') != state['scope'] or binding.get('choice') != state['next_choice'] or
            binding.get('init_state_id') != result['case'].get('init_state_id') or
            binding.get('execution_format') != 'experience_bank' or
            binding.get('mode') != result['case'].get('mode')):
        raise ValueError('Current worker was not bound to this state')
    selected = state['candidates'].get(binding['choice'])
    expected_skill = selected['artifact'] if selected else state['accepted'][binding['choice']]
    probation = bool(selected and selected['probation_uses'] < 2 and binding['mode'] == 'development')
    if (binding.get('skill') != expected_skill or binding.get('probation') != probation or
            binding.get('probation_use') != (selected['probation_uses'] + 1 if probation else None)):
        raise ValueError('Worker skill/probation binding differs')
    policy = state.get('lifecycle_policy')
    revalidation = bool(policy and selected and selected['status'] == 'revalidation_pending' and binding['mode'] == 'development')
    if policy:
        if (binding.get('lifecycle_policy_sha256') != canonical_sha(policy) or
                binding.get('revalidation') != revalidation or
                binding.get('revalidation_use') != (1 if revalidation else None)):
            raise ValueError('Worker revalidation binding differs')
        if revalidation and (selected['lifecycle']['revalidation_session'] != sid or selected['lifecycle']['revalidation_uses']):
            raise ValueError('Wrong or already consumed revalidation opportunity')
    refs = dict(binding=binding_ref, catalog=catalog_ref, result=result_ref)
    request = None; request_error = None
    if request_path is not None and Path(request_path).exists():
        # Read bytes even for an invalid request so rejection is reproducible.
        raw = Path(request_path).read_bytes()
        refs['request'] = {'path': str(Path(request_path).absolute()), 'sha256': _sha(raw)}
        try:
            request, ref = _read(request_path)
            validate_request(request, catalog, catalog_ref['sha256'])
        except (ValueError, TypeError, KeyError) as error:
            request = None; request_error = str(error)
    trace = live = None
    if trace_path is not None:
        trace, refs['trace'] = _read_trace(trace_path)
    if live_receipt_path is not None:
        live, refs['live_controls'] = _read(live_receipt_path)
    operation_id = canonical_sha({'state': canonical_sha(state), 'inputs': refs})
    output = Path(output_dir).absolute()
    if binding['mode'] == 'evaluation':
        # No files or persistent bookkeeping are written during evaluation.
        return dict(version=1, kind='capability_update_receipt', operation_id=operation_id,
                    status='EVALUATION_NO_UPDATE', session_id=sid, inputs=refs,
                    next_choice=state['next_choice'], state_sha256=canonical_sha(state),
                    operations=[], output_sha256={})
    if policy:
        remaining = policy['ordinary_sessions'][len(state['completed_sessions']):]
        if not remaining or sid != remaining[0]:
            raise ValueError('Lifecycle cannot replace or reorder registered ordinary tasks')
        _ordinary_completion(result, trace, refs.get('trace'), live)
    if output.exists():
        return _existing(output, operation_id)
    if sid in state['completed_sessions']:
        raise ValueError('A completed Session cannot pay for another update')
    # The caller supplies one shared ledger for its sequential development flow.
    # A request ID is the current Session plus catalog bytes; its entire inputs
    # are immutable. Retrying cannot create another candidate in another folder.
    ledger = Path(ledger_dir).absolute() if ledger_dir is not None else output.parent / '.capability-claims'
    ledger.mkdir(parents=True, exist_ok=True)
    request_id = canonical_sha({'session_id': sid, 'catalog_sha256': catalog_ref['sha256']})
    claim_path = ledger / (request_id + '.json')
    claim = dict(request_id=request_id, operation_id=operation_id, output_dir=str(output))
    try:
        _dump(claim_path, claim)
    except FileExistsError:
        previous, _ = _read(claim_path)
        if previous.get('operation_id') != operation_id:
            raise ValueError('The same request ID has a different immutable payload')
        prior_output = Path(previous['output_dir'])
        if prior_output.exists():
            return _existing(prior_output, operation_id)
        if prior_output != output:
            raise ValueError('Resume the original pending completion directory')
    updated = copy.deepcopy(state); actions = []; candidate_value = None; current = None
    root_success = _root_success(result)
    if selected:
        c = updated['candidates'][binding['choice']]
        try:
            usage = _actual_use(result, binding, trace, refs.get('trace'), live)
        except (ValueError, TypeError, KeyError) as error:
            usage = dict(actual_use_verified=False, reason=str(error))
        suitable = root_success and usage['actual_use_verified'] and not usage.get('validity_exit', False)
        if probation:
            c['probation_uses'] += 1
            if policy:
                _lifecycle_event(c, sid, result_ref['sha256'], 'PROBATION_SUCCESS' if suitable else 'PROBATION_FAILURE')
        if revalidation:
            c['lifecycle']['revalidation_uses'] += 1
            c['lifecycle']['revalidation_success'] = suitable
            _lifecycle_event(c, sid, result_ref['sha256'], 'REVALIDATION_SUCCESS' if suitable else 'REVALIDATION_FAILURE')
        if suitable:
            if probation:
                c['probation_successes'] += 1
            c['status'] = 'provisional'
            effect = 'PROVISIONAL_CONTINUATION'
            if c['probation_uses'] >= LIMITS['max_probation_uses']:
                effect = 'PROVISIONAL_DEPLOYMENT_NOT_CERTIFICATION'
            if revalidation:
                effect = 'REVALIDATED_PROVISIONAL_NOT_CERTIFICATION'
        else:
            updated['next_choice'] = c['restore_choice']
            can_suspend = bool(policy and c['probation_successes'] == 2 and not revalidation and
                c['lifecycle']['suspension_session'] is None and usage['actual_use_verified'] and
                not usage.get('validity_exit', False) and
                result['native_terminal_by_session'][sid]['native_success'] is False)
            if can_suspend:
                order = policy['ordinary_sessions']; index = order.index(sid)
                c['status'] = 'suspended'; record = c['lifecycle']
                record.update(suspension_session=sid,
                    cooldown_session=order[index+1] if index+1 < len(order) else None,
                    revalidation_session=order[index+2] if index+2 < len(order) else None)
                _lifecycle_event(c, sid, result_ref['sha256'], 'SUSPENDED_AFTER_FAILURE')
                effect = 'SUSPENDED_RESTORE'
            else:
                c['status'] = 'retired'; effect = 'RETIRED_RESTORE'
                if policy:
                    _lifecycle_event(c, sid, result_ref['sha256'], 'PERMANENTLY_RETIRED')
        observation = {'op': 'observe_revalidation' if revalidation else ('observe_probation' if probation else 'observe_provisional_use'), 'choice': binding['choice'], 'effect': effect,
                        'root_success': root_success, 'candidate_trial_success': suitable,
                        'actual_use': usage, 'probation_uses': c['probation_uses']}
        if policy:
            observation['revalidation_uses'] = c['lifecycle']['revalidation_uses']
        actions.append(observation)
    if request_error:
        actions.append({'op': 'request', 'effect': 'REJECTED', 'reason': request_error})
    for item in (request or {}).get('operations', []):
        op = item['op']
        try:
            if op == 'propose_current_success':
                if not root_success:
                    raise ValueError('Current root did not actually succeed')
                if len(updated['candidates']) >= LIMITS['max_candidates']:
                    raise ValueError('Lifetime candidate budget exhausted')
                if trace is None or live is None:
                    raise ValueError('Current live controller and root trace evidence required')
                if compiler is None:
                    from .experience_compiler import compile_live_success
                    compiler = compile_live_success
                provenance = {'result_sha256': result_ref['sha256'],
                              'trace_sha256': refs['trace']['sha256'],
                              'controller_trace_sha256': refs['live_controls']['sha256']}
                compile_tick = time.perf_counter()
                try:
                    skill = validate_compiled(compiler(result, trace, live, provenance=provenance))
                finally:
                    compile_seconds += time.perf_counter() - compile_tick
                evidence = skill['source_evidence']
                if (evidence.get('provenance_mode') != 'live_controller_v1' or
                        evidence.get('session_id') != sid or any(evidence.get(k) != v for k, v in provenance.items()) or
                        skill['source_goal'] != state['scope']['goal']):
                    raise ValueError('Compiler returned another source or provenance')
                origins = set(evidence.get('execution_origins', []))
                if not origins or not origins <= {'native_action', 'native_after_validity_exit', 'goal_skill_action'}:
                    raise ValueError('Unbound actual execution origins')
                if origins == {'goal_skill_action'}:
                    raise ValueError('Pure reuse of an existing skill is not a new capability')
                target_hash = evidence['waypoints_sha256']
                prior = [c['waypoints_sha256'] for c in updated['candidates'].values()]
                prior += [validate_compiled(_artifact(ref))['source_evidence']['waypoints_sha256']
                          for key, ref in updated['accepted'].items() if key != 'native']
                if target_hash in prior:
                    raise ValueError('Identical controller targets already exist')
                current = 'candidate_' + canonical_sha(skill)[:16]
                raw = (json.dumps(skill, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
                ref = {'path': str(output / 'CANDIDATE.json'), 'sha256': _sha(raw)}
                restore = binding['choice'] if binding['choice'] in updated['accepted'] else selected['restore_choice']
                updated['candidates'][current] = dict(artifact=ref, waypoints_sha256=target_hash,
                    source_session=sid, status='proposed', probation_uses=0, probation_successes=0, restore_choice=restore)
                if policy:
                    updated['candidates'][current]['lifecycle'] = _lifecycle_record()
                candidate_value = skill
                actions.append({'op': op, 'effect': 'CANDIDATE_COMPILED', 'choice': current,
                                'artifact': ref, 'source': 'current_real_root'})
            else:
                choice = current if item['choice'] == 'current' else item['choice']
                if choice is None:
                    raise ValueError('The current proposal did not produce a candidate')
                if choice in updated['accepted']:
                    if op == 'trial_candidate':
                        raise ValueError('Accepted choices do not consume probation')
                else:
                    c = updated['candidates'].get(choice)
                    if not c or c['status'] in ('retired', 'suspended'):
                        raise ValueError('Candidate unavailable')
                    if c['status'] == 'revalidation_pending':
                        raise ValueError('Revalidation is bound only to its preregistered ordinary task')
                    if op == 'trial_candidate' and c['probation_uses'] >= 2:
                        raise ValueError('Probation budget exhausted; only provisional deployment remains')
                    if op == 'set_next_choice' and c['probation_successes'] < 2:
                        raise ValueError('Two successful ordinary probation tasks required before provisional deployment')
                    if op == 'trial_candidate' and c['probation_uses'] == 0:
                        # Capture the incumbent when the trial is requested,
                        # which may be later than the source proposal Session.
                        c['restore_choice'] = (binding['choice'] if binding['choice'] in updated['accepted']
                                               else selected['restore_choice'])
                    c['status'] = 'probation' if c['probation_successes'] == 0 else 'provisional'
                updated['next_choice'] = choice
                actions.append({'op': op, 'effect': 'NEXT_ORDINARY_TASK_BOUND', 'choice': choice,
                                'extra_tasks_scheduled': 0})
        except (ValueError, TypeError, KeyError) as error:
            actions.append({'op': op, 'effect': 'REJECTED', 'reason': str(error)})
    if policy:
        # The opportunity is fixed when failure occurs: one intervening actual
        # task, then the next listed task. Explicit valid choice requests may
        # decline it, but cannot move it later or replenish its separate budget.
        explicit_choice = any(a.get('effect') == 'NEXT_ORDINARY_TASK_BOUND' for a in actions)
        for choice, c in updated['candidates'].items():
            record = c['lifecycle']
            if c['status'] != 'suspended' or record['cooldown_session'] != sid:
                continue
            _lifecycle_event(c, sid, result_ref['sha256'], 'COOLDOWN_COMPLETED')
            if record['revalidation_session'] is None:
                continue
            if explicit_choice:
                c['status'] = 'retired'
                _lifecycle_event(c, sid, result_ref['sha256'], 'REVALIDATION_CANCELLED')
                effect = 'REVALIDATION_DECLINED_NO_RESCHEDULE'
            else:
                c['status'] = 'revalidation_pending'; updated['next_choice'] = choice
                _lifecycle_event(c, sid, result_ref['sha256'], 'REVALIDATION_RESERVED')
                effect = 'NEXT_REGISTERED_TASK_REVALIDATION'
            actions.append(dict(op='lifecycle_cooldown', choice=choice, effect=effect,
                cooldown_session=sid, revalidation_session=record['revalidation_session'],
                separate_revalidation_budget=1, extra_tasks_scheduled=0))
    updated['completed_sessions'][sid] = result_ref['sha256']
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.' + output.name + '-', dir=str(output.parent)))
    try:
        digests = {}
        if candidate_value is not None:
            digests['CANDIDATE.json'] = _dump(staging / 'CANDIDATE.json', candidate_value)
        digests['STATE.json'] = _dump(staging / 'STATE.json', updated)
        receipt = dict(version=1, kind='capability_update_receipt', operation_id=operation_id,
            status='COMMITTED', session_id=sid, inputs=refs, parent_state_sha256=canonical_sha(state),
            state_sha256=canonical_sha(updated), next_choice=updated['next_choice'],
            root_success=root_success, operations=actions, output_sha256=digests,
            request_id=request_id, admission_status=result.get('experience_admission', {}).get('status'),
            costs=dict(compiler_seconds=compile_seconds,
                       processing_seconds_through_receipt=time.perf_counter()-tick,
                       scope='Measured executor CPU wall; compiler is a nested subset. Final commit I/O is covered by the caller outer span; do not add subsets to task wall.'),
            accepted_unchanged=updated['accepted'] == state['accepted'],
            scope='Inter-Session applications; at most two later ordinary probation uses. Success is provisional, not automatic acceptance.')
        if policy:
            receipt.update(lifecycle_policy=copy.deepcopy(policy),
                scope='Explicit finite lifecycle policy: two ordinary probation uses plus at most one separately counted revalidation after an actual ordinary cooldown task. All failures and task/update costs remain paid; no global certification or extra task scheduling.')
        _dump(staging / 'RECEIPT.json', receipt)
        # Recheck bound files before committing; no mutable agent-provided paths
        # or manufactured success labels determine the source.
        for ref in refs.values():
            if _sha(Path(ref['path']).read_bytes()) != ref['sha256']:
                raise ValueError('An input changed during completion')
        try:
            os.rename(str(staging), str(output))
        except OSError:
            if output.exists():
                return _existing(output, operation_id)
            raise
        return receipt
    finally:
        if staging.exists():
            shutil.rmtree(staging)
