"""Precisely bound, already-seen development interventions; no learned trigger or oracle."""
import hashlib
import json
import math

GOAL = 'put the yellow and white mug in the microwave and close it'
REMAINING_GOAL = 'close the microwave'
SESSION = 'net_t9_i15'
BOUNDARY = 200
STACK_GOAL = 'stack the left bowl on the right bowl and place them in the tray'
STACK_REMAINING_GOAL = 'put the bowl on the table on top of the bowl in the tray'
STACK_SESSION = 'net_t63_i0'
STACK_BOUNDARY = 120


def artifact(reference_trace_path, reference_trace_sha256,
             reference_cost_path, reference_cost_sha256):
    return {
        'version': 1, 'suite': 'libero_10', 'task_id': 9, 'init_state_id': 15,
        'session_id': SESSION, 'goal': GOAL, 'remaining_goal': REMAINING_GOAL,
        'boundary_step': BOUNDARY, 'max_steps': 520, 'feedback_window_mode': 'fixed10',
        'reference_trace': {'path': str(reference_trace_path), 'sha256': reference_trace_sha256},
        'reference_cost': {'path': str(reference_cost_path), 'sha256': reference_cost_sha256},
        'scope': 'Seen development boundary, not a learned trigger or a new test instance',
        'prefix_mode': 'Regenerate original full-goal actions and verify before submission',
    }


def stack_artifact(reference_trace_path, reference_trace_sha256,
                   reference_cost_path, reference_cost_sha256):
    """Task 63 only; post-hoc remaining operation after an exact original prefix."""
    value = artifact(reference_trace_path, reference_trace_sha256,
                     reference_cost_path, reference_cost_sha256)
    value.update(version=2, suite='libero_90', task_id=63, init_state_id=0,
                 session_id=STACK_SESSION, goal=STACK_GOAL, remaining_goal=STACK_REMAINING_GOAL,
                 boundary_step=STACK_BOUNDARY, max_steps=400,
                 development_suite_extension='libero90_stack_flow_v1',
                 scope='Post-hoc seen development operation probe; not a learned trigger, new test, '
                       'or paired wall-clock speed comparison')
    return value


def _validate_spec(value):
    try:
        trace, cost = value['reference_trace'], value['reference_cost']
        factory = {1: artifact, 2: stack_artifact}[value['version']]
        expected = factory(trace['path'], trace['sha256'], cost['path'], cost['sha256'])
    except (KeyError, TypeError) as error:
        raise ValueError('Unbound boundary specification') from error
    if value != expected:
        raise ValueError('Unbound boundary specification')
    for ref in (trace, cost):
        if (not isinstance(ref['path'], str) or not ref['path'] or
                not isinstance(ref['sha256'], str) or len(ref['sha256']) != 64 or
                any(x not in '0123456789abcdef' for x in ref['sha256'])):
            raise ValueError('Invalid reference binding')
    return value


def _digest(raw):
    return hashlib.sha256(raw).hexdigest()


def bind(case, raw, digest):
    if _digest(raw) != digest:
        raise ValueError('Changed boundary artifact')
    value = json.loads(raw)
    _validate_spec(value)
    required = dict(suite=value['suite'], task_id=value['task_id'], init_state_id=value['init_state_id'],
                    id=value['session_id'], task_description=value['goal'], mode='development',
                    max_steps=value['max_steps'], feedback_window_mode='fixed10')
    if value['version'] == 2:
        required['development_suite_extension'] = value['development_suite_extension']
    if any(case.get(k) != v for k, v in required.items()):
        raise ValueError('Boundary probe only allowed for its original development case')
    if type(case['boundary_probe'].get('enabled')) is not bool:
        raise ValueError('Boundary arm must be explicit')
    conflicts = ('procedure_probe', 'stage_flow_artifact', 'action_window_artifact',
                 'window_learning_artifact', 'task_window_artifact',
                 'prerequisite_component_probe', 'decision_validity_shadow',
                 'inference_horizon', 'commit_steps')
    if any(case.get(k) is not None for k in conflicts):
        raise ValueError('Boundary probe cannot combine interventions')
    return value


def reference(spec, trace_raw, cost_raw):
    """Retain only visible input bindings and executed actions up to the bound step.

    Native success, privileged replay facts and the original future are never
    stored in the guard and never determine the branch or the selected prompt.
    """
    _validate_spec(spec)
    boundary, session, goal = spec['boundary_step'], spec['session_id'], spec['goal']
    if (_digest(trace_raw) != spec['reference_trace']['sha256'] or
            _digest(cost_raw) != spec['reference_cost']['sha256']):
        raise ValueError('Changed original prefix evidence')
    trace = [json.loads(x) for x in trace_raw.splitlines() if x.strip()]
    rows = [r for r in trace if r.get('step', boundary + 1) <= boundary]
    if [r['step'] for r in rows] != list(range(0, boundary + 1, 10)):
        raise ValueError(f'Original prefix must retain all {boundary // 10 + 1} boundaries')
    events = [json.loads(x) for x in cost_raw.splitlines() if x.strip()]
    inputs = [e for e in events if e.get('event') == 'actual_policy_input'
              and e.get('case') == session][:len(rows)]
    if len(inputs) != len(rows):
        raise ValueError('Original prefix is missing actual policy inputs')
    result = {}
    for index, (row, policy) in enumerate(zip(rows, inputs)):
        step = row['step']
        if (row.get('session_id') != session or
                row.get('observation_id') != f'libero_obs_{step}' or
                row.get('method') != ('reset' if index == 0 else 'action_chunk') or
                policy.get('prompt') != goal):
            raise ValueError('Original prefix has a different session, goal or ordering')
        images = row.get('images', {})
        names = {f'{step:04d}_agentview_image.png', f'{step:04d}_robot0_eye_in_hand_image.png'}
        if set(images) != names:
            raise ValueError('Original prefix is missing image evidence')
        arrays = policy.get('array_bindings', {})
        if set(arrays) != {'observation/image', 'observation/wrist_image', 'observation/state'}:
            raise ValueError('Original prefix is missing array evidence')
        result[step] = dict(images=images, array_bindings=arrays)
        if index:
            actions = row.get('actions')
            if (not isinstance(actions, list) or len(actions) != 10 or
                    any(not isinstance(a, list) or len(a) != 7 or
                        any(type(x) not in (int, float) or not math.isfinite(x) for x in a)
                        for a in actions)):
                raise ValueError('Original prefix requires complete 10 by 7 action chunks')
            result[step - 10]['actions'] = actions
    return result


def guarded_commit(guard, commit):
    """Do not publish native experience from an incomplete or rejected prefix."""
    def checked(*args, **kwargs):
        guard.finish()
        return commit(*args, **kwargs)
    return checked


class PrefixGuard:
    """Reject mismatches before any changed action can be submitted."""
    def __init__(self, references, *, boundary=BOUNDARY, goal=GOAL, remaining_goal=REMAINING_GOAL):
        if type(boundary) is not int or boundary <= 0 or boundary % 10:
            raise ValueError('Prefix boundary must be a positive multiple of ten')
        if not isinstance(goal, str) or not goal or not isinstance(remaining_goal, str) or not remaining_goal:
            raise ValueError('Explicit original and remaining goals are required')
        if set(references) != set(range(0, boundary + 1, 10)):
            raise ValueError('Prefix references do not cover every required boundary')
        self.boundary, self.goal, self.remaining_goal = boundary, goal, remaining_goal
        self.references = references
        self.observations_verified = 0
        self.action_chunks_verified = 0
        self.next_observation = 0
        self.awaiting_actions = None
        self.matched = False
        self.failure = None

    def reject(self, reason):
        if self.failure is None:
            self.failure = reason
        raise ValueError(self.failure)

    def check_healthy(self):
        if self.failure is not None:
            raise ValueError(self.failure)

    def observe(self, step, array_bindings, image_bindings, prompt):
        self.check_healthy()
        if self.matched:
            if step <= self.boundary:
                self.reject('Root prefix restarted or repeated after boundary')
            return
        if step != self.next_observation or self.awaiting_actions is not None:
            self.reject(f'Unexpected prefix observation order at {step}')
        expected = self.references[step]
        if (prompt != self.goal or array_bindings != expected['array_bindings'] or
                image_bindings != expected['images']):
            self.reject(f'Original prefix observation mismatch at {step}')
        self.observations_verified += 1
        if step == self.boundary:
            if self.action_chunks_verified != self.boundary // 10:
                self.reject('Boundary reached without all original actions')
            self.matched = True
        else:
            self.awaiting_actions = step

    def actions(self, step, actions):
        self.check_healthy()
        if self.matched:
            if step < self.boundary:
                self.reject('Old prefix actions after boundary')
            return
        if step != self.awaiting_actions:
            self.reject(f'Unexpected prefix action order at {step}')
        if actions != self.references[step]['actions']:
            self.reject(f'Original prefix action mismatch before submission at {step}')
        self.action_chunks_verified += 1
        self.awaiting_actions = None
        self.next_observation = step + 10

    def finish(self):
        self.check_healthy()
        if not self.matched:
            self.reject('Root ended before the verified intervention boundary')

    def prompt(self, step, enabled):
        self.check_healthy()
        if step >= self.boundary:
            if not self.matched:
                self.reject('Intervention attempted without verified original prefix')
            if enabled:
                return self.remaining_goal
        return self.goal
