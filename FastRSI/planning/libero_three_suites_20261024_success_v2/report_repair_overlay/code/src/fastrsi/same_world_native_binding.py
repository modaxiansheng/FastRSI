"""Explicit, optional binding of registered development flows to the native OS.

The existing Agent/Watchdog/evaluation-isolation paths own execution and memory.
This module binds identity, a fixed W10 policy adapter, and clearly nonofficial
benchmark provenance; it neither dispatches Sessions nor changes feedback rules.
"""
import hashlib
import json
from pathlib import Path

from .same_world_continuation import SCOPE


FIELDS = ('same_world_registration', 'same_world_registration_sha256', 'same_world_arm')
JOINT_GOALS = {
    1: 'open the middle drawer of the cabinet and turn on the stove',
    2: 'put the cream cheese in the bowl and put the bowl on the plate',
}
CHOICE_MODE = 'native_registered_order_choice_v1'


def requested(case):
    return any(key in case for key in FIELDS)


def load_binding(case):
    """Validate case against the same frozen registration used by the target."""
    from check_same_world_policy_value_v1 import validate_registration, diagnostic_sequence
    if not all(isinstance(case.get(key), str) and case[key] for key in FIELDS):
        raise ValueError('Complete same-world registration path, SHA and arm required')
    if case.get('mode') not in ('development', 'evaluation'):
        raise ValueError('Bound development or fixed-version evaluation mode required')
    if not isinstance(case.get('id'), str) or not case['id'].strip():
        raise ValueError('Bound native root Session ID required')
    path = Path(case['same_world_registration'])
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != case['same_world_registration_sha256']:
        raise ValueError('Same-world registration changed')
    registration = validate_registration(json.loads(raw))
    arm = case['same_world_arm']
    if arm not in ('staged', 'order68', 'order86', 'choice'):
        raise ValueError('Only declared staged native flows are supported')
    choice = arm == 'choice'
    if choice:
        if registration['version'] != 2 or registration.get('selection_mode') != CHOICE_MODE:
            raise ValueError('Choice requires v2 and explicit registered selection_mode')
        if case.get('selection_mode', CHOICE_MODE) != CHOICE_MODE:
            raise ValueError('Case selection mode differs from registered choice')
        allowed_sequences = [list(registration['arm_orders'][key]) for key in ('order68', 'order86')]
        sequence = None  # Registration authorizes alternatives; it selects neither.
    else:
        sequence = diagnostic_sequence(registration, arm)
    joint_goal = JOINT_GOALS[registration['version']]
    expected = dict(suite=registration['suite'], task_id=registration['anchor_task_id'],
        init_state_id=registration['init_state_id'], max_steps=registration['total_steps'],
        task_description=joint_goal, feedback_window_mode='fixed10')
    # The existing fixed10 controller uses implicit H10/W10. Explicit H/W
    # fields belong to fixed_commitment and must not be mixed into this path.
    from .feedback_window import commitment_binding
    commitment_binding(case, 'fixed10')
    for key, value in expected.items():
        if case.get(key) != value or (type(value) is int and type(case.get(key)) is not int):
            raise ValueError('Case differs from bound same-world protocol: ' + key)
    by_key = {row['key']: row for row in registration['tasks']}
    stage_keys = list(by_key) if choice else sequence
    binding = dict(scope=SCOPE, registration_path=str(path), registration_sha256=digest,
        version=registration['version'], arm=arm, sequence=sequence,
        selection_mode=CHOICE_MODE if choice else 'fixed_registered_order',
        anchor_goal_key=next(row['key'] for row in registration['tasks']
                             if row['task_id'] == registration['anchor_task_id']),
        session_id=case['id'], joint_goal=joint_goal,
        stage_instructions=[by_key[key]['language'] for key in stage_keys],
        total_steps=registration['total_steps'], replan_steps=registration['replan_steps'],
        timeout_seconds=registration['timeout_seconds'],
        policy_timeout_seconds=registration['policy_timeout_seconds'],
        stage_feedback='native_oracle_stage_feedback_not_learned_recognition',
        recovery_child_session='UNSUPPORTED: bound target permits one root Session reset; no implicit replay',
        persistent_learning='Owned by unchanged native development/evaluation binding')
    if choice:
        binding['allowed_sequences'] = allowed_sequences
        binding['stage_definitions'] = {key: by_key[key]['language'] for key in stage_keys}
    return binding


def bind_workspace(case, workspace, spec, campaign_name):
    from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block, write_yaml_block
    from PhyAgentOS.runtime.schemas import SessionSpec
    from .same_world_policy_adapter import register_same_world_policy_adapter
    binding = load_binding(case)
    if (spec['session_id'] != binding['session_id'] or spec['task_description'] != binding['joint_goal']
            or spec['execution']['reset_policy'] != 'session_runner'):
        raise ValueError('Native root Session differs from bound stream')
    workspace = Path(workspace)
    skills = read_yaml_block(workspace/'SKILLRUNTIME.md')
    selected = [s for s in skills['skillruntimes']
                if 'skillruntime://' + s['id'] == spec['skillruntime_ref']]
    if len(selected) != 1 or selected[0]['runtime'] != 'OpenPISkillRuntime':
        raise ValueError('Expected one original OpenPI SkillRuntime')
    uri = register_same_world_policy_adapter(session_id=binding['session_id'],
        joint_goal=binding['joint_goal'], stage_instructions=binding['stage_instructions'])
    selected[0]['policy']['policy_adapter'] = uri
    spec['execution']['max_steps'] = binding['total_steps']
    spec['execution']['replan_every_steps'] = binding['replan_steps']
    spec['timeouts']['execute_timeout_s'] = binding['timeout_seconds']
    spec['timeouts']['policy_timeout_s'] = binding['policy_timeout_seconds']
    spec['benchmark']['task_name'] = 'synthetic_same_world_v%d:%s:not_official_libero_episode' % (
        binding['version'], binding['arm'])
    spec['benchmark']['run_id'] = 'synthetic_same_world_development/' + str(campaign_name)
    SessionSpec.model_validate(spec)
    write_yaml_block(workspace/'SKILLRUNTIME.md', 'Runtime Skill Runtimes', skills)
    binding['policy_adapter'] = uri
    (workspace/'SAME_WORLD_BINDING.json').write_text(json.dumps(binding, indent=2) + '\n')
    if binding['selection_mode'] == CHOICE_MODE:
        feedback = (
            'The native Agent must select one registered order in Session.execution.steps; '
            'no sequence is preselected by workspace preparation.\n'
            'Each step must contain only {"stage_key": <registered key>}.\n'
            'Registered stage instructions: ' + json.dumps(binding['stage_definitions']) + '\n'
            'Allowed complete sequences: ' + json.dumps(binding['allowed_sequences']) + '\n'
            'Native simulator stage feedback follows the selected sequence; it is not learned recognition.\n')
    else:
        feedback = 'Native simulator stage feedback drives the fixed registered sequence; it is not learned recognition.\n'
    (workspace/'ENVIRONMENT.md').write_text(
        'LIBERO same-world composed development setting; not an official single-task episode.\n'
        'The complete task is: ' + binding['joint_goal'] + '.\n'
        + feedback +
        'One root Session shares the entire physical budget. Recovery Sessions with a new reset are unsupported.\n')
    return binding
