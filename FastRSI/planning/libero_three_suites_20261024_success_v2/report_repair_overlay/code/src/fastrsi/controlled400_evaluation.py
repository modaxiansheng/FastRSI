"""Explicit fixed-version task0 evaluation at the existing controlled400 limit.

This is not official LIBERO520 evaluation. Original development contracts and
the frozen v3 executor remain unchanged. Only this opt-in case can use400 while
the entire original Agent lifetime has evaluation=True and persistent updates
are blocked, including terminal callbacks and inherited model/source files.
"""
import hashlib
import os
from pathlib import Path
import re
import sys

EXTENSION = 'libero10_task0_controlled400_evaluation_v1'
PROTOCOL = 'fixed_transport_evaluation_v1'
GOAL = 'put both the alphabet soup and the tomato sauce in the basket'
SELECTIONS = ('oldest_eligible_v1', 'nearest_eligible_v1', 'learned_source_choice_stage3_v1')
_original_step_limit = None
_protected = {}
_audit_installed = False


def validate_case(case):
    if (case.get('development_suite_extension') != EXTENSION
            or case.get('evaluation_protocol') != PROTOCOL
            or case.get('mode') != 'evaluation'
            or case.get('native_experience_maintenance') is not False
            or case.get('persistent_updates') is not False
            or case.get('suite') != 'libero_10'
            or type(case.get('task_id')) is not int or case['task_id'] != 0
            or case.get('task_description') != GOAL
            or type(case.get('max_steps')) is not int or case['max_steps'] != 400
            or type(case.get('init_state_id')) is not int or case['init_state_id'] < 0
            or case.get('id') != 'net_t0_i'+str(case['init_state_id'])
            or case.get('feedback_window_mode') != 'fixed10'
            or 'inference_horizon' in case or 'commit_steps' in case):
        raise ValueError('Explicit immutable controlled400 evaluation scope required')
    for key in ('budget_checkpoint', 'phase_transport_source_recording', 'trial_prefix_probe',
                'window_learning_artifact', 'task_window_artifact', 'instruction_experience',
                'goal_preserving_instruction', 'capability_interface'):
        if case.get(key):
            raise ValueError('Evaluation cannot enable a persistent update or another controller: '+key)
    return case


def validate_phase_case(case, config):
    validate_case(case)
    if (config.get('evaluation_protocol') != PROTOCOL or config.get('persistent_updates') is not False
            or config.get('source_selection') not in SELECTIONS
            or config.get('carrier_mode') != 'dense'
            or (config.get('suite'), config.get('task_id'), config.get('goal'), config.get('root_session'),
                config.get('max_steps'), config.get('init_state_id')) !=
               (case['suite'], case['task_id'], GOAL, case['id'], 400, case['init_state_id'])):
        raise ValueError('Frozen actor config belongs to another controlled400 evaluation case')
    return config


def install_task_scope():
    """Narrowly extend step_limit for this explicit evaluation version only."""
    global _original_step_limit
    from . import development_suite
    if _original_step_limit is not None:
        return
    _original_step_limit = development_suite.step_limit
    def step_limit(case):
        if case.get('development_suite_extension') == EXTENSION:
            validate_case(case)
            return 400
        return _original_step_limit(case)
    development_suite.step_limit = step_limit


def forbid_update(*args, **kwargs):
    raise PermissionError('Fixed controlled400 evaluation cannot update transport inventory, policy or completion pointers')


def install_update_guard():
    # Original development_case already refuses evaluation. These explicit
    # guards also prevent an accidental call using a projected development case.
    from . import phase_transport_updates
    for name in ('record_completion', 'finish_native_source', 'initialize_pointer'):
        setattr(phase_transport_updates, name, forbid_update)


def protect_inputs(paths, event_writer):
    """Process-lifetime guards on the frozen model/config/pool, not task logs."""
    global _audit_installed
    receipt = {}
    for candidate in paths:
        path = Path(candidate)
        if path.is_symlink():
            raise ValueError('Frozen evaluation input cannot be a symlink')
        path = path.resolve()
        if not path.is_file():
            raise ValueError('Missing fixed evaluation input: '+str(path))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        _protected[path] = (digest, event_writer)
        receipt[str(path)] = digest
    if _audit_installed:
        return receipt
    _audit_installed = True
    def audit(event, args):
        changed = []
        if event == 'open':
            path, mode, flags = args
            if (isinstance(path, (str, bytes, os.PathLike)) and
                    (mode and any(c in mode for c in 'wax+') or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))):
                changed = [path]
        elif event in ('os.rename', 'os.link', 'os.symlink'):
            changed = list(args[:2])
        elif event in ('os.remove', 'os.rmdir', 'os.chmod', 'os.truncate'):
            changed = [args[0]]
        for item in changed:
            if not isinstance(item, (str, bytes, os.PathLike)):
                continue
            path = Path(os.fsdecode(item)).resolve()
            matches = [fixed for fixed in _protected if path == fixed or
                       (event != 'open' and path in fixed.parents)]
            if matches:
                _, writer = _protected[matches[0]]
                writer(dict(event='controlled400_fixed_input_update_blocked', operation=event, path=str(path)))
                raise PermissionError('Fixed evaluation input is read-only: '+str(path))
    sys.addaudithook(audit)
    return receipt


def verify_inputs(receipt):
    for name, expected in receipt.items():
        path = Path(name)
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Fixed evaluation input changed: '+name)
    return dict(status='PASSED', actual_sha256=dict(receipt), persistent_updates=False,
                scope='Fixed controlled400 development-bound evaluation; not official520 or an efficacy result')
