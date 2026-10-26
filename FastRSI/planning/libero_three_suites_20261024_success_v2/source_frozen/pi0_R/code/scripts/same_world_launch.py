"""Bind the optional continuous target through data in the native worker."""
from pathlib import Path


def validate_step_choice(steps, allowed_sequences):
    """One narrow semantic choice; no task, prompt or action edits allowed."""
    allowed = [[{'stage_key': key} for key in sequence] for sequence in allowed_sequences]
    if not isinstance(steps, list) or steps not in allowed:
        raise ValueError('Choose one complete registered stage order')
    return [{'stage_key': row['stage_key']} for row in steps]


def target_command(argv, cases, base, project):
    fields = ('same_world_registration', 'same_world_registration_sha256', 'same_world_arm')
    selected = [case for case in cases if any(key in case for key in fields)]
    if not selected:
        return list(argv)
    if len(cases) != 1:
        raise ValueError('One root Session owns a same-world target instance')
    case = dict(selected[0])
    registration = Path(case['same_world_registration'])
    try:
        relative = registration.relative_to('/project')
    except ValueError:
        raise ValueError('Native target registration must use the mounted project path')
    if '..' in relative.parts:
        raise ValueError('Registration cannot escape the mounted project')
    probes = ('goal_skill_probe', 'action_memory_probe', 'procedure_probe', 'boundary_probe',
              'stall_artifact', 'prefix_artifact', 'native_stage_continuation_probe',
              'placement_recovery', 'action_window_artifact',
              'task_window_artifact', 'window_learning_artifact')
    if any(case.get(key) for key in probes) or any(value.startswith((
            'FASTRSI_GOAL_CONFIG=', 'FASTRSI_TRIAL_SPEC=', 'FASTRSI_LIVE_CONTROLS=')) for value in argv):
        raise ValueError('Same-world launch cannot replace another target intervention')
    case['same_world_registration'] = str(Path(project)/relative)
    from fastrsi.same_world_native_binding import load_binding
    binding = load_binding(case)
    result = list(argv)
    script = base + '/code/scripts/instrument_target.py'
    if result.count(script) != 1:
        raise ValueError('Expected the existing native target launch template')
    index = result.index(script)
    if result[index-1] != 'python':
        raise ValueError('Unexpected native target Python entry')
    return result[:index] + [base + '/code/scripts/serve_same_world_native_target.py',
        '--registration', str(registration),
        '--registration-sha256', binding['registration_sha256'],
        '--arm', binding['arm'], '--session-id', binding['session_id'],
        '--joint-goal', binding['joint_goal'], '--host', '127.0.0.1', '--port', '49502']
