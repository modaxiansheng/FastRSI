"""Explicit task bindings for development and fixed-experience evaluation.

Task identity and step limits do not grant learning permission. The native
Agent binding separately freezes persistent state throughout evaluation.
"""
GOAL = 'open the top drawer of the cabinet and put the bowl in it'
# Standard suites already advertised by the frozen native target. Limits match
# the frozen OpenPI examples/libero/main.py, not one campaign's custom budget.
STANDARD_STEP_LIMITS = {
    'libero_spatial': 220,
    'libero_object': 280,
    'libero_goal': 300,
    'libero_10': 520,
}
EXTENSION = 'libero90_prerequisite_v1'
STACK_EXTENSION = 'libero90_stack_flow_v1'
STACK_GOALS = {
    63: 'stack the left bowl on the right bowl and place them in the tray',
    64: 'stack the right bowl on the left bowl and place them in the tray',
}
COMPOSITION_EXTENSION = 'libero90_cabinet_composition_v1'
COMPOSITION_GOALS = {
    0: 'close the top drawer of the cabinet',
    1: 'close the top drawer of the cabinet and put the black bowl on top of it',
}
DEVELOPMENT_GOALS = {8: GOAL, **STACK_GOALS, **COMPOSITION_GOALS}
TASK0_CONTROLLED_EXTENSION = 'libero10_task0_controlled400_v1'
TASK0_CONTROLLED_GOAL = 'put both the alphabet soup and the tomato sauce in the basket'


def valid_goal(task_id, goal, limit):
    return type(task_id) is int and DEVELOPMENT_GOALS.get(task_id) == goal and type(limit) is int and limit == 400



def step_limit(case):
    from .same_world_native_binding import requested, load_binding
    if requested(case):
        return load_binding(case)['total_steps']
    suite = case.get('suite', 'libero_spatial')
    if suite == 'libero_90':
        task_id = case.get('task_id')
        expected_extension = (COMPOSITION_EXTENSION if task_id in COMPOSITION_GOALS
                              else EXTENSION if task_id == 8 else STACK_EXTENSION)
        if (case.get('mode') not in ('development', 'evaluation') or case.get('development_suite_extension') != expected_extension
                or not valid_goal(case.get('task_id'), case.get('task_description'), case.get('max_steps', 400))):
            raise ValueError('Outside bound task or data mode')
        limit = 400
    elif suite == 'libero_10' and case.get('development_suite_extension') == TASK0_CONTROLLED_EXTENSION:
        if (case.get('mode') != 'development' or type(case.get('task_id')) is not int or
                case['task_id'] != 0 or case.get('task_description') != TASK0_CONTROLLED_GOAL or
                type(case.get('max_steps')) is not int or case['max_steps'] != 400):
            raise ValueError('Outside explicit controlled task0 development binding')
        limit = 400
    elif suite in STANDARD_STEP_LIMITS:
        limit = STANDARD_STEP_LIMITS[suite]
    else:
        raise ValueError('Unbound suite')
    if type(case.get('max_steps', limit)) is not int or case.get('max_steps', limit) != limit:
        raise ValueError('Changed bound step limit')
    return limit


def extend_target(case, target):
    """Advertise only the reset-tested backend extension, in a fresh workspace."""
    step_limit(case)
    if case.get('suite') != 'libero_90':
        return
    capabilities = [c for c in target['benchmark_capabilities'] if c['benchmark_id'] == 'libero']
    if len(capabilities) != 1:
        raise ValueError('Ambiguous native LIBERO capability')
    if 'libero_90' not in capabilities[0]['suites']:
        capabilities[0]['suites'].append('libero_90')
