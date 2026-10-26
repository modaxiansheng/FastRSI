"""Development-only primitive feasibility probe; never changes the task evaluator."""
from .development_suite import step_limit, GOAL

PROMPT = 'open the top drawer of the cabinet'
PROBE = 'open_top_drawer_only_v1'


def install(adapter_class, case, model_sha, emit):
    step_limit(case)
    if (case.get('prerequisite_component_probe') != PROBE
            or case.get('suite') != 'libero_90'
            or case.get('init_state_id') != 0
            or case.get('component_model_sha256') != model_sha
            or case.get('feedback_window_mode') != 'fixed10'
            or any(case.get(k) for k in ('stage_flow_artifact', 'action_window_artifact',
                                        'window_learning_artifact', 'task_window_artifact',
                                        'decision_validity_shadow'))):
        raise ValueError('Unbound or conflicting component probe')
    original = adapter_class.to_policy_input

    def wrapped(adapter, observation, context):
        payload = original(adapter, observation, context)
        # Native replans keep their own goals and receive no component override.
        if context.get('session_id') != case['id'] or context.get('task_description') != GOAL:
            return payload
        if payload.get('prompt') != GOAL:
            raise ValueError('Unexpected native policy prompt')
        result = dict(payload)
        result['prompt'] = PROMPT
        emit(dict(event='prerequisite_component_input', case=case['id'],
                  observation_id=observation.get('observation_id'), original_goal=GOAL,
                  prompt=PROMPT, probe=PROBE, scope='component_feasibility_not_full_task_success'))
        return result

    adapter_class.to_policy_input = wrapped

    def restore():
        adapter_class.to_policy_input = original
    return restore
