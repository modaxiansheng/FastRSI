"""Low-cost fixed execution rules; no learning or hidden-state inputs."""
import math
import re
import time

MODES = ('fixed10', 'fixed5', 'gripper_transition', 'fixed_commitment')

def commitment_binding(case, mode):
    """Explicit H/W diagnostic only; legacy rules keep their original ten rows."""
    keys = ('inference_horizon', 'commit_steps')
    if mode != 'fixed_commitment':
        if any(key in case for key in keys):
            raise ValueError('Explicit H/W requires fixed_commitment mode')
        return None
    if any(case.get(key) for key in ('task_window_artifact', 'task_window_sha256',
            'window_learning_artifact', 'window_learning_sha256',
            'stage_flow_artifact', 'action_window_artifact')):
        raise ValueError('H/W diagnosis cannot reinterpret historical learning artifacts')
    h, w = (case.get(key) for key in keys)
    if type(h) is not int or type(w) is not int or (h, w) not in ((10, 10), (20, 10), (20, 20)):
        raise ValueError('Explicit H/W must be H10W10, H20W10 or H20W20')
    return dict(inference_horizon=h, commit_steps=w)

def prediction_revision_diagnostics(signs, step, prior_signs, prior_step):
    """Compare predictions for the same absolute action slots, before submission.

    No overlap is missing evidence, never a zero-disagreement observation.
    Gripper command agreement does not establish physical task success.
    """
    offset = None if prior_step is None else step - prior_step
    if prior_signs is None or offset is None or not 0 < offset < len(prior_signs):
        return {'available': False, 'overlap_steps': 0, 'changed_steps': None,
                'changed_fraction': None, 'reason': 'no_overlapping_prior_prediction'}
    old = prior_signs[offset:offset + len(signs)]
    changed = sum(a != b for a, b in zip(old, signs))
    return {'available': True, 'overlap_steps': len(old), 'changed_steps': changed,
            'changed_fraction': changed / len(old),
            'scope': 'Gripper commands for matching action indices; not physical correctness'}

def action_prediction_revision_diagnostics(actions, step, prior_actions, prior_step):
    """Compare raw action commands for matching absolute action indices.

    Inputs are already computed predictions, not extra policy queries. Rotation
    columns are compared in their native command representation, not as a pose
    error. Neither agreement nor disagreement is a physical reliability label.
    """
    diagnostics = {
        'available': False, 'overlap_steps': 0,
        'absolute_action_start': None, 'absolute_action_end_exclusive': None,
        'prior_prediction_start_index': None, 'current_prediction_start_index': None,
        'translation_l2_mean': None, 'translation_l2_max': None,
        'rotation_command_l2_mean': None, 'rotation_command_l2_max': None,
        'gripper_absolute_change_mean': None, 'gripper_absolute_change_max': None,
        'gripper_sign_changed_steps': None, 'gripper_sign_changed_fraction': None,
        'scope': 'Raw predicted commands at matching action indices; not pose error, safety or task success',
    }
    offset = None if prior_step is None else step - prior_step
    if prior_actions is None or offset is None or not 0 < offset < len(prior_actions):
        diagnostics['reason'] = 'no_overlapping_prior_prediction'
        return diagnostics
    overlap = min(len(actions), len(prior_actions) - offset)
    if not overlap:
        diagnostics['reason'] = 'no_overlapping_prior_prediction'
        return diagnostics
    old = prior_actions[offset:offset + overlap]
    current = actions[:overlap]
    translation = [math.dist(a[:3], b[:3]) for a, b in zip(old, current)]
    rotation = [math.dist(a[3:6], b[3:6]) for a, b in zip(old, current)]
    gripper = [abs(a[6] - b[6]) for a, b in zip(old, current)]
    sign_changes = sum((a[6] > 0) != (b[6] > 0) for a, b in zip(old, current))
    diagnostics.update(
        available=True, overlap_steps=overlap,
        absolute_action_start=step, absolute_action_end_exclusive=step + overlap,
        prior_prediction_start_index=offset, current_prediction_start_index=0,
        translation_l2_mean=sum(translation) / overlap, translation_l2_max=max(translation),
        rotation_command_l2_mean=sum(rotation) / overlap, rotation_command_l2_max=max(rotation),
        gripper_absolute_change_mean=sum(gripper) / overlap, gripper_absolute_change_max=max(gripper),
        gripper_sign_changed_steps=sign_changes, gripper_sign_changed_fraction=sign_changes / overlap,
    )
    return diagnostics

def transition_diagnostics(signs,previous,n,prior_direction,count):
    transition = next((i for i in range(1,len(signs)) if signs[i]!=signs[i-1]),None)
    if previous is not None and previous != signs[0]:transition=0
    direction=([previous,signs[0]] if previous is not None and previous != signs[0] else
           [signs[transition-1],signs[transition]] if transition is not None else None)
    diagnostics={'first_predicted_transition_index':transition,
             'transition_direction':direction,
             'same_direction_deferrals_before':count if direction==prior_direction else 0,
             'predicted_transition_outside_committed_prefix':transition is not None and transition>=n,
             'scope':'Prediction and past submitted commands only; not object state, outcome or recovery truth'}
    if diagnostics['predicted_transition_outside_committed_prefix']:
        count=diagnostics['same_direction_deferrals_before']+1
        prior_direction=direction
    else:count=0;prior_direction=None
    diagnostics['consecutive_deferrals_after']=count
    return diagnostics,prior_direction,count

class FeedbackWindow:
    def __init__(self, case, mode):
        if case.get('suite')=='libero_90':
            from .development_suite import step_limit
            step_limit(case)
            if mode!='fixed10':raise ValueError('Prerequisite development uses unchanged fixed10')
        elif case['suite'] in ('libero_goal', 'libero_object'):
            from .development_suite import step_limit
            step_limit(case)
        elif case['suite'] != 'libero_10':
            raise ValueError('Unsupported window scope')
        if mode not in MODES:
            raise ValueError('Unsupported window scope')
        self.case = dict(case)
        self.mode = mode
        self.commitment = commitment_binding(case, mode)
        self.inference_horizon = 10 if self.commitment is None else self.commitment['inference_horizon']
        self.previous_gripper = None
        self.previous_step = None
        self.deferred_direction = None
        self.consecutive_deferrals = 0
        self.prior_predicted_signs = None
        self.prior_predicted_actions = None

    def _choose_horizon(self, signs, within, boundary, step):
        """Original fixed/transition rule; diagnostic logging does not change it."""
        if self.commitment is not None:
            return self.commitment['commit_steps']
        return 5 if self.mode == 'fixed5' or (self.mode == 'gripper_transition' and (within or boundary)) else 10

    def apply(self, chunk, context):
        started = time.perf_counter()
        if commitment_binding(self.case, self.mode) != self.commitment:
            raise ValueError('Bound H/W changed during execution')
        if context.get('session_id') != self.case['id'] or context.get('task_description') != self.case['task_description']:
            raise ValueError('Window controller Session/goal mismatch')
        match = re.fullmatch(r'libero_obs_(\d+)', str(context.get('source_observation_id', '')))
        if not match:
            raise ValueError('Missing current observation step')
        step = int(match.group(1))
        if not 0 <= step < self.case['max_steps'] or (self.previous_step is not None and step <= self.previous_step):
            raise ValueError('Invalid/nonmonotonic execution step')
        actions = chunk['actions']
        if len(actions) != self.inference_horizon or any(len(a) != 7 or not all(math.isfinite(float(v)) for v in a) for a in actions):
            raise ValueError('Expected configured prediction length of finite seven-dimensional actions')
        # Detached JSON-compatible snapshot includes the unsubmitted tail.
        predicted_actions = [[float(v) for v in action] for action in actions]
        signs = [float(a[6]) > 0 for a in actions]
        within = any(s != signs[0] for s in signs[1:])
        boundary = self.previous_gripper is not None and self.previous_gripper != signs[0]
        h = self._choose_horizon(signs, within, boundary, step)
        n = min(h, self.case['max_steps'] - step)
        tag=chunk.get('provenance',{}).get('goal_skill')
        if tag is not None:
            # An explicit finite skill may end between normal VLA boundaries.
            # It is not a shorter policy inference horizon: placeholders are
            # validated and converted into real commands by the target.
            probe=self.case.get('goal_skill_probe',{})
            if (not probe.get('enabled') or self.mode!='fixed10' or
                tag.get('sha256')!=probe.get('sha256') or tag.get('session_id')!=context.get('session_id') or
                tag.get('start_step')!=step or type(tag.get('count')) is not int or not 1<=tag['count']<=n or
                any(float(v)!=0 for a in actions for v in a)):
                raise ValueError('Invalid explicit skill boundary')
            n=tag['count']
        out = {**chunk, 'actions': actions[:n].copy(),
               'timing': {**chunk['timing'], 'horizon_steps': n},
               'action_contract': {**chunk['action_contract'], 'shape': [n, 7]}}
        diagnostics,self.deferred_direction,self.consecutive_deferrals=transition_diagnostics(
            signs,self.previous_gripper,n,self.deferred_direction,self.consecutive_deferrals)
        diagnostics['prediction_revision_before_choice'] = prediction_revision_diagnostics(
            signs, step, self.prior_predicted_signs, self.previous_step)
        diagnostics['action_prediction_revision_before_choice'] = action_prediction_revision_diagnostics(
            predicted_actions, step, self.prior_predicted_actions, self.previous_step)
        event = {'event': 'feedback_window_choice', 'mode': self.mode, 'step': step,
                 'source_observation_id': context['source_observation_id'],
                 'within_chunk_transition': within, 'boundary_transition': boundary,
                 'previous_committed_gripper_positive': self.previous_gripper,
                 'predicted_actions': predicted_actions,
                 'predicted_gripper_positive': signs, 'requested_steps': h, 'submitted_steps': n,
                 'original_steps': len(actions),'decision_diagnostics':diagnostics}
        if self.commitment is not None:
            event.update(self.commitment)
        self.previous_gripper = signs[n - 1]
        self.prior_predicted_signs = signs.copy()
        # Event consumers and reused input buffers cannot change future diagnostics.
        self.prior_predicted_actions = [a.copy() for a in predicted_actions]
        self.previous_step = step
        event['selector_seconds'] = time.perf_counter() - started
        return out, event
