"""Reject inert candidates cheaply; never infer a counterfactual suffix outcome."""
from collections import Counter
import hashlib
import numpy as np

from .action_memory import state
from .decision_corpus import checked_image, png_pixels


def bound_observation(event, trace, image_bytes):
    """Project only recorded, pre-action policy inputs and verify their bindings."""
    if event['observation_id'] != trace['observation_id']:
        raise ValueError('Observation identity mismatch')
    value = state(event['robot_state'])
    binding = event['array_bindings']['observation/state']
    if (binding['shape'] != [8] or binding['dtype'] != 'float32' or
            hashlib.sha256(value.tobytes()).hexdigest() != binding['sha256']):
        raise ValueError('State does not match actual policy input')
    result = {'prompt': event['prompt'], 'observation/state': value}
    for key, camera in [('observation/image', 'agentview_image'),
                        ('observation/wrist_image', 'robot0_eye_in_hand_image')]:
        name = f'{trace["step"]:04d}_{camera}.png'
        blob = image_bytes(name)
        checked_image(blob, trace['images'][name], event['array_bindings'][key])
        pixels, shape = png_pixels(blob)
        result[key] = np.frombuffer(pixels, dtype=np.uint8).reshape(shape)
    return result


def first_intervention(records, selector):
    """records yields (audit identity, legal observation); stop on first action.

    No result, reward, or outcome label is given to the selector. Once it proposes
    an action, subsequent recorded states cease to describe its possible rollout.
    A hit is an intervention opportunity, not evidence of correctness or savings.
    """
    reasons = Counter()
    count = 0
    for identity, observation in records:
        actions, decision = selector.choose(observation)
        count += 1
        reasons[decision['reason']] += 1
        if actions is not None:
            a = np.asarray(actions, dtype=np.float32)
            if a.ndim != 2 or a.shape[1] != 7 or not np.isfinite(a).all():
                raise ValueError('Invalid proposed actions')
            return dict(status='POTENTIAL_INTERVENTION', checked=count,
                        reasons=dict(reasons), first=dict(identity=identity,
                        decision=decision, action_shape=list(a.shape),
                        action_sha256=hashlib.sha256(a.tobytes()).hexdigest()),
                        suffix_evaluated=False, outcome_claim=None)
    return dict(status='NO_INTERVENTION_ON_RECORDED_PATH', checked=count,
                reasons=dict(reasons), first=None, suffix_evaluated=False,
                outcome_claim=None)
