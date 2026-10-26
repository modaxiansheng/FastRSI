"""Bounded numerical-inference candidate; no weight, action horizon, or task edits."""
import functools
import hashlib
import inspect
import json
import time

ALLOWED_STEPS = (5, 10)


def bind(policy, steps, emit):
    if type(steps) is not int or steps not in ALLOWED_STEPS:
        raise ValueError('Unbound sampling budget')
    signature = inspect.signature(policy._sample_actions)
    if signature.parameters['num_steps'].default != 10:
        raise ValueError('Sampler default changed')
    if policy._sample_kwargs:
        raise ValueError('Existing sampling override: inspect before combining')
    original = policy._sample_actions

    @functools.wraps(original)
    def measured(*args, **kwargs):
        if kwargs.get('num_steps') != steps:
            raise ValueError('Sampler budget not actually forwarded')
        started = time.perf_counter()
        result = original(*args, **kwargs)
        emit({'event': 'sample_actions', 'num_steps': steps,
              'shape': list(result.shape), 'seconds': time.perf_counter()-started,
              'timing_scope': 'Host call interval, not synchronized GPU kernel time'})
        return result

    policy._sample_kwargs = {'num_steps': steps}
    policy._sample_actions = measured
    return {'num_steps': steps, 'native_default': 10, 'weights_changed': False,
            'action_horizon_changed': False}
