"""Count eager OpenPI sampler entries and actual component forward calls.

Instrumentation preserves arguments, bound-method identity, returned objects,
and RNG behavior. It does not synchronize the accelerator or change kernels.
"""
import functools
import inspect
import time
import types


def install(native, emit):
    policy = native.policy
    model = policy._model
    original = policy._sample_actions
    if not policy._is_pytorch_model or not inspect.ismethod(original) or original.__self__ is not model:
        raise ValueError('Meter requires native eager bound PyTorch sampler')
    if native.policy_config not in ('pi0_libero', 'pi05_libero'):
        raise ValueError('Unsupported model for pi meter')
    counter = {'generation': 0, 'forward': 0}

    @functools.wraps(original)
    def sampled(self, *args, **kwargs):
        assert self is model
        counter['generation'] += 1
        index = counter['generation']
        emit('action_generation_begin', index=index, scope='openpi_native_sample_actions')
        started = time.perf_counter()
        try:
            output = original(*args, **kwargs)
        except BaseException as exc:
            emit('action_generation_failed', index=index, error=type(exc).__name__)
            raise
        emit('action_generation_end', index=index, seconds=time.perf_counter() - started,
             timing_scope='host sampler call elapsed; accelerator not synchronized')
        return output

    # OpenPIStateGuard requires a bound method with __self__ == model. No new
    # Policy attributes are added and the native model sampler is retained.
    policy._sample_actions = types.MethodType(sampled, model)

    def wrap(obj, scope):
        method = obj.forward
        if not callable(method):
            raise ValueError('Missing native forward: ' + scope)
        @functools.wraps(method)
        def forward(self, *args, **kwargs):
            assert self is obj
            counter['forward'] += 1
            index = counter['forward']
            emit('neural_forward_begin', index=index, scope=scope)
            try:
                result = method(*args, **kwargs)
            except BaseException as exc:
                emit('neural_forward_failed', index=index, scope=scope, error=type(exc).__name__)
                raise
            emit('neural_forward_end', index=index, scope=scope)
            return result
        obj.forward = types.MethodType(forward, obj)

    # These are nonoverlapping component entries, rather than the outer joint
    # wrapper plus its nested children. Layer calls are not summed separately.
    joint = model.paligemma_with_expert
    wrap(joint.paligemma.model.vision_tower, 'openpi_vision_encoder')
    wrap(joint.paligemma.language_model, 'openpi_language_prefix')
    wrap(joint.gemma_expert.model, 'openpi_action_denoising')
    return {'version': 'pi_native_meter_v8', 'generation_site': 'Policy._sample_actions',
            'forward_scopes': ['openpi_vision_encoder', 'openpi_language_prefix', 'openpi_action_denoising'],
            'scope': 'Actual sampler/component entries, not FLOPs or total individual layer forwards; applied to P/F/R identically.',
            'accelerator_synchronization_added': False, 'sampling_parameters_changed': False}
