"""Observe the final existing action-expert pass without another encoder/denoiser.

The retained tensor precedes the FINAL velocity projection, not the integrated
action. It is a candidate-conditioned representation, not a predicted outcome.
No parameter or returned action is changed; production integration is absent.
"""
from contextlib import contextmanager
import time


@contextmanager
def capture_action_expert(model, extract=True):
    assert not model.training
    stats = dict(projection_passes=0, image_passes=0, feature_seconds=0.)
    retained = {}
    original_image = model.paligemma_with_expert.embed_image

    def image(*args, **kwargs):
        stats['image_passes'] += 1
        return original_image(*args, **kwargs)

    def projection(module, args):
        stats['projection_passes'] += 1
        if extract:
            # Replace rather than accumulate denoising history. No device copy
            # or pooling occurs in the model's forward path.
            retained['last'] = args[0].detach()

    handle = model.action_out_proj.register_forward_pre_hook(projection)
    model.paligemma_with_expert.embed_image = image
    try:
        yield stats
        if extract:
            tick = time.perf_counter()
            hidden = retained['last']
            assert hidden.ndim == 3 and hidden.shape[0] == 1
            assert hidden.shape[1] == model.config.action_horizon == 10
            stats['action_tokens'] = hidden.float().cpu().numpy()[0].copy()
            stats['feature_seconds'] = time.perf_counter() - tick
    finally:
        handle.remove()
        model.paligemma_with_expert.embed_image = original_image

