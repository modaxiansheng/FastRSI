"""Observe the existing native prefix pass; never replace actions or encode again."""
from contextlib import contextmanager
import time


@contextmanager
def capture_prefix(model, extract=True):
    import torch
    assert not model.training
    gemma = model.paligemma_with_expert
    original_embed = model.embed_prefix
    original_forward = gemma.forward
    original_image = gemma.embed_image
    stats = dict(prefix_passes=0, image_passes=0, feature_seconds=0., features=[])
    state = {}

    def embed(*args, **kwargs):
        result = original_embed(*args, **kwargs)
        state['mask'] = result[1].detach()
        return result

    def image(*args, **kwargs):
        stats['image_passes'] += 1
        return original_image(*args, **kwargs)

    def forward(*args, **kwargs):
        result = original_forward(*args, **kwargs)
        inputs = kwargs['inputs_embeds']
        if inputs[1] is None:
            stats['prefix_passes'] += 1
            if extract:
                tick = time.perf_counter()
                hidden = result[0][0].detach().float()
                mask = state['mask'].bool()
                assert hidden.shape[:2] == mask.shape and hidden.shape[0] == 1
                weights = mask.to(hidden.dtype)[..., None]
                mean = (hidden * weights).sum(1) / weights.sum(1).clamp_min(1.)
                indices = torch.arange(mask.shape[1], device=mask.device)[None, :]
                last_index = indices.expand_as(mask).masked_fill(~mask, -1).max(1).values
                assert bool((last_index >= 0).all())
                last = hidden[torch.arange(hidden.shape[0], device=hidden.device), last_index]
                feature = torch.cat((mean, last), dim=1).cpu().numpy()[0].copy()
                stats['features'].append(feature)
                stats['feature_seconds'] += time.perf_counter() - tick
        return result

    model.embed_prefix = embed
    gemma.forward = forward
    gemma.embed_image = image
    try:
        yield stats
    finally:
        model.embed_prefix = original_embed
        gemma.forward = original_forward
        gemma.embed_image = original_image
