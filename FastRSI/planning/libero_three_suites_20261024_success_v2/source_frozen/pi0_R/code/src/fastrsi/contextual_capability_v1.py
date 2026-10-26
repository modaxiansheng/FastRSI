"""Experimental local capability from current legal observations.

Frozen pi05 encoder; centered linear-kernel ridge on actual relative commands.
This module confers no execution authority or semantic grasp certificate.
"""
import hashlib
import json
from pathlib import Path

import numpy as np

VERSION = 'contextual_native_relative_capability_v1'
ROBOT_SCALES = np.asarray([5,5,5,.3125,.3125,.3125,25,25], np.float32)


def current_prefix(native, observation):
    """Run only the original current-input prefix, drawing no sampler noise."""
    import torch
    from openpi.models.model import Observation
    from openpi.models_pytorch.pi0_pytorch import make_att_2d_masks
    policy = native.policy
    model = policy._model
    assert model.config.pi05 and model.config.action_horizon == 10
    assert model.config.pytorch_compile_mode is None and not model.training
    with torch.no_grad():
        data = policy._input_transform(dict(observation))
        def tensor(value):
            if isinstance(value, dict):
                return {k:tensor(v) for k,v in value.items()}
            return torch.from_numpy(np.array(value)).to(policy._pytorch_device)[None,...]
        transformed = tensor(data)
        prepared = Observation.from_dict(transformed)
        images, image_masks, tokens, token_masks, state = model._preprocess_observation(prepared, train=False)
        embeds, masks, attention = model.embed_prefix(images, image_masks, tokens, token_masks)
        mask4 = model._prepare_attention_masks_4d(make_att_2d_masks(masks, attention))
        positions = torch.cumsum(masks, dim=1)-1
        model.paligemma_with_expert.paligemma.language_model.config._attn_implementation = 'eager'
        outputs, cache = model.paligemma_with_expert.forward(attention_mask=mask4,
            position_ids=positions, past_key_values=None, inputs_embeds=[embeds,None], use_cache=True)
        return outputs[0], masks


def feature_from_prefix(hidden, masks, raw_robot):
    import torch
    if tuple(hidden.shape) != (1,968,2048) or tuple(masks.shape) != (1,968):
        raise ValueError('Frozen prefix layout changed')
    if not bool(masks[0,:512].all()) or bool(masks[0,512:768].any()):
        raise ValueError('Expected two valid visual views and the native masked third view')
    visual = hidden[0,:512].float().reshape(2,16,16,2048)
    visual = visual.reshape(2,4,4,4,4,2048).mean(dim=(2,4)).reshape(-1)
    visual = visual / torch.linalg.vector_norm(visual).clamp_min(1e-12)
    robot = np.asarray(raw_robot,np.float32)
    if robot.shape != (8,) or not np.isfinite(robot).all():
        raise ValueError('Original public robot8 required')
    value = np.r_[visual.cpu().numpy(), robot*ROBOT_SCALES].astype(np.float32)
    if not np.isfinite(value).all():
        raise ValueError('Nonfinite feature')
    return value


def fit(features, actions, ridge=.01):
    x = np.asarray(features,np.float64)
    y = np.asarray(actions,np.float64).reshape(len(x),70)
    if (x.ndim != 2 or x.shape[1] != 65544 or len(x) < 2 or ridge != .01
            or not np.isfinite(x).all() or not np.isfinite(y).all()):
        raise ValueError('Registered finite feature and command recipe required')
    mean_x, mean_y = x.mean(axis=0), y.mean(axis=0)
    centered = x-mean_x
    coefficients = np.linalg.solve(centered@centered.T+ridge*np.eye(len(x)),y-mean_y)
    return dict(centered_features=centered, mean_feature=mean_x,
                mean_action=mean_y, coefficients=coefficients)


def predict(model, features):
    x = np.asarray(features,np.float64)
    prediction = (x-model['mean_feature'])@model['centered_features'].T@model['coefficients']+model['mean_action']
    return prediction.reshape(-1,10,7).astype(np.float32)


def command_is_in_scope(actions):
    a = np.asarray(actions)
    return (a.shape == (10,7) and bool(np.isfinite(a).all()) and bool((np.abs(a)<=2).all())
            and bool((a[:,6]>0).all()))


def save_model(path, model):
    path = Path(path)
    with path.open('xb') as handle:
        np.savez(handle, **model)
    return hashlib.sha256(path.read_bytes()).hexdigest()
