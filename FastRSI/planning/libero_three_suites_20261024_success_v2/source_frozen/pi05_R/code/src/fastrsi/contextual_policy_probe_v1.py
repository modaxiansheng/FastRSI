"""Opt-in, immutable local action capability; never changes the native default.

The client binds actual consumed history. This server enforces root identity,
ordered attempts and a two-chunk cap. Rejected proposals use the original policy
and pay the extra encoder cost. A used proposal advances only the original noise
draw, keeping later native randomness aligned without running the decoder.
"""
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np

from fastrsi.contextual_capability_v1 import current_prefix, feature_from_prefix, predict, command_is_in_scope

KEY = '__fastrsi_contextual_v1__'
AUTHORITY = 'CONTEXTUAL_CAPABILITY_DEVELOPMENT_VALUE'


class CarryAttemptSchedule:
    """At most two consecutive boundaries after real mixed-close/closed chunks."""
    def __init__(self):
        self.mixed = False
        self.last = []
        self.attempts = 0
        self.closed = False

    def committed(self, actions, consumed):
        a = np.asarray(actions)
        if (a.ndim != 2 or a.shape[1] != 7 or type(consumed) is not int
                or consumed < 0 or consumed > len(a)):
            raise ValueError('Actual consumed command count required')
        self.last = a[:consumed, 6].tolist()
        if self.last and min(self.last) < 0 < self.last[-1]:
            self.mixed = True

    def request(self):
        if self.closed or self.attempts >= 2:
            return None
        if not (self.mixed and len(self.last) == 10 and min(self.last) > 0):
            if self.attempts:
                self.closed = True
            return None
        self.attempts += 1
        return self.attempts

    def reply(self, used):
        if not used:
            self.closed = True


def load_artifact(path, digest, kind):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen local capability changed')
    if kind not in ('seed_frozen', 'feedback_updated', 'fixed_nearest'):
        raise ValueError('Unknown registered component')
    with np.load(path, allow_pickle=False) as z:
        result = {k: z[k].copy() for k in z.files}
    scope = {'positive_scope_features', 'negative_scope_features'}
    required = scope | ({'prototype_features', 'prototype_actions'} if kind == 'fixed_nearest'
                        else {'centered_features', 'mean_feature', 'mean_action', 'coefficients'})
    if set(result) != required or any(not np.isfinite(v).all() for v in result.values()):
        raise ValueError('Invalid local artifact arrays')
    for key in scope:
        if result[key].ndim != 2 or result[key].shape[1] != 65544 or not len(result[key]):
            raise ValueError('Missing valid scope examples')
    for value in result.values():
        value.setflags(write=False)
    return result


def protect_evaluation_component(path, digest, emit):
    """Lifetime protection beyond the root loop, including delayed callbacks."""
    fixed = Path(path).resolve()
    if hashlib.sha256(fixed.read_bytes()).hexdigest() != digest:
        raise ValueError('Evaluation component version changed')
    def audit(event, args):
        paths = []
        if event == 'open':
            name, mode, flags = args
            if (mode and any(c in mode for c in 'wax+')) or flags & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND):
                paths = [name]
        elif event in ('os.rename','os.link','os.symlink'):
            paths = args[:2]
        elif event in ('os.remove','os.chmod','os.truncate'):
            paths = args[:1]
        for item in paths:
            if isinstance(item,(str,bytes,os.PathLike)) and Path(os.fsdecode(item)).resolve()==fixed:
                emit(dict(event='contextual_evaluation_update_blocked',operation=event,path=str(fixed)))
                raise PermissionError('Frozen evaluation component is read-only')
    sys.addaudithook(audit)
    return dict(path=str(fixed),sha256=digest,lifetime='entire runner process including post-Session callbacks')


def validate_evaluation(case, cfg):
    import json
    raw = Path(cfg['evaluation_registration']).read_bytes()
    if hashlib.sha256(raw).hexdigest()!=cfg['evaluation_registration_sha256']:
        raise ValueError('Evaluation registration changed')
    reg = json.loads(raw)
    if (case.get('mode')!='evaluation' or case.get('native_experience_maintenance') is not False
            or case.get('persistent_updates') is not False or case.get('max_steps')!=reg['immutable']['steps']
            or case.get('suite')!=reg['suite'] or case.get('task_id')!=reg['task_id']
            or case.get('init_state_id') not in reg['initial_states'] or case.get('task_description')!=reg['goal']
            or cfg['sha256']!=reg['versions'][cfg['kind']]['sha256']):
        raise ValueError('Evaluation outside fixed scope or persistent learning enabled')
    return reg


def proposal(model, kind, feature):
    x = np.asarray(feature, np.float64)
    if x.shape != (65544,) or not np.isfinite(x).all():
        raise ValueError('Invalid current observation feature')
    distance = lambda bank: np.sum((bank - x) ** 2, axis=1)
    if kind == 'fixed_nearest':
        a = model['prototype_actions'][distance(model['prototype_features']).argmin()].astype(np.float32)
    else:
        a = predict(model, x[None])[0]
    pos = float(distance(model['positive_scope_features']).min())
    neg = float(distance(model['negative_scope_features']).min())
    return a, bool(pos < neg and command_is_in_scope(a)), pos, neg


def extract(native, observation):
    hidden, mask = current_prefix(native, observation)
    return feature_from_prefix(hidden, mask, observation['observation/state'])


def advance_native_draw(native):
    import torch
    policy = native.policy
    model = policy._model
    if (not model.config.pi05 or model.config.action_horizon != 10
            or policy._sample_kwargs.get('noise') is not None):
        raise ValueError('Original unconditioned H10 sampler required')
    with torch.no_grad():
        model.sample_noise((1, model.config.action_horizon, model.config.action_dim), policy._pytorch_device)


class ContextualPolicyProbe:
    def __init__(self, native, model, kind, *, session_id, goal, artifact_sha256,
                 emit=None, extractor=extract, noise_draw=advance_native_draw):
        self.native, self.model, self.kind = native, model, kind
        self.session_id, self.goal, self.digest = session_id, goal, artifact_sha256
        self.emit = emit or (lambda row: None)
        self.extractor, self.noise_draw = extractor, noise_draw
        self.attempts = 0
        self.used = 0
        self.closed = False
        self.rpc = 0

    def infer(self, observation):
        started = time.perf_counter()
        self.rpc += 1
        clean = dict(observation)
        marker = clean.pop(KEY, None)
        if marker is None:
            result = self.native.infer(clean)
            self.emit(dict(event='contextual_policy_cost', rpc=self.rpc, route='native',
                           teacher_decoder_calls=1, extra_encoder_calls=0,
                           seconds=time.perf_counter()-started))
            return result
        expected = dict(authority=AUTHORITY, session_id=self.session_id,
                        artifact_sha256=self.digest, attempt=self.attempts+1)
        if (marker != expected or self.closed or self.attempts >= 2
                or clean.get('prompt') != self.goal):
            raise ValueError('Foreign, stale, over-budget or wrong-goal capability request')
        self.attempts += 1
        tick = time.perf_counter()
        feature = self.extractor(self.native, clean)
        encoder_seconds = time.perf_counter()-tick
        tick = time.perf_counter()
        actions, accepted, pos, neg = proposal(self.model, self.kind, feature)
        decision_seconds = time.perf_counter()-tick
        if accepted:
            self.noise_draw(self.native)
            self.used += 1
            result = dict(actions=np.ascontiguousarray(actions), policy_meta=dict(
                backend='fastrsi_local_capability', chunk_size=10, action_dim=7))
        else:
            self.closed = True
            result = self.native.infer(clean)
        row = dict(event='contextual_policy_cost', rpc=self.rpc, attempt=self.attempts,
                   kind=self.kind, artifact_sha256=self.digest, used=accepted,
                   route='local_capability' if accepted else 'native_fallback',
                   teacher_decoder_calls=0 if accepted else 1, extra_encoder_calls=1,
                   prefix_feature_seconds=encoder_seconds, local_decision_seconds=decision_seconds,
                   positive_distance=pos, negative_distance=neg,
                   actions_float32_sha256=hashlib.sha256(np.asarray(result['actions'],np.float32).tobytes()).hexdigest(),
                   seconds=time.perf_counter()-started)
        self.emit(row)
        # Native mean latency must include the cheap path and rejected-prefix fee.
        result.setdefault('policy_meta', {})['policy_latency_ms'] = row['seconds'] * 1000
        result[KEY] = dict(used=accepted, attempt=self.attempts, artifact_sha256=self.digest)
        return result
