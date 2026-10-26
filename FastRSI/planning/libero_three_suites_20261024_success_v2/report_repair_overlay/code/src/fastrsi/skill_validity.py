"""One optional, artifact-bound check of a closed phase's public aperture.

This is a validity diagnostic, not object recognition or a learned threshold.
The bank supplies the boundary, threshold, mode and evidence digest. No hidden
object state is read. Missing policy input causes conservative native fallback;
it never becomes an empty-grasp label. A target rechecks available qpos against
its current formatted observation before accepting either action.
"""
import hashlib
import math
import numpy as np
from .experience_compiler import canonical_sha


def closed_gripper_phases(skill):
    """Return positive-command runs as zero-based [start,end) source indices.

    These are controller-command phases, not claims that an object is held.
    Nonpositive commands end a run. Only saved source waypoints are read.
    """
    commands = [w['gripper_command'] for w in skill['waypoints']]
    if any(not math.isfinite(x) for x in commands):
        raise ValueError('Nonfinite gripper command')
    phases = []; start = None
    for index, command in enumerate(commands):
        if command > 0 and start is None:
            start = index
        elif command <= 0 and start is not None:
            phases.append((start, index)); start = None
    if start is not None:
        phases.append((start, len(commands)))
    return phases


def validate_validity(config, skill):
    fields = {'type', 'boundary_step', 'min_aperture', 'source_sha256', 'mode'}
    if isinstance(config, dict) and config.get('type') == 'closed_phase_aperture_v2':
        fields.add('closed_phase_index')
    if not isinstance(config, dict) or set(config) != fields:
        raise ValueError('Unknown or missing skill validity fields')
    if config['type'] not in ('first_grasp_aperture_v1', 'closed_phase_aperture_v2'):
        raise ValueError('Unknown skill validity type')
    if config['mode'] not in ('aperture', 'always_exit'):
        raise ValueError('Unknown skill validity mode')
    boundary = config['boundary_step']
    waypoints = skill['waypoints']
    if (type(boundary) is not int or boundary <= 0 or boundary % 10
            or boundary >= len(waypoints) or boundary <= skill.get('start_step', 0)):
        raise ValueError('Validity boundary must be inside a ten-step skill prefix')
    threshold = config['min_aperture']
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or threshold <= 0:
        raise ValueError('Invalid aperture threshold')
    source = config['source_sha256']
    if not isinstance(source, str) or len(source) != 64 or any(c not in '0123456789abcdef' for c in source):
        raise ValueError('Validity source digest is not bound')
    commands = [w['gripper_command'] for w in waypoints]
    if any(not math.isfinite(x) for x in commands):
        raise ValueError('Nonfinite gripper command')
    # LIBERO delta_eef_gripper_v1: positive closes, negative opens. Boundary b
    # observes commands [0,b); it must follow a close and precede first release.
    if config['type'] == 'first_grasp_aperture_v1':
        first_close = next((i for i, x in enumerate(commands) if x > 0), len(commands))
        first_release = next((i for i in range(first_close, len(commands)) if commands[i] <= 0), len(commands))
        if not first_close < boundary < first_release:
            raise ValueError('Validity boundary is not inside the first closed-gripper phase')
    else:
        phase_index = config['closed_phase_index']
        phases = closed_gripper_phases(skill)
        if type(phase_index) is not int or not 1 <= phase_index <= len(phases):
            raise ValueError('Unknown one-based closed-gripper phase index')
        start, end = phases[phase_index-1]
        if not start < boundary < end:
            raise ValueError('Validity boundary is not inside the selected closed-gripper phase')
    return config


def _unavailable(status):
    return dict(status=status, gripper_qpos=None, qpos_sha256=None, aperture=None)


def qpos_evidence(qpos):
    """Exact float32[2] binding shared by public state8 and target _last_obs."""
    if qpos is None:
        return _unavailable('missing')
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            pair = np.asarray(qpos, dtype=np.float32)
        if pair.shape != (2,) or not np.isfinite(pair).all():
            return _unavailable('invalid')
    except (ValueError, TypeError, OverflowError):
        return _unavailable('invalid')
    pair = np.ascontiguousarray(pair, dtype='<f4')
    return dict(status='available', gripper_qpos=pair.tolist(),
                qpos_sha256=hashlib.sha256(pair.tobytes()).hexdigest(),
                aperture=abs(float(pair[0]) - float(pair[1])))


def state_evidence(state):
    if state is None:
        return _unavailable('missing')
    try:
        with np.errstate(over='ignore', invalid='ignore'):
            state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            return _unavailable('invalid')
    except (ValueError, TypeError, OverflowError):
        return _unavailable('invalid')
    return qpos_evidence(state[-2:])


def decision(config, evidence):
    if evidence['status'] != 'available':
        return dict(action='exit_to_native', reason='public_state_unavailable', evidence=evidence)
    if config['mode'] == 'always_exit':
        return dict(action='exit_to_native', reason='diagnostic_always_exit', evidence=evidence)
    keep = evidence['aperture'] >= config['min_aperture']
    return dict(action='continue' if keep else 'exit_to_native',
                reason='aperture_at_least_minimum' if keep else 'aperture_below_minimum', evidence=evidence)


def make_proof(config, *, bank_sha, choice, session, step, state):
    if type(step) is not int or step != config['boundary_step']:
        raise ValueError('Validity proof at unbound boundary')
    return dict(version=1, bank_sha=bank_sha, choice=choice, session=session, step=step,
                validity_sha256=canonical_sha(config), source_sha256=config['source_sha256'],
                **decision(config, state_evidence(state)))


def verify_proof(proof, config, *, bank_sha, choice, session, step, target_qpos):
    """Validate binding before physics. Unknown client input permits only exit.

    Available policy qpos must exactly match the target's current public qpos.
    For unavailable client input no aperture claim is accepted: the target logs
    its independent evidence and the conservative exit is explicitly unaligned.
    """
    keys = {'version', 'bank_sha', 'choice', 'session', 'step', 'validity_sha256',
            'source_sha256', 'action', 'reason', 'evidence'}
    if not isinstance(proof, dict) or set(proof) != keys:
        raise ValueError('Malformed validity proof')
    if (type(proof['version']) is not int or proof['version'] != 1
            or type(proof['step']) is not int or proof['step'] != step
            or step != config['boundary_step'] or proof['bank_sha'] != bank_sha
            or proof['choice'] != choice or proof['session'] != session
            or proof['validity_sha256'] != canonical_sha(config)
            or proof['source_sha256'] != config['source_sha256']):
        raise ValueError('Validity proof scope mismatch')
    evidence = proof['evidence']
    if not isinstance(evidence, dict) or set(evidence) != {'status', 'gripper_qpos', 'qpos_sha256', 'aperture'}:
        raise ValueError('Malformed public aperture evidence')
    target = qpos_evidence(target_qpos)
    if evidence['status'] == 'available':
        recomputed = qpos_evidence(evidence['gripper_qpos'])
        if recomputed['status'] != 'available' or evidence != recomputed or evidence != target:
            raise ValueError('Policy and target public gripper evidence differ')
    elif evidence['status'] in ('missing', 'invalid'):
        if evidence != _unavailable(evidence['status']):
            raise ValueError('Unavailable evidence contains an aperture claim')
    else:
        raise ValueError('Unknown public evidence status')
    expected = decision(config, evidence)
    if proof['action'] != expected['action'] or proof['reason'] != expected['reason']:
        raise ValueError('Validity action contradicts the frozen guard')
    return dict(proof=proof, target_evidence=target,
                exact_public_qpos_alignment=evidence['status'] == 'available')
