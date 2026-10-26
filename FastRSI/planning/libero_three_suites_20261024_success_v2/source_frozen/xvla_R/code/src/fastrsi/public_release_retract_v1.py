"""Development-only public-proprio operation; no object-release certificate.

Only SameWorldContinuation owns physics, goal, deadline and the shared budget.
Call AFTER a normal activation of a nonterminal stage, never to bypass done.
Control semantics verified against the frozen target's robosuite OSC / Panda:
OSC source cfa62a0e719bc53ef0c701efa66f7b3e2272d4fca2150ec73c05bb145eba85cb
base source 61f99a2daeb4eb6e39bfb996d94c5a5689ef3bc771d030d070193ea56d6a8332
gripper a3760a8fc4599fa6f79914304b82466f3c1bda1a945943ca693bb57f00dc59eb
EEF xyz is public robot0_eef_pos (MuJoCo site_xpos); registration uses native
world-length units, never interprets a normalized action as metres. Negative
Panda command opens the fingers; qpos bounds do not establish object release.
"""
import copy
import math
import time

PROTOCOL = 'public_release_retract_v1'
UNIT = 'native_world_length'


def _vector(value, length):
    values = list(value)
    if len(values) != length or any(isinstance(v, bool) or not math.isfinite(float(v)) for v in values):
        raise ValueError('Finite vector of length %s required' % length)
    return [float(v) for v in values]


def describe_control_contract(owner):
    """Read static controller configuration, not controller position or object state."""
    robot = owner.runtime.env.robots[0]
    c, g = robot.controller, robot.gripper
    return dict(controller_class=type(c).__name__, gripper_class=type(g).__name__,
                use_delta=bool(c.use_delta), use_ori=bool(c.use_ori),
                impedance_mode=c.impedance_mode, control_dim=c.control_dim,
                input_min=_vector(c.input_min, 6), input_max=_vector(c.input_max, 6),
                output_min=_vector(c.output_min, 6), output_max=_vector(c.output_max, 6),
                gripper_dof=g.dof, gripper_speed=float(g.speed))


def validate_registration(registration):
    r = copy.deepcopy(registration)
    fields = {'protocol', 'development_only', 'stage_key', 'coordinate_units',
              'controller_contract', 'open_hold_steps', 'open_qpos_bounds',
              'retract_delta_z', 'max_translation_delta_per_step', 'max_retract_steps',
              'z_tolerance', 'max_normalized_z_command'}
    if set(r) != fields or r['protocol'] != PROTOCOL or r['development_only'] is not True:
        raise ValueError('Explicit development-only operation registration required')
    if r['coordinate_units'] != UNIT or not isinstance(r['stage_key'], str) or not r['stage_key']:
        raise ValueError('Bind native public coordinate units and active stage')
    for k in ('open_hold_steps', 'max_retract_steps'):
        if type(r[k]) is not int or r[k] <= 0:
            raise ValueError('Positive finite phase step limits required')
    for k in ('retract_delta_z', 'max_translation_delta_per_step', 'z_tolerance', 'max_normalized_z_command'):
        if type(r[k]) not in (float, int) or not math.isfinite(r[k]) or r[k] <= 0:
            raise ValueError('Explicit positive finite motion limits required')
    if r['z_tolerance'] >= r['retract_delta_z']:
        raise ValueError('Tolerance must be below the registered lift')
    if len(r['open_qpos_bounds']) != 2:
        raise ValueError('Two public finger-coordinate bounds required')
    r['open_qpos_bounds'] = [_vector(b, 2) for b in r['open_qpos_bounds']]
    if any(b[0] > b[1] for b in r['open_qpos_bounds']):
        raise ValueError('Invalid qpos bounds')
    c = r['controller_contract']
    exact = dict(controller_class='OperationalSpaceController', gripper_class='PandaGripper',
                 use_delta=True, use_ori=True, impedance_mode='fixed', control_dim=6,
                 gripper_dof=1, gripper_speed=0.01)
    if set(c) != set(exact) | {'input_min','input_max','output_min','output_max'} or any(c[k] != v for k,v in exact.items()):
        raise ValueError('Unverified controller or gripper semantics')
    for k in ('input_min','input_max','output_min','output_max'):
        c[k] = _vector(c[k], 6)
    for a,b in zip(c['input_min'] + c['output_min'], c['input_max'] + c['output_max']):
        if b <= 0 or a != -b:
            raise ValueError('This operation requires symmetric zero-centred scaling')
    if r['max_normalized_z_command'] > c['input_max'][2]:
        raise ValueError('Command amplitude exceeds registered controller input range')
    return r


class PublicReleaseRetract:
    def __init__(self, owner, registration=None, *, enabled=False, clock=time.perf_counter):
        if type(enabled) is not bool:
            raise ValueError('Explicit boolean opt-in required')
        self.owner, self.enabled, self.clock = owner, enabled, clock
        self.registration = validate_registration(registration) if enabled else None
        self.last_report = None
        self.used = False

    def run(self, *, event_sink=None):
        if not self.enabled:
            self.last_report = dict(protocol=PROTOCOL, status='DISABLED', physical_steps=0, rows=[])
            return self.last_report
        if self.used:
            raise RuntimeError('Operation instance is single-use')
        self.used = True
        o, r, cfg = self.owner, self.owner.runtime, self.registration
        t0, start = self.clock(), r.step_idx
        report = self.last_report = dict(protocol=PROTOCOL, status='RUNNING',
            scope='seen_development_operation_not_learning_or_release_certificate',
            registration=copy.deepcopy(cfg), start_step=start, end_step=start, physical_steps=0,
            rows=[], public_qpos_in_registered_bounds=False, public_z_target_reached=False,
            costs=dict(action_seconds=0., observe_seconds=0., event_seconds=0.))
        binding = (o.active, o.activation_count, r.language, copy.deepcopy(r.env.env.parsed_problem['goal_state']),
                   o.deadline, o.total_steps, r.config['max_steps'], getattr(r,'session_id',None))

        def identity():
            o._check_identity()
            now = (o.active, o.activation_count, r.language, r.env.env.parsed_problem['goal_state'],
                   o.deadline, o.total_steps, r.config['max_steps'], getattr(r,'session_id',None))
            if now != binding:
                raise RuntimeError('Operation owner/goal/budget binding changed')
            if describe_control_contract(o) != cfg['controller_contract']:
                raise RuntimeError('Controller configuration differs from registration')
            c = r.env.robots[0].controller
            expected = [b/a for a,b in zip(cfg['controller_contract']['input_max'],cfg['controller_contract']['output_max'])]
            for key, values in (('action_scale', expected), ('action_input_transform',[0.]*6), ('action_output_transform',[0.]*6)):
                cached = getattr(c,key,None)
                if cached is not None and _vector(cached,6) != values:
                    raise RuntimeError('Controller cached scaling differs from registration')

        def stopped():
            identity()
            if r.execution_status().get('cancelled'):
                return 'CANCELLED'
            if r.success:
                return 'NATIVE_SUCCESS'
            if r.step_idx >= min(o.total_steps, r.config['max_steps']):
                return 'BUDGET_EXHAUSTED'
            if r.done:
                return 'NATIVE_DONE'
            if o.clock() >= o.deadline:
                return 'TIMEOUT'
            return None

        def public():
            tick = self.clock()
            try:
                obs = o.observe()
                result = dict(observation_id=obs.get('observation_id'),
                    eef_pos=_vector(obs['robot0_eef_pos'],3),
                    gripper_qpos=_vector(obs['robot0_gripper_qpos'],2))
                report['latest_public'] = result
                report['public_qpos_in_registered_bounds'] = all(lo <= v <= hi for v,(lo,hi) in zip(result['gripper_qpos'],cfg['open_qpos_bounds']))
                return result
            finally:
                report['costs']['observe_seconds'] += self.clock()-tick

        def step(phase, action, before):
            row = dict(phase=phase, action=action, step_before=r.step_idx,
                       public_before=before, status='SUBMITTED')
            report['rows'].append(row)
            tick = self.clock()
            try:
                status = o.action_chunk({'chunk_id':'%s_%s_%d' % (PROTOCOL,phase,r.step_idx), 'actions':[action]})
            finally:
                row['action_seconds'] = self.clock()-tick
                report['costs']['action_seconds'] += row['action_seconds']
                row['step_after'] = r.step_idx
                row['physical_steps'] = r.step_idx-row['step_before']
                row['native_success'], row['native_done'] = bool(r.success), bool(r.done)
            if row['physical_steps'] != 1 or bool(status.get('success')) != bool(r.success) or bool(status.get('done')) != bool(r.done):
                raise RuntimeError('Native action status/count mismatch')
            identity()
            after = public()
            row.update(public_after=after,status='EXECUTED')
            if event_sink is not None:
                tick = self.clock()
                try:
                    event_sink(copy.deepcopy(row))
                finally:
                    report['costs']['event_seconds'] += self.clock()-tick
            identity()
            return after

        try:
            if o.active != cfg['stage_key']:
                raise ValueError('Activate the registered stage normally before this operation')
            reason = stopped()
            if reason:
                report['status'] = reason
                return report
            obs = public()
            report['initial_public'] = obs
            report['target_z'] = obs['eef_pos'][2] + cfg['retract_delta_z']
            for _ in range(cfg['open_hold_steps']):
                reason = stopped()
                if reason:
                    report['status'] = reason
                    return report
                obs = step('open_hold',[0.,0.,0.,0.,0.,0.,-1.],obs)
            reason = stopped()
            if reason:
                report['status'] = reason
                return report
            report['public_qpos_in_registered_bounds'] = all(lo <= v <= hi for v,(lo,hi) in zip(obs['gripper_qpos'],cfg['open_qpos_bounds']))
            if not report['public_qpos_in_registered_bounds']:
                report['status'] = 'OPEN_QPOS_UNCONFIRMED'
                return report
            c = cfg['controller_contract']
            scale_z = c['output_max'][2]/c['input_max'][2]
            for _ in range(cfg['max_retract_steps']):
                reason = stopped()
                if reason:
                    report['status'] = reason
                    return report
                if not report['public_qpos_in_registered_bounds']:
                    report['status'] = 'OPEN_QPOS_UNCONFIRMED'
                    return report
                error = report['target_z']-obs['eef_pos'][2]
                if error <= cfg['z_tolerance']:
                    break
                dz = min(error,cfg['max_translation_delta_per_step'])
                command = min(dz/scale_z,cfg['max_normalized_z_command'])
                obs = step('retract',[0.,0.,command,0.,0.,0.,-1.],obs)
            reason = stopped()
            if reason:
                report['status'] = reason
                return report
            if not report['public_qpos_in_registered_bounds']:
                report['status'] = 'OPEN_QPOS_UNCONFIRMED'
                return report
            report['public_z_target_reached'] = obs['eef_pos'][2] >= report['target_z']-cfg['z_tolerance']
            report['status'] = 'PUBLIC_TARGET_REACHED' if report['public_z_target_reached'] else 'RETRACT_LIMIT'
            return report
        except BaseException as exc:
            report.update(status='FAILED' if isinstance(exc, Exception) else 'ABORTED',error='%s: %s' % (type(exc).__name__,exc))
            raise
        finally:
            report.update(end_step=r.step_idx,physical_steps=r.step_idx-start,
                          native_success=bool(r.success),native_done=bool(r.done))
            report['costs']['total_seconds'] = self.clock()-t0
