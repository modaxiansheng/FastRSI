"""Opt-in recording of goals actually set by the existing robot controller.

This adds no controller update, simulator step, observation, or object-state
read. A completed env.step owns one set_goal record; incomplete steps cannot
produce a compilable receipt. Reset settling is outside the armed chunk.
"""
import contextlib
import hashlib
import inspect
import json
import math
from pathlib import Path
import struct
import time

from .experience_compiler import canonical_sha


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    # Copy arrays immediately: controller goal buffers can change on the next step.
    return json.loads(json.dumps(value, allow_nan=False))


def action_f32(value, size=7):
    values = list(value)
    if len(values) != size:
        raise ValueError('Unexpected controller action dimension')
    values = [struct.unpack('<f', struct.pack('<f', float(v)))[0] for v in values]
    if not all(math.isfinite(v) for v in values):
        raise ValueError('Nonfinite controller action')
    return values


def controller_config(controller):
    keys = ('input_min', 'input_max', 'output_min', 'output_max', 'use_delta',
            'use_ori', 'impedance_mode', 'position_limits', 'orientation_limits')
    config = {key: plain(getattr(controller, key)) for key in keys}
    if (config['use_delta'] is not True or config['use_ori'] is not True or
            config['impedance_mode'] != 'fixed' or config['position_limits'] is not None or
            config['orientation_limits'] is not None):
        raise ValueError('Live compilation supports the existing relative fixed OSC configuration only')
    for key in ('input_min', 'input_max', 'output_min', 'output_max'):
        if len(config[key]) != 6 or not all(math.isfinite(v) for v in config[key]):
            raise ValueError('Invalid OSC range')
    return config


class LiveControllerTrace:
    def __init__(self, runtime, root, *, runtime_source, recording_role='online_execution'):
        if recording_role not in ('online_execution', 'replay_compatibility'):
            raise ValueError('Unknown controller recording role')
        self.recording_role = recording_role
        if runtime.step_idx != 0:
            raise ValueError('Live recording must start after reset and before the first action')
        self.runtime = runtime
        self.session = runtime.session_id
        if not self.session or Path(self.session).name != self.session or self.session in ('.', '..'):
            raise ValueError('Invalid recording Session')
        self.env = runtime.env
        self.controller = runtime.env.env.robots[0].controller
        self.config = controller_config(self.controller)
        self.config_sha = canonical_sha(self.config)
        controller_file = inspect.getsourcefile(type(self.controller))
        self.sources = dict(runtime_sha256=file_sha(runtime_source),
                            controller_sha256=file_sha(controller_file))
        self.runtime_config = {k: plain(runtime.config[k]) for k in
                               ('benchmark_name', 'task_id', 'init_state_id', 'control_mode', 'max_steps')}
        self.folder = Path(root) / self.session
        self.folder.mkdir(parents=True, exist_ok=False)
        self.raw_path = self.folder / 'CONTROLLER_TARGETS.jsonl'
        self.raw_path.touch(exist_ok=False)
        self.controls = []
        self.pending = []
        self.context = None
        self.armed = False
        self.failed = False
        self.finalized = False
        self.logging_seconds = 0.0
        self.chunk_index = 0
        self.original_step = self.env.step
        self.original_goal = self.controller.set_goal
        self.step_hook = self._step
        self.goal_hook = self._goal
        self.env.step = self.step_hook
        self.controller.set_goal = self.goal_hook

    def detach(self):
        # Restore before another reset, including its settling actions.
        if self.env.step is self.step_hook:
            self.env.step = self.original_step
        if self.controller.set_goal is self.goal_hook:
            self.controller.set_goal = self.original_goal
        self.armed = False
        self.context = None

    def bind_execution(self, receipt, *, source_skill_sha256=None):
        """Bind the actual execute() branch, rather than the transport placeholder."""
        if not self.armed or self.context is not None or self.execution_bound:
            raise ValueError('Execution branch cannot be rebound')
        origin = 'native_action'
        skill_sha = None
        if receipt is not None:
            if receipt.get('skill_execution') is False:
                if receipt.get('origin') != 'native_after_validity_exit':
                    raise ValueError('Unknown non-skill execution receipt')
                origin = 'native_after_validity_exit'
            else:
                origin = 'goal_skill_action'
            skill_sha = receipt.get('candidate_sha256', source_skill_sha256)
            if source_skill_sha256 is not None and skill_sha != source_skill_sha256:
                raise ValueError('Execution receipt refers to another candidate')
            if (not isinstance(skill_sha, str) or len(skill_sha) != 64 or
                    any(c not in '0123456789abcdef' for c in skill_sha)):
                raise ValueError('Actual skill requires its canonical candidate SHA')
        elif source_skill_sha256 is not None:
            raise ValueError('Native execution cannot claim a skill')
        for row in self.pending:
            row.update(execution_origin=origin, source_skill_sha256=skill_sha)
        self.execution_bound = True

    def _goal(self, action, *args, **kwargs):
        result = self.original_goal(action, *args, **kwargs)
        if self.armed:
            start = time.perf_counter()
            try:
                if self.context is None:
                    raise ValueError('set_goal outside a bound env.step')
                arm = action_f32(action, 6)
                if arm != self.context['actual_action'][:6]:
                    raise ValueError('Robot controller received another arm action')
                c = self.controller
                self.context['goals'].append(dict(
                    controller_action=arm, target_pos=plain(c.goal_pos), target_rot=plain(c.goal_ori),
                    zero_ori_preserves_goal=all(v == 0 for v in arm[3:])))
            finally:
                self.logging_seconds += time.perf_counter() - start
        return result

    def _step(self, action, *args, **kwargs):
        if not self.armed:
            return self.original_step(action, *args, **kwargs)
        start = time.perf_counter()
        if self.context is not None:
            raise ValueError('Nested env.step cannot be recorded')
        before = self.runtime.step_idx
        context = dict(actual_action=action_f32(action), goals=[])
        self.context = context
        self.logging_seconds += time.perf_counter() - start
        try:
            result = self.original_step(action, *args, **kwargs)
            start = time.perf_counter()
            try:
                if len(context['goals']) != 1:
                    raise ValueError('Expected exactly one set_goal per executed action')
                goal = context['goals'][0]
                self.pending.append(dict(session_id=self.session, step_before=before, step_after=before+1,
                    chunk_index=self.chunk_index, chunk_id=self.chunk_id, action_index=len(self.pending),
                    actual_action=context['actual_action'], gripper_command=context['actual_action'][-1],
                    controller_config_sha256=self.config_sha, **goal))
            finally:
                self.logging_seconds += time.perf_counter() - start
            return result
        finally:
            self.context = None

    @contextlib.contextmanager
    def chunk(self, chunk):
        if (self.failed or self.finalized or self.armed or self.runtime.session_id != self.session or
                self.runtime.step_idx != len(self.controls)):
            raise ValueError('Invalid live recording state')
        if controller_config(self.controller) != self.config:
            raise ValueError('Controller configuration changed during recording')
        if {key: plain(self.runtime.config[key]) for key in self.runtime_config} != self.runtime_config:
            raise ValueError('Runtime configuration changed during recording')
        self.chunk_id = str(chunk.get('chunk_id', 'libero_chunk'))
        self.pending = []
        self.execution_bound = False
        self.armed = True
        try:
            yield
            expected = self.runtime.step_idx - len(self.controls)
            if expected <= 0 or expected != len(self.pending) or not self.execution_bound:
                raise ValueError('Completed action count differs from controller records')
            for index, row in enumerate(self.pending, len(self.controls)):
                if row['step_before'] != index or row['step_after'] != index+1:
                    raise ValueError('Controller action indices are not contiguous')
            start = time.perf_counter()
            try:
                with self.raw_path.open('a') as handle:
                    for row in self.pending:
                        handle.write(json.dumps(row, allow_nan=False) + '\n')
                self.controls.extend(self.pending)
                self.chunk_index += 1
            finally:
                self.logging_seconds += time.perf_counter() - start
        except BaseException:
            self.failed = True
            raise
        finally:
            self.armed = False
            self.context = None
            self.pending = []

    def boundary(self, trace_path):
        """Finalize only after the ordinary TRACE recorded the terminal action."""
        if not self.runtime.done:
            return None
        if self.failed or self.finalized or self.armed or self.runtime.step_idx != len(self.controls):
            raise ValueError('Cannot finalize an incomplete controller trace')
        start = time.perf_counter()
        try:
            trace_raw = Path(trace_path).read_bytes()
            trace = [json.loads(line) for line in trace_raw.splitlines()]
            if (not trace or trace[-1]['session_id'] != self.session or
                    trace[-1]['step'] != len(self.controls)):
                raise ValueError('Controller trace and terminal TRACE differ')
            targets = [{k: row[k] for k in ('target_pos', 'target_rot', 'gripper_command')}
                       for row in self.controls]
            raw_controls = self.raw_path.read_bytes()
            if [json.loads(line) for line in raw_controls.splitlines()] != self.controls:
                raise ValueError('Persisted controller records differ')
            receipt = dict(version=1, kind='live_controller_trace', status='LIVE_RECORDED',
                provenance_mode='live_controller_v1', recording_role=self.recording_role, session_id=self.session,
                runtime_config=self.runtime_config, controller_config=self.config,
                controller_config_sha256=self.config_sha, source_code=self.sources,
                controls=self.controls, controls_count=len(self.controls),
                controls_sha256=canonical_sha(self.controls), targets_sha256=canonical_sha(targets),
                actual_actions_sha256=canonical_sha([row['actual_action'] for row in self.controls]),
                controller_jsonl_sha256=hashlib.sha256(raw_controls).hexdigest(),
                source_trace_sha256=hashlib.sha256(trace_raw).hexdigest(),
                source_trace_rows_sha256=canonical_sha(trace),
                terminal_native_success=bool(self.runtime.success), terminal_done=bool(self.runtime.done),
                logging_seconds_before_final_write=self.logging_seconds + time.perf_counter()-start,
                scope='Only actual robot commands and controller goals; no object state. Not an exact replay.')
            path = self.folder / 'LIVE_CONTROLS.json'
            with path.open('x') as handle:
                json.dump(receipt, handle, allow_nan=False)
            self.finalized = True
            return path
        except BaseException:
            self.failed = True
            raise
        finally:
            self.logging_seconds += time.perf_counter() - start
