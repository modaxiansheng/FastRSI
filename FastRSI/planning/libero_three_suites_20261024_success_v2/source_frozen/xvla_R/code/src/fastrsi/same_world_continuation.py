"""Opt-in target-side stage continuation for a separate development task stream.

The caller binds same-world task definitions before execution. This is not an
official LIBERO episode evaluator, a learned selector, or an Agent replacement.
It owns no policy: the existing native target still executes every action.
"""
import copy
import time


SCOPE = 'same_world_development_stream_not_official_libero_episode'


class SameWorldContinuation:
    def __init__(self, runtime, stages, *, world_signature, total_steps,
                 timeout_seconds, clock=time.monotonic):
        if runtime.env is None or runtime._last_obs is None or runtime.step_idx != 0:
            raise ValueError('Bind an initialized, unadvanced environment')
        if type(total_steps) is not int or total_steps <= 0:
            raise ValueError('Positive shared physical budget required')
        if not isinstance(timeout_seconds, (int, float)) or not 0 < timeout_seconds < float('inf'):
            raise ValueError('Finite shared deadline required')
        if not isinstance(world_signature, str) or len(world_signature) != 64 or not stages:
            raise ValueError('Bind same-world source definitions')
        for key, stage in stages.items():
            if (not isinstance(key,str) or not key or stage['world_signature'] != world_signature
                    or not stage['goal_state'] or not stage['language'].strip()
                    or type(stage['max_steps']) is not int or stage['max_steps'] <= 0):
                raise ValueError('Invalid or incompatible stage definition')
        self.runtime = runtime
        self.stages = copy.deepcopy(stages)
        self.total_steps = total_steps
        self.clock = clock
        self.deadline = clock() + timeout_seconds
        self.environment = runtime.env
        self.anchor_key = runtime._env_key
        self.active = None
        self.start_step = 0
        self.history = []
        self.activation_count = 0
        self.expected_step = runtime.step_idx
        self.expected_observation = runtime._last_obs

    def _check_identity(self):
        if self.runtime.env is not self.environment or self.runtime._env_key != self.anchor_key:
            raise RuntimeError('Environment was replaced outside continuation owner')
        if (self.runtime.step_idx != self.expected_step
                or self.runtime._last_obs is not self.expected_observation):
            raise RuntimeError('External reset or action changed the owned stream')

    def _check(self):
        self._check_identity()
        if self.clock() >= self.deadline:
            raise TimeoutError('Shared stream deadline exhausted')
        if self.runtime.step_idx >= self.total_steps:
            raise RuntimeError('Shared stream action budget exhausted')

    def activate(self, key, *, abandon_incomplete=False):
        """Rebind evaluator/instruction only; never reset physics or counters.

        Goal predicates remain target-side evaluation data, not selector input.
        An incomplete previous stage must be explicitly recorded as abandoned.
        """
        self._check()
        if key not in self.stages:
            raise ValueError('Unregistered stage')
        if self.active is not None and not self.runtime.success and not abandon_incomplete:
            raise ValueError('Previous stage incomplete; explicit abandonment required')
        r = self.runtime
        if self.active is not None:
            self.history.append(dict(stage=self.active, start_step=self.start_step,
                end_step=r.step_idx, success=bool(r.success), abandoned=not bool(r.success)))
        stage = self.stages[key]
        r.env.env.parsed_problem['goal_state'] = copy.deepcopy(stage['goal_state'])
        r.language = stage['language']
        self.active = key
        self.start_step = r.step_idx
        self.activation_count += 1
        r.config['max_steps'] = min(self.total_steps, self.start_step + stage['max_steps'])
        # Never carry a previous goal's sticky success into the new stage.
        r.success = bool(r.env.check_success())
        r.done = r.success
        r._last_obs = dict(r._last_obs, task_description=r.language,
                           stream_stage=key, evaluation_scope=SCOPE)
        self.expected_observation = r._last_obs
        r._last_video_path = None
        r._last_status = dict(accepted=True, executed_steps=r.step_idx,
                             success=r.success, done=r.done, obs=r._last_obs,
                             stream_stage=key, evaluation_scope=SCOPE)
        return self.observe()

    def observe(self):
        # Final evidence remains readable after execution budget/deadline expiry.
        self._check_identity()
        return self.runtime.observe()

    def action_chunk(self, chunk):
        self._check()
        if self.active is None or self.runtime.done:
            raise RuntimeError('Activate a nonterminal stage before sending actions')
        status = self.runtime.action_chunk(chunk)
        self.expected_step = self.runtime.step_idx
        self.expected_observation = self.runtime._last_obs
        self.runtime._last_obs.update(stream_stage=self.active, evaluation_scope=SCOPE)
        status.update(stream_stage=self.active, evaluation_scope=SCOPE)
        return status

    def snapshot(self):
        r = self.runtime
        return dict(scope=SCOPE, active_stage=self.active, activation_count=self.activation_count,
                    physical_steps=r.step_idx, total_steps=self.total_steps,
                    remaining_steps=max(0,self.total_steps-r.step_idx),
                    active_stage_success=bool(r.success), stage_terminal=bool(r.done),
                    history=copy.deepcopy(self.history))
