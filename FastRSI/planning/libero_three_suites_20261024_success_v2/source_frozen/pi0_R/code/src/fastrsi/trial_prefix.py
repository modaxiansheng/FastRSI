"""Re-execute a paid development prefix without rendering unused camera frames.

This is experiment reuse, not an online robot shortcut. Physics actions, rewards,
and success checks still execute. Only images that no new decision consumes are
not rendered. The restored decision boundary must match the source exactly.
"""
import hashlib
import time


CAMERAS = ('agentview_image', 'robot0_eye_in_hand_image')


def signature(value):
    import numpy as np
    arr = np.ascontiguousarray(value)
    return dict(shape=list(arr.shape), dtype=str(arr.dtype),
                sha256=hashlib.sha256(arr.tobytes()).hexdigest())


def validate_prefix(rows, boundary):
    if type(boundary) is not int or boundary <= 0 or boundary % 10:
        raise ValueError('Positive H10 boundary required')
    if not rows or rows[0]['method'] != 'reset' or rows[0]['step'] != 0:
        raise ValueError('Source reset missing')
    prefix = [r for r in rows if r['step'] <= boundary]
    if [r['step'] for r in prefix] != list(range(0, boundary+1, 10)):
        raise ValueError('Missing or non-H10 source prefix')
    for r in prefix:
        if r.get('native_success') or r['session_id'] != rows[0]['session_id']:
            raise ValueError('Terminal or cross-session source prefix')
    for r in prefix[1:]:
        if r['method'] != 'action_chunk' or len(r['actions']) != 10:
            raise ValueError('Unexpected prefix actions')
    return prefix


class UnconsumedCameras:
    """Keep observable clocks/filter calls intact; replace only camera rendering."""
    def __init__(self, runtime):
        self.runtime = runtime
        self.saved = {}

    def __enter__(self):
        import numpy as np
        from robosuite.utils import observables
        env = self.runtime.env.env
        try:
            for name in CAMERAS:
                observable = env._observables[name]
                for field, expected in (('_corrupter', 'NO_CORRUPTION'),
                                        ('_filter', 'NO_FILTER'), ('_delayer', 'NO_DELAY')):
                    if getattr(observable, field) is not getattr(observables, expected):
                        raise ValueError('Stateful/noisy camera operator unsupported')
                zero = np.zeros_like(self.runtime._raw_obs[name])
                self.saved[name] = observable._sensor
                replacement = lambda cache, image=zero: image
                replacement.__modality__ = observable._sensor.__modality__
                observable._sensor = replacement
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, *unused):
        env = self.runtime.env.env
        for name, sensor in self.saved.items():
            env._observables[name]._sensor = sensor



def execute_prefix(runtime, rows, boundary, *, on_boundary=None):
    prefix = validate_prefix(rows, boundary)
    if runtime.step_idx != 0 or runtime.done or runtime.config.get('record_dir'):
        raise ValueError('Prefix reuse requires fresh unrecorded development episode')
    tick = time.perf_counter()
    def execute(row):
        runtime.action_chunk({'actions':row['actions']})
        if runtime.step_idx != row['step'] or runtime.done:
            raise ValueError('Prefix execution changed termination/step')
        if on_boundary is not None:on_boundary(runtime,row)
    with UnconsumedCameras(runtime):
        for row in prefix[1:-1]:execute(row)
    # The final chunk uses native sensors and their original sampling phase.
    # A manually rendered boundary can differ from the simulator's sampled image.
    execute(prefix[-1])
    return dict(physical_steps_reexecuted=boundary, boundary=boundary,
                omitted_camera_step_frames=2*(boundary-10), seconds=time.perf_counter()-tick,
                scope='No online decision observes omitted frames; physics still executed')
