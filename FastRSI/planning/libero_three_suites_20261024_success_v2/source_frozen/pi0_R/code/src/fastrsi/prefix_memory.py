"""One-shot action-prefix transfer probe, never an approved general skill.

Only the first H10 request may be replaced. Subsequent observations always go
back to the native policy. A successful source trajectory is candidate evidence,
not evidence that its actions transfer to another scene.
"""
import hashlib
import json
import math
from pathlib import Path


def load(path, expected_sha):
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError('Prefix artifact changed')
    value = json.loads(raw)
    if value.get('version') != 'prefix_memory_probe_v1' or value.get('status') != 'PROPOSED':
        raise ValueError('Only a declared development candidate is supported')
    actions = value.get('actions', [])
    if len(actions) != 10 or any(len(a) != 7 or any(
            type(v) not in (float, int) or not math.isfinite(v) for v in a) for a in actions):
        raise ValueError('Expected ten finite seven-dimensional actions')
    if not value.get('source_files') or not value.get('goal'):
        raise ValueError('Source binding required')
    return value


class PrefixProbe:
    def __init__(self, artifact, case, model_sha, enabled):
        if (case['mode'] != 'development' or case['suite'] != 'libero_10'
                or case.get('feedback_window_mode') != 'fixed10'
                or case['task_description'] != artifact['goal']
                or model_sha != artifact['model_sha256']
                or case['task_id'] != artifact['source_task_id']
                or case['init_state_id'] == artifact['source_init_state_id']):
            raise ValueError('Probe is limited to a new development initial state of the source task')
        self.artifact, self.case, self.enabled = artifact, dict(case), enabled
        self.calls = 0
        self.reused = 0

    def choose(self, prompt, current_case):
        if current_case != self.case['id']:
            raise ValueError('Prefix probe cannot cross cases')
        first = self.calls == 0
        self.calls += 1
        # A replan with a changed goal must receive ordinary policy inference.
        use = self.enabled and first and prompt == self.artifact['goal']
        if use:
            self.reused += 1
            return [list(a) for a in self.artifact['actions']]
        return None
