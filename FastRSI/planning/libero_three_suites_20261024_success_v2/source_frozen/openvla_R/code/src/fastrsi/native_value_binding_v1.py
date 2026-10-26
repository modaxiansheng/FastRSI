"""Read fixed external value experience at each native Session policy request.

This extends the original pi adapter, not AgentLoop or the robot controller.
The observation ID supplies actual submitted progress; no simulated object truth
or future task outcome enters the model. Model selection is an owned, versioned
development configuration rather than an LLM-generated success label.
"""
import hashlib
import json
import re
from pathlib import Path

KEY = '_fastrsi_value_v1'
GOAL = 'stack the right bowl on the left bowl and place them in the tray'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(workspace):
    root = Path(workspace).resolve()
    path = root / 'policies/external_value_v1/CURRENT.json'
    value = json.loads(path.read_text())
    assert value['version'] == 'native_external_value_v1'
    assert value['goal'] == GOAL and value['max_steps'] == 400
    assert value['extra_candidate_cap'] == 20
    model = (root / value['model_relative']).resolve()
    assert model.is_relative_to(root) and sha(model) == value['model_sha256']
    assert type(value['enabled']) is bool and type(value['noise_base']) is int
    assert value['qualification'] == 'DEVELOPMENT_CANDIDATE_NOT_PRODUCTION'
    return value, sha(path)


def request(value, manifest_sha, context):
    sid = context['session_id']
    assert isinstance(sid, str) and sid
    match = re.fullmatch(r'libero_obs_(\d+)', context.get('source_observation_id', ''))
    assert match, 'Require actual observation progress, not infer count'
    step = int(match.group(1))
    assert 0 <= step < value['max_steps']
    return dict(session_id=sid, step=step, remaining_budget=value['max_steps']-step,
                noise_base=value['noise_base'], model_sha256=value['model_sha256'],
                manifest_sha256=manifest_sha,
                enabled=value['enabled'] and context['task_description'] == GOAL)


def install(adapter_class, workspace, emit):
    value, manifest_sha = load(workspace)
    original = adapter_class.to_policy_input
    reads = set()

    def adapted(self, observation, context):
        payload = original(self, observation, context)
        # A replan with a different instruction continues through unchanged pi,
        # with deterministic noise shared by both comparison conditions.
        tag = request(value, manifest_sha, context)
        payload[KEY] = tag
        if tag['session_id'] not in reads:
            reread, current_sha = load(workspace)
            assert reread == value and current_sha == manifest_sha
            emit(dict(event='native_value_experience_read', **tag,
                      reader='original_openpi_adapter', model_relative=value['model_relative'],
                      source_feedback=value['source_feedback']))
            reads.add(tag['session_id'])
        return payload

    adapter_class.to_policy_input = adapted

    def restore():
        adapter_class.to_policy_input = original
    return restore
