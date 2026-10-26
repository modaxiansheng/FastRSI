"""Opt-in native component drafts and a real, narrow execution consumer.

This restores development authority, not acceptance of an effective capability.
Only pre-registered runtime implementations are usable. Evaluation can select
fixed components but never acquire write authority from this flag.
"""
import hashlib
from pathlib import Path

VERSION = 'registered_v1'
PREFIX_KEY = 'preferred_replan_after_steps'


def requested(case):
    value = case.get('native_component_updates')
    if value not in (None, VERSION):
        raise ValueError('Unknown native component update mode')
    return value == VERSION


def validate_case(case):
    physical = case.get('registered_goal_component_physical_v1')
    if physical is not None and physical is not True:
        raise ValueError('Physical goal component mode must be explicitly True')
    if physical and (not requested(case) or case.get('registered_goal_component') is not True):
        raise ValueError('Physical goal program requires registered component authority')
    if not requested(case):
        return False
    if case.get('native_registry_lifecycle') != 'incremental_v1':
        raise ValueError('Registered component updates require incremental registry lifecycle')
    if case.get('mode') not in ('development', 'evaluation'):
        raise ValueError('Registered component phase must be development or evaluation')
    conflicts = ('same_world_registration', 'same_world_arm', 'stage_flow_artifact',
                 'action_window_artifact', 'feedback_window_mode',
                 'prerequisite_component_probe', 'placement_recovery')
    if any(case.get(key) for key in conflicts):
        raise ValueError('Registered component consumer cannot compose with another execution controller')
    return True


def component_relative_path(workspace, value):
    """Return a component path only for ordinary files inside this workspace."""
    root = Path(workspace).resolve()
    path = Path(value)
    if not path.is_absolute():
        path = root/path
    try:
        rel = path.relative_to(root)
    except ValueError:
        return None
    if '..' in rel.parts:
        return None
    component = rel.as_posix() in ('TARGETS.md', 'SKILLRUNTIME.md') or (
        len(rel.parts) > 1 and rel.parts[0] == 'configs')
    if not component:
        return None
    for parent in (path, *path.parents):
        if parent == root:
            break
        if parent.is_symlink():
            return None
    if not path.resolve().is_relative_to(root):
        return None
    return rel


def registry_hashes(workspace):
    """Hash the exact native registries/configs after real structural validation."""
    from .native_registry_lifecycle_v1 import capture_inherited
    return capture_inherited(workspace, enabled=True)['sha256']


def prefix_steps(action_contract):
    chunk = action_contract.get('chunk', {})
    if not isinstance(chunk, dict):
        raise ValueError('Native action chunk contract must be a mapping')
    if PREFIX_KEY not in chunk:
        return None  # Legacy contract remains exactly unchanged.
    steps = chunk[PREFIX_KEY]
    if type(steps) is not int or steps not in (5, 10):
        raise ValueError('Registered execution prefix must be integer 5 or 10')
    return steps


def install_prefix_consumer(workspace, emit):
    """Consume a current skill field carried by the unchanged native runtime.

    OpenPISkillRuntime already copies output_contract.action to session context.
    The original pi05 adapter still produces and validates the action values;
    this explicit host extension submits its prefix. No extra policy inference,
    image, state or hidden task predicate is introduced.
    """
    from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
    from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
    from PhyAgentOS.runtime.schemas import SkillRuntimeDocument
    root = Path(workspace).resolve()
    original = OpenPIPi05Adapter.from_policy_output
    bound = {}

    def consume(adapter, payload, context):
        ref = context.get('skillruntime_ref', '')
        if not isinstance(ref, str) or not ref.startswith('skillruntime://'):
            raise ValueError('Registered prefix lacks native SkillRuntime identity')
        sid = context.get('session_id')
        if sid not in bound:
            digest = registry_hashes(root)
            document = SkillRuntimeDocument.model_validate(read_yaml_block(root/'SKILLRUNTIME.md'))
            matches = [skill for skill in document.skillruntimes if 'skillruntime://'+skill.id == ref]
            if len(matches) != 1:
                raise ValueError('Runtime-selected SkillRuntime not found uniquely')
            skill = matches[0]
            if (skill.runtime != 'OpenPISkillRuntime' or skill.policy is None or
                    skill.policy.policy_adapter != 'policy_adapter://openpi_pi05_adapter'):
                raise ValueError('Prefix consumer requires unchanged native pi05 runtime/adapter')
            contract = skill.output_contract.get('action', {})
            value = prefix_steps(contract)
            if context.get('policy_action_contract') != contract:
                raise ValueError('Runtime action contract differs from selected registry')
            bound[sid] = (ref, value, contract)
            emit(event='registered_component_consumed', session_id=sid,
                 skillruntime_ref=ref, field='output_contract.action.chunk.'+PREFIX_KEY,
                 value=value, registry_sha256=digest,
                 scope='Actual native runtime contract read; not capability approval')
        expected_ref, steps, contract = bound[sid]
        if ref != expected_ref or context.get('policy_action_contract') != contract:
            raise ValueError('Component contract changed during native execution')
        output = original(adapter, payload, context)
        if steps is None:
            return output
        actions = output['actions']
        count = min(steps, len(actions))
        if count < 1:
            raise ValueError('Registered prefix cannot submit an empty action sequence')
        updated = {**output, 'actions': actions[:count].copy(),
                   'timing': {**output['timing'], 'horizon_steps': count},
                   'action_contract': {**output['action_contract'], 'shape': [count, 7]}}
        emit(event='registered_component_action', session_id=sid, skillruntime_ref=ref,
             source_observation_id=context.get('source_observation_id'),
             predicted_actions=len(actions), submitted_actions=count,
             field_value=steps, action_values_changed=False, extra_policy_calls=0)
        return updated

    OpenPIPi05Adapter.from_policy_output = consume

    restored = False
    def restore():
        nonlocal restored
        if restored:
            return
        if OpenPIPi05Adapter.from_policy_output is not consume:
            raise RuntimeError('Registered prefix consumer ownership changed')
        OpenPIPi05Adapter.from_policy_output = original
        restored = True
    return restore
