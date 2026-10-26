"""Bind a fixed window rule to an actually queued native recovery Session.

Shared lifecycle repair only. No new planning, verifier call, policy inference,
RNG reset, or experience update. Session suffixes alone are never authority.
"""
import hashlib
import json
from pathlib import Path

from PhyAgentOS.runtime.schemas import SessionResult, SessionSpec, SessionStatus
from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block
from .feedback_window import FeedbackWindow, MODES, commitment_binding


_DYNAMIC = {'session_id', 'parent_session_id', 'replan_attempt', 'task_description',
            'status', 'created_at', 'updated_at', 'claimed_by', 'claim_token', 'result'}


def _invariants(spec):
    value = spec.model_dump(mode='json')
    for key in _DYNAMIC:
        value.pop(key)
    value['retry']['attempted'] = 0  # Native retry accounting is per Session.
    # Same two inert policy_loop defaults accepted by validate_queued_session.
    known = (value['target_ref'] == 'target://libero_real_remote'
             and value['skillruntime_ref'] == 'skillruntime://pi05_libero_remote'
             and (value.get('benchmark') or {}).get('execution_mode') == 'policy_loop'
             and value['safety_profile']['profile'] == 'default_simulation')
    if known:
        if value['runtime_hints']['preferred_replan_every_steps'] == value['execution']['replan_every_steps']:
            value['runtime_hints']['preferred_replan_every_steps'] = None
        if value['safety_profile']['workspace_bounds'] == 'default':
            value['safety_profile']['workspace_bounds'] = None
    return value


def rebind_native_replan(workspace, case, mode, outcome, attempt, previous_session_id, root_spec,
                         *, previous_commitment=None):
    if mode not in MODES or any(case.get(key) for key in (
            'window_learning_artifact', 'stage_flow_artifact',
            'action_window_artifact')):
        raise ValueError('Native recovery window binding does not define experimental or learned controllers')
    commitment = commitment_binding(case, mode)
    if commitment != previous_commitment:
        raise ValueError('Native recovery must retain the previous bound inference horizon and commitment')
    snapshot_binding = None
    if case.get('task_window_artifact') or case.get('task_window_sha256'):
        if mode not in ('fixed5', 'fixed10'):
            raise ValueError('Task snapshot recovery requires a fixed5/fixed10 root choice')
        artifact, digest = case.get('task_window_artifact'), case.get('task_window_sha256')
        if not isinstance(artifact, str) or not artifact or not isinstance(digest, str) or not digest:
            raise ValueError('Task snapshot recovery requires both artifact and SHA')
        from .task_window import load, select
        state = load(artifact, digest)
        # Validate the already selected root decision. Never select on the
        # changed recovery goal or update the immutable task table here.
        root_selection = select(state, case['task_description'])
        if root_selection['selected_mode'] != mode:
            raise ValueError('Task snapshot root selection differs from bound window mode')
        snapshot_binding = {
            'artifact': artifact, 'snapshot_sha256': digest,
            'root_goal': case['task_description'], 'root_selection': root_selection,
            'recovery_window_policy': 'retain_root_fixed_window',
            'recovery_goal_reselection': False, 'snapshot_updated': False,
        }
    if type(attempt) is not int or not 1 <= attempt <= 2:
        raise ValueError('Native recovery attempt outside existing bound')
    if (outcome.get('ok') is not True or outcome.get('code') != 'verified'
            or outcome.get('status') != 'replanned' or outcome.get('verdict') != 'replan'
            or outcome.get('session_id') != previous_session_id):
        raise ValueError('No matching completed native replan outcome')
    child_id = outcome.get('replanned_session_id')
    if not isinstance(child_id, str) or not child_id or child_id == previous_session_id:
        raise ValueError('Missing explicit native child Session identity')
    path = Path(workspace)/'SESSIONS.md'
    source = path.read_bytes()
    document = read_yaml_block(path)
    sessions = [SessionSpec.model_validate(item) for item in document['sessions']]
    if len({item.session_id for item in sessions}) != len(sessions):
        raise ValueError('Ambiguous Session identities')
    by_id = {item.session_id:item for item in sessions}
    try:
        root, parent, child = (by_id[case['id']], by_id[previous_session_id], by_id[child_id])
    except KeyError as exc:
        raise ValueError('Native recovery Session record missing') from exc
    original = SessionSpec.model_validate(root_spec)
    expected = _invariants(original)
    if (root.session_id != original.session_id or original.session_id != case['id']
            or root.replan_attempt != 0 or root.parent_session_id != original.parent_session_id
            or root.task_description != case['task_description']
            or original.task_description != case['task_description']):
        raise ValueError('Original queued goal/Session differs from bound case')
    if any(_invariants(item) != expected for item in (root, parent, child)):
        raise ValueError('Native recovery changed target, execution, routing or benchmark fields')
    if (child.parent_session_id != parent.session_id or child.replan_attempt != attempt
            or parent.replan_attempt != attempt-1 or parent.status != SessionStatus.REPLANNED
            or parent.result.metadata.get('replanned_session_id') != child.session_id):
        raise ValueError('Native parent/child recovery lineage mismatch')
    # Walk actual parent records, rather than recognizing a naming convention.
    cursor = parent
    for _ in range(attempt-1):
        if cursor.parent_session_id not in by_id:
            raise ValueError('Recovery ancestry missing')
        ancestor = by_id[cursor.parent_session_id]
        if (ancestor.result.metadata.get('replanned_session_id') != cursor.session_id
                or ancestor.replan_attempt != cursor.replan_attempt-1):
            raise ValueError('Recovery ancestry changed')
        cursor = ancestor
    if cursor.session_id != root.session_id:
        raise ValueError('Recovery does not descend from this root case')
    verification = parent.result.metadata.get('verification', {})
    applied_id = verification.get('applied_attempt_id')
    applied = [item for item in verification.get('attempts', []) if item.get('attempt_id') == applied_id]
    if len(applied) != 1:
        raise ValueError('Missing or ambiguous applied native verdict')
    verdict = applied[0]
    details = verdict.get('details') or {}
    if (verdict.get('source') != 'auto' or verdict.get('mode') != 'apply'
            or verdict.get('error') is not None or verdict.get('verdict') != 'replan'
            or details.get('verdict') != 'replan'
            or details.get('replan_task_description') != child.task_description
            or not child.task_description.strip()):
        raise ValueError('Child goal is not the applied native replan instruction')
    if (child.status != SessionStatus.PENDING or child.claimed_by is not None
            or child.claim_token is not None or child.retry.attempted != 0
            or child.result.model_dump() != SessionResult().model_dump()):
        raise ValueError('Native recovery child is not a fresh pending Session')
    benchmark = child.benchmark
    if (benchmark is None or benchmark.suite_id != case['suite']
            or benchmark.task_index != case['task_id'] or benchmark.instance_id != case['init_state_id']
            or child.execution.max_steps != case['max_steps']):
        raise ValueError('Recovery benchmark or execution bound differs from case')
    bound_case = {**case, 'id':child.session_id, 'task_description':child.task_description,
                  'max_steps':child.execution.max_steps, 'suite':benchmark.suite_id,
                  'native_replan_attempt':attempt}
    controller = FeedbackWindow(bound_case, mode)
    if path.read_bytes() != source:
        raise ValueError('Session records changed while binding native recovery')
    if snapshot_binding is not None and hashlib.sha256(
            Path(snapshot_binding['artifact']).read_bytes()).hexdigest() != snapshot_binding['snapshot_sha256']:
        raise ValueError('Task table changed during native recovery binding')
    child_value = child.model_dump(mode='json')
    record = {'scope':'Common native recovery lifecycle repair; not FastRSI learning',
              'attempt':attempt, 'root_session_id':root.session_id,
              'parent_session_id':parent.session_id, 'session_id':child.session_id,
              'original_goal':case['task_description'], 'native_attempt_goal':child.task_description,
              'mode':mode, 'max_steps':child.execution.max_steps,
              'target_ref':child.target_ref, 'skillruntime_ref':child.skillruntime_ref,
              'session_spec':child_value,
              'session_spec_sha256':hashlib.sha256(json.dumps(child_value,sort_keys=True).encode()).hexdigest(),
              'sessions_file_sha256':hashlib.sha256(source).hexdigest(),
              'applied_verification_attempt_id':applied_id,
              'diagnostic_history_reset':True, 'policy_rng_reset':False}
    if commitment is not None:
        record.update(commitment)
        record['commitment_policy'] = 'retain_root_inference_horizon_and_commit_steps'
    if snapshot_binding is not None:
        record['task_window_snapshot_binding'] = snapshot_binding
    return controller, record
