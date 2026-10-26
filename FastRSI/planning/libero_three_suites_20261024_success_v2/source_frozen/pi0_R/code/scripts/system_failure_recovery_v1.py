"""Bounded execution recovery from actual unsuccessful canonical feedback.

The visual model verdict is preserved. The controller may request a fresh
native child attempt under the original retry budget; it cannot change success.
"""
import hashlib
import json
from pathlib import Path


def apply_with_bounded_retry(verifier, session, verdict, source, *, native_apply):
    if source != 'auto' or session.replan_attempt >= verifier.max_replans or verdict.verdict == 'replan':
        return native_apply(verifier, session, verdict, source)
    artifact = session.result.artifact_dir
    if not artifact:
        return native_apply(verifier, session, verdict, source)
    root = Path(verifier.workspace).resolve()
    path = (root / artifact / 'episode.json').resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Bounded recovery requires a canonical episode')
    raw = path.read_bytes()
    episode = json.loads(raw)
    if episode['session_id'] != session.session_id:
        raise ValueError('Recovery episode identity mismatch')
    success = episode['success']
    if type(success) is not bool or success != episode['benchmark']['success']:
        raise ValueError('Recovery outcome is not an unambiguous canonical boolean')
    if success:
        return native_apply(verifier, session, verdict, source)
    receipt = {'version': 'system_failure_recovery_v1', 'session_id': session.session_id,
               'original_visual_verdict': verdict.model_dump(mode='json'),
               'canonical_environment_success': False,
               'episode_sha256': hashlib.sha256(raw).hexdigest(),
               'episode': str(path.relative_to(root)),
               'decision': 'request_fresh_native_attempt',
               'attempt_before': session.replan_attempt, 'maximum_replans': verifier.max_replans,
               'unchanged_task_description': session.task_description,
               'scope': 'Recovery action only; original semantic verdict and native physical failure preserved.'}
    output = root / 'reports/evidence' / ('retry-policy-' + session.session_id + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError('Refusing to replace a recovery decision')
    output.write_text(json.dumps(receipt, ensure_ascii=False, indent=2))
    session.result.metadata['system_failure_recovery'] = receipt
    effective = verdict.model_copy(update={'verdict': 'replan',
                                           'replan_task_description': session.task_description})
    status, child = native_apply(verifier, session, effective, source)
    assert status.value == 'replanned' and child
    return status, child


def install():
    from PhyAgentOS.agent.session_verifier import SessionVerifier
    original = SessionVerifier._apply_verdict

    def apply(self, session, verdict, source):
        return apply_with_bounded_retry(self, session, verdict, source, native_apply=original)

    SessionVerifier._apply_verdict = apply
