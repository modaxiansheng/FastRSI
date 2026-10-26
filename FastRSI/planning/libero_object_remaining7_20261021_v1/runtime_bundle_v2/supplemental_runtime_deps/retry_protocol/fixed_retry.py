"""Fixed simulator-feedback retry protocol. No training, Qwen or skill selection.

The native executor is injected. This module never launches a process, changes
an initial state, or promotes an infrastructure exception to physical failure.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from typing import Optional
from .records import EvidenceError, plain

MODELS = ('pi0', 'pi05', 'openvla', 'xvla')
LIMITS = {'libero_spatial': 220, 'libero_object': 280, 'libero_goal': 300, 'libero_10': 520}
VARIANTS = ('object', 'position', 'semantic', 'task', 'environment')


def digest_valid(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


@dataclass(frozen=True)
class Identity:
    identity_id: str
    suite: str
    perturbation: str
    task_id: int
    initial_state_sha256: str
    physical_state_sha256: str
    asset_sha256: str
    source: str
    source_receipt_sha256: str
    split: str = 'development'

    def validate(self):
        if not self.identity_id or self.suite not in LIMITS or self.perturbation not in VARIANTS:
            raise EvidenceError('invalid development identity')
        if type(self.task_id) is not int or self.task_id not in range(10) or self.split != 'development':
            raise EvidenceError('invalid task/split')
        if not self.source or not all(digest_valid(s) for s in
                (self.initial_state_sha256, self.physical_state_sha256, self.asset_sha256, self.source_receipt_sha256)):
            raise EvidenceError('physical source/state evidence required')


def freeze_roster(identities, excluded_physical_states, source_authorization):
    """Metadata only; exclusion data must not contain holdout outcomes."""
    if not source_authorization:
        raise EvidenceError('development source not authorized')
    if len(identities) != 32:
        raise EvidenceError('exactly 32 real development identities required')
    for identity in identities: identity.validate()
    if len({r.identity_id for r in identities}) != 32 or len({r.physical_state_sha256 for r in identities}) != 32:
        raise EvidenceError('run IDs or model names do not create new physical identities')
    if {r.physical_state_sha256 for r in identities}.intersection(excluded_physical_states):
        raise EvidenceError('development/holdout physical-state overlap')
    if len({(r.suite,r.task_id) for r in identities}) < 3 or len({r.perturbation for r in identities}) < 2:
        raise EvidenceError('insufficient task/perturbation coverage')
    body=dict(schema_version=1, identities=[asdict(r) for r in identities], models=list(MODELS),
              source_authorization=source_authorization, max_attempts=3,
              max_native_task_attempts=384, success_feedback='simulator_assisted',
              first_attempt_shared_baseline=True, learning=False, training=False, qwen=False,
              limits=LIMITS, openvla_identical_second_trace_stops_third=True)
    raw=json.dumps(body,sort_keys=True,separators=(',',':')).encode()
    return dict(body, protocol_sha256=hashlib.sha256(raw).hexdigest())


@dataclass(frozen=True)
class Attempt:
    index: int
    physical_success: Optional[bool]
    software_status: str
    verification_status: str
    initial_state_sha256: str
    actions: int
    policy_inferences: Optional[int]
    forward_calls: Optional[int]
    seconds: Optional[float]
    qwen_calls: int
    qwen_tokens: int
    observation_sha256: Optional[str]
    action_trace_sha256: Optional[str]
    final_state_sha256: Optional[str]
    raw_receipt_sha256: str
    rng_operation: str
    # This value is native evidence, not an inferred boolean from process exit.
    episode_completed: bool
    error: Optional[str] = None

    def validate(self, identity):
        if type(self.index) is not int or self.index not in (1,2,3):
            raise EvidenceError('attempt index out of protocol')
        if self.physical_success is not None and type(self.physical_success) is not bool:
            raise EvidenceError('physical result must be bool or unknown')
        if self.software_status not in ('OK','ERROR') or self.verification_status != 'NOT_RUN':
            raise EvidenceError('software/visual verification categories mixed')
        if self.initial_state_sha256 != identity.initial_state_sha256:
            raise EvidenceError('retry changed physical initial state')
        if type(self.actions) is not int or not 0 <= self.actions <= LIMITS[identity.suite]:
            raise EvidenceError('per-attempt official action budget exceeded')
        for v in (self.policy_inferences,self.forward_calls):
            if v is not None and (type(v) is not int or v < 0):raise EvidenceError('invalid native count')
        if self.qwen_calls != 0 or self.qwen_tokens != 0:
            raise EvidenceError('Qwen is outside this fixed retry experiment')
        if self.rng_operation != ('begin' if self.index == 1 else 'handoff'):
            raise EvidenceError('retry must preserve policy RNG stream')
        if self.physical_success is False and (not self.episode_completed or self.software_status != 'OK'):
            raise EvidenceError('engineering/unfinished result is not a confirmed physical failure')
        if self.software_status == 'OK' and (not self.episode_completed or self.physical_success is None):
            raise EvidenceError('normal completion missing native outcome')
        if not digest_valid(self.raw_receipt_sha256):raise EvidenceError('raw receipt binding required')
        if self.seconds is not None and (type(self.seconds) not in (float,int) or self.seconds < 0):
            raise EvidenceError('invalid duration')
        plain(asdict(self))


def run_identity(identity, model, execute, emit, *, deterministic_decoding_verified=False):
    """execute(index, max_steps, rng_operation) must durably account native runs.

    No automatic technical restart is performed. Resume after an interrupted
    dispatch needs a separate audited receipt; never call this again blindly.
    """
    identity.validate()
    if model not in MODELS:raise EvidenceError('unregistered model')
    attempts=[];reason='ATTEMPT_LIMIT'
    for index in (1,2,3):
        op='begin' if index==1 else 'handoff'
        emit(dict(event='attempt_requested',identity_id=identity.identity_id,model=model,
                  index=index,max_steps=LIMITS[identity.suite],rng_operation=op))
        # Exceptions remain exceptions with the requested-attempt event retained.
        # The executor must preserve physical/log costs before raising.
        attempt=execute(index,LIMITS[identity.suite],op)
        if not isinstance(attempt,Attempt) or attempt.index!=index:raise EvidenceError('executor receipt mismatch')
        attempt.validate(identity);attempts.append(attempt)
        emit(dict(event='attempt_recorded',identity_id=identity.identity_id,model=model,attempt=asdict(attempt)))
        if attempt.physical_success is True:
            reason='PHYSICAL_SUCCESS';break
        if attempt.software_status!='OK' or attempt.physical_success is None:
            reason='ENGINEERING_OR_UNKNOWN_STOP';break
        if index==2 and model=='openvla' and deterministic_decoding_verified:
            first=attempts[0]
            fields=('observation_sha256','action_trace_sha256','final_state_sha256')
            if first.actions==attempt.actions and all(digest_valid(getattr(first,k)) and
                    getattr(first,k)==getattr(attempt,k) for k in fields):
                reason='IDENTICAL_DETERMINISTIC_RETRY_STOP';break
    success=any(a.physical_success is True for a in attempts)
    complete=success or reason in ('ATTEMPT_LIMIT','IDENTICAL_DETERMINISTIC_RETRY_STOP')
    return dict(identity=asdict(identity),model=model,attempts=[asdict(a) for a in attempts],
                first_success=attempts[0].physical_success,
                final_success=True if success else (False if complete else None),stop_reason=reason,
                software_ok=all(a.software_status=='OK' for a in attempts),
                verification_status='NOT_RUN',scope='fixed simulator-assisted retry, not learned recovery')


def report_model(model, identities, results):
    """One row; incomplete observations produce bounds, never fake failures."""
    if model not in MODELS:raise EvidenceError('unregistered model')
    for identity in identities:identity.validate()
    roster={r.identity_id for r in identities}
    if len(roster)!=len(identities):raise EvidenceError('duplicate roster identity')
    selected=[r for r in results if r['model']==model]
    ids=[r['identity']['identity_id'] for r in selected]
    if len(ids)!=len(set(ids)) or not set(ids)<=roster:raise EvidenceError('duplicate/unregistered result')
    frozen={r.identity_id:asdict(r) for r in identities}
    if any(r['identity']!=frozen[r['identity']['identity_id']] for r in selected):
        raise EvidenceError('result identity differs from frozen roster')
    first=sum(r['first_success'] is True for r in selected)
    first_fail=sum(r['first_success'] is False for r in selected)
    first_unknown=len(identities)-first-first_fail
    rescued=[0,0]
    for r in selected:
        if r['first_success'] is True and r['final_success'] is not True:
            raise EvidenceError('original success was not retained')
        for a in r['attempts'][1:]:
            if a['physical_success'] is True:rescued[a['index']-2]+=1;break
    final=sum(r['final_success'] is True for r in selected)
    unknown=len(identities)-sum(r['final_success'] is not None for r in selected)
    pending_failed=sum(r['first_success'] is False and r['final_success'] is None for r in selected)
    extra=[a for r in selected for a in r['attempts'][1:]]
    all_attempts=[a for r in selected for a in r['attempts']]
    def measured_sum(key, rows=extra):
        return sum(a[key] for a in rows) if all(a[key] is not None for a in rows) else None
    n=len(identities)
    rescue_incomplete=bool(first_unknown or pending_failed)
    return dict(model=model,independent_identities=n,started=len(selected),first_success=first,
                first_failure=first_fail,first_unknown=first_unknown,
                first_completion_rate=first/n if n and not first_unknown else None,
                second_new_success=rescued[0],third_new_success=rescued[1],
                final_success=final,final_unknown=unknown,
                final_completion_rate=final/n if n and not unknown else None,
                final_rate_bounds=[final/n,(final+unknown)/n] if n else None,
                failure_rescue_rate=(sum(rescued)/first_fail if first_fail and not rescue_incomplete else None),
                failure_rescue_rate_status=('INCOMPLETE' if rescue_incomplete else
                                            ('NOT_APPLICABLE' if first_fail==0 else 'MEASURED')),
                total_recorded_attempts=len(all_attempts),
                recorded_attempt_costs={k:measured_sum(k,all_attempts) for k in
                    ('actions','policy_inferences','forward_calls','seconds','qwen_calls','qwen_tokens')},
                extra_attempts=len(extra),extra_actions=measured_sum('actions'),
                extra_policy_inferences=measured_sum('policy_inferences'),extra_forward_calls=measured_sum('forward_calls'),
                extra_seconds=measured_sum('seconds'),extra_money=None,
                cost_scope='Attempt subtotal only. Initialization, probes, failures and service lifecycle must be joined separately.')
