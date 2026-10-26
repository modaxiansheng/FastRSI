"""One development-only decision: spend on another matched candidate pair or stop.

This is a transparent finite-horizon heuristic, not a calibrated confidence bound,
not candidate discovery, and not authority to promote an execution policy.
"""
import copy
import hashlib
import json
import math

VERSION = 'development_validation_budget_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def number(value, name, positive=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (positive and value == 0):
        raise ValueError('Invalid '+name)
    return value


def initial(candidate_sha256, scope_sha256):
    for value in [candidate_sha256, scope_sha256]:
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):raise ValueError('Missing immutable identity')
    return {'version': VERSION, 'candidate_sha256': candidate_sha256, 'scope_sha256': scope_sha256,
            'revision': 0, 'observations': [], 'frozen': False}


def observe(state, pair):
    """Append all paid development evidence, including failures, without overwriting."""
    if state['version'] != VERSION or state['frozen']:raise ValueError('State is not writable development state')
    if pair['mode'] != 'development':raise ValueError('Evaluation outcomes cannot update validation policy')
    for key in ['candidate_sha256', 'scope_sha256']:
        if pair[key] != state[key]:raise ValueError('Changed candidate or experiment scope')
    if pair['pair_id'] in {p['pair_id'] for p in state['observations']}:raise ValueError('Duplicate feedback')
    if pair['init_id'] in {p['init_id'] for p in state['observations']}:raise ValueError('Repeated init is not an independent development pair')
    if not pair.get('sources') or any(len(h)!=64 for h in pair['sources'].values()):raise ValueError('Unbound source evidence')
    for key in ['native_task_seconds', 'candidate_task_seconds', 'project_source_run_seconds']:
        number(pair[key], key, positive=True)
    if type(pair['quality_resolved']) is not bool:raise ValueError('Missing explicit quality status')
    out=copy.deepcopy(state);out['observations'].append(copy.deepcopy(pair));out['revision']+=1
    return out


def decide(state, request):
    if state['version'] != VERSION or state['frozen']:raise ValueError('Validation requires writable development state')
    if request.get('operation') != 'matched_development_pair':raise ValueError('Unsupported validation operation')
    if request['mode'] != 'development':raise ValueError('Additional candidate validation is development-only')
    for key in ['candidate_sha256', 'scope_sha256']:
        if state[key] != request[key]:raise ValueError('Job identity mismatch')
    horizon=request['remaining_uses']
    if type(horizon) is not int or horizon<0:raise ValueError('Invalid finite horizon')
    remaining=number(request['remaining_budget_seconds'],'remaining budget')
    estimate=number(request['estimated_next_pair_seconds'],'next pair estimate',True)
    cap=number(request['maximum_next_pair_seconds'],'next pair cap',True)
    if estimate>cap:raise ValueError('Estimate exceeds declared execution cap')
    pairs=state['observations'];mean_gain=None;net=None
    if not pairs:reason='no_development_evidence'
    elif any(not p['quality_resolved'] for p in pairs):reason='unresolved_quality_evidence'
    else:
        mean_gain=sum(p['native_task_seconds']-p['candidate_task_seconds'] for p in pairs)/len(pairs)
        net=horizon*mean_gain-estimate
        if horizon==0:reason='no_remaining_opportunity'
        elif cap>remaining:reason='hard_budget_insufficient'
        elif mean_gain<=0:reason='no_observed_incremental_benefit'
        elif net<=0:reason='projected_benefit_below_next_validation_cost'
        else:reason='positive_projected_value_within_budget'
    action='validate_next_development_pair' if reason=='positive_projected_value_within_budget' else 'stop_candidate'
    return {'version':VERSION,'action':action,'reason':reason,'state_sha256':digest(state),
            'request_sha256':digest(request),'evidence_pairs':len(pairs),'observed_mean_task_saving_seconds':mean_gain,
            'projected_remaining_net_seconds':net,
            'project_source_run_seconds':sum(p['project_source_run_seconds'] for p in pairs),
            'uncertainty':'Point-estimate planning heuristic; no confidence bound or guaranteed return. No autonomous bootstrap from zero evidence.',
            'sunk_cost_rule':'Past source-run costs remain in the project ledger; mandatory method costs need separate attribution and are not subtracted again from continuation value.',
            'promotion_authority':False}


def dispatch(state, request, decision, executor):
    """Guard the real experiment launch boundary; stop never invokes the executor.

    An eligible executor must enforce the supplied wall cap, preserve all outcomes,
    and return a receipt. This function itself does not supervise remote processes.
    """
    expected=decide(state, request)
    if decision != expected:raise ValueError('Stale or modified decision')
    if decision['action']=='stop_candidate':
        return {'status':'NOT_DISPATCHED','executor_calls':0,'decision':decision,'measured_cost_saving':None}
    if request['init_id'] in {p['init_id'] for p in state['observations']}:raise ValueError('Next pair repeats acquired evidence')
    result=executor(copy.deepcopy(request))
    return {'status':'DISPATCHED','executor_calls':1,'decision':decision,'execution_receipt':result,
            'promotion_authority':False}
