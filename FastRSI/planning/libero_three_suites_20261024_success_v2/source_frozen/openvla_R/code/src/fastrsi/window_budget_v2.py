"""Opt-in continuation rule over cumulative verified window feedback.

A bounded capability confirmation is distinct from cost-payback validation.
This is a transparent rule baseline, not a learned value-of-information model.
Legacy validation_budget and deployed defaults are unchanged.
"""
import math
from .validation_budget import digest
VERSION='window_budget_v2'

def decide(ledger, request):
    if request.get('phase')!='development':raise ValueError('Development-only allocation')
    if request.get('ledger_sha256')!=digest(ledger):raise ValueError('Stale ledger binding')
    if request.get('candidate_sha256')!=ledger.get('candidate_sha256'):raise ValueError('Candidate mismatch')
    records=ledger['records']
    uses=request['remaining_uses'];followup_cap=request['max_quality_followups']
    if type(uses) is not int or uses<0 or type(followup_cap) is not int or followup_cap<0:
        raise ValueError('Finite nonnegative integer limits required')
    if type(request['allow_quality_confirmation']) is not bool:raise ValueError('Explicit confirmation authority required')
    for key in ('remaining_budget_seconds','estimated_pair_seconds','maximum_pair_seconds'):
        v=request[key]
        if type(v) not in (float,int) or not math.isfinite(v) or v<0 or (key!='remaining_budget_seconds' and v==0):
            raise ValueError('Invalid cost bound')
    if request['estimated_pair_seconds']>request['maximum_pair_seconds']:raise ValueError('Estimate exceeds cap')
    ids=[r['init_id'] for r in records]
    if request['next_init_id'] in ids:raise ValueError('Repeated initial state')
    wins=[i for i,r in enumerate(records) if r['native_quality_relation']=='win']
    harms=[i for i,r in enumerate(records) if r['native_quality_relation']=='harm']
    # Already paid independent observations after the first win count as followups.
    # Later reruns keep the first quality win and never count as independent
    # confirmations. Their physical outcomes and fees remain in the ledger.
    seen=set();independent=[]
    for r in records:
        fresh=r['init_id'] not in seen;seen.add(r['init_id'])
        if 'independent_observation' in r and r['independent_observation'] is not fresh:
            raise ValueError('Invalid independent-instance marker')
        if not fresh and ledger.get('repeat_policy')!='retain':raise ValueError('Unbound repeated observation')
        independent.append(fresh)
    paid_followups=sum(independent[wins[0]+1:]) if wins else 0
    eligible=[r for r,fresh in zip(records,independent) if fresh and r['eligible_cost']]
    gains=[r['native_task_seconds']-r['candidate_task_seconds'] for r in eligible]
    mean=sum(gains)/len(gains) if gains else None
    net=uses*mean-request['estimated_pair_seconds'] if mean is not None else None
    action='stop';purpose=None
    if not records:reason='no_evidence_bootstrap_requires_separate_research_plan'
    elif any(r['measurement_integrity']!='valid' for r in records):
        action='repair_evidence';reason='measurement_integrity_unresolved'
    elif wins and harms:
        action='hold_task_wide_update';reason='observed_quality_heterogeneity_requires_new_control_signal'
    elif harms:reason='observed_candidate_quality_harm'
    elif wins:
        purpose='capability_confirmation_not_cost_payback'
        if paid_followups>=followup_cap:reason='paid_quality_followup_limit_reached'
        elif not request['allow_quality_confirmation']:reason='quality_confirmation_not_enabled'
        elif uses==0:reason='no_remaining_opportunity'
        elif request['maximum_pair_seconds']>request['remaining_budget_seconds']:reason='hard_budget_insufficient'
        else:action='validate_next';reason='bounded_confirmation_of_observed_quality_gain'
    elif not gains:reason='no_comparable_both_success_cost_evidence'
    elif mean<=0:reason='no_observed_cost_benefit'
    elif uses==0:reason='no_remaining_opportunity'
    elif request['maximum_pair_seconds']>request['remaining_budget_seconds']:reason='hard_budget_insufficient'
    elif net<=0:reason='projected_cost_gain_below_validation_fee'
    else:action='validate_next';reason='positive_projected_cost_value';purpose='cost_validation'
    result={'version':VERSION,'action':action,'reason':reason,'purpose':purpose,
        'ledger_sha256':digest(ledger),'request_sha256':digest(request),
        'evidence_count':len(records),'quality_wins':len(wins),'quality_harms':len(harms),
        'already_paid_quality_followups':paid_followups,'comparable_cost_pairs':len(gains),
        'observed_mean_cost_saving_seconds':mean,'projected_net_seconds':net if purpose=='cost_validation' else None,
        'promotion_authority':False,'learning_gain_established':False,
        'limitations':'Empirical rule; no confidence bound, noninferiority result, information-value learning or guaranteed return.'}
    if ledger.get('repeat_policy')=='retain':result.update(independent_evidence_count=sum(independent),repeated_execution_count=len(records)-sum(independent))
    return result

def dispatch(ledger, request, decision, executor):
    if decision!=decide(ledger,request):raise ValueError('Stale decision')
    if decision['action']!='validate_next':
        return {'status':'NOT_DISPATCHED','executor_calls':0,'decision':decision,'measured_saving_seconds':None}
    receipt=executor(dict(request))
    return {'status':'DISPATCHED','executor_calls':1,'decision':decision,'execution_receipt':receipt,
            'measured_saving_seconds':None,'promotion_authority':False}
