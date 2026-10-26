"""Compare fixed, frozen and feedback budgets without changing the OS executor.

The feedback policy prices ONE more measurement using a declared Gaussian
planning model. This is not a calibrated success model or promotion authority.
Noise/prior parameters must be fixed before a prospective study; a sensitivity
analysis on already-seen logs is development work, not an efficacy experiment.
"""
import copy
import math

from .validation_budget import digest, number

VERSION = 'validation_policy_v2'
UNIT = 'two_gpu_reservation_seconds'
MODES = ('fixed', 'frozen', 'feedback')


def create(candidate_sha256, scope_sha256, config, mode='feedback'):
    if mode not in MODES:
        raise ValueError('Unknown policy')
    for value in (candidate_sha256, scope_sha256):
        if len(value) != 64 or any(c not in '0123456789abcdef' for c in value):
            raise ValueError('Unbound identity')
    if config['cost_unit'] != UNIT:
        raise ValueError('Mixed cost units')
    for key in ('prior_sd', 'observation_sd', 'total_budget', 'next_pair_estimate', 'next_pair_cap'):
        number(config[key], key, positive=True)
    if isinstance(config['prior_mean'], bool) or not math.isfinite(config['prior_mean']):
        raise ValueError('Nonfinite prior')
    if type(config['remaining_uses']) is not int or config['remaining_uses'] < 0:
        raise ValueError('Invalid remaining uses')
    for key in ('fixed_pairs', 'max_pairs', 'minimum_quality_pairs'):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError('Invalid '+key)
    if not (config['minimum_quality_pairs'] <= config['fixed_pairs'] <= config['max_pairs']):
        raise ValueError('Inconsistent sample allocations')
    if config['next_pair_estimate'] > config['next_pair_cap']:
        raise ValueError('Estimate exceeds cap')
    return {'version': VERSION, 'candidate_sha256': candidate_sha256,
            'scope_sha256': scope_sha256, 'mode': mode, 'config': copy.deepcopy(config),
            'records': [], 'frozen_record_ids': None}


def record(state, row):
    """Always account paid development work, including invalid measurements."""
    if state['version'] != VERSION or row['mode'] != 'development':
        raise ValueError('Only development evidence may enter policy state')
    for key in ('candidate_sha256', 'scope_sha256'):
        if row[key] != state[key]:
            raise ValueError('Changed candidate/protocol')
    if row['cost_unit'] != state['config']['cost_unit']:
        raise ValueError('Mixed cost units')
    if any(row['pair_id'] == r['pair_id'] or row['instance_id'] == r['instance_id'] for r in state['records']):
        raise ValueError('Duplicate paid evidence')
    if row['measurement'] not in ('valid', 'invalid', 'unknown'):
        raise ValueError('Missing measurement disposition')
    if row['quality'] not in ('pass', 'candidate_failure', 'unresolved'):
        raise ValueError('Missing quality disposition')
    number(row['paid_cost'], 'paid cost')
    if row['measurement'] == 'valid':
        for key in ('baseline_cost', 'candidate_cost'):
            number(row[key], key, positive=True)
    if not row['sources'] or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h)
                                 for h in row['sources'].values()):
        raise ValueError('Missing source hashes')
    result = copy.deepcopy(state)
    result['records'].append(copy.deepcopy(row))
    return result


def freeze(state):
    """Freeze selection evidence, not accounting or common quality checks."""
    if state['mode'] != 'frozen' or state['frozen_record_ids'] is not None:
        raise ValueError('Freeze the initial selection snapshot exactly once')
    out = copy.deepcopy(state)
    out['frozen_record_ids'] = [r['pair_id'] for r in out['records']]
    return out


def posterior(config, rows):
    precision = 1 / config['prior_sd'] ** 2
    weighted = config['prior_mean'] * precision
    for row in rows:
        if row['measurement'] == 'valid' and row['quality'] == 'pass':
            precision += 1 / config['observation_sd'] ** 2
            weighted += (row['baseline_cost'] - row['candidate_cost']) / config['observation_sd'] ** 2
    return weighted / precision, 1 / precision


def information_value(mean, variance, noise_variance, uses):
    """Expected increase in max(0, posterior mean) after one noisy observation.

Posterior-mean variance before seeing y is v^2/(v+noise_variance).
E[max(0, X)] for X~N(m,s^2) is s*phi(m/s)+m*Phi(m/s).
This values information about cost ONLY, assuming quality eligibility. It does
not price all candidate savings as if the next validation produced them.
"""
    sd = math.sqrt(variance * variance / (variance + noise_variance))
    z = mean / sd
    phi = math.exp(-z*z/2) / math.sqrt(2*math.pi)
    cdf = .5 * math.erfc(-z/math.sqrt(2))
    return max(0., uses * (sd*phi + mean*cdf - max(0., mean)))


def decide(state, request):
    if state['version'] != VERSION or request['mode'] != 'development':
        raise ValueError('Evaluation cannot allocate development experiments')
    if request['operation'] != 'matched_development_pair':
        raise ValueError('Unknown operation')
    for key in ('candidate_sha256', 'scope_sha256'):
        if request[key] != state[key]:
            raise ValueError('Changed job identity')
    if any(request['instance_id'] == r['instance_id'] for r in state['records']):
        raise ValueError('Next instance has already been consumed')
    if type(request['bootstrap']) is not bool:
        raise ValueError('Explicit bootstrap required')
    cfg = state['config']; rows = state['records']
    if state['mode'] == 'frozen':
        if state['frozen_record_ids'] is None:
            raise ValueError('Missing frozen initial snapshot')
        selected = [r for r in rows if r['pair_id'] in state['frozen_record_ids']]
    else:
        selected = rows
    mean, variance = posterior(cfg, selected)
    info = information_value(mean, variance, cfg['observation_sd']**2, cfg['remaining_uses'])
    spent = sum(r['paid_cost'] for r in rows)
    quality_n = sum(r['quality'] == 'pass' and r['measurement'] == 'valid' for r in rows)
    action, reason, disposition = 'hold', 'insufficient_evidence', 'UNRESOLVED'
    # These common checks see every new record in ALL arms, including frozen.
    if any(r['paid_cost'] > cfg['next_pair_cap'] for r in rows):
        reason = 'executor_cap_breach_reconcile_before_continuing'
    elif any(r['quality'] == 'candidate_failure' for r in rows):
        reason, disposition = 'observed_candidate_failure', 'REJECT_FOR_CURRENT_SCOPE'
    elif any(r['measurement'] != 'valid' or r['quality'] == 'unresolved' for r in rows):
        reason = 'measurement_or_quality_requires_resolution'
    elif cfg['remaining_uses'] == 0:
        reason = 'no_remaining_use'
    elif len(rows) >= cfg['max_pairs']:
        reason = 'pair_limit'
    elif spent + cfg['next_pair_cap'] > cfg['total_budget']:
        reason = 'allocation_exhausted'
    elif request['bootstrap'] and not rows:
        action, reason = 'run_pair', 'declared_paid_bootstrap'
    elif not rows:
        reason = 'bootstrap_not_requested'
    elif state['mode'] == 'fixed':
        if len(rows) < cfg['fixed_pairs']:
            action, reason = 'run_pair', 'fixed_allocation'
        else:
            reason = 'fixed_allocation_complete'
    elif quality_n < cfg['minimum_quality_pairs']:
        # Common minimum evidence; not marketed as information-value selection.
        action, reason = 'run_pair', 'minimum_quality_evidence'
    elif info > cfg['next_pair_estimate']:
        action, reason = 'run_pair', 'positive_modeled_information_value'
    else:
        reason = 'information_value_below_next_cost'
    if (action == 'hold' and quality_n >= cfg['minimum_quality_pairs'] and
            all(r['quality'] == 'pass' and r['measurement'] == 'valid' for r in rows)):
        disposition = 'INDEPENDENT_REVIEW_REQUIRED' if mean > 0 else 'NO_CURRENT_COST_CASE'
    return {'version': VERSION, 'action': action, 'reason': reason,
            'candidate_disposition': disposition, 'promotion_authority': False,
            'paid_cost': spent, 'cost_unit': cfg['cost_unit'], 'quality_pairs': quality_n,
            'selection_record_ids': [r['pair_id'] for r in selected],
            'mean_saving': mean, 'posterior_variance': variance,
            'modeled_next_information_value': info,
            'modeled_next_net_value': info-cfg['next_pair_estimate'],
            'state_sha256': digest(state), 'request_sha256': digest(request),
            'interpretation': 'Declared cost-only Gaussian planning model, not calibrated confidence, quality noninferiority, realized savings, or cross-candidate learning.'}


def step(state, request, executor, emit):
    """Write decision before execution; executor enforces caps and returns costs.

No shell/remote launch is implicit. A real driver supplies its existing bounded
pair executor. Exceptions preserve the request and do not invent zero cost.
"""
    decision = decide(state, request)
    emit({'event': 'decision', 'decision': decision, 'request': copy.deepcopy(request)})
    if decision['action'] != 'run_pair':
        return state, {'decision': decision, 'executor_calls': 0, 'realized_saving': None}
    job = copy.deepcopy(request)
    job['maximum_pair_cost'] = state['config']['next_pair_cap']
    try:
        row = executor(job)
        # Preserve malformed/over-budget executor receipts before validation.
        emit({'event': 'executor_return', 'record': copy.deepcopy(row)})
        if row['instance_id'] != request['instance_id']:
            raise ValueError('Executor returned another instance')
        updated = record(state, row)
    except Exception as exc:
        emit({'event': 'executor_error', 'error': repr(exc), 'cost_status': 'reconcile_driver_receipt'})
        raise
    receipt = {'decision': decision, 'executor_calls': 1, 'realized_saving': None,
               'over_cap': row['paid_cost'] > job['maximum_pair_cost']}
    emit({'event': 'recorded', 'state_sha256': digest(updated), 'receipt': receipt})
    return updated, receipt
