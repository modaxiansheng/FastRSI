"""Audit stopping decisions against sealed, already executed development pairs.

This is chronological historical replay, not counterfactual robot simulation.
Skipped outcomes are visible only to the scorer, never to the allocation rule.
No candidate is promoted, and recorded lifecycle sums are not realized savings.
"""
import copy
import math

from .validation_budget import digest
from .window_budget_v2 import decide

MODES = ('fixed_one', 'fixed_two', 'full_catalog', 'frozen', 'feedback')


def validate(rows):
    ids = set()
    previous = -math.inf
    for row in rows:
        if (type(row['observed_at']) not in (int, float) or not math.isfinite(row['observed_at'])
                or type(row['init']) is not int or row['init'] < 0):
            raise ValueError('Invalid ordering or initial-state identity')
        if row['phase'] != 'development':
            raise ValueError('Evaluation feedback is forbidden')
        if row['id'] in ids or row['observed_at'] < previous:
            raise ValueError('Duplicate or out-of-order outcome')
        ids.add(row['id']); previous = row['observed_at']
        for key in ('native_seconds', 'candidate_seconds'):
            if type(row[key]) not in (int, float) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError('Invalid lifecycle measurement')
        for key in ('native_success', 'candidate_success', 'semantic_aligned'):
            if type(row[key]) is not bool:
                raise ValueError('Outcome must be explicit')
        if not row['sources'] or any(not isinstance(v, str) or len(v) != 64
                                   or any(c not in '0123456789abcdef' for c in v)
                                   for v in row['sources'].values()):
            raise ValueError('Missing evidence binding')


def scope(row):
    # No initial-state identifier, future outcome or runtime enters task scope.
    return digest({'goal': row['goal'], 'model': row['model_sha256'],
                   'candidate': 'fixed5_vs_fixed10_H10'})


def relation(row):
    if row['candidate_success'] and not row['native_success']: return 'win'
    if row['native_success'] and not row['candidate_success']: return 'harm'
    return 'equal'


def choose(mode, observed, proposal, config):
    """Only paid prefix evidence and a whitelisted proposal are accessible."""
    if mode not in MODES: raise ValueError('Unknown policy')
    if not observed: return True, 'common_one_pair_bootstrap', []
    # A reasonable fixed baseline also stops after observed quality harm.
    if any(relation(r) == 'harm' for r in observed):
        return False, 'common_observed_quality_harm', [r['id'] for r in observed]
    if mode.startswith('fixed_') or mode == 'full_catalog':
        limit = {'fixed_one': 1, 'fixed_two': 2, 'full_catalog': math.inf}[mode]
        return len(observed) < limit, 'declared_fixed_allocation', [r['id'] for r in observed]
    selected = observed[:1] if mode == 'frozen' else observed
    records = []
    for r in selected:
        records.append({'init_id': r['init'], 'native_quality_relation': relation(r),
                        'measurement_integrity': 'valid',
                        'eligible_cost': r['protocol_group'] == proposal['protocol_group']
                            and r['native_success'] and r['candidate_success'],
                        'native_task_seconds': r['native_seconds'],
                        'candidate_task_seconds': r['candidate_seconds']})
    ledger = {'candidate_sha256': proposal['scope'], 'records': records}
    request = dict(config, phase='development', ledger_sha256=digest(ledger),
                   candidate_sha256=proposal['scope'], next_init_id=proposal['init'])
    d = decide(ledger, request)
    return d['action'] == 'validate_next', d['reason'], [r['id'] for r in selected]


def replay(rows, config):
    validate(rows)
    # Repeated starts retain their paid costs but never count as new evidence.
    seen = set(); independent = []; repeats = []
    for r in rows:
        key = (scope(r), r['init'])
        (repeats if key in seen else independent).append(r)
        seen.add(key)
    results = {}
    for mode in MODES:
        histories = {}; trace = []; selected = []; skipped = []
        for row in independent:
            key = scope(row); history = histories.setdefault(key, [])
            proposal = {k: row[k] for k in ('init', 'protocol_group')}
            proposal['scope'] = key
            run, reason, evidence = choose(mode, copy.deepcopy(history), proposal, config)
            trace.append({'proposal_id': row['id'], 'selected': run, 'reason': reason,
                          'evidence_ids': evidence, 'proposal': proposal})
            # Outcomes enter policy state only AFTER the logged decision to buy.
            if run:
                history.append(copy.deepcopy(row)); selected.append(row)
            else: skipped.append(row)
        def ids(items, pred): return [r['id'] for r in items if pred(r)]
        results[mode] = {
            'trace': trace, 'selected_pairs': len(selected), 'skipped_pairs': len(skipped),
            'selected_reference_lifecycle_seconds': sum(r['native_seconds']+r['candidate_seconds'] for r in selected),
            'skipped_reference_lifecycle_seconds': sum(r['native_seconds']+r['candidate_seconds'] for r in skipped),
            'selected_native_quality_wins': ids(selected, lambda r: relation(r) == 'win'),
            'missed_native_quality_wins': ids(skipped, lambda r: relation(r) == 'win'),
            'skipped_native_quality_harms': ids(skipped, lambda r: relation(r) == 'harm'),
            # A single faster pair is only an opportunity, not a stable benefit.
            'missed_raw_both_success_cost_opportunities': ids(skipped, lambda r:
                r['native_success'] and r['candidate_success'] and r['candidate_seconds'] < r['native_seconds']),
            'missed_aligned_cost_opportunities': ids(skipped, lambda r:
                r['native_success'] and r['candidate_success'] and r['semantic_aligned']
                and r['candidate_seconds'] < r['native_seconds']),
            'candidate_promotions': 0, 'measured_gpu_savings': None,
        }
    return {'policies': results, 'unique_pairs': len(independent),
            'coverage': {
                'independent_native_quality_wins': [r['id'] for r in independent if relation(r) == 'win'],
                'independent_native_quality_harms': [r['id'] for r in independent if relation(r) == 'harm'],
                'aligned_both_success_faster_pairs': [r['id'] for r in independent
                    if r['native_success'] and r['candidate_success'] and r['semantic_aligned']
                    and r['candidate_seconds'] < r['native_seconds']],
                'post_first_pair_native_quality_wins': [r['id'] for index,r in enumerate(independent)
                    if relation(r) == 'win' and any(scope(p)==scope(r) for p in independent[:index])],
                'final_candidate_quality_tested_by_this_replay': False,
            },
            'repeated_pair_ids': [r['id'] for r in repeats],
            'all_paid_reference_lifecycle_seconds': sum(r['native_seconds']+r['candidate_seconds'] for r in rows),
            'new_robot_episodes': 0, 'new_model_calls': 0, 'policy_training_updates': 0,
            'learned_policy_advantage_established': False,
            'scope': 'Retrospective development replay. Reference costs mix historical deployments; not a current speedup, calibrated quality guarantee or cold-start result.'}
