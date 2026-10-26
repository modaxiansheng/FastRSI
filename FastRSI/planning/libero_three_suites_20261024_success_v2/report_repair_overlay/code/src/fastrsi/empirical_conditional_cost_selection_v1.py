"""Opt-in development selection by complete cost within an observable scope.

This is a draft selector, not a deployed classifier or capability certificate.
The adapter, not this pure function, must bind raw outcomes, public features,
the complete preregistered roster and homogeneous resource/timing receipts.
Legacy componentwise selection and the native default remain unchanged.
"""
import copy
import hashlib
import math
from pathlib import Path
import re
import time

from .empirical_scope_update_v1 import METRICS, VERSION, digest, select_update
from .empirical_stream_selection_v1 import _cohort

SELECTION_POLICY = 'empirical_conditional_cost_selection_v1'
FEATURE_SCHEMA = 'initial_public_binding_v1'
# Already available at the root initial binding; no task/draw IDs or labels.
FEATURES = {'task_goal', 'phase', 'anchor0_dx_m', 'anchor0_dy_m',
            'anchor1_dx_m', 'anchor1_dy_m'}
COST_SCOPE = 'PER_ARM_START_THROUGH_OWNED_SERVICE_STOP_INCLUSIVE'


def _sha(value):
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _seconds(value):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Finite nonnegative actual seconds required')
    return float(value)


def features_from_initial_binding(prepared, robot8):
    """Reuse already-computed public matching; no new encoding/model/physics.

    This is a feature extraction interface, not a learned applicability rule.
    The caller must bind the prepared receipt and robot8 to the actual initial
    observation. Hidden simulator state, draw IDs and terminal results never
    enter these values or the public-observation fingerprint.
    """
    declaration = prepared['declaration']
    receipt = prepared['derivation_receipt']
    anchors = [part['anchor'] for part in declaration['transform']['anchor_plan']]
    if (receipt.get('status') != 'UNVALIDATED_DERIVED_OPERATION_CANDIDATE'
            or len(anchors) != 2 or len(set(anchors)) != 2
            or not _sha(receipt.get('current_rgb_array_sha256'))
            or not isinstance(receipt.get('offsets'), dict)
            or set(receipt['offsets']) != set(anchors)
            or not isinstance(robot8, (list, tuple)) or len(robot8) != 8
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in robot8)):
        raise ValueError('Actual initial public binding and complete robot8 required')
    goal = declaration['bindings']['source_goal']
    if not isinstance(goal, str) or not goal.strip():
        raise ValueError('Actual complete task goal required')
    values = dict(task_goal=goal, phase='root_initial')
    for index, anchor in enumerate(anchors):
        offset = receipt['offsets'][anchor]
        if (not isinstance(offset, (list, tuple)) or len(offset) != 3
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in offset)
                or offset[2] != 0):
            raise ValueError('Only actual planar public offsets are supported')
        values['anchor%d_dx_m' % index] = float(offset[0])
        values['anchor%d_dy_m' % index] = float(offset[1])
    return dict(feature_schema=FEATURE_SCHEMA,
        extractor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), values=values,
        public_observation_sha256=digest(dict(front_rgb_array_sha256=receipt['current_rgb_array_sha256'], robot8=list(robot8))))


def validate_scope(scope):
    """Fixed predicates only; this function never fits a threshold to outcomes."""
    if (not isinstance(scope, dict) or set(scope) != {'feature_schema', 'extractor_sha256', 'conditions'}
            or scope['feature_schema'] != FEATURE_SCHEMA or not _sha(scope['extractor_sha256'])
            or not isinstance(scope['conditions'], dict) or not scope['conditions']):
        raise ValueError('Explicit observable scope and bound extractor required')
    for key, condition in scope['conditions'].items():
        if key not in FEATURES:
            raise ValueError('Only current public initial-binding features may define scope')
        if key in ('task_goal', 'phase'):
            if not isinstance(condition, str) or not condition.strip():
                raise ValueError('Exact textual condition required')
            if key == 'phase' and condition != 'root_initial':
                raise ValueError('Only the initial binding is supported')
        elif (not isinstance(condition, list) or len(condition) != 2
              or any(type(x) not in (int, float) or not math.isfinite(x) for x in condition)
              or condition[0] > condition[1]):
            raise ValueError('Finite ordered inclusive feature interval required')
    return copy.deepcopy(scope)


def scope_membership(scope, features):
    """IN / OUT / UNKNOWN; missing or malformed evidence is never absence."""
    validate_scope(scope)
    if (not isinstance(features, dict)
            or features.get('feature_schema') != scope['feature_schema']
            or features.get('extractor_sha256') != scope['extractor_sha256']
            or not _sha(features.get('public_observation_sha256'))
            or not isinstance(features.get('values'), dict)):
        return 'UNKNOWN'
    values = features['values']
    # Unknown required facts take precedence over an apparently excluding fact.
    for key, condition in scope['conditions'].items():
        value = values.get(key)
        if isinstance(condition, str):
            if not isinstance(value, str) or not value.strip():
                return 'UNKNOWN'
        elif type(value) not in (int, float) or not math.isfinite(value):
            return 'UNKNOWN'
    for key, condition in scope['conditions'].items():
        value = values[key]
        if isinstance(condition, str):
            if value != condition:
                return 'OUT'
        elif not condition[0] <= value <= condition[1]:
            return 'OUT'
    return 'IN'


def _cost(row, expected):
    cost = row.get('complete_cost')
    if not isinstance(cost, dict):
        return None, 'COMPLETE_COST_MISSING'
    if (cost.get('version') != 'inclusive_arm_cost_v1' or cost.get('scope') != COST_SCOPE
            or cost.get('comparison_eligible') is not True
            or cost.get('comparison_protocol_sha256') != expected
            or not _sha(cost.get('receipt_sha256'))):
        return None, 'COST_PROTOCOL_UNRESOLVED'
    return _seconds(cost.get('seconds')), None


def select_conditional_cost_update(observations, candidate_ids, *, incumbent='native',
                                   cohort, scopes, cost_protocol_sha256,
                                   minimum_saving_seconds, update_costs_seconds,
                                   previous=None):
    """Compare observed successful capabilities, with every failed bill retained.

    Counts can increase. Only complete inclusive, same-protocol per-arm costs
    rank candidates. Candidate-specific update costs are charged once in this
    finite comparison; shared acquisition costs are retained in history, not
    made free or duplicated in an operational ranking. No future amortization
    or statistical noninferiority claim is inferred from this development test.
    """
    started = time.perf_counter()
    if previous is not None and (not isinstance(previous, dict)
            or previous.get('version') != VERSION
            or previous.get('selection_policy') != SELECTION_POLICY
            or not _sha(previous.get('evidence_sha256'))
            or not _sha(previous.get('policy_sha256'))):
        raise ValueError('Bound previous conditional-policy result required')
    minimum = _seconds(minimum_saving_seconds)
    if not _sha(cost_protocol_sha256):
        raise ValueError('Bound homogeneous complete-cost protocol required')
    keys = _cohort(cohort)
    ids = set(candidate_ids) | {'native'}
    if set(scopes) != ids - {'native'} or set(update_costs_seconds) != ids:
        raise ValueError('Explicit scope and once-only update cost for each catalogue option required')
    scopes = {key: validate_scope(value) for key, value in scopes.items()}
    update_costs = {key: _seconds(value) for key, value in update_costs_seconds.items()}
    output = copy.deepcopy(select_update(observations, ids, incumbent=incumbent))
    for field in ('status', 'reason', 'next_requirement', 'seconds', 'frontier', 'withdrawal_required'):
        output.pop(field, None)
    rows = output['observations']
    groups = {}
    for row in rows:
        groups.setdefault((row['protocol_sha256'], row['context_sha256']), {})[row['candidate']] = row
    policy = dict(selection_policy=SELECTION_POLICY, incumbent=incumbent, cohort=[list(key) for key in keys],
                  scopes=scopes, cost_protocol_sha256=cost_protocol_sha256,
                  minimum_saving_seconds=minimum, update_costs_seconds=update_costs)
    output.update(policy, policy_sha256=digest(policy), candidate_assessments={},
                  wall_time_used_for_ranking=True, quality_scope='OBSERVED_DECLARED_SCOPE_ONLY',
                  ranking='Finite inclusive per-arm job seconds plus once-only candidate update costs',
                  same_capability_speedup_claimed=False, statistical_quality_certified=False,
                  conditional_contract_required=True, runtime_scope_installed=False,
                  update_costs_verified=False,
                  update_cost_receipt_responsibility='Caller must bind once-only update receipts and ledger boundary; no net-benefit certificate from supplied numbers',
                  withdrawal_required=False, incumbent_reassessment_required=False,
                  scope_withdrawals=[], frontier=[])
    revoked = False
    for candidate in sorted(ids - {'native'}):
        historical = [row for row in rows if row['candidate'] == candidate]
        failures = {state: [row['result_sha256'] for row in historical
                           if not row['native_success']
                           and scope_membership(scopes[candidate], row.get('decision_features')) == state]
                    for state in ('IN', 'OUT', 'UNKNOWN')}
        if candidate == incumbent and failures['IN']:
            revoked = True
            output['scope_withdrawals'].append(dict(candidate=candidate,
                scope_sha256=digest(scopes[candidate]), failed_result_sha256=failures['IN']))
        if candidate == incumbent and failures['UNKNOWN']:
            output['incumbent_reassessment_required'] = True
        reasons, comparisons, selected = [], [], []
        if failures['IN']:
            reasons.append('KNOWN_FAILURE_WITHIN_SCOPE')
        if failures['UNKNOWN']:
            reasons.append('FAILURE_SCOPE_UNKNOWN')
        for key in keys:
            group = groups.get(key, {})
            reference, alternative = group.get(incumbent), group.get(candidate)
            if reference is None or alternative is None:
                reasons.append('UNEXECUTED_BRANCH'); continue
            # Both paired arms must bind the same decision-time public facts.
            features = alternative.get('decision_features')
            if features != reference.get('decision_features'):
                reasons.append('PAIR_PUBLIC_FEATURES_DIFFER'); continue
            member = scope_membership(scopes[candidate], features)
            if member != 'IN':
                reasons.append('DECLARED_COHORT_OUTSIDE_SCOPE' if member == 'OUT' else 'PUBLIC_FEATURES_UNKNOWN')
                continue
            selected.append((reference, alternative))
            if not alternative['native_success']:
                reasons.append('OBSERVED_NATIVE_FAILURE')
            comparison = dict(context_sha256=key[1], protocol_sha256=key[0],
                reference_result_sha256=reference['result_sha256'], candidate_result_sha256=alternative['result_sha256'],
                quality_repair=not reference['native_success'] and alternative['native_success'],
                quality_regression=reference['native_success'] and not alternative['native_success'])
            if comparison['quality_regression']:
                reasons.append('QUALITY_REGRESSION')
            comparisons.append(comparison)
        reference_total = alternative_total = None
        if len(selected) == len(keys):
            reference_costs, alternative_costs = [], []
            for reference, alternative in selected:
                for row, values in ((reference, reference_costs), (alternative, alternative_costs)):
                    cost, error = _cost(row, cost_protocol_sha256)
                    if error:
                        reasons.append(error)
                    else:
                        values.append(cost)
            if len(reference_costs) == len(keys) and len(alternative_costs) == len(keys):
                reference_total = math.fsum(reference_costs) + update_costs[incumbent]
                alternative_total = math.fsum(alternative_costs) + update_costs[candidate]
                if reference_total - alternative_total <= minimum:
                    reasons.append('NO_SUFFICIENT_COMPLETE_COST_REDUCTION')
        work_delta = ({metric: sum(alt['work'][metric] - ref['work'][metric] for ref, alt in selected)
                       for metric in METRICS} if len(selected) == len(keys) else None)
        output['candidate_assessments'][candidate] = dict(eligible=not reasons,
            reasons=sorted(set(reasons)), comparisons=comparisons, known_failures=failures,
            scope_sha256=digest(scopes[candidate]), aggregate_work_delta=work_delta,
            reference_complete_job_seconds=reference_total, candidate_complete_job_seconds=alternative_total,
            unknown_historical_results=[row['result_sha256'] for row in historical
                if scope_membership(scopes[candidate], row.get('decision_features')) == 'UNKNOWN'])
    # A replacement draft is not installed. It cannot cancel withdrawal of an
    # already failed incumbent scope; lifecycle consumers must act on both.
    output['withdrawal_required'] = revoked
    # Scope-level failure must also override replay of the same cohort.
    if (previous is not None and previous.get('selection_policy') == SELECTION_POLICY
            and previous.get('policy_sha256') == output['policy_sha256']
            and previous.get('evidence_sha256') == output['evidence_sha256'] and not revoked):
        output['status'] = 'NO_NEW_EVIDENCE'
    else:
        eligible = [key for key, value in output['candidate_assessments'].items()
                    if key != incumbent and value['eligible']]
        if eligible:
            best = min(output['candidate_assessments'][key]['candidate_complete_job_seconds'] for key in eligible)
            output['frontier'] = [key for key in eligible
                if output['candidate_assessments'][key]['candidate_complete_job_seconds'] == best]
        if len(output['frontier']) == 1:
            output.update(status='UNVALIDATED_CONDITIONAL_COST_DRAFT', candidate=output['frontier'][0],
                next_requirement='Bind predicate consumer to the native runtime and validate new real contexts before admission')
        elif revoked:
            output.update(status='INCUMBENT_SCOPE_SUPPORT_REVOKED', withdrawal_required=True)
        else:
            output['status'] = 'AMBIGUOUS_IMPROVEMENTS' if output['frontier'] else 'NO_SUPPORTED_UPDATE'
    output['seconds'] = time.perf_counter() - started
    return output
