"""Read-only cumulative numeric evidence for a declared fixed10/fixed5 candidate.

Only explicitly ordered pair paths are read. Caller-declared protocol groups
express cost compatibility; hashes bind the declarations and original evidence,
not an automatic proof that two different protocols are equivalent. No policy,
promotion, experiment dispatch, or narrative relabelling happens here.
"""
import hashlib
import json
import math
from pathlib import Path

from task_window_study import pair_record


VERSION = 'window_budget_evidence_v1'


def _digest(value, name):
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('Invalid SHA256: ' + name)
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing ' + name)
    return value


def _number(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Missing or invalid ' + name)
    return value


def _path(root, relative):
    _text(relative, 'source path')
    path = (root / relative).resolve()
    if Path(relative).is_absolute() or not path.is_relative_to(root):
        raise ValueError('Evidence path must remain relative to project')
    return path


def _sources(root, bindings):
    if not isinstance(bindings, dict) or not bindings:
        raise ValueError('Explicit source bindings required')
    for relative, expected in bindings.items():
        _digest(expected, relative)
        if hashlib.sha256(_path(root, relative).read_bytes()).hexdigest() != expected:
            raise ValueError('Source SHA mismatch: ' + relative)


def _alignment(native, outcomes):
    if (not isinstance(outcomes, list) or not outcomes
            or any(o.get('code') != 'verified' for o in outcomes)):
        return 'unresolved'
    verdicts = [o.get('verdict') for o in outcomes]
    if len(verdicts) != 1 or verdicts[0] not in ('success', 'failure'):
        return 'unresolved'
    return 'aligned' if (verdicts[0] == 'success') == native else 'conflict'


def arm_groups(root, spec):
    """Bind original directories, including pairs executed in separate branches."""
    root=Path(root).resolve()
    pair=_path(root,spec.get('path'))
    if not pair.is_dir():raise ValueError('Pair identity must name an existing source directory')
    arms=spec.get('arms',['N','F'])
    if (not isinstance(arms,(list,tuple)) or len(arms)!=2 or len(set(arms))!=2
            or any(a not in ('N','F','R','S') for a in arms)):
        raise ValueError('Two distinct explicit execution arms required')
    explicit=spec.get('arm_paths')
    if explicit is not None:
        if not isinstance(explicit,dict) or set(explicit)!=set(arms):raise ValueError('Arm path binding mismatch')
        groups=[_path(root,explicit[arm]) for arm in arms]
    else:groups=[pair/arm for arm in arms]
    if len(set(groups))!=2 or not all(p.is_dir() for p in groups):raise ValueError('Two distinct existing arm directories required')
    return groups


def build_ledger(root, pair_specs, *, candidate_sha256, goal, model_sha256,
                 protocol_group, protocol_sources, repeat_policy='reject'):
    """Build one goal/model/candidate ledger in observed completion order.

    Each spec has path, protocol_group, protocol_sources and cost_compatible,
    plus optional arms (default N,F, interpreted as fixed10,fixed5). Historical
    protocol groups may differ: retain their facts/costs but exclude them from
    this scope's cost estimate. Corrupt, evaluation, duplicate or future-ordered
    inputs are rejected rather than represented as unsuccessful robot trials.
    """
    root = Path(root).resolve()
    if repeat_policy not in ('reject','retain'):raise ValueError('Unknown repeat policy')
    _digest(candidate_sha256, 'candidate'); _digest(model_sha256, 'model')
    _text(goal, 'goal'); _text(protocol_group, 'protocol_group')
    _sources(root, protocol_sources)
    if not isinstance(pair_specs, list):
        raise ValueError('Explicit ordered pair list required')
    records, seen_instances, seen_paths, seen_executions = [], {}, set(), set()
    previous_time = None
    for spec in pair_specs:
        if not isinstance(spec, dict):
            raise ValueError('Invalid pair specification')
        pair_path = _path(root, spec.get('path'))
        if pair_path in seen_paths:
            raise ValueError('Duplicate pair path')
        seen_paths.add(pair_path)
        group = _text(spec.get('protocol_group'), 'pair protocol_group')
        _sources(root, spec.get('protocol_sources'))
        compatible = spec.get('cost_compatible')
        if type(compatible) is not bool:
            raise ValueError('Explicit cost_compatible flag required')
        arms = spec.get('arms', ['N', 'F'])
        if (not isinstance(arms, (list, tuple)) or len(arms) != 2
                or len(set(arms)) != 2 or any(a not in ('N', 'F', 'R', 'S') for a in arms)):
            raise ValueError('Two distinct explicit execution arms required')
        groups=arm_groups(root,spec)
        # Reuse the established checks for actual policy modes, original
        # episode facts, successful chains and matched images/model config.
        try:
            pair = pair_record(root, pair_path, tuple(arms), compatible,
                               arm_paths=spec.get('arm_paths'),record_schema=spec.get('record_schema','feedback_window'))
        except (AssertionError, KeyError, IndexError, TypeError) as exc:
            raise ValueError('Invalid original development pair: ' + spec['path']) from exc
        if pair['goal'] != goal:
            raise ValueError('Pair belongs to another task goal')
        instance = 'libero_10/task' + str(pair['task']) + '/init' + str(pair['init'])
        independent=instance not in seen_instances
        if not independent and repeat_policy=='reject':
            raise ValueError('Repeated init is not independent evidence')
        if spec.get('repeat_of') != (seen_instances.get(instance) if not independent else None):
            raise ValueError('Repeat must reference the earliest original instance evidence')
        first_evidence=seen_instances.setdefault(instance,pair['evidence_id'])
        sources = dict(pair['sources'])
        for path, digest in spec['protocol_sources'].items():
            if path in sources and sources[path] != digest:
                raise ValueError('Conflicting source SHA declarations')
            sources[path] = digest
        _sources(root, sources)
        finished, alignments, admissions = [], {}, {}
        for arm, group_path, label, fixed in zip(arms, groups, ('native', 'candidate'), ('fixed10', 'fixed5')):
            prefix = str(group_path.relative_to(root))
            model_path = prefix + '/MODEL_CONFIG.json'
            if sources[model_path] != model_sha256:
                raise ValueError('Pair model differs from ledger model')
            result_path = prefix + '/stream/RESULT.json'
            execution = (sources[result_path], str((root / result_path).resolve()))
            if any(execution[i] in seen_executions for i in (0, 1)):
                raise ValueError('Original execution reused')
            seen_executions.update(execution)
            status = json.loads((root / prefix / 'STATUS.json').read_text())
            finished.append(_number(status.get('finished_at'), 'STATUS.finished_at'))
            result = json.loads((root / result_path).read_text())
            if len(result['cases']) != 1:
                raise ValueError('Source cost must bind exactly one case per arm')
            row = result['cases'][0]
            if type(row['case'].get('task_id')) is not int or type(row['case'].get('init_state_id')) is not int:
                raise ValueError('Invalid task/init identity')
            if row['case']['task_id'] < 0 or row['case']['init_state_id'] < 0:
                raise ValueError('Invalid task/init identity')
            native = pair[fixed]['native_success']
            if type(native) is not bool:
                raise ValueError('Missing explicit native outcome')
            _number(pair[fixed]['seconds'], 'paid arm lifecycle seconds')
            alignments[label] = _alignment(native, row.get('outcomes'))
            admission = row.get('experience_admission', {})
            admissions[label] = {'status': _text(admission.get('status'), 'narrative admission status'),
                                 'reasons': admission.get('reasons', [])}
        observed_at = max(finished)
        if previous_time is not None and observed_at < previous_time:
            raise ValueError('Pair order would consume future evidence before its prefix')
        previous_time = observed_at
        a, b = pair['fixed10'], pair['fixed5']
        relation = ('win' if b['native_success'] and not a['native_success'] else
                    'harm' if a['native_success'] and not b['native_success'] else 'equal')
        same_protocol = group == protocol_group
        comparable = same_protocol and compatible
        reason = ('different_protocol' if not same_protocol else
                  'explicitly_excluded' if not compatible else 'declared_same_protocol')
        records.append({
            'evidence_id': pair['evidence_id'], 'init_id': pair['init'],
            'independent_instance_key': instance, 'observed_at': observed_at,
            'candidate_sha256': candidate_sha256, 'goal': goal, 'model_sha256': model_sha256,
            'protocol_group': group, 'native_quality_relation': relation,
            'native_success': a['native_success'], 'candidate_success': b['native_success'],
            'measurement_integrity': 'valid', 'semantic_alignment': alignments,
            'narrative_admission': admissions,
            'cost_comparability': {'comparable': comparable, 'same_protocol': same_protocol,
                                   'explicit_cost_compatible': compatible, 'reason': reason},
            'eligible_cost': independent and comparable and a['native_success'] and b['native_success'],
            'native_task_seconds': a['seconds'], 'candidate_task_seconds': b['seconds'],
            'project_source_run_seconds': a['seconds'] + b['seconds'],
            'sources': sources,
        })
        if repeat_policy=='retain':
            records[-1].update(independent_observation=independent,first_evidence_id=first_evidence,
                               repeat_of=None if independent else first_evidence,
                               original_record_schema=spec.get('record_schema','feedback_window'))
        # Detect changed inputs during the multi-file read without altering them.
        _sources(root, sources)
    _sources(root, protocol_sources)
    result={'version': VERSION, 'phase': 'development', 'candidate_sha256': candidate_sha256,
            'goal': goal, 'model_sha256': model_sha256, 'protocol_group': protocol_group,
            'protocol_sources': dict(protocol_sources), 'records': records,
            'paid_source_seconds': sum(r['project_source_run_seconds'] for r in records),
            'cost_unit': 'two_gpu_reservation_seconds',
            'eligible_cost_evidence_ids': [r['evidence_id'] for r in records if r['eligible_cost']],
            'scope': 'Cumulative SHA-bound numeric development facts; protocol compatibility is declared, not inferred. Native failure is an outcome, not corrupt measurement. Narrative status is preserved.',
            'policy_decision': None, 'promotion_authority': False,
        'method_cost_attribution': 'All paid source costs retained; this ledger does not claim they must recur at each method cold start.'}
    if repeat_policy=='retain':
        result.update(version='window_budget_evidence_v2',repeat_policy=repeat_policy,
                      independent_instances=len(seen_instances),repeated_executions=len(records)-len(seen_instances))
    return result
