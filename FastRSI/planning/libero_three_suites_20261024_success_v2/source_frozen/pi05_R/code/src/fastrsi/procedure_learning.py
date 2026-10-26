"""Small, offline adapter from full-Agent branches to procedure-choice examples.

This does not collect observations, approve skills, or train a model. Observed
inputs and actual branch facts can be retained without a full simulator snapshot.
The separate strict comparative path still requires an original starting-state
snapshot; this is a restriction of that path, not a universal learning requirement.
Receipts are trace bindings, not an independent proof of the runner's honesty.
Single-branch success establishes feasibility only. A comparative target requires
all declared alternatives to have been executed from the same bound start.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import struct

from .decision_corpus import digest, project_input
from .decision_validity import validate_choice


VERSION = 'procedure_learning_v2'


def _sha(value, name):
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value):
        raise ValueError('Missing or invalid ' + name)
    return value


def _text(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError('Missing ' + name)
    return value


def _finite(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError('Invalid ' + name)
    return value


def _json_copy(value):
    # Reject NaNs, Python objects and mutable aliases before computing bindings.
    return json.loads(json.dumps(value, allow_nan=False))


def _start_binding(value):
    """Keep observed/configured/reconstructed/original evidence distinct.

    Strength is derived from kind, never chosen by the caller. Source hashes are
    provenance references; the importer does not independently verify files.
    A reconstructed hash may not enter the original-snapshot field.
    """
    value = _json_copy(value or {'kind': 'observed_inputs'})
    kinds = {
        'observed_inputs': ('observed_only', set()),
        'sdk_initialization': ('configured_start_not_snapshot',
                               {'suite', 'task_id', 'init_state_id', 'seed', 'source_sha256'}),
        'reconstructed_replay': ('reconstructed_not_original', {'replay_receipt_sha256'}),
        'original_snapshot': ('original_complete_snapshot', {'snapshot_sha256', 'snapshot_source'}),
    }
    if value.get('kind') not in kinds:
        raise ValueError('Unknown start binding kind')
    strength, fields = kinds[value['kind']]
    if set(value) != {'kind'} | fields:
        raise ValueError('Start binding fields do not match its evidence kind')
    for key in fields:
        if key.endswith('sha256'):
            _sha(value[key], key)
        elif key in ('task_id', 'init_state_id', 'seed'):
            if type(value[key]) is not int or value[key] < 0:
                raise ValueError('Invalid start binding ' + key)
        else:
            _text(value[key], key)
    value['strength'] = strength
    return value


def procedure(spec):
    """Bind a concrete, already available procedure; not an effectiveness claim.

    Steps are the ordered instructions the runner will execute. This first
    adapter chooses complete, parameter-bound plans; it does not invent skill
    arguments. Changing any instruction therefore changes the procedure hash.
    """
    if set(spec) != {'id', 'version', 'steps', 'status'}:
        raise ValueError('Expected id, version, steps and status')
    for key in ('id', 'version'):
        if not isinstance(spec[key], str) or not re.fullmatch(r'[A-Za-z0-9_.-]+', spec[key]):
            raise ValueError('Invalid procedure ' + key)
    if spec['status'] not in ('active', 'retired'):
        raise ValueError('Invalid procedure status')
    if not isinstance(spec['steps'], list) or not spec['steps']:
        raise ValueError('Missing executable steps')
    for step in spec['steps']:
        _text(step, 'step instruction')
    value = _json_copy(spec)
    value['key'] = spec['id'] + '@' + spec['version']
    # Retirement changes permission, not the identity of the historical steps.
    value['sha256'] = digest({k: value[k] for k in ('id', 'version', 'steps')})
    return value


def make_decision(event, procedures, *, execution_rule, stage, provenance):
    """Whitelist current policy inputs; keep split/start/source IDs out of Clef.

    ``provenance`` must identify development mode, task group, observation,
    source event and shared protocol. ``start_binding`` is optional and defaults
    to observed inputs only. Original snapshots enable the existing strict
    comparison path; SDK settings and replay evidence remain clearly weaker.
    All start/protocol metadata stay outside model features.
    """
    if provenance.get('mode') != 'development':
        raise ValueError('Only development data may enter learning')
    projected, missing = project_input(event, execution_rule)
    if missing:
        raise ValueError('Incomplete current input: ' + ', '.join(missing))
    _text(execution_rule, 'execution rule')
    _text(stage, 'current stage')
    state_binding = projected['array_bindings']['observation/state']
    if (state_binding.get('dtype') != 'float32' or state_binding.get('shape') != [8]
            or hashlib.sha256(struct.pack('<8f', *projected['robot_state'])).hexdigest() != state_binding['sha256']):
        raise ValueError('Robot state does not match recorded input bytes')
    for key, binding in projected['array_bindings'].items():
        _sha(binding.get('sha256'), key)
        # Drop incidental source fields, including outcomes or hidden state.
        projected['array_bindings'][key] = {'sha256': binding['sha256']}
    candidates = [procedure(p) for p in procedures]
    if not candidates or len({p['key'] for p in candidates}) != len(candidates):
        raise ValueError('Empty or duplicate procedure choices')
    if any(p['status'] != 'active' for p in candidates):
        raise ValueError('Retired procedures cannot be new choices')
    projected['stage'] = stage
    projected['procedures'] = [{k: p[k] for k in ('key', 'sha256', 'steps')} for p in candidates]
    meta = {k: _text(provenance.get(k), k) for k in ('mode', 'task_group', 'observation_id')}
    meta.update({k: _sha(provenance.get(k), k) for k in
                 ('source_event_sha256', 'protocol_sha256')})
    if 'start_state_sha256' in provenance:
        raise ValueError('Bare start_state_sha256 is ambiguous; use explicit start_binding')
    meta['start_binding'] = _start_binding(provenance.get('start_binding'))
    meta['strict_comparison_start_bound'] = meta['start_binding']['kind'] == 'original_snapshot'
    value = dict(version=VERSION, model_input=_json_copy(projected), metadata=meta)
    value['input_sha256'] = digest(value['model_input'])
    value['decision_id'] = digest(value)
    return value


def _check_decision(decision):
    if decision.get('version') != VERSION or decision['metadata']['mode'] != 'development':
        raise ValueError('Invalid development decision')
    body = {k: v for k, v in decision.items() if k != 'decision_id'}
    if digest(body) != decision.get('decision_id') or digest(decision['model_input']) != decision['input_sha256']:
        raise ValueError('Decision bytes changed')


def clef_request(decision, current_procedures):
    """Typed choice in the existing Clef schema; no result/label is included.

    Image bytes must be resolved against the retained input hashes by the caller.
    The existing continue/inspect/escalate encoder has a fixed schema and should
    not be used unchanged for this procedure-choice request.
    """
    _check_decision(decision)
    versions = [procedure(s) for s in current_procedures]
    current = {p['key']: p for p in versions}
    if len(current) != len(versions):
        raise ValueError('Duplicate current procedure versions')
    choices = decision['model_input']['procedures']
    for p in choices:
        live = current.get(p['key'])
        if live is None or live['status'] != 'active' or live['sha256'] != p['sha256']:
            raise ValueError('Procedure version missing, changed or retired')
    return {'model': 'clef-flash', 'state': copy.deepcopy(decision['model_input']),
            'questions': {'procedure': {'type': 'choice',
                'instructions': 'Select one available procedure using current observations only. '
                                'Previously successful procedures are not guaranteed to apply here.',
                'criteria': {p['key']: json.dumps(p['steps'], ensure_ascii=False) for p in choices}}}}


def validate_answer(response, request):
    if not isinstance(response, dict) or set(response.get('answers', {})) != {'procedure'}:
        raise ValueError('Missing or extra procedure answer')
    return validate_choice(response['answers']['procedure'],
                           list(request['questions']['procedure']['criteria']))


def import_native_branch(decision, receipt, result_bytes, *, result_sha256):
    """Import one actual, single-attempt full-Agent case, retaining all layers.

    Receipt fields bind the runner's executed version to its decision and source
    trace. The result is the existing stream RESULT.json (or one native case).
    Multi-attempt recovery is deliberately unsupported until its entire chain
    can be bound; it must not become a first-choice success label accidentally.
    """
    _check_decision(decision)
    if hashlib.sha256(result_bytes).hexdigest() != _sha(result_sha256, 'result_sha256'):
        raise ValueError('Result bytes changed')
    if receipt.get('execution_source') != 'native_full_agent':
        raise ValueError('A teacher proposal or replay is not an executed branch')
    for key, expected in (('decision_id', decision['decision_id']),
                          ('input_sha256', decision['input_sha256']),
                          ('start_binding_sha256', digest(decision['metadata']['start_binding'])),
                          ('protocol_sha256', decision['metadata']['protocol_sha256'])):
        if receipt.get(key) != expected:
            raise ValueError('Execution binding mismatch: ' + key)
    _sha(receipt.get('execution_trace_sha256'), 'execution trace')
    _text(receipt.get('case_id'), 'case_id')
    available = {p['key']: p for p in decision['model_input']['procedures']}
    chosen = available.get(receipt.get('procedure_key'))
    if chosen is None or receipt.get('executed_procedure_sha256') != chosen['sha256']:
        raise ValueError('Executed procedure does not match declared choice')
    result = json.loads(result_bytes)
    if 'cases' in result:
        if result.get('status') != 'COMPLETED':
            raise ValueError('Unfinished native stream')
        matches = [r for r in result['cases'] if r['case']['id'] == receipt['case_id']]
        if len(matches) != 1:
            raise ValueError('Missing or duplicate native case')
        result = matches[0]
    case = result['case']
    if (case['id'] != receipt['case_id'] or case['mode'] != 'development'
            or case['task_description'] != decision['model_input']['goal']):
        raise ValueError('Native case scope differs from decision')
    probe = case.get('procedure_probe', {})
    if (probe.get('procedure_id') + '@' + str(probe.get('version'))
            if isinstance(probe.get('procedure_id'), str) else None) != chosen['key']:
        raise ValueError('Native case does not bind the executed procedure version')
    success = result.get('environment_success')
    if type(success) is not bool:
        raise ValueError('Missing native environment result')
    terminals = result.get('native_terminal_by_session', {})
    verdicts = result.get('outcomes', [])
    if set(terminals) != {case['id']} or len(verdicts) != 1 or verdicts[0].get('session_id') != case['id']:
        raise ValueError('Expected a bound single-attempt execution')
    terminal = terminals[case['id']]
    if (terminal.get('session_id') != case['id'] or terminal.get('event') != 'native_observation'
            or type(terminal.get('native_success')) is not bool
            or type(terminal.get('runtime_success')) is not bool
            or terminal['native_success'] != success or terminal['runtime_success'] != success):
        raise ValueError('Native/runtime outcome mismatch')
    semantic = verdicts[0].get('verdict')
    aligned = semantic == ('success' if success else 'failure')
    completed = result.get('chain_status') == 'PASSED' and result.get('status') == 'PASSED'
    known = aligned and completed and verdicts[0].get('code') == 'verified'
    branch = dict(decision_id=decision['decision_id'], input_sha256=decision['input_sha256'],
                start_binding=copy.deepcopy(decision['metadata']['start_binding']),
                strict_comparison_start_bound=decision['metadata']['strict_comparison_start_bound'],
                protocol_sha256=decision['metadata']['protocol_sha256'],
                procedure_key=chosen['key'], procedure_sha256=chosen['sha256'],
                environment_success=success, semantic_verdict=semantic,
                chain_status=result.get('chain_status'),
                label_status=('OBSERVED_SUCCESS' if success else 'OBSERVED_FAILURE') if known else 'UNKNOWN',
                seconds=_finite(result.get('seconds'), 'native case seconds'),
                cost_scope='native_case_seconds_excludes_worker_startup_and_training',
                result_sha256=result_sha256, receipt=copy.deepcopy(receipt),
                receipt_sha256=digest(receipt), selection_label=None)
    branch['branch_sha256'] = digest(branch)
    return branch


def comparison(decision, branches, *, minimum_cost_saving_fraction=None):
    """Pair-local outcome supervision, not a generalization/causality claim.

    Cost targets are opt-in: a caller must declare a threshold before observing
    results. Even then they are observed preferences, not statistical proof.
    Missing, conflicting and all-failed alternatives yield no training label.
    Records without original snapshots are accepted as observations but remain
    unlabelled by this strict path; no weaker comparison algorithm is added here.
    """
    _check_decision(decision)
    if minimum_cost_saving_fraction is not None:
        _finite(minimum_cost_saving_fraction, 'cost threshold')
        if minimum_cost_saving_fraction >= 1:
            raise ValueError('Cost threshold must be below one')
    available = {p['key']: p for p in decision['model_input']['procedures']}
    indexed = {}
    for branch in branches:
        if digest({k: v for k, v in branch.items() if k != 'branch_sha256'}) != branch.get('branch_sha256'):
            raise ValueError('Imported branch changed')
        for key in ('input_sha256', 'protocol_sha256'):
            expected = decision.get(key, decision['metadata'].get(key))
            if branch.get(key) != expected:
                raise ValueError('Branches must share the declared start and protocol: ' + key)
        if decision['metadata']['strict_comparison_start_bound']:
            start = branch.get('start_binding', {})
            expected = decision['metadata']['start_binding']
            # Separate files may contain identical original snapshot bytes.
            if start.get('kind') != 'original_snapshot' or start.get('snapshot_sha256') != expected['snapshot_sha256']:
                raise ValueError('Branches must share the original start_binding snapshot')
        key = branch['procedure_key']
        if key in indexed or key not in available or branch['procedure_sha256'] != available[key]['sha256']:
            raise ValueError('Duplicate, unknown or changed procedure branch')
        indexed[key] = branch
    summary = dict(label=None, label_status='UNLABELLED',
                   outcomes={k: indexed[k]['label_status'] if k in indexed else 'UNEXECUTED' for k in available},
                   reason='incomplete_or_unresolved_comparison',
                   scope='development branch observations; comparative labels require original matched snapshots; '
                         'no unseen-instance or net learning gain claim')
    if not decision['metadata']['strict_comparison_start_bound']:
        summary['reason'] = 'observations_retained_original_snapshot_required_by_strict_comparison'
        return summary
    if len(available) < 2 or set(indexed) != set(available) or any(
            b['label_status'] not in ('OBSERVED_SUCCESS', 'OBSERVED_FAILURE') for b in branches):
        return summary
    successful = [b for b in branches if b['label_status'] == 'OBSERVED_SUCCESS']
    if len(successful) == 1:
        summary.update(label=successful[0]['procedure_key'], label_status='OBSERVED_COMPARATIVE_CHOICE',
                       reason='only_observed_successful_branch')
    elif len(successful) > 1 and minimum_cost_saving_fraction is not None:
        ordered = sorted(successful, key=lambda b: b['seconds'])
        best, next_best = ordered[:2]
        if next_best['seconds'] > 0 and best['seconds'] < next_best['seconds'] * (1 - minimum_cost_saving_fraction):
            summary.update(label=best['procedure_key'], label_status='OBSERVED_COMPARATIVE_CHOICE',
                           reason='observed_case_cost_preference_not_total_cost_proof')
    else:
        summary['reason'] = 'no_unique_supported_choice'
    return summary


def training_example(decision, branches, current_procedures, *, minimum_cost_saving_fraction=None):
    """Return separate request/target/evidence; never inject outcome into state."""
    request = clef_request(decision, current_procedures)
    result = comparison(decision, branches, minimum_cost_saving_fraction=minimum_cost_saving_fraction)
    if result['label'] is None:
        return None
    return dict(request=request, target={'procedure': result['label']},
                provenance={'task_group': decision['metadata']['task_group'],
                            'decision_id': decision['decision_id'],
                            'result_sha256': [b['result_sha256'] for b in branches],
                            'receipt_sha256': [b['receipt_sha256'] for b in branches],
                            'supervision': result})
