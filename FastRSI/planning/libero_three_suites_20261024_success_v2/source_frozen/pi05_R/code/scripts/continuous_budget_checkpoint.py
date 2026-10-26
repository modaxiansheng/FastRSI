"""CPU-only allocation checkpoint between admitted native development cases.

The current one-sided execution binds this decision to a stream and snapshot;
it is never converted into fixed10/fixed5 evidence. No executor is provided.
"""
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastrsi.experience_admission import SEAL, validate_snapshot
from fastrsi.validation_budget import digest
from window_budget_continuation import SCOPED_VERSION, continuation_scoped

VERSION = 'continuous_budget_checkpoint_v1'


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inside(project, value, name):
    path = Path(value)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError(name + ' must be an absolute non-symlink project path')
    path = path.resolve()
    if not path.is_relative_to(project):
        raise ValueError(name + ' is outside project')
    return path


def _relative_source(project, value):
    path = Path(value)
    if path.is_absolute() or not (project / path).resolve().is_relative_to(project):
        raise ValueError('Historical source must remain relative to project')


def _save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def _binding(project, path):
    return {'path': str(path.relative_to(project)), 'sha256': _sha(path)}


def _brief(receipt, path):
    return {key: receipt[key] for key in (
        'version', 'status', 'reason', 'checkpoint_wall_seconds', 'checkpoint_cpu_seconds',
        'additional_validation_pairs', 'actual_worker_dispatch', 'learning_gain_established',
    )} | {'receipt_path': str(path), 'receipt_sha256': _sha(path)}


def run_checkpoint(project, campaign, stream, row, accepted, elapsed_stream_seconds):
    """Save an immutable allocation receipt; failures are explicit exceptions.

    An enclosing task loop may catch checkpoint failures and continue mandatory
    tasks. Evaluation calls are rejected before any file is written.
    """
    started, cpu_started = time.perf_counter(), time.process_time()
    case = row['case']
    if case.get('mode') != 'development':
        raise ValueError('Budget checkpoint requires development; evaluation is read-only')
    project = Path(project).resolve()
    campaign = _inside(project, campaign, 'Campaign')
    stream = _inside(project, stream, 'Stream')
    if stream == campaign or not stream.is_relative_to(campaign):
        raise ValueError('Stream must be inside its campaign')
    case_id = case.get('id')
    if not isinstance(case_id, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', case_id):
        raise ValueError('Invalid case ID')
    if (type(elapsed_stream_seconds) not in (int, float)
            or not math.isfinite(elapsed_stream_seconds) or elapsed_stream_seconds < 0):
        raise ValueError('Invalid elapsed stream seconds')
    config = case['budget_checkpoint']
    if set(config) != {'template_path', 'template_sha256', 'proposal'}:
        raise ValueError('Checkpoint requires only template path, SHA and proposal')
    template_path = _inside(project, config['template_path'], 'Template')
    template_bytes = template_path.read_bytes()
    if hashlib.sha256(template_bytes).hexdigest() != config['template_sha256']:
        raise ValueError('Budget template SHA mismatch')
    protocol = json.loads(template_bytes)
    if protocol.get('budget_policy_version') != SCOPED_VERSION:
        raise ValueError('Explicit scoped_v3 template required')
    for spec in protocol.get('budget_history_specs', []):
        _relative_source(project, spec['path'])
        for value in spec.get('arm_paths',{}).values():_relative_source(project,value)
        for source in spec['protocol_sources']:
            _relative_source(project, source)
    proposal = config['proposal']
    identity = ('suite', 'task_id', 'task_description', 'max_steps')
    if (proposal.get('mode') != 'development'
            or any(proposal.get(key) != case.get(key) for key in identity)):
        raise ValueError('Proposal must be development on the same task and protocol')
    if (type(proposal.get('init_state_id')) is not int or proposal['init_state_id'] < 0
            or proposal['init_state_id'] == case['init_state_id']):
        raise ValueError('Proposal must name a distinct development initial state')
    if 'budget_checkpoint' in proposal:
        raise ValueError('Recursive checkpoint proposal is unsupported')
    protocol = copy.deepcopy(protocol)
    protocol['first_batch'] = [copy.deepcopy(proposal)]
    source_row = _inside(project, stream / case_id / 'RESULT.json', 'Case result')
    row_bytes = source_row.read_bytes()
    if json.loads(row_bytes) != json.loads(json.dumps(row, default=str)):
        raise ValueError('Current row differs from saved case RESULT')
    campaign_status = _inside(project, campaign / 'STATUS.json', 'Campaign status')
    status = json.loads(campaign_status.read_text())
    campaign_started = status['started_at']
    if (type(campaign_started) not in (int, float) or not math.isfinite(campaign_started)
            or not 0 < campaign_started <= time.time()):
        raise ValueError('Invalid actual campaign start')
    model = _inside(project, campaign / 'MODEL_CONFIG.json', 'Model config')
    output = _inside(project, stream / ('budget_checkpoint_' + case_id), 'Output')
    if accepted is not None:
        accepted = _inside(project, accepted, 'Accepted snapshot')
        if output.is_relative_to(accepted):
            raise ValueError('Checkpoint output cannot modify accepted snapshot')
    output.mkdir(exist_ok=False)
    receipt_path = output / 'RECEIPT.json'
    receipt = {
        'version': VERSION, 'case_id': case_id,
        'campaign': str(campaign.relative_to(project)), 'stream': str(stream.relative_to(project)),
        'stream_elapsed_seconds_before_checkpoint': elapsed_stream_seconds,
        'campaign_started_at': campaign_started,
        'additional_validation_pairs': 0, 'extra_pair_lifecycle_seconds': 0.0,
        'actual_worker_dispatch': False, 'model_calls': 0, 'robot_episodes': 0,
        'learning_gain_established': False, 'measured_saving_seconds': None,
        'sources': {'template': _binding(project, template_path),
                    'case_result': _binding(project, source_row),
                    'campaign_status': _binding(project, campaign_status),
                    'model_config': _binding(project, model)},
        'current_case_used_as_paired_evidence': False,
        'scope': 'Allocation only. Snapshot is context binding; current single execution is not paired evidence. '
                 'No model, container, promotion, or mandatory-task/evaluation scheduling occurs here.',
    }
    try:
        (output / 'ROW_SNAPSHOT.json').write_bytes(row_bytes)
        _save(output / 'PROTOCOL.json', protocol)
        (output / 'MODEL_CONFIG.json').write_bytes(model.read_bytes())
        _save(output / 'STATUS.json', {'status': 'ALLOCATION_CHECK', 'started_at': campaign_started,
                                     'source': str(campaign_status.relative_to(project))})
        receipt['sources']['row_snapshot'] = _binding(project, output / 'ROW_SNAPSHOT.json')
        receipt['sources']['derived_protocol'] = _binding(project, output / 'PROTOCOL.json')
        if accepted is None:
            receipt.update(status='DEFERRED_NO_ACCEPTED_SOURCE', reason='no_admitted_snapshot',
                           accepted_snapshot=None, allocation=None)
        else:
            seal = validate_snapshot(accepted)
            seal_sha = _sha(accepted / SEAL)
            next_source = row.get('experience_admission', {}).get('next_source')
            if next_source is None or _inside(project, next_source, 'Admission next source') != accepted:
                raise ValueError('Accepted snapshot differs from case admission next_source')
            receipt['accepted_snapshot'] = {'path': str(accepted.relative_to(project)),
                'seal_sha256': seal_sha, 'source_case': seal.get('source_case'),
                'seal_version': seal['version'], 'files_count': len(seal['files'])}
            allocation = continuation_scoped(project, output, protocol, 0)
            if allocation.get('status') not in ('NOT_DISPATCHED', 'DISPATCHED'):
                raise ValueError('Unsupported upstream allocation status')
            if allocation.get('paid_current_campaign_pair_seconds') != 0:
                raise ValueError('Checkpoint unexpectedly contains paid pair executions')
            normalized = copy.deepcopy(allocation)
            authorized = allocation['status'] == 'DISPATCHED'
            normalized['upstream_allocation_status'] = normalized['status']
            normalized['status'] = 'AUTHORIZED_PENDING' if authorized else 'NOT_DISPATCHED'
            normalized['actual_worker_dispatch_is_next_loop'] = False
            normalized['iteration_action'] = 'no_executor_bound_continue_mandatory_stream'
            receipt.update(status=normalized['status'], reason=allocation.get('allocation_reason',
                allocation.get('decision', {}).get('reason')), allocation=normalized)
            receipt['ledger_sha256'] = digest(allocation['ledger'])
            if validate_snapshot(accepted) != seal or _sha(accepted / SEAL) != seal_sha:
                raise ValueError('Accepted snapshot changed during checkpoint')
        if _sha(template_path) != config['template_sha256'] or source_row.read_bytes() != row_bytes:
            raise ValueError('Checkpoint source changed during read')
    except Exception as exc:
        receipt.update(status='CHECKPOINT_FAILED', reason=str(exc), error_type=type(exc).__name__)
        receipt.update(checkpoint_wall_seconds=time.perf_counter()-started,
                       checkpoint_cpu_seconds=time.process_time()-cpu_started)
        _save(receipt_path, receipt)
        raise
    receipt.update(checkpoint_wall_seconds=time.perf_counter()-started,
                   checkpoint_cpu_seconds=time.process_time()-cpu_started)
    _save(receipt_path, receipt)
    return _brief(receipt, receipt_path)
