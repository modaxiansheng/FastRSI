"""Opt-in cumulative budget gate for the existing feedback-window orchestrator."""
import hashlib,json
import math,time,sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastrsi import window_budget_v2
from fastrsi.validation_budget import digest
from window_budget_evidence import build_ledger, arm_groups

SCOPED_VERSION = 'window_budget_scoped_v3'


def scoped_gate_enabled(protocol):
    return protocol.get('budget_policy_version') == SCOPED_VERSION


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('Invalid ' + name)
    return value


def recheck_dispatch_deadline(receipt, absolute_deadline, *, now=None):
    """Recheck after CPU/preflight work; never revive a rejected allocation."""
    checked_at = time.time() if now is None else _positive(now, 'current time')
    remaining = _positive(absolute_deadline, 'absolute deadline') - checked_at
    result = dict(receipt)
    result['prelaunch_remaining_wall_seconds'] = max(0, remaining)
    if result['status'] == 'DISPATCHED' and remaining < 850:
        result.update(status='NOT_DISPATCHED', allocation_reason='prelaunch_wall_budget_insufficient',
                      actual_worker_dispatch_is_next_loop=False,
                      iteration_action='skip_proposal_continue_stream')
    return result


def continuation_scoped(root, run, protocol, index, *, now=None):
    """Read paid task-scoped evidence before *every* proposed validation pair.

    This schedules development validation only. Skipping a pair does not skip
    deployment tasks or the shared final evaluation. It neither promotes a
    candidate nor claims a counterfactual cost saving.
    """
    root, run = Path(root).resolve(), Path(run).resolve()
    if not scoped_gate_enabled(protocol):
        raise ValueError('Explicit scoped budget version required')
    if type(index) is not int or not 0 <= index < len(protocol['first_batch']):
        raise ValueError('Invalid proposal index')
    if protocol.get('active_arms') != ['N', 'F'] or protocol.get('arms') != {'N': 'fixed10', 'F': 'fixed5'}:
        raise ValueError('Scoped validation requires real fixed10/fixed5 arms')
    if protocol.get('sampling_steps') != 10:
        raise ValueError('Only the bound native H10 sampling10 candidate is supported')
    case = protocol['first_batch'][index]
    overrides = ('task_window_', 'window_learning_', 'staged_flow_', 'stage_flow_', 'action_window_')
    if any(key.startswith(overrides) for key in case):
        raise ValueError('Fixed validation case cannot carry a learned execution override')
    if case['mode'] != 'development' or case['suite'] != 'libero_10' or case['max_steps'] != 520:
        raise ValueError('Only declared LIBERO development validation is allocatable')
    clock = time.time() if now is None else _positive(now, 'current time')
    status = json.loads((run / 'STATUS.json').read_text())
    elapsed = clock - _positive(status['started_at'], 'campaign start')
    if elapsed < 0:
        raise ValueError('Clock precedes campaign start')
    wall_limit = _positive(protocol['lifecycle_budget_seconds'], 'wall budget')
    wall_limit = min(2100, wall_limit)  # Existing orchestrator absolute execution ceiling.
    allocation_limit = _positive(protocol['validation_budget_pair_seconds'], 'allocation budget')
    run_source = str((run / 'PROTOCOL.json').relative_to(root))
    run_binding = {run_source: sha(run / 'PROTOCOL.json')}
    model_sha = sha(run / 'MODEL_CONFIG.json')
    candidate = digest({'operation': 'fixed5_vs_fixed10', 'model_sha256': model_sha,
                        'sampling_steps': 10, 'inference_horizon': 10})
    specs = list(protocol.get('budget_history_specs', []))
    paid_current = 0.0
    completed = []
    # Account for all prior tasks, not only evidence matching the next goal.
    for i in range(index):
        pair = run / ('pair' + str(i))
        if not pair.exists():
            receipt = json.loads((run / ('CONTINUATION_' + str(i) + '.json')).read_text())
            if receipt.get('status') != 'NOT_DISPATCHED':
                raise ValueError('Missing previously authorized pair')
            continue
        if (pair / 'ALLOCATION_CANCELLED.json').exists():
            cancelled=json.loads((pair / 'ALLOCATION_CANCELLED.json').read_text())
            if cancelled.get('status')!='NOT_DISPATCHED' or any((pair / a / 'DISPATCH.json').exists() for a in ('N','F')):
                raise ValueError('Cancellation conflicts with actual dispatch')
            continue
        for arm in ('N', 'F'):
            arm_status = json.loads((pair / arm / 'STATUS.json').read_text())
            paid_current += _positive(arm_status['full_wall_seconds'], 'paid arm time')
            if arm_status['status'] != 'COMPLETED':
                raise ValueError('Prior failed execution needs explicit repair, not automatic retry')
        completed.append(str(pair.relative_to(root)))
        specs.append({'path': str(pair.relative_to(root)), 'arms': ['N', 'F'],
                      'protocol_group': protocol['budget_cost_protocol'],
                      'protocol_sources': run_binding, 'cost_compatible': True})
    selected, excluded, seen = [], [], set()
    for spec in specs:
        pair = (root / spec['path']).resolve()
        if not pair.is_relative_to(root) or pair in seen:
            raise ValueError('Invalid or duplicate evidence path')
        seen.add(pair)
        arms = spec.get('arms', ['N', 'F'])
        if len(arms) != 2 or len(set(arms)) != 2 or any(a not in ('N', 'F', 'R', 'S') for a in arms):
            raise ValueError('Invalid evidence arms')
        groups=arm_groups(root,spec)
        rows = [json.loads((g / 'stream/RESULT.json').read_text())['cases'][0]['case'] for g in groups]
        times = [json.loads((g / 'STATUS.json').read_text())['finished_at'] for g in groups]
        observed = max(_positive(t, 'observation time') for t in times)
        if observed > clock:
            raise ValueError('Future evidence cannot affect allocation')
        identity = ('mode', 'suite', 'task_id', 'task_description', 'max_steps')
        if any(any(row.get(k) != case.get(k) for k in identity) for row in rows):
            excluded.append({'path': spec['path'], 'reason': 'outside_exact_task_scope'})
        elif any(sha(g / 'MODEL_CONFIG.json') != model_sha for g in groups):
            excluded.append({'path': spec['path'], 'reason': 'different_model'})
        else:
            selected.append((observed, spec))
    selected.sort(key=lambda item: item[0])
    ledger = build_ledger(root, [s for _, s in selected], candidate_sha256=candidate,
        goal=case['task_description'], model_sha256=model_sha,
        protocol_group=protocol['budget_cost_protocol'], protocol_sources=run_binding,
        repeat_policy=protocol.get('budget_repeat_policy','reject'))
    request = {'phase': 'development', 'ledger_sha256': digest(ledger), 'candidate_sha256': candidate,
        'remaining_uses': protocol['remaining_uses'], 'max_quality_followups': protocol['max_quality_followups'],
        'allow_quality_confirmation': protocol['allow_quality_confirmation'],
        'remaining_budget_seconds': max(0, allocation_limit - paid_current),
        'estimated_pair_seconds': protocol['estimated_pair_seconds'], 'maximum_pair_seconds': 1700,
        'next_init_id': case['init_state_id']}
    # Repeated development instances are no longer independent new evidence.
    duplicate = case['init_state_id'] in [r['init_id'] for r in ledger['records']]
    if duplicate:
        decision = {'action': 'stop', 'reason': 'already_observed_development_instance',
                    'ledger_sha256': digest(ledger), 'promotion_authority': False}
    else:
        decision = window_budget_v2.decide(ledger, request)
    authorize = decision['action'] == 'validate_next'
    reason = decision['reason']
    if wall_limit - elapsed < 850:
        authorize = False
        reason = 'campaign_wall_budget_insufficient'
    return {'version': SCOPED_VERSION, 'status': 'DISPATCHED' if authorize else 'NOT_DISPATCHED',
        'decision': decision, 'allocation_reason': reason, 'ledger': ledger, 'request': request,
        'excluded_evidence': excluded, 'paid_current_campaign_pair_seconds': paid_current,
        'completed_current_pairs': completed, 'campaign_elapsed_wall_seconds': elapsed,
        'remaining_campaign_wall_seconds': max(0, wall_limit - elapsed),
        'allocation_cost_unit': 'sum_of_two_gpu_arm_lifecycle_seconds',
        'wall_cap_per_arm_seconds': 850, 'allocation_only': True,
        'budget_limit_scope': 'Execution admission caps; cleanup overrun remains charged in measured lifecycle, not a guaranteed hard lifecycle bound',
        'actual_worker_dispatch_is_next_loop': authorize,
        'iteration_action': 'execute_pair' if authorize else 'skip_proposal_continue_stream',
        'measured_saving_seconds': None, 'learning_gain_established': False,
        'scope': 'Exact-task evidence reuse for development allocation; no contextual generalization or task/evaluation skipping'}

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def continuation_v2(root,run,protocol,index):
    root,run=Path(root),Path(run)
    if type(index) is not int or index<1:raise ValueError('Need a completed prior development round')
    if protocol['budget_policy_version']!='window_budget_v2':raise ValueError('Explicit v2 required')
    if protocol['maximum_pair_seconds']!=1700:raise ValueError('Existing orchestrator enforces two 850-second arm caps')
    next_case=protocol['first_batch'][index]
    if next_case['mode']!='development':raise ValueError('No evaluation feedback allocation')
    run_source=str((run/'PROTOCOL.json').relative_to(root))
    specs=list(protocol.get('budget_history_specs',[]))
    current=[]
    for i in range(index):
        spec={'path':str((run/('pair'+str(i))).relative_to(root)), 'arms':['N','F'],
              'protocol_group':protocol['budget_cost_protocol'],
              'protocol_sources':{run_source:sha(run/'PROTOCOL.json')},'cost_compatible':True}
        current.append(spec);specs.append(spec)
    ledger=build_ledger(root,specs,candidate_sha256=protocol['task_window_sha256'],
        goal=next_case['task_description'],model_sha256=sha(run/'MODEL_CONFIG.json'),
        protocol_group=protocol['budget_cost_protocol'],protocol_sources={run_source:sha(run/'PROTOCOL.json')})
    current_ids={s['path'] for s in current}
    spent=sum(r['project_source_run_seconds'] for r in ledger['records'] if r['evidence_id'] in current_ids)
    request={'phase':'development','ledger_sha256':digest(ledger),'candidate_sha256':protocol['task_window_sha256'],
        'remaining_uses':protocol['remaining_uses'],'max_quality_followups':protocol['max_quality_followups'],
        'allow_quality_confirmation':protocol['allow_quality_confirmation'],
        'remaining_budget_seconds':max(0,protocol['lifecycle_budget_seconds']-spent),
        'estimated_pair_seconds':protocol['estimated_pair_seconds'],'maximum_pair_seconds':1700,
        'next_init_id':next_case['init_state_id']}
    decision=window_budget_v2.decide(ledger,request)
    dispatch=window_budget_v2.dispatch(ledger,request,decision,
        lambda req:{'authorized_next_loop_index':index,'wall_cap_per_arm_seconds':850})
    return {**dispatch,'ledger':ledger,'request':request,'paid_current_campaign_seconds':spent,
        'allocation_only':True,'actual_worker_dispatch_is_next_loop':dispatch['status']=='DISPATCHED',
        'scope':'Opt-in empirical continuation rule; historical prefix bound, no default promotion or measured learning gain'}
