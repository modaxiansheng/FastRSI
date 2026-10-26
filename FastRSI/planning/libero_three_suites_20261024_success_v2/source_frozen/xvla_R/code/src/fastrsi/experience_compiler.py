"""Compile successful source experience into robot-only controller targets.

Compilation is automatic. A compiled candidate still needs transfer validation;
source success does not certify its applicability to another initial state.
Simulator object poses in the diagnostic are deliberately not copied.
"""
import hashlib,json

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def compile_success(result, trace, replay, *, provenance):
    if result.get('chain_status')!='PASSED' or result.get('environment_success') is not True:
        raise ValueError('Only real source task success may produce a candidate')
    if replay.get('status') not in ('EXACT_REPLAY','EXACT_REPLAY_RECHECKED') or replay.get('inversion_check') is not True:
        raise ValueError('Controller reconstruction was not verified')
    case=result['case'];sid=case['id'];steps=trace[-1]['step']
    if 'native_sampler_binding' in case:
        raise ValueError('Sampler source requires live compilation with bound initialization records')
    if not (case['mode']=='development' and 0<steps<=400 and trace[-1]['native_success'] is True):
        raise ValueError('Invalid source data role or terminal outcome')
    if replay['case']['id']!=sid or any(row['session_id']!=sid for row in trace):
        raise ValueError('Source Session mismatch')
    controls=replay['controls']
    if len(controls)!=steps or replay.get('controls_count')!=steps:
        raise ValueError('Incomplete source controller records')
    if set(provenance)!={'result_sha256','trace_sha256','replay_sha256'}:
        raise ValueError('Missing source provenance')
    if provenance['trace_sha256']!=replay['source_trace_sha256']:
        raise ValueError('Replay refers to another trace')
    waypoints=[{k:c[k] for k in ('target_pos','target_rot','gripper_command')} for c in controls]
    return dict(version=2,kind='native_success_controller_candidate',source_goal=case['task_description'],
                source_completed_step=steps,waypoints=waypoints,
                source_evidence=dict(**provenance,session_id=sid,suite=case['suite'],task_id=case['task_id'],
                    init_state_id=case['init_state_id'],native_success=True,waypoints_sha256=canonical_sha(waypoints)),
                transfer_status='UNVALIDATED',supervision='Native development success; semantic prose is not a target.',
                scope='Public robot commands and robot controller targets only; no source or target object poses. Exact source replay is distinct from transfer success.')

def compile_live_success(result, trace, live, *, provenance, initialization_records=None):
    """Compile goals captured during execution; this is not replay verification.

    File hashes are supplied by the artifact reader, as for compile_success.
    The terminal root Session and every actually executed action are additionally
    checked here. A final successful retry cannot certify a failed root trace.
    """
    from .live_controller_trace import action_f32
    if live.get('recording_role')!='online_execution':
        raise ValueError('Only online controller records may produce a live candidate')
    if result.get('chain_status')!='PASSED' or result.get('environment_success') is not True:
        raise ValueError('Only real source task success may produce a candidate')
    case=result['case'];sid=case['id']
    initialization = None
    if 'native_sampler_binding' in case:
        if initialization_records is None:
            raise ValueError('Sampler source requires bound initialization records; slot0 is not a fixed initial state')
        from .source_initialization_identity_v1 import validate_source_initialization
        initialization = validate_source_initialization(case, live.get('runtime_config', {}), **initialization_records)
    elif initialization_records is not None:
        raise ValueError('Sampler initialization records cannot relabel a fixed source')
    if case.get('mode')!='development' or not trace:
        raise ValueError('Only a development source may be compiled')
    steps=trace[-1].get('step')
    if type(steps) is not int or not 0<steps<=400 or trace[-1].get('native_success') is not True:
        raise ValueError('Invalid source terminal outcome')
    terminal=result.get('native_terminal_by_session',{}).get(sid,{})
    if (terminal.get('session_id')!=sid or terminal.get('native_success') is not True or
            terminal.get('steps')!=steps):
        raise ValueError('The successful root Session is not bound')
    if (live.get('version')!=1 or live.get('kind')!='live_controller_trace' or
            live.get('provenance_mode')!='live_controller_v1' or live.get('status')!='LIVE_RECORDED' or
            live.get('session_id')!=sid or live.get('terminal_done') is not True or
            live.get('terminal_native_success') is not True):
        raise ValueError('Incomplete or mismatched live recording')
    if set(provenance)!={'result_sha256','trace_sha256','controller_trace_sha256'}:
        raise ValueError('Missing live source provenance')
    if (provenance['trace_sha256']!=live.get('source_trace_sha256') or
            canonical_sha(trace)!=live.get('source_trace_rows_sha256')):
        raise ValueError('Live recording refers to another trace')
    rc=live.get('runtime_config',{})
    if (rc.get('benchmark_name')!=case['suite'] or rc.get('task_id')!=case['task_id'] or
            rc.get('init_state_id')!=case['init_state_id'] or rc.get('control_mode')!='relative' or
            rc.get('max_steps')!=case.get('max_steps',400)):
        raise ValueError('Source runtime configuration differs')
    if (canonical_sha(live.get('controller_config'))!=live.get('controller_config_sha256') or
            live.get('controller_config',{}).get('use_delta') is not True or
            live['controller_config'].get('use_ori') is not True or
            live['controller_config'].get('impedance_mode')!='fixed' or
            live['controller_config'].get('position_limits') is not None or
            live['controller_config'].get('orientation_limits') is not None):
        raise ValueError('Unsupported or changed controller configuration')
    if (trace[0].get('method')!='reset' or trace[0].get('step')!=0 or
            any(row.get('session_id')!=sid for row in trace)):
        raise ValueError('The trace is not one complete root Session')
    actual=[];previous=0
    for row in trace[1:]:
        step=row.get('step')
        if type(step) is not int:raise ValueError('Malformed executed action boundary')
        count=step-previous
        if (row.get('method')!='action_chunk' or count<=0 or
                not isinstance(row.get('actions'),list) or len(row['actions'])<count):
            raise ValueError('Malformed executed action boundary')
        # A terminal native chunk can include an unexecuted requested suffix.
        actual.extend(action_f32(a) for a in row['actions'][:count]);previous=step
    controls=live.get('controls',[])
    if len(actual)!=steps or len(controls)!=steps or live.get('controls_count')!=steps:
        raise ValueError('Incomplete source controller records')
    for index,(action,control) in enumerate(zip(actual,controls)):
        if (control.get('session_id')!=sid or control.get('step_before')!=index or
                control.get('step_after')!=index+1 or control.get('actual_action')!=action or
                control.get('controller_action')!=action[:6] or
                control.get('gripper_command')!=action[-1] or
                control.get('controller_config_sha256')!=live['controller_config_sha256']):
            raise ValueError('A controller record does not match its actual action')
        if control.get('execution_origin') not in ('native_action','native_after_validity_exit','goal_skill_action'):
            raise ValueError('Unbound execution branch')
    waypoints=[{k:c[k] for k in ('target_pos','target_rot','gripper_command')} for c in controls]
    if (canonical_sha(controls)!=live.get('controls_sha256') or
            canonical_sha(actual)!=live.get('actual_actions_sha256') or
            canonical_sha(waypoints)!=live.get('targets_sha256')):
        raise ValueError('Controller record or target digest mismatch')
    sources=live.get('source_code',{})
    if set(sources)!={'runtime_sha256','controller_sha256'}:
        raise ValueError('Missing controller code provenance')
    evidence=dict(**provenance,provenance_mode='live_controller_v1',session_id=sid,
        suite=case['suite'],task_id=case['task_id'],init_state_id=case['init_state_id'],native_success=True,
        waypoints_sha256=canonical_sha(waypoints),actual_actions_sha256=live['actual_actions_sha256'],
        controller_config_sha256=live['controller_config_sha256'],source_code=sources,
        controls_sha256=live['controls_sha256'],controller_jsonl_sha256=live['controller_jsonl_sha256'],
        execution_origins=sorted(set(c['execution_origin'] for c in controls)))
    if initialization is not None:
        evidence['init_state_id'] = None
        evidence['initialization_source'] = 'native_sampler_extension'
        evidence['initialization'] = initialization
    value=dict(version=2,kind='native_success_controller_candidate',source_goal=case['task_description'],
        source_completed_step=steps,waypoints=waypoints,source_evidence=evidence,transfer_status='UNVALIDATED',
        supervision='Native development success and goals captured during actual execution.',
        scope='Robot commands and actual controller targets only; no object states. Live recording, not exact replay or transfer validation.')
    return validate_compiled(value)

def validate_compiled(value):
    import math
    from .retarget_geometry import pose
    if value.get('version')!=2 or value.get('kind')!='native_success_controller_candidate':
        raise ValueError('Unknown compiled skill format')
    waypoints=value.get('waypoints',[]);n=len(waypoints);e=value.get('source_evidence',{})
    if not 0<n<=400 or value.get('source_completed_step')!=n or e.get('native_success') is not True:
        raise ValueError('Invalid source completion binding')
    if not isinstance(value.get('source_goal'),str) or not value['source_goal'].strip():
        raise ValueError('Missing full task goal')
    mode=e.get('provenance_mode','exact_replay')
    if mode=='exact_replay':
        hashes=('result_sha256','trace_sha256','replay_sha256','waypoints_sha256')
    elif mode=='live_controller_v1':
        if 'replay_sha256' in e:raise ValueError('Live provenance cannot claim exact replay')
        hashes=('result_sha256','trace_sha256','controller_trace_sha256','waypoints_sha256',
                'actual_actions_sha256','controller_config_sha256','controls_sha256','controller_jsonl_sha256')
        source=e.get('source_code',{})
        if set(source)!={'runtime_sha256','controller_sha256'}:raise ValueError('Missing controller code provenance')
        for h in source.values():
            if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
                raise ValueError('Malformed controller code hash')
    else:raise ValueError('Unknown candidate provenance mode')
    for k in hashes:
        h=e.get(k)
        if not isinstance(h,str) or len(h)!=64 or any(c not in '0123456789abcdef' for c in h):
            raise ValueError('Malformed provenance hash')
    if e['waypoints_sha256']!=canonical_sha(waypoints):raise ValueError('Compiled waypoints changed')
    for waypoint in waypoints:
        if set(waypoint)!={'target_pos','target_rot','gripper_command'}:raise ValueError('Non-robot fields in compiled waypoint')
        pose(waypoint['target_pos'],waypoint['target_rot'])
        if not math.isfinite(waypoint['gripper_command']):raise ValueError('Nonfinite gripper command')
    return value
