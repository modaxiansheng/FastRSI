"""Explicit fixed controlled400 evaluation worker; no source/learning update."""
"""One isolated full-OS development worker for the staged-flow study."""
import hashlib,json,shutil,socket,sys,time,traceback,signal,re
from pathlib import Path
import orchestrate_full_os as life
from native_service_config import configure_qwen,model_env


def bind_planning_entry(campaign, protocol, arm, project):
    """Keep original Agent planning available alongside public construction."""
    entries=protocol.get('planning_entries')
    if entries is not None and (not isinstance(entries,dict) or arm not in entries):
        raise ValueError('Explicit planning entries must include the selected arm')
    entry='public_session_flow' if entries is None else entries[arm]
    if entry not in ('native_agent','public_session_flow'):
        raise ValueError('Unknown planning entry')
    if entry=='native_agent':return {'planning_entry':'native_agent'}
    flow=(Path(campaign)/'FLOW.json').resolve();project=Path(project).resolve()
    if not flow.is_relative_to(project):raise ValueError('Flow must remain within project')
    return {'planning_entry':entry,
            'session_flow_artifact':str(Path('/project')/flow.relative_to(project)),
            'session_flow_sha256':hashlib.sha256(flow.read_bytes()).hexdigest()}


def bind_horizon_commitment(protocol, arm):
    mode=protocol.get('arm_modes',{}).get(arm)
    specs=protocol.get('horizon_commitments',{})
    if not isinstance(specs,dict):raise ValueError('Horizon commitments must be an arm map')
    if arm not in specs:
        if mode=='fixed_commitment':raise ValueError('Explicit H/W binding missing for arm')
        return {}
    if protocol.get('control_kind')!='feedback_window' or mode!='fixed_commitment':
        raise ValueError('H/W binding requires explicit fixed_commitment feedback controller')
    spec=specs[arm]
    if not isinstance(spec,dict) or set(spec)!={'inference_horizon','commit_steps'}:
        raise ValueError('H/W binding requires only inference_horizon and commit_steps')
    from fastrsi.feedback_window import commitment_binding
    return commitment_binding(spec,mode)


def bind_task_snapshot(campaign, protocol, arm, project):
    """Bind the exact protocol snapshot before ROSTER publication; read only."""
    snapshots=protocol.get('task_window_snapshots',{})
    expected=protocol.get('task_window_snapshot_sha256')
    if not isinstance(snapshots,dict) or ('task_window_snapshot_sha256' in protocol and not isinstance(expected,dict)):
        raise ValueError('Task snapshot bindings must be mappings')
    if arm not in snapshots:
        if expected is not None and arm in expected:
            raise ValueError('Task snapshot SHA has no artifact for arm '+arm)
        return {}
    name=snapshots[arm]
    if not isinstance(name,str) or not name.strip():
        raise ValueError('Task snapshot requires an artifact path')
    artifact=(Path(campaign)/name).resolve()
    project=Path(project).resolve()
    if not artifact.is_relative_to(project):
        raise ValueError('Task snapshot artifact must remain in project')
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    if expected is not None:
        bound=expected.get(arm)
        if not isinstance(bound,str) or not re.fullmatch(r'[0-9a-f]{64}',bound):
            raise ValueError('Missing or invalid task snapshot SHA for arm '+arm)
        if digest!=bound:
            raise ValueError('Task snapshot SHA mismatch for arm '+arm)
    return {'task_window_artifact':str(Path('/project')/artifact.relative_to(project)),
            'task_window_sha256':digest}


def bind_boundary_probe(campaign, protocol, arm, project, case_count):
    probe=protocol.get('boundary_probe')
    if probe is None:return {}
    if (not isinstance(probe,dict) or case_count!=1 or
            protocol.get('control_kind')!='feedback_window' or protocol.get('arm_modes',{}).get(arm)!='fixed10' or
            any(protocol.get(k) for k in ('procedure_probe','stall_recovery_probe','prefix_memory_probe',
                                         'trial_prefix_probe','resident_services'))):
        raise ValueError('Boundary probe requires one isolated fixed10 case')
    if type(probe.get('enabled',{}).get(arm)) is not bool:
        raise ValueError('Boundary arm requires an explicit boolean')
    artifact=(Path(campaign)/'BOUNDARY.json').resolve();project=Path(project).resolve()
    if not artifact.is_relative_to(project):raise ValueError('Boundary artifact outside project')
    digest=hashlib.sha256(artifact.read_bytes()).hexdigest()
    if digest!=probe.get('sha256'):raise ValueError('Boundary artifact changed')
    return {'boundary_probe':{'artifact':str(Path('/project')/artifact.relative_to(project)),
            'sha256':digest,'enabled':probe['enabled'][arm]}}


def resident_spec(protocol, shift, cases):
    """Only exact service IDs and this worker's already-bound ports are legal."""
    spec=protocol.get('resident_services')
    if spec is None:
        if any(c.get('inference_horizon',10)!=10 for c in cases):
            raise ValueError('H20 worker requires an explicitly bound resident service')
        return None
    if not isinstance(spec,dict) or len(cases)!=1:raise ValueError('Resident worker requires one case and a service map')
    required={'policy_port','qwen_port','policy_container','qwen_container','session_id','max_inferences'}
    if not required.issubset(spec):raise ValueError('Incomplete resident service specification')
    for key,expected in [('policy_port',49500+shift),('qwen_port',49510+shift)]:
        if type(spec[key]) is not int or spec[key]!=expected:raise ValueError('Resident port does not match worker code binding: '+key)
    for key in ['policy_container','qwen_container']:
        if not isinstance(spec[key],str) or not re.fullmatch(r'[0-9a-f]{64}',spec[key]):raise ValueError('Resident services require full Docker IDs')
    if spec['policy_container']==spec['qwen_container']:raise ValueError('Policy and Qwen require distinct services')
    if not isinstance(spec['session_id'],str) or not re.fullmatch(r'[A-Za-z0-9_.:-]{1,160}',spec['session_id']):raise ValueError('Invalid resident Session identity')
    if type(spec['max_inferences']) is not int or spec['max_inferences']<1:raise ValueError('Invalid resident inference bound')
    expected=spec.get('inference_horizon',10)
    if (type(expected) is not int or expected not in (10,20)
            or type(cases[0].get('inference_horizon',10)) is not int
            or cases[0].get('inference_horizon',10)!=expected):
        raise ValueError('Resident inference horizon does not match case binding')
    return dict(spec)


def resident_roles(spec):
    return [('target','LAUNCH_target.json')] if spec else [('pi05','LAUNCH_pi05_v2.json'),('qwen','LAUNCH_qwen.json'),('target','LAUNCH_target.json')]


def inspect_borrowed_services(spec, cfg, command=None, health=None):
    """Read-only borrow: never add parent-owned services to life.active."""
    command=command or life.command;health=health or life.health
    rows=[]
    for role,key in [('policy','policy_container'),('qwen','qwen_container')]:
        cid=spec[key];port=spec[role+'_port']
        inspect=command(['docker','inspect',cid])
        if inspect.get('returncode')!=0:raise RuntimeError('Cannot inspect borrowed '+role)
        values=json.loads(inspect['stdout'])
        if len(values)!=1:raise RuntimeError('Ambiguous borrowed service')
        info=values[0]
        if info.get('Id')!=cid or info.get('State',{}).get('Running') is not True:raise RuntimeError('Borrowed service identity/running mismatch')
        pid=info.get('State',{}).get('Pid')
        if type(pid) is not int or pid<=0:raise RuntimeError('Borrowed service missing host PID')
        top=command(['docker','top',cid,'-eo','pid'])
        if top.get('returncode')!=0:raise RuntimeError('Cannot inspect borrowed service processes')
        owned={int(x.strip()) for x in top['stdout'].splitlines() if x.strip().isdigit()}
        listeners=command(['ss','-H','-ltnp',f'sport = :{port}'])
        listening={int(x) for x in re.findall(r'pid=(\d+)',listeners.get('stdout',''))}
        if listeners.get('returncode')!=0 or not listening or pid not in owned or not listening.issubset(owned):raise RuntimeError('Resident endpoint not bound to exact borrowed Docker process')
        if role=='qwen' and 'FASTRSI_AGENT_MODEL='+cfg['model_id'] not in info.get('Config',{}).get('Env',[]):raise RuntimeError('Borrowed Qwen model differs from bound config')
        health(port,limit=15)
        rows.append({'role':role,'container_id':cid,'name':info.get('Name'),'image_id':info.get('Image'),'port':port,
                     'host_pid':pid,'container_pids':sorted(owned),'listening_pids':sorted(listening),'health':'PASSED',
                     'ownership':'parent_owned_borrowed_not_added_to_worker_stop_all','startup_charged_by':'parent_service_pool',
                     'worker_usage_charged_by':'worker_lifecycle_and_parent_pool_reservation','docker_inspect':inspect,'docker_top':top,'listener_check':listeners})
    return {'status':'PASSED','checked_at':time.time(),'services':rows,'no_worker_model_launch_or_stop':True}


def wait_target(port, container_id, limit=60):
    """Resident models are ready immediately; the fresh target may still boot."""
    until=min(time.monotonic()+limit,time.monotonic()+max(0,life.deadline-time.time()))
    while time.monotonic()<until:
        try:
            with socket.create_connection(('127.0.0.1',port),timeout=min(1,max(.01,until-time.monotonic()))):return
        except OSError:
            receipt=life.command(['docker','inspect',container_id],timeout=5)
            if receipt.get('returncode')!=0:raise RuntimeError('Fresh target container cannot be inspected')
            info=json.loads(receipt['stdout'])
            if len(info)!=1 or info[0].get('Id')!=container_id or info[0].get('State',{}).get('Running') is not True:
                raise RuntimeError('Fresh target exited before becoming ready')
            time.sleep(min(.25,max(0,until-time.monotonic())))
    raise TimeoutError('Fresh target did not become ready within bound')

def capability_binding(campaign, protocol, case, project):
    """Optional shared catalog, bound to this actual normal development task."""
    spec=protocol.get('capability_interface')
    if spec is None:return {}
    if (set(spec)!={'sha256'} or case.get('mode')!='development' or
            case.get('native_experience_maintenance') is not True):
        raise ValueError('Capability requests require native development maintenance')
    path=campaign/'CAPABILITY_CATALOG.json';raw=path.read_bytes();cat=json.loads(raw)
    if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('Capability catalog changed')
    if (cat.get('session_id')!=case['id'] or cat.get('scope')!=dict(suite=case['suite'],
            task_id=case['task_id'],goal=case['task_description'])):
        raise ValueError('Capability catalog belongs to another actual task')
    return {'capability_interface':dict(artifact='/project/'+str(path.relative_to(project)),sha256=spec['sha256'])}


def live_control_env(protocol, cases, base):
    enabled=protocol.get('live_controller_trace',False)
    if type(enabled) is not bool:raise ValueError('Live recording flag must be boolean')
    if not enabled:return []
    if (len(cases)!=1 or cases[0].get('mode')!='development' or
            not protocol.get('goal_skill_probe') or not protocol.get('diagnostic_trace')):
        raise ValueError('Live controller capture requires one traced development goal task')
    return ['-e','FASTRSI_LIVE_CONTROLS='+base+'/live_controls']


def main():
    name,arm=sys.argv[1:];assert arm in ('N','F','R','S')
    label=name.replace('/','-')
    campaign=life.PROJECT/'runs'/name;group=campaign/arm
    life.CAMPAIGN=group;life.BASE='/project/runs/'+name+'/'+arm
    protocol=json.loads((campaign/'PROTOCOL.json').read_text());life.deadline=protocol['absolute_deadline']
    from fastrsi.controlled400_evaluation import PROTOCOL,install_task_scope,install_update_guard,validate_phase_case
    if protocol.get('evaluation_protocol')!=PROTOCOL or protocol.get('persistent_updates') is not False:
        raise ValueError('Explicit immutable evaluation protocol required')
    if protocol.get('phase_transport_source_recording'):
        raise ValueError('Evaluation cannot collect a new transport teacher')
    install_task_scope();install_update_guard()
    phase_probe=protocol.get('phase_transfer_probe')
    if not isinstance(phase_probe,dict) or set(phase_probe)!={'artifact','sha256'}:
        raise ValueError('Independent phase worker requires one explicitly frozen phase config')
    forbidden=('goal_skill_probe','trial_prefix_probe','prefix_memory_probe','stall_recovery_probe','live_controller_trace',
        'action_memory_probe','procedure_probe','boundary_probe','capability_interface')
    if any(protocol.get(k) for k in forbidden):raise ValueError('Cannot combine old probes with phase carrier')
    phase_path=campaign/phase_probe['artifact']
    if not phase_path.is_file() or hashlib.sha256(phase_path.read_bytes()).hexdigest()!=phase_probe['sha256']:
        raise ValueError('Phase configuration changed')
    from fastrsi.phase_transfer import validate_candidate,validate_config
    phase_config=json.loads(phase_path.read_text())
    host_artifact=Path(phase_config['artifact'].replace('/project',str(life.PROJECT),1))
    if not host_artifact.resolve().is_relative_to(life.PROJECT.resolve()):raise ValueError('Candidate escapes project')
    if hashlib.sha256(host_artifact.read_bytes()).hexdigest()!=phase_config['sha256']:raise ValueError('Phase candidate changed')
    phase_candidate=validate_candidate(json.loads(host_artifact.read_text()));validate_config(phase_config,phase_candidate)
    if not protocol.get('diagnostic_trace'):raise ValueError('Phase trace/images are required')
    protocol_commitment=bind_horizon_commitment(protocol,arm)
    plan=json.loads((campaign/'PLAN.json').read_text());cfg=json.loads((campaign/'MODEL_CONFIG.json').read_text())
    initial=campaign/'initial_experience'
    planning_binding=bind_planning_entry(campaign,protocol,arm,life.PROJECT)
    slot=protocol['slot'];arm_index=('N','F','R','S').index(arm);shift=protocol.get('port_offset',0)+slot*1000+arm_index*100
    gpu_base=protocol.get('gpu_bases',{}).get(arm,slot*4+arm_index*2)
    def port(p):return p+shift
    def relocate(argv,arm):
        if not shift:return argv
        out=[]
        for v in argv:
            for p in (49500,49502,49510,49520):v=v.replace(str(p),str(p+shift))
            for prefix in ('HIP_VISIBLE_DEVICES=','CUDA_VISIBLE_DEVICES='):
                if v==prefix+'0':v=prefix+str(gpu_base)
                elif v==prefix+'1':v=prefix+str(gpu_base+1)
            out.append(v)
        return out
    def interrupt(n,f):raise KeyboardInterrupt('Interrupted worker '+str(n))
    signal.signal(signal.SIGTERM,interrupt);signal.signal(signal.SIGINT,interrupt)
    group=campaign/arm;stream=group/'stream';stream.mkdir(parents=True,exist_ok=False);gstart=time.time();base=str(stream).replace(str(life.PROJECT),'/project')
    shutil.copyfile(campaign/'MODEL_CONFIG.json',group/'MODEL_CONFIG.json')
    cases=[]
    for item in plan['cases']:
        case={**item,'id':f"net_t{item['task_id']}_i{item['init_state_id']}",'session_intent':True,
              'require_discovery_reads':True,'require_clean_model_calls':True,'read_files_tool':True}
        case['terminal_prefetch']=False
        case['terminal_evidence_view']=bool(protocol.get('terminal_evidence_view',False))
        case.update(bind_boundary_probe(campaign,protocol,arm,life.PROJECT,len(plan['cases'])))
        if protocol.get('goal_skill_probe'):
            probe=protocol['goal_skill_probe']
            if (len(plan['cases'])!=1 or protocol.get('control_kind')!='feedback_window' or
                    protocol['arm_modes'][arm]!='fixed10' or type(probe['enabled'].get(arm)) is not bool or
                    any(protocol.get(k) for k in ('action_memory_probe','procedure_probe','boundary_probe',
                        'stall_recovery_probe','prefix_memory_probe','trial_prefix_probe'))):
                raise ValueError('Goal skill requires one isolated fixed10 case')
            artifact=campaign/'GOAL_SKILL.json'
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=probe['sha256']:raise ValueError('Goal skill changed')
            case['goal_skill_probe']=dict(artifact=str(artifact).replace(str(life.PROJECT),'/project'),
                sha256=probe['sha256'],enabled=probe['enabled'][arm])
        if protocol.get('action_memory_probe'):
            probe=protocol['action_memory_probe']
            if (len(plan['cases'])!=1 or protocol.get('control_kind')!='feedback_window' or
                    protocol['arm_modes'][arm]!='fixed10' or type(probe['enabled'].get(arm)) is not bool or
                    any(protocol.get(k) for k in ('boundary_probe','procedure_probe','stall_recovery_probe',
                        'prefix_memory_probe','trial_prefix_probe','resident_services','horizon_commitments',
                        'window_learning_snapshots','task_window_snapshots'))):
                raise ValueError('Action memory requires an isolated fixed10 case')
            artifact=campaign/'MEMORY.json'
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=probe['sha256']:raise ValueError('Memory changed')
            case['action_memory_probe']=dict(artifact=str(artifact).replace(str(life.PROJECT),'/project'),
                sha256=probe['sha256'],enabled=probe['enabled'][arm])
        if protocol.get('procedure_probe'):
            probe=protocol['procedure_probe']
            if (protocol.get('control_kind')!='feedback_window' or protocol['arm_modes'][arm]!='fixed10'
                    or any(protocol.get(k) for k in ('stall_recovery_probe','prefix_memory_probe','trial_reuse','resident_services'))):
                raise ValueError('Procedure probe requires isolated fixed10 execution')
            artifact=campaign/'PROCEDURE.json'
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=probe['sha256']:
                raise ValueError('Procedure artifact changed')
            case['procedure_probe']={'artifact':str(artifact).replace(str(life.PROJECT),'/project'),
                'sha256':probe['sha256'],'enabled':probe['enabled'][arm],
                'procedure_id':'remaining_subgoal' if probe['enabled'][arm] else 'full_goal','version':'1'}
        if protocol.get('stall_recovery_probe'):
            probe=protocol['stall_recovery_probe']
            if (protocol.get('prefix_memory_probe') or protocol.get('resident_services') or
                    protocol.get('control_kind')!='feedback_window' or
                    protocol['arm_modes'][arm]!='fixed10' or len(plan['cases'])!=1):
                raise ValueError('Stall probe requires a single isolated fixed10 case')
            artifact=campaign/'STALL.json'
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=probe['artifact_sha256']:
                raise ValueError('Stall artifact changed')
            case.update(stall_artifact=str(artifact).replace(str(life.PROJECT),'/project'),
                        stall_sha256=probe['artifact_sha256'],stall_enabled=probe['enabled'][arm])
        if protocol.get('prefix_memory_probe'):
            probe=protocol['prefix_memory_probe']
            if (protocol.get('control_kind')!='feedback_window' or
                    protocol['arm_modes'][arm]!='fixed10' or
                    arm not in probe['enabled'] or len(plan['cases'])!=1):
                raise ValueError('Prefix probe requires one fixed10 development case')
            artifact=campaign/'PREFIX.json'
            if hashlib.sha256(artifact.read_bytes()).hexdigest()!=probe['artifact_sha256']:
                raise ValueError('Prefix artifact SHA mismatch')
            case.update(prefix_artifact=str(artifact).replace(str(life.PROJECT),'/project'),
                        prefix_sha256=probe['artifact_sha256'],prefix_probe_enabled=probe['enabled'][arm])
        if protocol.get('control_kind')=='feedback_window':
            case['feedback_window_mode']=protocol['arm_modes'][arm]
            commitment=protocol_commitment
            if any(key in case and case[key]!=commitment.get(key) for key in ('inference_horizon','commit_steps')):
                raise ValueError('Plan H/W differs from explicit protocol binding')
            case.update(commitment)
            case['terminal_bind_report_destination']=True
            case.update(bind_task_snapshot(campaign,protocol,arm,life.PROJECT))
            if arm in protocol.get('window_learning_snapshots', {}):
                artifact=campaign/protocol['window_learning_snapshots'][arm]
                case.update(window_learning_artifact=str(artifact).replace(str(life.PROJECT),'/project'),
                            window_learning_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest())
        elif arm=='F':
            artifact=campaign/'STAGE_FLOW.json'
            if protocol.get('control_kind')=='action_window':
                case.update(action_window_artifact=str(artifact).replace(str(life.PROJECT),'/project'),
                            action_window_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            action_window_probe=case['mode']=='development')
            else:
                case['placement_recovery']=True
                case.update(stage_flow_artifact=str(artifact).replace(str(life.PROJECT),'/project'),
                            stage_flow_sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),
                            stage_flow_probe=case['mode']=='development')
        for key in ('planning_entry','session_flow_artifact','session_flow_sha256'):
            if key in case and case[key]!=planning_binding.get(key):
                raise ValueError('Case planning path differs from explicit group binding')
        case.update(planning_binding)
        if protocol_commitment:
            from fastrsi.feedback_window import commitment_binding
            commitment_binding(case,case['feedback_window_mode'])
        case.update(capability_binding(campaign,protocol,case,life.PROJECT))
        cases.append(case)
    if len(cases)!=1:raise ValueError('Phase worker requires exactly one case')
    phase_case=cases[0]
    if (phase_case['id']!=phase_config['root_session'] or phase_case['suite']!=phase_config['suite']
            or phase_case['task_id']!=phase_config['task_id'] or phase_case['task_description']!=phase_config['goal']
            or phase_case['max_steps']!=phase_config['max_steps'] or phase_case.get('feedback_window_mode')!='fixed10'
            or phase_case['init_state_id']!=phase_config['init_state_id']
            or phase_case['mode']!='evaluation'):
        raise ValueError('Phase configuration belongs to another evaluation case')
    validate_phase_case(phase_case,phase_config)
    phase_env=['-e','FASTRSI_CONTROLLED400_EVALUATION='+PROTOCOL,'-e','FASTRSI_PHASE_CONFIG='+str(phase_path).replace(str(life.PROJECT),'/project'),
               '-e','FASTRSI_PHASE_CONFIG_SHA256='+phase_probe['sha256']]
    resident=resident_spec(protocol,shift,cases)
    trial=protocol.get('trial_prefix_probe')
    trial_env=[]
    if trial:
        if not resident or len(cases)!=1 or cases[0]['mode']!='development' or cases[0].get('feedback_window_mode')!='fixed10':
            raise ValueError('Trial reuse requires one resident fixed10 development case')
        if protocol.get('prefix_memory_probe') or protocol.get('stall_recovery_probe'):
            raise ValueError('Cannot combine trial and online intervention probes')
        path=campaign/'TRIAL.json'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=trial['sha256']:
            raise ValueError('Trial specification changed')
        trial_env=['-e','FASTRSI_TRIAL_SPEC='+str(path).replace(str(life.PROJECT),'/project')]
    life.save(group/'ROSTER.json',cases);g={'arm':arm,'status':'RUNNING','started_at':gstart};life.save(group/'STATUS.json',g)
    goal_env=[]
    live_env=live_control_env(protocol,cases,base)
    if protocol.get('goal_skill_probe'):
        spec={**cases[0]['goal_skill_probe'],'root_session':cases[0]['id']}
        life.save(group/'GOAL_TARGET.json',spec)
        goal_env=['-e','FASTRSI_GOAL_CONFIG='+str(group/'GOAL_TARGET.json').replace(str(life.PROJECT),'/project')]
    try:
        if resident:
            borrowed=inspect_borrowed_services(resident,cfg)
            borrowed['session_id']=resident['session_id'];borrowed['borrow_started_at']=time.time()
            life.save(stream/'BORROWED_SERVICES.json',borrowed)
        else:shutil.copytree(life.PROJECT/'models/openpi_data',stream/'tokenizer_cache')
        target_container=None
        for role,receipt in resident_roles(resident):
            argv=json.loads((life.OLD/receipt).read_text())['argv'];argv[argv.index('--name')+1]='yuchen-fastrsi-'+label+'-'+arm.lower()+'-'+role
            argv=[x.replace(str(life.OLD),str(stream)).replace('/project/runs/pilot_20261001',base) for x in argv]
            if role=='pi05':
                argv=[x.replace(str(life.PROJECT/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv];assert '--torch-seed 7' in argv[-1]
            if role=='qwen':argv=configure_qwen(argv,cfg,life)
            if role=='target':argv=[x.replace('/project/scripts/instrument_target.py',life.BASE+'/code/scripts/instrument_target.py').replace(':/project/scripts',':'+life.BASE+'/code/scripts') for x in argv]
            if role=='target' and protocol.get('diagnostic_trace'):
                pos=argv.index('-e');argv[pos:pos]=['-e','FASTRSI_STAGE_TRACE='+base+'/stage_trace']
            if role=='target' and goal_env:
                if not protocol.get('diagnostic_trace'):raise ValueError('Goal skill actual-action trace required')
                argv=[x.replace('/instrument_target.py','/instrument_goal_target.py') for x in argv]
                pos=argv.index('-e');argv[pos:pos]=goal_env
            if role=='target' and live_env:
                pos=argv.index('-e');argv[pos:pos]=live_env
            if role=='target' and trial:
                argv=[x.replace('/instrument_target.py','/instrument_trial_target.py') for x in argv]
                pos=argv.index('-e');argv[pos:pos]=trial_env
            if role=='target':
                argv=[x.replace('/instrument_target.py','/instrument_phase_transfer_target.py') for x in argv]
                pos=argv.index('-e');argv[pos:pos]=phase_env
            argv=relocate(argv,arm);launched=life.launch(role,argv,stream)
            if role=='target':target_container=launched
        life.health(port(49510));life.health(port(49500))
        wait_target(port(49502),target_container)
        g['startup_seconds']=time.time()-gstart
        source=str(initial).replace(str(life.PROJECT),'/project')
        remaining=int(life.deadline-time.time()-30);assert remaining>60
        resident_env=[]
        runner_script='run_native_baseline.py'
        if protocol.get('goal_skill_probe'):
            runner_script='run_goal_skill_native_baseline.py'
        if protocol.get('action_memory_probe'):
            runner_script='run_action_memory_native_baseline.py'
        if protocol.get('procedure_probe'):
            runner_script='run_procedure_native_baseline.py'
        if protocol.get('boundary_probe'):
            runner_script='run_boundary_native_baseline.py'
        if protocol.get('stall_recovery_probe'):
            runner_script='run_stall_native_baseline.py'
        if protocol.get('prefix_memory_probe'):
            if resident:raise ValueError('Prefix probe resident RNG accounting not implemented')
            runner_script='run_prefix_native_baseline.py'
        if resident:
            runner_script='run_resident_goal_baseline.py' if protocol.get('goal_skill_probe') else 'run_resident_native_baseline.py'
            resident_env=['-e','FASTRSI_RESIDENT_POLICY_PORT='+str(resident['policy_port']),
                          '-e','FASTRSI_RESIDENT_SESSION_ID='+resident['session_id'],
                          '-e','FASTRSI_RESIDENT_INFERENCE_HORIZON='+str(resident.get('inference_horizon',10)),
                          '-e','FASTRSI_RESIDENT_MAX_INFERENCES='+str(resident['max_inferences'])]
        if trial:runner_script='run_trial_native_baseline.py'
        runner_script='run_controlled400_evaluation_baseline.py'
        resident_env+=phase_env
        argv=life.cpu_args('yuchen-fastrsi-'+label+'-'+arm.lower()+'-runner',stream)+model_env(cfg)+resident_env+trial_env+['-e','FASTRSI_NATIVE_RUN='+base,'-e','FASTRSI_CONSUME_TERMINAL=1','-e','FASTRSI_EXPERIENCE_ADMISSION=guarded_v1','-e','FASTRSI_INITIAL_EXPERIENCE='+source,
             life.IMAGE,'timeout',str(remaining),'bash','-lc','source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+life.BASE+'/code/scripts/'+runner_script]
        cid=life.launch('runner',argv,stream);life.wait_exit(cid,remaining+5)
        if resident:
            lifecycle=json.loads((stream/'RESIDENT_SESSION.json').read_text())
            assert lifecycle['status']=='COMPLETED' and lifecycle['release_confirmed'] is True, 'Resident Session not cleanly released'
        report=json.loads((stream/'RESULT.json').read_text())
        assert report['status']=='COMPLETED' and len(report['cases'])==1
        assert sum(len(row['outcomes']) for row in report['cases'])<=3
        for row in report['cases']:
            assert row['chain_status']=='PASSED'
            if row['case']['mode']=='development':assert 'experience_admission' in row
            assert row['terminal_report']['read_coverage']['complete']
            assert row['case']['mode']=='evaluation' and row['experience_before']==row['experience_after']
            assert row['controlled400_evaluation']['status']=='PASSED' and 'experience_admission' not in row
            # The original runner copies the entire accepted workspace, including
            # historical runtime receipts. Only new output paths outside that
            # frozen inherited workspace can be mistaken for fresh promotion.
            assert not any(p.name in ('NEXT_POINTER.json','ADMISSION.json','RECEIPT.json')
                for p in stream.rglob('*') if p.is_file() and 'workspace' not in p.relative_to(stream).parts)
        g.update(status='COMPLETED',controller_seconds=report['seconds'],
                 outcomes=[{'case':row['case']['id'],'environment_success':row['environment_success'],
                            'verdicts':[o.get('verdict') for o in row['outcomes']]} for row in report['cases']])
    except BaseException:g.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        life.save(stream/'STOP_RECEIPT.json',life.stop_all())
        if resident and (stream/'BORROWED_SERVICES.json').exists():
            borrowed=json.loads((stream/'BORROWED_SERVICES.json').read_text());borrowed['borrow_finished_at']=time.time()
            borrowed['borrow_seconds']=borrowed['borrow_finished_at']-borrowed['borrow_started_at']
            life.save(stream/'BORROWED_SERVICES.json',borrowed)
        g.update(finished_at=time.time(),full_wall_seconds=time.time()-gstart);life.save(group/'STATUS.json',g);print(json.dumps(g),flush=True)

if __name__=='__main__':main()
