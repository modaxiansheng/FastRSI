"""Prepare and explicitly dispatch one complete native-Agent sampler arm.

No launch during prepare/check. A fresh exact snapshot and actual SDK composition
receipt precede --execute --allow-real-physical-calls. The original Agent,
verification, recovery and evaluation freeze remain the executable implementation.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from types import SimpleNamespace

VERSION='native_sampler_agent_arm_v1'
PROJECT_REMOTE=Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
GOAL='stack the right bowl on the left bowl and place them in the tray'
IMAGE='sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
TARGET_IMAGE='sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text())
def save(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')
def manifest(root):
    result={}
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink():raise ValueError('Symlink snapshot input rejected')
        if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ('.pyc','.pyo'):result[p.relative_to(root).as_posix()]=sha(p)
    return result


def validate_sampler_partition(registration, phase, seed):
    from fastrsi.native_sampler_restore_v1 import registration_role
    role = registration_role(registration)
    if (phase not in ('development', 'evaluation') or type(seed) is not int
            or seed not in registration['seeds']
            or (phase == 'development') != (role == 'DEVELOPMENT_INPUT')):
        raise ValueError('Sampler seed/phase partition differs')
    return role


def registered_boundary(binding, component):
    """Forward an explicit development operation, never an ignored preference."""
    key = 'registered_program_boundary_v1'
    if key not in binding:
        return None
    value = binding[key]
    if (binding.get('mode') != 'development' or component is None
            or component.get('execution_contract') is not None
            or not isinstance(value, dict) or set(value) != {'version', 'step', 'decision'}
            or value.get('version') != key or type(value.get('step')) is not int
            or value['step'] != 120 or value.get('decision') not in ('continue', 'handoff')):
        raise ValueError('Invalid registered development boundary operation')
    return copy.deepcopy(value)

def ref(binding,key):
    value=binding[key];path=Path(value['path']).resolve()
    if sha(path)!=value['sha256']:raise ValueError('Bound input changed: '+key)
    return path

def declaration(workspace):
    import yaml
    text=(Path(workspace)/'SKILLRUNTIME.md').read_text();m=re.search(r'```yaml\s*\n(.*?)\n```',text,re.S)
    if not m:raise ValueError('Missing SkillRuntime YAML')
    rows=[r for r in yaml.safe_load(m.group(1))['skillruntimes'] if r['id']=='pi05_libero_remote']
    if len(rows)!=1:raise ValueError('Original skill not unique')
    return rows[0].get('output_contract',{}).get('action',{}).get('chunk',{}).get('native_goal_component')


def project_recorded_path(value, namespace=None):
    """Resolve only an explicitly declared /project mount; never rewrite evidence."""
    path=Path(value)
    if not path.is_absolute() or '..' in path.parts:
        raise ValueError('Recorded project path must be absolute and contained')
    if namespace is not None:
        if (not isinstance(namespace,dict) or set(namespace)!={'version','container_root','host_root'}
                or namespace['version']!='project_namespace_v1' or namespace['container_root']!='/project'):
            raise ValueError('Unsupported explicit project namespace')
        host=Path(namespace['host_root'])
        if not host.is_absolute() or '..' in host.parts:
            raise ValueError('Project host root must be absolute and contained')
        host=host.resolve()
        if path==Path('/project') or Path('/project') in path.parents:
            path=host/path.relative_to('/project')
        elif path!=host and host not in path.parents:
            raise ValueError('Recorded path is outside explicit project namespace')
        if path!=host and host not in path.parents:
            raise ValueError('Mapped project path escaped host root')
        for parent in (path,*path.parents):
            if parent==host:break
            if parent.is_symlink():raise ValueError('Symlinked project record rejected')
        if path.resolve()!=path:
            raise ValueError('Mapped project path aliases another location')
    return path.resolve()


def empirical_update_source(binding,workspace,decl):
    """Independently replay a non-LLM update; never relax the teacher branch."""
    import fastrsi.empirical_scope_update_v1 as algorithm
    import run_empirical_scope_update_native_v1 as updater
    import probe_native_goal_component_update_v1 as protocol
    result_path=ref(binding,'update_result');result=read(result_path)
    binding_path=ref(binding,'update_binding');campaign=binding_path.parent
    input_path=ref(binding,'update_input_manifest');code_path=ref(binding,'update_code_manifest')
    namespace=binding.get('project_namespace')
    if namespace is not None:
        for actual_path in (campaign,result_path,workspace):
            if project_recorded_path(str(actual_path),namespace)!=actual_path:
                raise ValueError('Actual empirical inputs differ from explicit project mount')
    if (binding_path.name!='BINDING.json' or input_path!=campaign/'INPUT_MANIFEST.json'
            or code_path!=campaign/'CODE_MANIFEST.json'):
        raise ValueError('Empirical update campaign references differ')
    if (result.get('version')!='empirical_scope_update_native_v1'
            or result.get('update_origin')!='EMPIRICAL_DEVELOPMENT_FEEDBACK'
            or result.get('status')!='COMPLETED_EMPIRICAL_SCOPE_UPDATE'
            or any(type(result.get(k)) is not int or result[k]!=0 for k in
                   ('model_calls','robot_episodes','policy_inference_calls','training_updates'))
            or any(result.get(k) is not False for k in
                   ('empirical_admission','teacher_adopted','distillation_performed','learning_net_benefit'))
            or any(result.get(k) is not True for k in
                   ('registry_version_changed','evaluation_updates_blocked','original_campaign_inputs_unchanged'))):
        raise ValueError('Actual non-LLM empirical update receipt incomplete')
    if workspace!=(result_path.parent/'update_session/workspace').resolve() or decl is None:
        raise ValueError('Empirical C must inherit actual updated development workspace')
    def bounded(root,relative):
        rel=Path(relative)
        if rel.is_absolute() or '..' in rel.parts:raise ValueError('Empirical manifest path escapes snapshot')
        path=root/rel
        if not path.is_file() or any(x.is_symlink() for x in (path,*path.parents) if x!=root):
            raise ValueError('Empirical input missing or symlinked')
        return path
    for relative,expected in read(input_path).items():
        if sha(bounded(campaign,relative))!=expected:raise ValueError('Empirical frozen input changed')
    code=read(code_path)
    for relative,expected in code.items():
        if sha(bounded(campaign/'code',relative))!=expected:raise ValueError('Empirical frozen code changed')
    for relative,module in [('src/fastrsi/empirical_scope_update_v1.py',algorithm),
                            ('scripts/run_empirical_scope_update_native_v1.py',updater)]:
        if relative not in code or sha(Path(module.__file__))!=code[relative]:
            raise ValueError('Recomputed empirical algorithm differs from executed frozen code')
    contract=updater.checked_inputs(campaign,binding['update_binding']['sha256'])
    if contract['input_manifest_sha256']!=binding['update_input_manifest']['sha256']:
        raise ValueError('Empirical input manifest does not match update binding')
    observations=algorithm.load_development_observations(campaign/'successor_package',**contract['evidence_refs'])
    catalogue_doc=read(campaign/'successor_package'/contract['evidence_refs']['catalogue']['path'])
    model_sha=sha(campaign/'MODEL_CONFIG.json')
    catalogue=protocol.finalize_semantics(catalogue_doc,model_sha)['candidates']
    actual_update=read(result_path.parent/'UPDATE.json')
    if result.get('update')!=actual_update:raise ValueError('Saved empirical update differs from result')
    # Current CLI begins at native. Do not imply implicit rollback or adoption
    # from another incumbent without its own verified lifecycle.
    recomputed=algorithm.select_update(observations,list(catalogue),incumbent='native')
    semantic=lambda value:{k:v for k,v in value.items() if k!='seconds'}
    if (semantic(actual_update)!=semantic(recomputed) or recomputed.get('withdrawal_required')
            or recomputed.get('status')!='UNVALIDATED_DEVELOPMENT_UPDATE'
            or recomputed['candidate'] not in catalogue):
        raise ValueError('Empirical choice does not follow bound development observations')
    chosen=recomputed['candidate'];decision=result['decision']
    if (decision.get('origin')!='EMPIRICAL_DEVELOPMENT_FEEDBACK_NOT_QWEN'
            or project_recorded_path(decision['workspace'],namespace)!=workspace or decision.get('candidate')!=chosen
            or decision.get('declaration')!=decl or catalogue[chosen]!=decl
            or decision.get('registry_sha256')!=sha(workspace/'SKILLRUNTIME.md')):
        raise ValueError('Actual empirical registry/choice differs')
    transactions=decision.get('transactions',[])
    if len(transactions)!=1:raise ValueError('Exactly one empirical native transaction required')
    transaction=transactions[0]
    if (transaction!=read(result_path.parent/'update_session/TRANSACTION.json')
            or transaction.get('candidate_id')!=chosen or transaction.get('status')!='UNVALIDATED_DRAFT_REGISTERED'
            or transaction.get('declaration')!=decl or transaction.get('registry_after_sha256')!=decision['registry_sha256']
            or transaction.get('registry_before_sha256')==transaction.get('registry_after_sha256')
            or transaction.get('robot_episodes')!=0 or transaction.get('empirical_admission') is not False
            or transaction.get('learning_gain_claimed') is not False):
        raise ValueError('Actual empirical transaction differs from UPDATE')
    state=read(workspace/protocol.STATE_REL)
    if state.get('component_is_registered') is not False or state['registry_sha256']['SKILLRUNTIME.md']!=transaction['registry_before_sha256']:
        raise ValueError('Empirical update did not begin from native incumbent')
    events=[json.loads(line) for line in (result_path.parent/'update_session/AGENT_EVENTS.jsonl').read_text().splitlines() if line.strip()]
    writes=[e for e in events if e.get('event')=='typed_component_transaction']
    if len(writes)!=1 or any(writes[0].get(k)!=v for k,v in transaction.items()):
        raise ValueError('Actual native transaction event missing or different')
    # Native transaction records do not carry TOOL_ORIGIN; the enclosing
    # actual tool call does, and its result must be this same transaction.
    commits=[e for e in events if e.get('event')=='tool' and e.get('origin')=='empirical_scope_update'
             and e.get('name')=='commit_goal_component']
    if len(commits)!=1 or json.loads(commits[0].get('result','null'))!=transaction:
        raise ValueError('Actual empirical commit tool provenance missing')
    required=tuple(dict.fromkeys((*protocol.REQUIRED_READS,protocol.MANIFEST_REL,protocol.STATE_REL,
        contract['evidence_refs']['catalogue']['path'],'configs/empirical_scope_update/UPDATE.json',
        *(value['execution_contract']['evidence']['path'] for value in catalogue.values() if 'execution_contract' in value))))
    def tool_params(event):
        params=event.get('params',{})
        if isinstance(params.get('path'),str) and Path(params['path']).is_absolute():
            params=dict(params,path=str(project_recorded_path(params['path'],namespace)))
        return params
    tracker=protocol.CompleteReads(workspace,required)
    for event in events:
        if event.get('event')=='tool' and event.get('origin')=='empirical_scope_update':
            tracker.observe(event.get('name'),tool_params(event),event.get('result'))
    # TARGETS/SKILLRUNTIME were read before the actual write. The immutable
    # inputs are independently verified; pre-write registry is bound above.
    immutable=[p for p in required if p not in ('SKILLRUNTIME.md',)]
    if not set(immutable).issubset(tracker.complete()) or not set(required).issubset(decision.get('complete_reads',[])):
        raise ValueError('Actual empirical complete input reads missing')
    selection=result['new_session'];child=(result_path.parent/'new_session/workspace').resolve()
    saved=read(result_path.parent/'new_session/SELECTION.json')
    if any(selection.get(k)!=v for k,v in saved.items()):raise ValueError('Readonly receipt differs from saved consumer')
    if (project_recorded_path(selection['workspace'],namespace)!=child or selection.get('selected_declaration')!=decl
            or selection.get('registry_bytes_inherited') is not True or selection.get('task_executed') is not False
            or selection.get('public_flow_status')!='QUEUED' or selection.get('queued')!=1 or selection.get('model_calls')!=0
            or selection.get('public_flow') is not True or not (child/'.FASTRSI_EVALUATION_CLOSED.json').is_file()
            or (child/'SKILLRUNTIME.md').read_bytes()!=(workspace/'SKILLRUNTIME.md').read_bytes()):
        raise ValueError('Actual H10 sampler readonly consumer/isolation missing')
    planned=selection['planned']
    if (planned['execution'].get('max_steps')!=400 or planned['execution'].get('replan_every_steps')!=10
            or planned['benchmark'].get('instance_id') is not None or planned['benchmark'].get('suite_id')!='libero_90'
            or planned['benchmark'].get('task_index')!=64 or planned['task_description']!=GOAL):
        raise ValueError('Readonly consumer changed actual H10/sampler protocol')
    current=read(child/protocol.STATE_REL)
    if (sha(child/protocol.STATE_REL)!=selection.get('current_component_state_sha256')
            or current.get('native_goal_component')!=decl or current.get('component_is_registered') is not True
            or current['registry_sha256']['SKILLRUNTIME.md']!=decision['registry_sha256']):
        raise ValueError('Readonly current-state evidence does not reflect actual update')
    child_events=[json.loads(line) for line in (result_path.parent/'new_session/AGENT_EVENTS.jsonl').read_text().splitlines() if line.strip()]
    reads=protocol.CompleteReads(child,('SKILLRUNTIME.md','TARGETS.md','REQUEST.json',protocol.STATE_REL))
    for event in child_events:
        if event.get('event')=='tool':reads.observe(event.get('name'),tool_params(event),event.get('result'))
    reads.require()
    outcomes=[e for e in child_events if e.get('event')=='session_flow_outcome']
    if len(outcomes)!=1 or outcomes[0].get('outcome',{}).get('status')!='QUEUED':
        raise ValueError('Readonly native queue evidence differs')
    # Confirm the immutable inherited inputs independently, allowing only the
    # newly computed current-state fact and native queue history to differ.
    staged=read(workspace/protocol.MANIFEST_REL)['staged_sha256']
    for relative,digest in staged.items():
        if sha(bounded(workspace,relative))!=digest or sha(bounded(child,relative))!=digest:
            raise ValueError('Readonly finish/background changed staged update input')
    for relative in (protocol.MANIFEST_REL,'configs/empirical_scope_update/UPDATE.json'):
        if (workspace/relative).read_bytes()!=(child/relative).read_bytes():
            raise ValueError('Readonly finish/background changed fixed update inputs')
    return dict(update_result_sha256=sha(result_path),algorithm_evidence_sha256=recomputed['evidence_sha256'],candidate=chosen)


def source_gate(binding):
    workspace=Path(binding['snapshot_workspace']).resolve()
    expected=read(ref(binding,'snapshot_manifest'))
    if manifest(workspace)!=expected:raise ValueError('Actual source workspace differs from pinned complete manifest')
    if (workspace/'.FASTRSI_EVALUATION_CLOSED.json').exists():raise ValueError('Use development snapshot, never a completed evaluation')
    decl=declaration(workspace)
    arm=binding['arm']
    if arm not in ('A_fixed_primitive','B_frozen_prior','C_real_agent_update','C_empirical_scope_update'):raise ValueError('Unknown arm')
    if arm=='B_frozen_prior' and decl is not None:raise ValueError('Frozen prior ablation expected no executable component')
    if arm=='A_fixed_primitive' and decl is None:raise ValueError('Strong fixed arm must contain the common primitive')
    if arm=='C_empirical_scope_update':
        empirical_update_source(binding,workspace,decl)
    if arm=='C_real_agent_update':
        result_path=ref(binding,'teacher_result');result=read(result_path)
        version=result.get('version')
        if version=='native_goal_learning_cycle_v1':
            if (result.get('phase')!='update' or result.get('provider_mode')!='ORIGINAL_MEASURED_QWEN' or
                result.get('status') not in ('COMPLETED_TYPED_DRAFT_CHANGE','COMPLETED_LEGAL_NO_COMPONENT_CHANGE')):
                raise ValueError('Actual cycle update evidence missing')
            expected_workspace=result_path.parent/'agent_session/workspace';decision=result['decision']
        elif version=='native_goal_component_update_probe_v1':
            partial=binding.get('source_status')=='PARTIAL_TYPED_COMMIT_WITH_SEPARATE_READONLY_RECEIPT'
            if result.get('status') not in ('COMPLETED_REAL_AGENT_COMPONENT_SELECTION','COMPLETED_LEGAL_NO_COMPONENT_CHANGE') and not (partial and result.get('status')=='FAILED'):
                raise ValueError('Actual typed proposal terminal result missing')
            # This older result schema has no provider_mode. Bind the real CLI
            # launch instead of inventing that absent evidence field.
            launch=read(ref(binding,'teacher_runner_launch'));argv=launch.get('argv',[])
            command=' '.join(argv)
            if (launch.get('returncode')!=0 or not re.fullmatch(r'[a-f0-9]{64}',launch.get('stdout','').strip()) or
                'probe_native_goal_component_update_v1.py' not in command or '--allow-real-model-calls' not in command):
                raise ValueError('Actual typed-probe model CLI launch missing')
            expected_workspace=result_path.parent/'proposal/workspace';decision=result['proposal']
        else:raise ValueError('Unsupported teacher result schema')
        recovery=None
        if binding.get('source_status')=='PARTIAL_TYPED_COMMIT_WITH_SEPARATE_READONLY_RECEIPT':
            if version!='native_goal_component_update_probe_v1' or result.get('status')!='FAILED':raise ValueError('Partial recovery requires original failed typed probe')
            from recover_native_typed_readonly_v1 import verify_partial, VERSION as recovery_version
            rp=ref(binding,'readonly_recovery_result');recovery=read(rp)
            if (recovery.get('version')!=recovery_version or recovery.get('status')!='PASSED' or
                recovery.get('source_status')!=binding['source_status'] or recovery.get('queued')!=1 or recovery.get('model_calls')!=0 or
                recovery.get('public_flow_status')!='QUEUED' or any(recovery.get(k) is not True for k in
                    ('registry_bytes_inherited','evaluation_updates_blocked','background_updates_blocked','source_inputs_unchanged','teacher_outputs_unchanged')) or
                any(recovery.get(k)!=0 for k in ('models','resets','actions'))):raise ValueError('Separate real CPU readonly receipt incomplete')
            current=verify_partial(Path(binding['teacher_campaign']),result_path.parent,ref(binding,'teacher_runner_launch'),
                result_sha=sha(result_path),launch_sha=binding['teacher_runner_launch']['sha256'],registry_sha=decision['registry_sha256'])
            ep=rp.parent/'PARTIAL_COMMIT_EVIDENCE.json'
            if sha(ep)!=recovery['partial_commit_evidence_sha256'] or read(ep)!=current:raise ValueError('Recovered typed evidence changed')
            if (recovery['teacher_result_sha256']!=sha(result_path) or recovery['teacher_launch_sha256']!=binding['teacher_runner_launch']['sha256'] or
                recovery['source_workspace_manifest']!=expected or recovery['registry_sha256']!=decision['registry_sha256'] or recovery['selected_declaration']!=decl):raise ValueError('Recovery bound to another actual proposal')
            if sha(rp.parent/'selection/AGENT_EVENTS.jsonl')!=recovery['events_sha256'] or sha(rp.parent/'PLANNED_SESSION.json')!=recovery['planned_session_sha256']:
                raise ValueError('Actual readonly queue evidence changed')
            import probe_native_goal_component_update_v1 as proposal_protocol
            recovered_workspace=rp.parent/'selection/workspace';state_path=recovered_workspace/proposal_protocol.STATE_REL
            if sha(state_path)!=recovery.get('current_component_state_sha256') or recovery.get('current_component_state_registered') is not True:
                raise ValueError('Recovered current native component state is not bound')
            state=read(state_path)
            if (state.get('component_is_registered') is not True or state.get('native_goal_component')!=decl or
                state.get('registry_sha256',{}).get('SKILLRUNTIME.md')!=decision['registry_sha256']):raise ValueError('Recovered state disagrees with actual proposal')
            tracker=proposal_protocol.CompleteReads(recovered_workspace,(proposal_protocol.STATE_REL,))
            for event in (json.loads(line) for line in (rp.parent/'selection/AGENT_EVENTS.jsonl').read_text().splitlines()):
                if event.get('event')=='tool' and event.get('origin')=='public_component_state_read':
                    tracker.observe(event.get('name'),event.get('params',{}),event.get('result'))
            tracker.require()
        elif result.get('evaluation_updates_blocked') is not True or result.get('original_campaign_inputs_unchanged') is not True:
            raise ValueError('Actual terminal teacher isolation evidence missing')
        if workspace!=expected_workspace.resolve():raise ValueError('C must inherit the actual update workspace')
        if sha(workspace/'SKILLRUNTIME.md')!=decision['registry_sha256'] or decl!=decision['declaration']:
            raise ValueError('C registry does not match actual teacher decision')
        selection=recovery if recovery is not None else result['new_session']
        if selection.get('selected_declaration')!=decl or not selection.get('registry_bytes_inherited'):
            raise ValueError('Real read-only next-Session choice differs')
        if not result.get('model_calls') or not list(result_path.parent.glob('*/*_call_*/RESPONSE.json')):
            raise ValueError('No real measured model outputs')
        if decl is None:return workspace,expected,None,'SKIPPED_REAL_TEACHER_NO_COMPONENT'
    import yaml
    session_text=(workspace/'SESSIONS.md').read_text()
    match=re.search(r'```yaml\s*\n(.*?)\n```',session_text,re.S)
    if not match:raise ValueError('Inherited native Session registry missing')
    rows=yaml.safe_load(match.group(1)).get('sessions',[])
    terminal={'succeeded','failed','replanned','timed_out','cancelled','rejected'}
    if any(row.get('status') not in terminal for row in rows):
        raise ValueError('Inherited nonterminal Session could dispatch unrelated work')
    evidence=read(ref(binding,'common_evidence_manifest'))
    for relative,digest in evidence.items():
        path=Path(relative)
        if path.is_absolute() or '..' in path.parts or sha(workspace/path)!=digest:
            raise ValueError('Common evidence unavailable in this arm: '+relative)
    return workspace,expected,decl,'ELIGIBLE'


def freeze_code(project,dest):
    roots=['scripts/run_native_sampler_goal_component_v1.py','scripts/instrument_native_sampler_draw_target_v1.py',
        'scripts/run_registered_goal_component_native.py','scripts/run_native_baseline.py','scripts/qwen_agent_server.py',
        'scripts/engine_overlay.py','scripts/orchestrate_full_os.py','scripts/native_service_config.py',
        'scripts/orchestrate_native_development_operation_v1.py','scripts/orchestrate_native_sampler_agent_arm_v1.py','scripts/recover_native_typed_readonly_v1.py',
        'scripts/run_empirical_scope_update_native_v1.py']
    def resolve(name):
        for base in (project/'src',project/'scripts',project):
            p=(base/Path(*name.split('.'))).with_suffix('.py')
            if p.is_file() and 'third_party' not in p.relative_to(project).parts:return p
    pending=[project/p for p in roots];seen=set()
    while pending:
        p=pending.pop()
        if p in seen:continue
        seen.add(p)
        for n in ast.walk(ast.parse(p.read_text())):
            if isinstance(n,ast.ImportFrom):
                q=None
                if n.level and n.module:
                    base=p.parent
                    for _ in range(n.level-1):base=base.parent
                    q=(base/Path(*n.module.split('.'))).with_suffix('.py')
                elif n.module:q=resolve(n.module)
                if q and q.is_file():pending.append(q)
            elif isinstance(n,ast.Import):
                for item in n.names:
                    q=resolve(item.name)
                    if q:pending.append(q)
    # Registered consumers also import components lazily by version. Preserve
    # the complete local package, rather than silently dropping an optional
    # module because the older host checkout did not reference it yet.
    seen.update((project/'src/fastrsi').rglob('*.py'))
    for p in seen:
        q=dest/p.relative_to(project);q.parent.mkdir(parents=True,exist_ok=True);before=sha(p);shutil.copyfile(p,q)
        if sha(q)!=before or sha(p)!=before:raise ValueError('Source changed while freezing')


def prepare(project,campaign,binding_path):
    project,campaign=Path(project).resolve(),Path(campaign).resolve()
    relative=campaign.relative_to(project).as_posix()
    if not re.fullmatch(r'[A-Za-z0-9_./-]+',relative) or campaign.exists():raise ValueError('Fresh ordinary campaign path required')
    binding=read(binding_path);workspace,source,decl,status=source_gate(binding)
    boundary = registered_boundary(binding, decl)
    campaign.mkdir(parents=True)
    if status!='ELIGIBLE':
        save(campaign/'REGISTRATION.json',dict(version=VERSION,status=status,arm=binding['arm'],models=0,resets=0));return read(campaign/'REGISTRATION.json')
    phase=binding['mode'];seed=binding['draw_seed']
    resources=binding['resources'];gpus=resources['gpus'];ports=resources['ports']
    if len(gpus)!=2 or len(set(gpus))!=2 or any(type(g) is not int or not 0<=g<8 for g in gpus):
        raise ValueError('Single arm requires two distinct physical GPU IDs, one per model')
    device_map=resources.get('device_map',{})
    if set(device_map)!={str(g) for g in gpus}:raise ValueError('Exact two HCU PCI/render mappings required')
    for item in device_map.values():
        if set(item)!={'pci_bus','render_node'} or not re.fullmatch(r'[0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7]',item['pci_bus']) or not re.fullmatch(r'/dev/dri/renderD[0-9]+',item['render_node']):raise ValueError('Invalid registered GPU mapping')
    from service_endpoints import validate_ports,shifted_ports
    ports=validate_ports(ports);shift=ports['policy']-49500
    if ports!=shifted_ports(shift):raise ValueError('Use one consistent original service port shift')
    model=ref(binding,'model_config');qwen_binding=ref(binding,'qwen_weights_binding')
    collection=ref(binding,'sampler_manifest').parent;reg=ref(binding,'sampler_registration');registration=read(reg)
    input_role = validate_sampler_partition(registration, phase, seed)
    if read(collection/'RESULT.json')['registration_sha256']!=sha(reg):raise ValueError('Collection identity differs')
    shutil.copytree(workspace,campaign/'initial_experience');save(campaign/'INHERITED_MANIFEST.json',source)
    shutil.copyfile(model,campaign/'MODEL_CONFIG.json');shutil.copyfile(qwen_binding,campaign/'QWEN_WEIGHTS_BINDING.json')
    shutil.copyfile(binding_path,campaign/'SOURCE_BINDINGS.json');save(campaign/'COMMON_EVIDENCE_MANIFEST.json',read(ref(binding,'common_evidence_manifest')))
    runtime_source = project
    if 'runtime_code_manifest' in binding:
        runtime_manifest = ref(binding, 'runtime_code_manifest')
        runtime_source = runtime_manifest.parent / 'code'
        runtime_source.resolve().relative_to(project)
        if manifest(runtime_source) != read(runtime_manifest):
            raise ValueError('Explicit tested runtime snapshot differs from its complete manifest')
    freeze_code(runtime_source,campaign/'code');save(campaign/'CODE_MANIFEST.json',manifest(campaign/'code'))
    sdk_binding=binding.get('sampler_sdk_preflight')
    if sdk_binding is None:
        cpu=project/'planning/native_sampler_agent_glue_sdk_v1_20261007'
    else:
        bundle=ref(binding,'sampler_sdk_preflight')
        cpu=bundle.parent
        if bundle.name!='MANIFEST.json' or cpu.parent!=project/'planning':
            raise ValueError('Sampler SDK receipt must bind an explicit project planning bundle')
        for sdk_relative,digest in read(bundle).items():
            p=Path(sdk_relative)
            if p.is_absolute() or '..' in p.parts or sha(cpu/p)!=digest:
                raise ValueError('Bound sampler SDK bundle changed')
    if read(cpu/'outputs/STATUS.json')['status']!='PASSED':raise ValueError('Actual sampler SDK/import preflight unavailable')
    tested=read(cpu/'MANIFEST.json')
    paths=['src/fastrsi/native_sampler_agent_binding_v1.py','scripts/run_native_sampler_goal_component_v1.py','scripts/instrument_native_sampler_draw_target_v1.py']
    if sdk_binding is not None:
        paths.append('scripts/run_native_baseline.py')
        sdk_status=read(cpu/'outputs/STATUS.json')
        sdk_result=read(cpu/'outputs/SDK_RESULT.json')
        target_result=read(cpu/'outputs/TARGET_RESULT.json')
        if (sdk_status.get('manifest_sha256')!=sha(cpu/'MANIFEST.json') or
                sdk_result.get('status')!='PASSED' or target_result.get('status')!='PASSED' or
                target_result.get('rpc_reset_completed')!=1 or target_result.get('settling_actions')!=10 or
                target_result.get('policy_actions')!=0 or target_result.get('model_calls')!=0 or
                target_result.get('tampered_requests_rejected_before_reset') is not True or
                target_result.get('native_reset_exact') is not True):
            raise ValueError('Actual sampler RPC/reset receipt incomplete')
    for path in paths:
        if tested['code/'+path]!=sha(campaign/'code'/path):raise ValueError('Passed sampler glue source changed')
    save(campaign/'SDK_REUSE.json',dict(bundle=str(cpu.relative_to(project)),manifest_sha256=sha(cpu/'MANIFEST.json'),sdk_result_sha256=sha(cpu/'outputs/SDK_RESULT.json'),target_result_sha256=sha(cpu/'outputs/TARGET_RESULT.json')))
    sid='sampler_'+binding['arm'][0].lower()+'_'+hashlib.sha256(relative.encode()).hexdigest()[:12]
    base='/project/'+relative;w=base+'/stream/'+sid+'/workspace'
    cfg=dict(version='native_sampler_agent_binding_v1',workspace=w,run_id='native_sampler_'+campaign.name,
        restore=dict(version='native_sampler_draw_restore_v1',root_session=sid,registration='/project/'+reg.relative_to(project).as_posix(),registration_sha256=sha(reg),
            collection_root='/project/'+collection.relative_to(project).as_posix(),collection_manifest_sha256=sha(collection/'MANIFEST.json'),sample_id='sample_'+str(seed)))
    save(campaign/'SAMPLER_CONFIG.json',cfg)
    case=dict(id=sid,mode=phase,suite='libero_90',task_id=64,init_state_id=0,max_steps=400,task_description=GOAL,
        development_suite_extension='libero90_stack_flow_v1',native_registry_lifecycle='incremental_v1',native_component_updates='registered_v1',
        registered_goal_component=True,registered_goal_component_physical_v1=True,native_experience_maintenance=(phase=='development'),
        require_discovery_reads=True,session_intent=True,read_files_tool=True,
        model_config=base+'/MODEL_CONFIG.json',model_config_sha256=sha(model),
        native_sampler_binding=base+'/SAMPLER_CONFIG.json',native_sampler_binding_sha256=sha(campaign/'SAMPLER_CONFIG.json'))
    if boundary is not None:
        case['registered_program_boundary_v1'] = boundary
    # Same public construction artifact built from the actual common benchmark skill.
    sys.path.insert(0,str(project/'src'))
    from fastrsi.session_flow import artifact
    save(campaign/'FLOW.json',artifact((workspace/'skills/benchmarking/SKILL.md').read_bytes(),'common_original_benchmark_contract'))
    case.update(session_flow_artifact=base+'/FLOW.json',session_flow_sha256=sha(campaign/'FLOW.json'))
    save(campaign/'ROSTER.json',[case]);save(campaign/'TARGET_CONFIG.json',dict(workspace=w,selection_path=w+'/reports/NATIVE_GOAL_COMPONENT_SELECTION.json',
        root_session=sid,target_ref='target://libero_real_remote',task_description=GOAL,model_config=case['model_config'],model_config_sha256=case['model_config_sha256']))
    for role,name in [('pi05','pi05_v2'),('qwen','qwen'),('target','target')]:shutil.copyfile(project/'runs/pilot_20261001'/('LAUNCH_'+name+'.json'),campaign/('TEMPLATE_'+role+'.json'))
    record=dict(version=VERSION,status='PREPARED_NOT_DISPATCHED',arm=binding['arm'],mode=phase,draw_seed=seed,root_session=sid,
        input_role=input_role, sampler_registration_version=registration['version'],
        source_status=binding.get('source_status','TERMINAL_SOURCE'),teacher_failure_costs_retained=True,
        declaration=decl,resources=resources,port_shift=shift,code_manifest_sha256=sha(campaign/'CODE_MANIFEST.json'),
        default_entry_sha256=sha(project/'CURRENT_NATIVE_ENTRY.json'),model_config_sha256=sha(model),source_bindings_sha256=sha(campaign/'SOURCE_BINDINGS.json'),
        budget=dict(max_root_cases=1,max_replans=2,max_sessions=3,max_steps_per_session=400,max_counted_actions=1200,max_settling_actions=30),
        source_sampler=dict(registration=str(reg.relative_to(project)),registration_sha256=sha(reg),manifest=str((collection/'MANIFEST.json').relative_to(project)),manifest_sha256=sha(collection/'MANIFEST.json')),
        service_timeout_seconds=6300,runner_timeout_seconds=5400,controller_timeout_seconds=6000,
        scope='Complete original Agent on registered sampler extension; not official fixed50 score',
        B_equal_executable_action_space=False if binding['arm']=='B_frozen_prior' else None,
        learning_gain_claimed=False,snapshot_preparation_is_not_empirical_admission=True)
    if boundary is not None:
        record['registered_program_boundary_v1'] = copy.deepcopy(boundary)
        record['registered_program_boundary_sha256'] = hashlib.sha256(
            json.dumps(boundary, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    save(campaign/'REGISTRATION.json',record)
    save(campaign/'MANIFEST.json',manifest(campaign))
    return dict(status=record['status'],manifest_sha256=sha(campaign/'MANIFEST.json'),campaign=str(campaign))


def check(project,campaign,manifest_sha):
    project,campaign=Path(project),Path(campaign)
    if sha(campaign/'MANIFEST.json')!=manifest_sha:raise ValueError('Campaign manifest identity differs')
    for rel,digest in read(campaign/'MANIFEST.json').items():
        if sha(campaign/rel)!=digest:raise ValueError('Frozen input changed: '+rel)
    reg=read(campaign/'REGISTRATION.json')
    if manifest(campaign/'initial_experience')!=read(campaign/'INHERITED_MANIFEST.json'):raise ValueError('Inherited source changed')
    if sha(project/'CURRENT_NATIVE_ENTRY.json')!=reg['default_entry_sha256']:raise ValueError('Default pointer changed')
    for key in ('registration','manifest'):
        if sha(project/reg['source_sampler'][key])!=reg['source_sampler'][key+'_sha256']:raise ValueError('Sampler source identity changed')
    return reg


def cpu_composition(project,campaign,manifest_sha):
    import asyncio
    check(project,campaign,manifest_sha)
    campaign=Path(campaign);out=campaign/'preflight';out.mkdir(exist_ok=False)
    # CPU argv binds its trace here; create it before native spans emit.
    # Physical dispatch creates its stream separately, and remains unchanged.
    (campaign/'preflight_logs').mkdir(exist_ok=True)
    sys.path[:0]=[str(campaign/'code/scripts'),str(campaign/'code/src')]
    os.environ['FASTRSI_NATIVE_RUN']=str(out/'stream');os.environ['FASTRSI_EXPERIENCE_ADMISSION']='legacy_raw'
    from fastrsi.native_binding import bind_agent,finish_agent,hashes
    import run_native_baseline as native
    import run_native_sampler_goal_component_v1 as glue
    import fastrsi.session_flow as flow
    from PhyAgentOS.providers.base import LLMProvider
    from PhyAgentOS.runtime.schemas import SessionSpec
    class NoModel(LLMProvider):
        def __init__(self):super().__init__();self.calls=0
        def get_default_model(self):return 'CPU_COMPOSITION_NO_MODEL'
        async def chat(self,*a,**kw):self.calls+=1;raise AssertionError('Public construction fell back to model')
    async def run():
        cfg=read(campaign/'SAMPLER_CONFIG.json');case=read(campaign/'ROSTER.json')[0];folder=out/'setup';folder.mkdir()
        cfg['workspace']=str(folder/'workspace');cfg['restore']['registration']=str(Path(project)/read(campaign/'REGISTRATION.json')['source_sampler']['registration'])
        cfg['restore']['collection_root']=str((Path(project)/read(campaign/'REGISTRATION.json')['source_sampler']['manifest']).parent)
        save(out/'SAMPLER_CONFIG.json',cfg);case.update(native_sampler_binding=str(out/'SAMPLER_CONFIG.json'),native_sampler_binding_sha256=sha(out/'SAMPLER_CONFIG.json'),
            model_config=str(campaign/'MODEL_CONFIG.json'),session_flow_artifact=str(campaign/'FLOW.json'))
        restore=glue.install_runner(native,flow);provider=NoModel();agent=None
        try:
            w,spec,old=native.setup_workspace(folder,case,campaign/'initial_experience')
            if (w/'SKILLRUNTIME.md').read_bytes()!=(campaign/'initial_experience/SKILLRUNTIME.md').read_bytes():raise ValueError('Setup changed inherited actual registry')
            SessionSpec.model_validate(spec)
            log=folder/'EVENTS.jsonl';agent=bind_agent(w,provider,log,evaluation=case['mode']=='evaluation',component_updates=True);before=hashes(w);queued=[]
            # The original registered runner grants validation of this exact component.
            validation=native.validate_component_selection
            native.validate_component_selection=lambda w,p,a:validation(w,p,a,goal_component_case=case)
            try:_,planned=await native.plan_session(agent,case,w,spec,old,log,on_queued=queued.append)
            finally:native.validate_component_selection=validation
            if len(queued)!=1 or provider.calls or planned['benchmark']['instance_id'] is not None:raise ValueError('Native queue composition did not match')
            await finish_agent(agent,log);agent=None
            if case['mode']=='evaluation' and hashes(w)!=before:raise ValueError('Evaluation end updated persistent experience')
            return dict(status='PASSED',queued=1,model_calls=0,robot_resets=0,actions=0,registry_bytes_inherited=True,instance_id=None)
        finally:
            if agent is not None:await finish_agent(agent,folder/'EVENTS.jsonl')
            restore()
    result=asyncio.run(run());result.update(manifest_sha256=manifest_sha,registration_sha256=sha(campaign/'REGISTRATION.json'))
    save(out/'CHECK_RESULT.json',result);return result


def launch_argv(project,campaign,role):
    import orchestrate_full_os as life
    from service_endpoints import relocate_service_command,docker_port_env
    from native_service_config import configure_qwen,model_env
    project,campaign=Path(project),Path(campaign);reg=read(campaign/'REGISTRATION.json');base='/project/'+campaign.relative_to(project).as_posix();stream=campaign/'stream'
    life.PROJECT=project;life.CAMPAIGN=campaign;life.BASE=base
    life.CORE=project.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/dependencies/PhyAgentOS-core'
    name='yuchen-fastrsi-'+reg['root_session']+'-'+role;cfg=read(campaign/'MODEL_CONFIG.json');ports=reg['resources']['ports']
    if role in ('runner','cpu'):
        argv=life.cpu_args(name,stream if role=='runner' else campaign/'preflight_logs')
        argv=[x.replace(str(project)+':/project:rw',str(project)+':/project:ro') for x in argv]
        argv+=['-v',str(campaign)+':'+base+':rw','-e','FASTRSI_NATIVE_RUN='+base+'/stream',
            '-e','FASTRSI_INITIAL_EXPERIENCE='+base+'/initial_experience','-e','FASTRSI_EXPERIENCE_ADMISSION=legacy_raw',
            '-e','PYTHONPATH='+base+'/code/scripts:'+base+'/code/src:/core:/project/planning/native_goal_program_sdk_preflight_v1_20261007/linux_deps:/project/envs/structured-decode-v1']
        argv+=model_env(cfg)+docker_port_env(ports)
        if role=='cpu':
            argv[argv.index('--network')+1]='none'
            command='source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+base+'/code/scripts/orchestrate_native_sampler_agent_arm_v1.py --cpu-composition --project /project --campaign '+base+' --manifest-sha256 '+sha(campaign/'MANIFEST.json')
            limit=180
        else:
            command='source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python '+base+'/code/scripts/run_native_sampler_goal_component_v1.py'
            limit=reg['runner_timeout_seconds']
        return argv+[IMAGE,'timeout',str(limit),'bash','-lc',command]
    argv=read(campaign/('TEMPLATE_'+role+'.json'))['argv'];old=project/'runs/pilot_20261001'
    argv=[x.replace(str(PROJECT_REMOTE),str(project)).replace(str(old),str(stream)).replace('/project/runs/pilot_20261001',base+'/stream').replace(str(project)+':/project:rw',str(project)+':/project:ro') for x in argv]
    argv[argv.index('--name')+1]=name
    idx=next(i for i,x in enumerate(argv) if x.startswith('sha256:'))
    argv[idx:idx]=['-v',str(campaign)+':'+base+':rw']+docker_port_env(ports)
    if role=='pi05':
        argv=[x.replace(str(project/'models/openpi_data'),str(stream/'tokenizer_cache')) for x in argv]
    elif role=='qwen':argv=configure_qwen(argv,cfg,life)
    else:
        argv=[x.replace('/project/scripts/instrument_target.py',base+'/code/scripts/instrument_native_sampler_draw_target_v1.py') for x in argv]
        idx=argv.index(TARGET_IMAGE)
        argv[idx:idx]=['-e','PYTHONPATH=/workspace/openpi/third_party/libero:'+base+'/code/scripts:'+base+'/code/src:/core',
            '-e','FASTRSI_NATIVE_SAMPLER_CONFIG='+base+'/SAMPLER_CONFIG.json','-e','FASTRSI_COMPONENT_TARGET_CONFIG='+base+'/TARGET_CONFIG.json',
            '-e','FASTRSI_STAGE_TRACE='+base+'/stream/stage_trace','-e','FASTRSI_LIVE_CONTROLS='+base+'/stream/live_controls']
    argv[argv.index('timeout')+1]=str(reg['service_timeout_seconds'])
    argv=relocate_service_command(argv,reg['port_shift'],0)
    if role in ('pi05','qwen'):
        physical=reg['resources']['gpus'][0 if role=='pi05' else 1]
        remapped=[]
        for item in argv:
            prefix=next((p for p in ('HIP_VISIBLE_DEVICES=','CUDA_VISIBLE_DEVICES=') if item.startswith(p)),None)
            remapped.append(prefix+str(physical) if prefix else item)
        argv=remapped
    return argv



def parse_visibility(entries):
    values=[]
    for entry in entries:
        if entry.startswith(('HIP_VISIBLE_DEVICES=','CUDA_VISIBLE_DEVICES=')):
            value=entry.split('=',1)[1]
            if not re.fullmatch(r'[0-7](?:,[0-7])*',value):return None
            values.append(set(map(int,value.split(','))))
    if not values or any(v!=values[0] for v in values):return None
    return values[0]


def known_driver_daemon(device, cmd, exe, cgroup):
    """Only named host driver/telemetry services; never a generic Python holder."""
    if device != '/dev/mkfd':
        return None
    if cmd.strip() == '/usr/local/hyhal/bin/hymgr' and Path(exe).name == 'hymgr':
        return 'hymgr'
    if (cmd.split() == ['/opt/gridview/agent/python3.12.12/bin/python3',
                        '/opt/gridview/agent/collector-agent/collector_agent.py']
            and exe == '/opt/gridview/agent/python3.12.12/bin/python3.12'
            and cgroup.strip() == '0::/system.slice/gridview-collect-agent.service'):
        return 'gridview_collect_agent'
    return None


def resource_gate(project,reg,output):
    """Selected physical cards only; never infer availability from utilization alone."""
    import socket
    import subprocess
    selected=set(reg['resources']['gpus']);mapping=reg['resources']['device_map']
    receipt=dict(status='CHECKING',time=time.time(),host=socket.gethostname(),selected_hcu=sorted(selected),mapping=mapping,
                 visibility_only_allocation=True,driver_mounts_shared=True,containers=[],holders=[])
    def call(args):
        p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=45)
        if p.returncode:raise RuntimeError(dict(argv=args,code=p.returncode,stderr=p.stderr))
        return p.stdout
    try:
        receipt['hcu']=call(['bash','-lc','source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse --showbus'])
        text=receipt['hcu'];uses=dict((int(i),float(v)) for i,v in re.findall(r'HCU\[(\d+)\].*?HCU use \(%\):\s*([0-9.]+)',text))
        memory=dict((int(i),float(v)) for i,v in re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)',text))
        buses=dict((int(i),v.lower()) for i,v in re.findall(r'HCU\[(\d+)\][^\n]*?(?:PCI Bus|PCI bus|PCI Bus ID):\s*([0-9a-fA-F]{4}:[0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-7])',text))
        receipt.update(parsed_use=uses,parsed_memory=memory,parsed_pci=buses)
        for gpu in selected:
            item=mapping[str(gpu)];render=Path(item['render_node'])
            uevent=Path('/sys/class/drm')/render.name/'device/uevent'
            native_bus=next((r.split('=',1)[1].lower() for r in uevent.read_text().splitlines() if r.startswith('PCI_SLOT_NAME=')),None)
            if buses.get(gpu)!=item['pci_bus'].lower() or native_bus!=item['pci_bus'].lower():raise ValueError('HCU/PCI/render mapping unconfirmed; root must inspect actual host interface')
            if uses.get(gpu)!=0 or memory.get(gpu)!=0:raise ValueError('Selected HCU busy or not parsed')
        ids=call(['docker','ps','-q']).split()
        for cid in ids:
            info=json.loads(call(['docker','inspect',cid]))[0];devices=info['HostConfig'].get('Devices') or []
            gpu_device=any(d.get('PathOnHost','').startswith(('/dev/kfd','/dev/mkfd','/dev/dri')) for d in devices)
            claims=parse_visibility(info['Config'].get('Env') or [])
            row=dict(container_id=info['Id'],name=info['Name'],gpu_devices=devices,visible_hcu=None if claims is None else sorted(claims));receipt['containers'].append(row)
            if gpu_device and (claims is None or selected&claims):raise ValueError('Live container claims selected GPUs or lacks attributable GPU visibility')
        devices=['/dev/kfd','/dev/mkfd']+[mapping[str(i)]['render_node'] for i in selected]
        for device in devices:
            p=subprocess.run(['fuser',device],capture_output=True,text=True,timeout=15)
            if p.returncode not in (0,1):raise RuntimeError('Cannot inspect selected GPU holders')
            for token in p.stdout.split():
                if not token.isdigit():raise ValueError('Unparsed device holder')
                proc=Path('/proc')/token
                try:
                    cmd=proc.joinpath('cmdline').read_bytes().replace(b'\0',b' ').decode();exe=os.readlink(proc/'exe')
                    env=proc.joinpath('environ').read_bytes().decode().split('\0')
                    cgroup=proc.joinpath('cgroup').read_text()
                except FileNotFoundError:continue  # exited during this read
                claims=parse_visibility(env);daemon_kind=known_driver_daemon(device,cmd,exe,cgroup);daemon=daemon_kind is not None
                receipt['holders'].append(dict(device=device,pid=int(token),argv=cmd,exe=exe,visible_hcu=None if claims is None else sorted(claims),system_daemon=daemon,system_daemon_kind=daemon_kind,cgroup=cgroup))
                if not daemon and (claims is None or selected&claims):raise ValueError('Selected GPU or unattributable driver holder; never preempt')
        for port in reg['resources']['ports'].values():
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',port))==0:raise ValueError('Assigned port already in use')
        for image in (IMAGE,TARGET_IMAGE):
            if call(['docker','image','inspect',image,'--format','{{.Id}}']).strip()!=image:raise ValueError('Frozen image unavailable')
        available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:')))*1024
        receipt['available_memory_bytes']=available;receipt['cpu_count']=os.cpu_count();receipt['disk_free']=shutil.disk_usage(project).free
        if available<176*1024**3 or (os.cpu_count() or 0)<32 or receipt['disk_free']<4*1024**3:raise ValueError('Insufficient CPU, memory or scratch headroom')
        receipt['status']='PASSED'
    except BaseException as error:
        receipt.update(status='FAILED',error=repr(error));raise
    finally:save(Path(output)/'RESOURCE_RECEIPT.json',receipt)

def execute(project,campaign,manifest_sha):
    import fcntl
    import signal
    import socket
    import subprocess
    import traceback
    from datetime import datetime
    import orchestrate_full_os as life
    from orchestrate_native_development_operation_v1 import parse_docker_timestamp
    project,campaign=Path(project).resolve(),Path(campaign).resolve()
    if project!=PROJECT_REMOTE:raise ValueError('Physical dispatch requires the registered shared project namespace')
    reg=check(project,campaign,manifest_sha)
    cpu=read(campaign/'preflight/CHECK_RESULT.json')
    if cpu.get('status')!='PASSED' or cpu.get('manifest_sha256')!=manifest_sha or cpu.get('registration_sha256')!=sha(campaign/'REGISTRATION.json'):
        raise ValueError('Current exact snapshot actual SDK composition not passed')
    lock=(campaign/'dispatch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stream=campaign/'stream';stream.mkdir(exist_ok=False)
    started=time.time();owned={};state=dict(version=VERSION,status='CHECKING',arm=reg['arm'],mode=reg['mode'],draw_seed=reg['draw_seed'],budget=reg['budget'],started_at=started)
    life.deadline=started+reg['controller_timeout_seconds'];life.PROJECT=project;life.CAMPAIGN=campaign
    def inspect(cid):return json.loads(subprocess.check_output(['docker','inspect',cid]))[0]
    def launch(role):
        argv=launch_argv(project,campaign,role);tick=time.time();row=life.command(argv)
        row.update(role=role,launched_at=tick);save(stream/('LAUNCH_'+role+'.json'),row)
        if row['returncode']!=0:raise RuntimeError(row['stderr'])
        cid=row['stdout'].strip()
        if not re.fullmatch(r'[a-f0-9]{64}',cid):raise ValueError('Unrecognized launch container identity')
        name=argv[argv.index('--name')+1];owned[cid]=(role,name)
        info=inspect(cid)
        if info['Id']!=cid or info['Name']!='/'+name:raise ValueError('Actual launch ownership differs')
        return cid
    def interrupted(signum,frame):raise RuntimeError('Interrupted: '+str(signum))
    original_signals={sig:signal.signal(sig,interrupted) for sig in (signal.SIGINT,signal.SIGTERM)}
    try:
        resource_gate(project,reg,stream)
        cfg=read(campaign/'MODEL_CONFIG.json');binding=read(campaign/'QWEN_WEIGHTS_BINDING.json');tick=time.time()
        for name,expected in binding['current_file_sha256'].items():
            if sha(Path(cfg['weights'])/name)!=expected:raise ValueError('Original Qwen model bytes changed')
        from orchestrate_native_development_operation_v1 import MODEL_FILES
        pi_root=project.parent/'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1/weights/pi05_libero_pytorch_fp32'
        for name,expected in MODEL_FILES.items():
            if sha(pi_root/name)!=expected:raise ValueError('Original pi05 model bytes changed')
        save(stream/'MODEL_SOURCE_CHECK.json',dict(seconds=time.time()-tick,qwen_files=binding['current_file_sha256'],pi05_files=MODEL_FILES,model_config_sha256=sha(campaign/'MODEL_CONFIG.json')))
        shutil.copytree(project/'models/openpi_data',stream/'tokenizer_cache')
        tick=time.time();launch('pi05');launch('qwen');launch('target')
        ports=reg['resources']['ports'];life.health(ports['policy'],limit=600);life.health(ports['qwen'],limit=600)
        until=min(life.deadline,time.time()+180)
        while True:
            try:
                with socket.create_connection(('127.0.0.1',ports['target']),timeout=1):break
            except OSError:
                if time.time()>until:raise TimeoutError('Native target TCP readiness failed')
                time.sleep(1)
        state.update(status='RUNNING',model_and_target_startup_seconds=time.time()-tick);save(campaign/'STATUS.json',state)
        runner=launch('runner');life.wait_exit(runner,reg['runner_timeout_seconds']+10)
        raw=read(stream/'RESULT.json');cases=raw.get('cases',[])
        if len(cases)!=1:raise ValueError('Exactly one original root-case chain required')
        state.update(status='COMPLETED',native_result_status=raw.get('status'),raw_result=dict(path='stream/RESULT.json',sha256=sha(stream/'RESULT.json')),
            environment_success=cases[0].get('environment_success'),verification_status=cases[0].get('verification_status'),chain_status=cases[0].get('chain_status'),
            full_agent_chain=True,learning_gain_claimed=False,primary_metric='original_native_benchmark_success',
            physical_release_status='not_inferred_from_native_success',original_verifier_and_recovery_preserved=True)
        if raw.get('status')!='COMPLETED':state['status']='FAILED_NATIVE_CHAIN'
    except BaseException:state.update(status='FAILED',error=traceback.format_exc())
    finally:
        stops=[];stop_started=time.time()
        # Durable exact launch receipts recover a CID even after an interrupted
        # post-launch inspect. Names must still match this one campaign.
        for path in stream.glob('LAUNCH_*.json'):
            row=read(path);cid=row.get('stdout','').strip();role=path.stem[len('LAUNCH_'):]
            if re.fullmatch(r'[a-f0-9]{64}',cid):owned[cid]=(role,'yuchen-fastrsi-'+reg['root_session']+'-'+role)
        for cid,(role,name) in sorted(owned.items(),key=lambda item:0 if item[1][0]=='runner' else 1):
            record=dict(container_id=cid,role=role,name=name)
            try:
                info=inspect(cid)
                if info['Id']!=cid or info['Name']!='/'+name:raise ValueError('Refuse cleanup of unrelated container')
                if info['State']['Running']:
                    stopped=life.command(['docker','stop','-t','10',cid],timeout=40)
                    if stopped['returncode']:raise RuntimeError(stopped['stderr'])
                after=inspect(cid);logs=subprocess.run(['docker','logs',cid],capture_output=True)
                (stream/(role+'.log')).write_bytes(logs.stdout+logs.stderr)
                record.update(state=after['State'],running=after['State']['Running'],
                    residency_seconds=(parse_docker_timestamp(after['State']['FinishedAt'])-parse_docker_timestamp(info['State']['StartedAt'])).total_seconds())
                if record['running']:raise RuntimeError('Owned container remained running')
            except BaseException:record['cleanup_error']=traceback.format_exc();state['status']='CLEANUP_FAILED'
            stops.append(record)
        if (stream/'RESULT.json').is_file():
            try:
                raw=read(stream/'RESULT.json');rows=raw.get('cases',[])
                state['raw_result']=dict(path='stream/RESULT.json',sha256=sha(stream/'RESULT.json'))
                if len(rows)==1:
                    state.update(environment_success=rows[0].get('environment_success'),verification_status=rows[0].get('verification_status'),chain_status=rows[0].get('chain_status'))
            except Exception:state['raw_result_parse_error']=traceback.format_exc()
        save(stream/'STOP_RECEIPT.json',stops)
        state.update(stop_seconds=time.time()-stop_started,finished_at=time.time(),full_wall_seconds=time.time()-started,
            model_residencies={r['role']:r.get('residency_seconds') for r in stops if r['role'] in ('pi05','qwen')},
            cost_relationship='Startup, reset, policy, verification, recovery and stop are included in full_wall; residency overlaps it. Native span costs remain separate nested raw events.',
            default_entry_unchanged=sha(project/'CURRENT_NATIVE_ENTRY.json')==reg['default_entry_sha256'])
        if not state['default_entry_unchanged']:state['status']='FAILED_DEFAULT_CHANGED'
        save(campaign/'STATUS.json',state)
        for sig,handler in original_signals.items():signal.signal(sig,handler)
        fcntl.flock(lock,fcntl.LOCK_UN);lock.close()
    return state


def main():
    p=argparse.ArgumentParser(description=__doc__);m=p.add_mutually_exclusive_group(required=True)
    for name in ('prepare','check','print-argv','cpu-composition','execute'):m.add_argument('--'+name,action='store_true')
    p.add_argument('--project',type=Path,required=True);p.add_argument('--campaign',type=Path,required=True);p.add_argument('--bindings',type=Path)
    p.add_argument('--manifest-sha256');p.add_argument('--allow-real-physical-calls',action='store_true');a=p.parse_args()
    if a.prepare:
        if a.bindings is None:p.error('--prepare requires explicit source bindings')
        result=prepare(a.project,a.campaign,a.bindings)
    else:
        if not a.manifest_sha256:p.error('Exact frozen manifest SHA required')
        if a.cpu_composition:result=cpu_composition(a.project,a.campaign,a.manifest_sha256)
        elif a.execute:
            if not a.allow_real_physical_calls:p.error('Physical dispatch needs explicit --allow-real-physical-calls')
            result=execute(a.project,a.campaign,a.manifest_sha256)
        elif a.print_argv:
            check(a.project,a.campaign,a.manifest_sha256);result={role:launch_argv(a.project,a.campaign,role) for role in ('cpu','pi05','qwen','target','runner')}
        else:result=check(a.project,a.campaign,a.manifest_sha256)
    print(json.dumps(result,indent=2))
    if result.get('status','').startswith(('FAILED','CLEANUP_FAILED')):raise SystemExit(1)
if __name__=='__main__':main()
