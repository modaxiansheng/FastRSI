"""Bounded native Agent planning -> unchanged OS execution/feedback binding."""
import asyncio
import copy
import hashlib
import json
import os
import shutil
import time
import traceback
import yaml
from pydantic import ValidationError
from pathlib import Path
from types import SimpleNamespace
from cost_trace import span, emit
from service_endpoints import service_ports
from scripts.run_pi05_libero_real_e2e import _write_workspace
from PhyAgentOS.runtime.workspace.manager import _RUNTIME_INSTRUCTIONS
from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block, write_yaml_block
from PhyAgentOS.runtime.watchdog.supervisor import WatchdogSupervisor
from PhyAgentOS.runtime.schemas import SessionSpec
from PhyAgentOS.runtime.watchdog.result_writer import ResultWriter
from PhyAgentOS.runtime.policy.openpi.client import OpenPIClientPolicyWrapper
from PhyAgentOS.agent.session_verifier import SessionVerifier
from PhyAgentOS.providers.custom_provider import CustomProvider
from PhyAgentOS.providers.base import GenerationSettings
from fastrsi.native_binding import bind_agent, finish_agent, hashes, record, native_maintenance_enabled
from fastrsi.queue_receipt import accepted_queue_receipt
from fastrsi.evidence_layers import evidence_layers
from fastrsi.experience_admission import mode as admission_mode, validate_snapshot, commit_experience, SEAL

ROOT=Path(os.environ['FASTRSI_NATIVE_RUN'])
CAMPAIGN=ROOT.parent
CORE=Path('/core')
PORTS=service_ports((49500,49502,49510,49520))
QWEN_API=f"http://127.0.0.1:{PORTS['qwen']}/v1"


def measured_provider(folder, role, max_tokens):
    provider=CustomProvider(api_base=QWEN_API,default_model=os.environ.get('FASTRSI_AGENT_MODEL','qwen25vl7b'),timeout_s=300)
    provider.generation=GenerationSettings(temperature=0,max_tokens=max_tokens)
    original=provider.chat
    calls=0
    async def chat(*args,**kw):
        nonlocal calls
        calls+=1
        if calls>40:raise RuntimeError('Bounded provider call limit')
        # Native archival returns a complete memory document. Do not silently
        # apply the shorter action-planning cap to save_memory. This common
        # baseline setting preserves the frozen GenerationSettings default.
        names={t.get('function',{}).get('name') for t in kw.get('tools',[]) or []}
        if names=={'save_memory'}:
            kw={**kw,'max_tokens':4096,'temperature':0.7}
        target=folder/(role+'_call_'+str(calls));target.mkdir()
        (target/'REQUEST.json').write_text(json.dumps(kw,ensure_ascii=False,default=str))
        with span(role+'_model_call'):
            response=await original(*args,**kw)
        (target/'RESPONSE.json').write_text(json.dumps(response.__dict__,ensure_ascii=False,default=lambda x:x.__dict__))
        return response
    provider.chat=chat
    return provider


def setup_workspace(folder, case, accepted):
    from fastrsi.native_component_updates_v1 import validate_case
    validate_case(case)
    registry_lifecycle = case.get('native_registry_lifecycle')
    if registry_lifecycle not in (None, 'incremental_v1'):
        raise ValueError('Unknown native registry lifecycle')
    w=folder/'workspace'
    previous_sessions=[]
    inherited={}
    if accepted:
        if admission_mode()=='guarded_v1':validate_snapshot(accepted)
        # Carry the full accepted OS workspace, including prior logs, sessions and
        # artifacts. A new conversation key does not erase native experience paths.
        shutil.copytree(accepted,w)
        if (w/SEAL).exists():(w/SEAL).unlink()  # mutable task branch, not a published snapshot
        previous_sessions=read_yaml_block(w/'SESSIONS.md').get('sessions',[])
        inherited={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in w.rglob('*') if p.is_file() and
                   (p.relative_to(w).parts[0] in ('memory','skills','sessions') or
                    p.relative_to(w).as_posix() in ('LESSONS.md','MEMORY_SPATIAL.md','TIMELINE.md'))}
    inherited_registry = None
    if registry_lifecycle == 'incremental_v1' and accepted:
        from fastrsi.native_registry_lifecycle_v1 import capture_inherited
        inherited_registry = capture_inherited(w, enabled=True)
    suite=case.get('suite','libero_spatial')
    from fastrsi.development_suite import step_limit, extend_target
    max_steps=step_limit(case)
    args=SimpleNamespace(policy_endpoint=f"openpi://127.0.0.1:{PORTS['policy']}",target_endpoint=f"targetws://127.0.0.1:{PORTS['target']}",
        benchmark_name=suite,task_id=case['task_id'],init_state_id=case['init_state_id'],
        task_description=case['task_description'],max_steps=max_steps,
        replan_every_steps=10 if case.get('registered_goal_component_physical_v1') is True else 5)
    _write_workspace(w,args)
    registry_receipt = None
    if inherited_registry is not None:
        from fastrsi.native_registry_lifecycle_v1 import merge_with_fresh_bindings
        registry_receipt = merge_with_fresh_bindings(w, inherited_registry, enabled=True)
    targets=read_yaml_block(w/'TARGETS.md')
    bound_target=next(t for t in targets['targets'] if t['id']=='libero_real_remote')
    if inherited_registry is None:
        target_template=read_yaml_block(CORE/'PhyAgentOS/templates/TARGETS.md')
        native_target=next(t for t in target_template['targets'] if t['id']=='libero_real_remote')
        bound_target['benchmark_capabilities']=copy.deepcopy(native_target['benchmark_capabilities'])
    capabilities_before=copy.deepcopy(bound_target.get('benchmark_capabilities'))
    extend_target(case,bound_target)
    if inherited_registry is None:
        write_yaml_block(w/'TARGETS.md','Runtime Targets',targets)
    elif capabilities_before!=bound_target.get('benchmark_capabilities'):
        # A bound suite extension changes one capability list, not surrounding
        # accepted Agent instructions. Preserve the inherited Markdown text.
        from PhyAgentOS.runtime.state_io.markdown_yaml import _YAML_BLOCK_RE
        target_text=(w/'TARGETS.md').read_text()
        match=_YAML_BLOCK_RE.search(target_text)
        if match is None:raise ValueError('Native target YAML block missing')
        start,end=match.span('body')
        body=yaml.safe_dump(targets,sort_keys=False,allow_unicode=True).rstrip('\n')
        (w/'TARGETS.md').write_text(target_text[:start]+body+target_text[end:])
    if inherited_registry is None:
        skills=read_yaml_block(w/'SKILLRUNTIME.md')
        skill_template=read_yaml_block(CORE/'PhyAgentOS/templates/SKILLRUNTIME.md')
        native_skill=next(s for s in skill_template['skillruntimes'] if s['id']=='pi05_libero_remote')
        bound_skill=next(s for s in skills['skillruntimes'] if s['id']=='pi05_libero_remote')
        bound_skill['benchmark']=copy.deepcopy(native_skill['benchmark'])
        write_yaml_block(w/'SKILLRUNTIME.md','Runtime Skill Runtimes',skills)
    (w/'RUNTIME.md').write_text(_RUNTIME_INSTRUCTIONS)
    (w/'ENVIRONMENT.md').write_text('LIBERO simulation. Runtime will update observations during execution.\n')
    doc=read_yaml_block(w/'SESSIONS.md');spec=doc['sessions'][0]
    spec['session_id']=case['id'];spec['verification_profile']='recovery'
    spec['execution']['reset_policy']='session_runner'
    spec['timeouts']['execute_timeout_s']=1200
    spec['benchmark']={'benchmark_id':'libero','suite_id':suite,'task_index':case['task_id'],
        'instance_id':case['init_state_id'],'policy_id':'pi05_libero','run_id':CAMPAIGN.name,'execution_mode':'policy_loop'}
    from fastrsi.same_world_native_binding import requested as same_world_requested
    if same_world_requested(case):
        from fastrsi.same_world_native_binding import bind_workspace as bind_same_world_workspace
        bind_same_world_workspace(case,w,spec,CAMPAIGN.name)
        from same_world_decision import bind_order_evidence
        bind_order_evidence(case,w)
    if case.get('instruction_experience'):
        from fastrsi.instruction_memory import install, select, intent_request, CANDIDATE
        state=install(w,case['instruction_experience'],case['instruction_experience_sha256'])
        config_sha=hashlib.sha256((CAMPAIGN/'MODEL_CONFIG.json').read_bytes()).hexdigest()
        decision=select(state,case,config_sha)
        if case.get('instruction_probe')=='candidate':
            if case['mode']!='development' or decision['reason']=='outside_validated_scope':raise ValueError('Unvalidated candidate only allowed in matching development scope')
            decision={'action':'candidate_validation','instruction':CANDIDATE,'reason':'predeclared_paid_development_probe','version':state['version']}
        elif case.get('instruction_probe')=='native':
            decision={'action':'native','instruction':case['task_description'],'reason':'matched_native_control','version':state['version']}
        payload=intent_request(spec,case['task_description'],decision)
    elif case.get('session_intent'):
        from fastrsi.session_flow import intent_from_spec
        payload=intent_from_spec(spec,case['task_description'])
    else:
        payload={'session':spec,'goal':case['task_description']}
    (w/'REQUEST.json').write_text(json.dumps(payload,indent=2))
    doc['sessions']=previous_sessions;write_yaml_block(w/'SESSIONS.md','Runtime Sessions',doc)
    (w/'memory').mkdir(exist_ok=True);(w/'skills').mkdir(exist_ok=True);(w/'sessions').mkdir(exist_ok=True)
    skill_path=w/'skills/benchmarking/SKILL.md';skill_path.parent.mkdir(exist_ok=True)
    if not skill_path.exists():
        shutil.copyfile(CORE/'PhyAgentOS/skills/benchmarking/SKILL.md',skill_path)
    if not (w/'LESSONS.md').exists():
        write_yaml_block(w/'LESSONS.md','Runtime Lessons',{'version':'runtime_lessons_v1','lessons':[]})
    preserved={name:hashlib.sha256((w/name).read_bytes()).hexdigest() for name in inherited}
    if inherited!=preserved:raise ValueError('Workspace setup altered inherited learning files')
    if registry_receipt is not None:
        registry_receipt['case_capability_extension']=capabilities_before!=bound_target.get('benchmark_capabilities')
        registry_receipt['final_registry_sha256']={name:hashlib.sha256((w/name).read_bytes()).hexdigest()
                                                 for name in ('TARGETS.md','SKILLRUNTIME.md')}
        (folder/'REGISTRY_INHERITANCE.json').write_text(json.dumps(registry_receipt,indent=2))
    (folder/'INHERITANCE.json').write_text(json.dumps({
        'source':str(accepted) if accepted else None,
        'source_seal_sha256':hashlib.sha256((accepted/SEAL).read_bytes()).hexdigest() if accepted and (accepted/SEAL).exists() else None,
        'inherited_learning_files':inherited,'preserved_before_agent':True,
        'conversation_key':case.get('conversation_key','cli:'+case['id']),
        'scope':'File inheritance only; model reads and decision effects require separate evidence'},indent=2))
    return w,spec,previous_sessions


def discovery_reads(events,workspace):
    """Absent initial LOG is legitimate discovery, not invented empty history."""
    root=Path(workspace).resolve();names=set()
    for e in events:
        if e.get('event')!='tool' or e.get('name')!='read_file':continue
        raw=Path(e.get('params',{}).get('path',''))
        path=raw.resolve() if raw.is_absolute() else (root/raw).resolve()
        if path.parent!=root:continue
        result=e.get('result','')
        if not result.startswith('Error'):
            names.add(path.name)
        elif path==root/'LOG.md' and not path.exists() and result.startswith('Error: File not found:'):
            names.add('LOG.md')
    return names


def effective_execution_view(spec):
    """Normalize two inert native-example defaults only on this bound policy path.

    The frozen policy_loop does not consume these fields. The alternative
    target_native runtime is deliberately outside this exception.
    """
    value=SessionSpec.model_validate(spec).model_dump(mode='json')
    known_path=(value['target_ref']=='target://libero_real_remote' and
        value['skillruntime_ref']=='skillruntime://pi05_libero_remote' and
        (value.get('benchmark') or {}).get('execution_mode')=='policy_loop' and
        value['safety_profile']['profile']=='default_simulation')
    if known_path:
        hint=value['runtime_hints']['preferred_replan_every_steps']
        if hint==value['execution']['replan_every_steps']:
            value['runtime_hints']['preferred_replan_every_steps']=None
        if value['safety_profile']['workspace_bounds']=='default':
            value['safety_profile']['workspace_bounds']=None
    return value



def component_selection_anchor(w, spec):
    """Capture the immutable physical/model contract before native tool writes."""
    from PhyAgentOS.runtime.schemas import TargetsDocument, SkillRuntimeDocument, TargetRuntimeContractDocument
    from PhyAgentOS.runtime.watchdog.scheduler import SessionScheduler
    from fastrsi.native_registry_lifecycle_v1 import _path
    targets=TargetsDocument.model_validate(read_yaml_block(w/'TARGETS.md'))
    skills=SkillRuntimeDocument.model_validate(read_yaml_block(w/'SKILLRUNTIME.md'))
    scheduled=SessionScheduler().resolve_session(SessionSpec.model_validate(spec),targets,skills)
    target=scheduled.target_spec.model_dump(mode='json')
    contract=TargetRuntimeContractDocument.model_validate(yaml.safe_load(
        _path(w, target['runtime']['runtime_contract_ref']).read_bytes())).model_dump(mode='json')
    return dict(target=target,skill=scheduled.skillruntime_spec.model_dump(mode='json'),contract=contract)


def validate_component_selection(w, planned, anchor, *, goal_component_case=None):
    """Resolve a real SDK component, preserving the initial physical/VLA contract."""
    from PhyAgentOS.runtime.schemas import TargetsDocument, SkillRuntimeDocument, TargetRuntimeContractDocument
    from PhyAgentOS.runtime.watchdog.scheduler import SessionScheduler
    from PhyAgentOS.runtime.preflight.runtime_compatibility_preflight import RuntimeCompatibilityPreflight
    from PhyAgentOS.runtime.perception.perception_runtime import PerceptionRuntime
    from fastrsi.native_registry_lifecycle_v1 import _path, _validate, _configs
    from fastrsi.native_component_updates_v1 import registry_hashes
    before=registry_hashes(w)
    targets=read_yaml_block(w/'TARGETS.md');skills=read_yaml_block(w/'SKILLRUNTIME.md')
    _validate(targets,skills,_configs(w))
    scheduled=SessionScheduler().resolve_session(SessionSpec.model_validate(planned),
        TargetsDocument.model_validate(targets),SkillRuntimeDocument.model_validate(skills))
    target=scheduled.target_spec.model_dump(mode='json')
    skill=scheduled.skillruntime_spec.model_dump(mode='json')
    contract=TargetRuntimeContractDocument.model_validate(yaml.safe_load(
        _path(w,target['runtime']['runtime_contract_ref']).read_bytes())).model_dump(mode='json')
    if not target['enabled']:raise ValueError('Selected target is disabled')
    # Target identity and advertised compatible skills may differ; the physical
    # task, sensors, controller and endpoint must still be exactly the anchor.
    physical=copy.deepcopy(target);baseline=copy.deepcopy(anchor['target'])
    for item in (physical,baseline):
        item.pop('id');item.pop('supported_skillruntimes')
        item['runtime'].pop('runtime_contract_ref')
    if physical!=baseline:raise ValueError('Selected target changes the bound physical task or observations')
    c=copy.deepcopy(contract);b=copy.deepcopy(anchor['contract'])
    c.pop('target_id');b.pop('target_id')
    if c!=b:raise ValueError('Selected target changes the runtime action/safety contract')
    selected=copy.deepcopy(skill);original=copy.deepcopy(anchor['skill'])
    selected.pop('id');original.pop('id')
    # This is the single registered action-prefix field consumed by the
    # opt-in host bridge, not permission to change VLA outputs or contracts.
    for item in (selected,original):
        chunk=item.get('output_contract',{}).get('action',{}).get('chunk',{})
        if 'preferred_replan_after_steps' in chunk:
            value=chunk.pop('preferred_replan_after_steps')
            if type(value) is not int or value not in (5,10):
                raise ValueError('Registered action-prefix steps must be 5 or 10')
    # Only the explicit registered-program runner enables this typed field.
    for item in (selected,original):
        action=item.get('output_contract',{}).get('action',{})
        chunk=action.get('chunk',{})
        if 'native_goal_component' in chunk:
            if not goal_component_case or goal_component_case.get('registered_goal_component') is not True:
                raise ValueError('Registered goal component requires its explicit native runner')
            from fastrsi.native_goal_component_v1 import extract_component, prepare_component
            if 'preferred_replan_after_steps' in skill.get('output_contract',{}).get('action',{}).get('chunk',{}):
                raise ValueError('Registered goal program cannot compose with prefix slicing')
            model_path=Path(goal_component_case['model_config'])
            model_sha=goal_component_case['model_config_sha256']
            if hashlib.sha256(model_path.read_bytes()).hexdigest()!=model_sha:
                raise ValueError('Registered goal model configuration changed')
            declaration=extract_component(item)
            prepare_component(w,declaration,expected_goal=planned['task_description'],
                expected_model_config_sha256=model_sha,expected_target_ref=planned['target_ref'])
            chunk.pop('native_goal_component')
    if selected!=original:raise ValueError('Selected skill changes the frozen VLA/runtime contract')
    if skill['runtime']!='OpenPISkillRuntime':raise ValueError('Only the native OpenPI runtime is allowed')
    plan=PerceptionRuntime(w,w).resolve_and_check(scheduled)
    checked=RuntimeCompatibilityPreflight(w).check(scheduled,plan)
    if checked.verdict!='accepted':
        raise ValueError('Native component compatibility rejected: '+checked.model_dump_json())
    if registry_hashes(w)!=before:raise ValueError('Component files changed during validation')
    return dict(target_ref=planned['target_ref'],skillruntime_ref=planned['skillruntime_ref'],
                registry_sha256=before,planned_session=SessionSpec.model_validate(planned).model_dump(mode='json'),
                preflight=checked.model_dump(mode='json'))



def scope_component_verification(registry, active_ids):
    """Select only this run's awaiting Sessions, leaving historical rows alone."""
    original=registry.first_awaiting_verification
    def first(document=None):
        document=document or registry.load()
        return next((s for s in document.sessions if s.session_id in active_ids
                     and s.status.value=='awaiting_verification'),None)
    registry.first_awaiting_verification=first
    return original


def validate_component_drafts(w, spec, planned, anchor):
    """Do not let terminal drafts redefine the next Session's trusted anchor.

    Checking the selected alias alone misses a separately edited base binding.
    Both base and consumed bindings must retain the original physical/model
    contract. A valid draft is still not an empirically accepted capability.
    """
    return {'base':validate_component_selection(w,spec,anchor),
            'selected':validate_component_selection(w,planned,anchor)}


def dispatch_registered_session(supervisor,w,session_id,component_receipt,*,expected_parent=None,active_ids=None):
    """Never let run_once's preparation revive a completed/historical Session."""
    from fastrsi.native_component_updates_v1 import registry_hashes
    if registry_hashes(w)!=component_receipt['registry_sha256']:
        raise ValueError('Consumed component registry changed before dispatch')
    doc=read_yaml_block(w/'SESSIONS.md')
    matches=[s for s in doc['sessions'] if s['session_id']==session_id]
    if len(matches)!=1:raise ValueError('Requested dispatch Session is missing or duplicated')
    session=SessionSpec.model_validate(matches[0])
    if session.status.value!='pending' or session.claimed_by or session.claim_token:
        raise ValueError('Requested dispatch Session is not unclaimed pending')
    if session.parent_session_id!=expected_parent:
        raise ValueError('Dispatch Session lineage mismatch')
    current=session.model_dump(mode='json')
    original=copy.deepcopy(component_receipt['planned_session'])
    if expected_parent is None:
        if current!=original:raise ValueError('Queued Session changed after acceptance')
    else:
        parents=[s for s in doc['sessions'] if s['session_id']==expected_parent]
        if len(parents)!=1:raise ValueError('Replan parent is absent or duplicated')
        parent=SessionSpec.model_validate(parents[0])
        if (parent.status.value!='replanned' or
            parent.result.metadata.get('replanned_session_id')!=session_id or
            session.replan_attempt!=parent.replan_attempt+1):
            raise ValueError('Child was not produced by the verified parent replan')
        # These are the exact lifecycle/description changes of the original
        # SessionVerifier._build_replanned_session; execution rights stay fixed.
        for item in (current,original):
            for key in ('session_id','parent_session_id','replan_attempt','task_description',
                        'status','created_at','updated_at','claimed_by','claim_token','result'):
                item.pop(key)
            item['retry']['attempted']=0
        if current!=original:raise ValueError('Replan child changed execution invariants')
    if (session.target_ref,session.skillruntime_ref)!=(component_receipt['target_ref'],component_receipt['skillruntime_ref']):
        raise ValueError('Dispatch changed the selected component references')
    if active_ids is not None:active_ids.add(session_id)
    return supervisor.run_once(session_id=session_id)


def raise_unverifiable_runtime_failure(supervisor, workspace, session_id, log):
    """Expose a stopped native execution failure that cannot reach the verifier.

    run_once returns True for both an executed episode and a caught protocol /
    preflight exception. The latter writes no verification bundle. Do not spend
    the semantic-verifier wait budget hiding that already recorded first error.
    """
    session=supervisor.registry.get_session(session_id)
    status=getattr(session.status,'value',session.status)
    if status not in ('failed','rejected','timed_out','cancelled','pending'):return
    result=session.result
    if result is None:return
    artifact=getattr(result,'artifact_dir',None)
    if artifact and (Path(workspace)/artifact/'verification_bundle.json').is_file():return
    # Applied semantic results are allowed to finish through the existing
    # outcome callback, including a completed failure or a verified replan.
    if getattr(result,'metadata',{}).get('verification'):return
    error_code=getattr(result,'error_code',None)
    if not error_code:return
    error_message=getattr(result,'error_message',None)
    record(log,event='native_runtime_failed_before_verification',session_id=session_id,
        status=status,error_code=error_code,error_message=error_message,
        verification_bundle_available=False)
    raise RuntimeError('Native runtime terminated before verification: '+session_id+
                       ' ['+str(error_code)+'] '+str(error_message))


def validate_queued_session(case,w,spec,previous_sessions,log,component_anchor=None):
    doc=read_yaml_block(w/'SESSIONS.md')
    events=[json.loads(line) for line in log.read_text().splitlines()]
    assert any(e.get('event')=='tool' and e.get('name')=='read_file' and
        e.get('params',{}).get('path','').endswith('benchmarking/SKILL.md') and
        not e.get('result','').startswith('Error') for e in events),'Native benchmarking skill was not read'
    old_ids={s['session_id'] for s in previous_sessions}
    new=[s for s in doc['sessions'] if s['session_id'] not in old_ids]
    assert len(new)==1,'Agent did not queue exactly one new session'
    assert [s for s in doc['sessions'] if s['session_id'] in old_ids]==previous_sessions,'Agent changed historical sessions'
    planned=new[0]
    from fastrsi.native_component_updates_v1 import requested as component_updates_requested
    component_mode=component_updates_requested(case)
    if component_mode:
        if planned['session_id']!=case['id']:raise ValueError('Queued Session ID differs from the current request')
        if component_anchor is None:raise ValueError('Component selection requires a pre-tool contract anchor')
        validate_component_selection(w,planned,component_anchor)
    expected=effective_execution_view(spec)
    actual=effective_execution_view(planned)
    if component_mode:
        # Normalize the same inert native defaults for a validated alias too.
        alias=copy.deepcopy(planned)
        alias['target_ref']=spec['target_ref'];alias['skillruntime_ref']=spec['skillruntime_ref']
        actual=effective_execution_view(alias)
        assert planned['task_description']==spec['task_description'],'Current immutable goal changed'
    if case.get('same_world_arm')=='choice':
        from fastrsi.same_world_native_binding import load_binding
        from same_world_launch import validate_step_choice
        binding=load_binding(case)
        expected['execution']['steps']=validate_step_choice(actual['execution']['steps'],binding['allowed_sequences'])
    expected.pop('task_description');actual.pop('task_description')
    diff={k:{'expected':expected[k],'actual':actual[k]} for k in expected if expected[k]!=actual[k]}
    assert not diff,'Frozen execution fields differ: '+json.dumps(diff,ensure_ascii=False)
    assert isinstance(planned['task_description'],str) and planned['task_description'].strip()
    if case.get('instruction_experience'):
        decision=json.loads((w/'REQUEST.json').read_text())['instruction_decision']
        assert planned['task_description']==decision['instruction'],'Agent did not apply the declared instruction decision'
        assert any(e.get('event')=='tool' and e.get('name')=='read_file' and e.get('params',{}).get('path','').endswith('instruction-reuse/EXPERIENCE.json') and not e.get('result','').startswith('Error') for e in events),'Experience was not actually read'
    if case.get('session_intent'):
        assert planned['task_description']==json.loads((w/'REQUEST.json').read_text())['goal'],'Current immutable goal changed'
    if case.get('same_world_registration'):
        assert planned['task_description']==case['task_description'],'Complete same-world goal changed'
    if case.get('require_discovery_reads'):
        read_names=discovery_reads(events,w)
        required={'RUNTIME.md','TARGETS.md','SKILLRUNTIME.md','SESSIONS.md','LOG.md','REQUEST.json'}
        assert required<=read_names,'Native discovery files not read: '+str(sorted(required-read_names))
    return planned


async def plan_session(agent, case, w, spec, previous_sessions, log, on_queued=None):
    queued=None
    draft_seen=False
    from fastrsi.native_component_updates_v1 import requested as component_updates_requested
    component_mode=component_updates_requested(case)
    component_anchor=component_selection_anchor(w,spec) if component_mode else None
    if component_mode:agent._fastrsi_component_anchor=component_anchor
    original_execute=agent.tools.execute
    async def execute(name,params):
        nonlocal queued,draft_seen
        path=Path(params.get('path',''))
        changes_queue=name in ('write_file','edit_file','append_yaml_item','patch_yaml_item') and path.name=='SESSIONS.md'
        if queued is not None and changes_queue:
            return 'Error: the requested Session has already been handed to the Watchdog; do not rewrite its live state.'
        if changes_queue and name=='patch_yaml_item' and (params.get('match_key')!='session_id' or params.get('match_value')!=case['id']):
            return 'Error: only the requested Session draft may be patched; preserve historical sessions.'
        queue_before=(w/'SESSIONS.md').read_bytes() if changes_queue else None
        prior_draft=draft_seen
        result=await original_execute(name,params)
        if changes_queue and not result.startswith('Error'):draft_seen=True
        # A saved draft can become acceptable after the missing discovery read.
        # Do not require a dummy rewrite merely to dispatch an unchanged draft.
        check=queued is None and draft_seen and (changes_queue or name=='read_file')
        if check:
            try:
                candidate=validate_queued_session(case,w,spec,previous_sessions,log,component_anchor=component_anchor)
            except yaml.YAMLError as exc:
                if changes_queue:
                    (w/'SESSIONS.md').write_bytes(queue_before)
                    draft_seen=prior_draft
                    record(log,event='invalid_queue_edit_rolled_back',error=str(exc))
                    return 'Error: invalid YAML edit was rolled back; no Session was dispatched. Read the restored queue and correct your tool arguments: '+str(exc)
                raise
            except ValidationError as exc:
                if changes_queue:
                    (w/'SESSIONS.md').write_bytes(queue_before)
                    draft_seen=prior_draft
                    record(log,event='invalid_session_schema_rolled_back',error=str(exc),after_tool=name)
                    action=('Retry append_yaml_item with one corrected Session item.'
                            if name=='append_yaml_item' else
                            'Read the restored draft and correct it with the existing file tools.')
                    return ('Error: Session schema invalid; this edit was rolled back and no Session was dispatched. '
                            'The item is a single Session, not the YAML document: document-level version/sessions keys do not belong inside it. '
                            +action+' Validation details: '+str(exc))
                record(log,event='queue_still_pending',error=str(exc),after_tool=name)
                return result+'\n\n[Runtime queue diagnostic] Invalid Session draft; correct it before dispatch. '+str(exc)
            except (AssertionError,KeyError,ValueError) as exc:
                if changes_queue:
                    record(log,event='queue_not_accepted',error=str(exc))
                    if str(exc)=='Native benchmarking skill was not read':
                        return ('Error: draft saved but NOT dispatched. Missing required read: '
                                'call read_file with path="skills/benchmarking/SKILL.md". '
                                'Patching Session fields cannot fix a missing read. Do not append again. '
                                'The read itself will recheck the saved draft; then follow any remaining diagnostic.')
                    if str(exc).startswith('Native discovery files not read:'):
                        return ('Error: draft saved but NOT dispatched. Read the missing files with read_file or read_files: '
                                +str(exc)+'. Patching Session fields cannot fix missing reads. '
                                'Do not append again; reads recheck the saved draft automatically.')
                    return 'Error: draft was saved but no Session was dispatched. Correct the following fields using the existing file tools: '+str(exc)
                record(log,event='queue_still_pending',error=str(exc),after_tool=name)
                return result+'\n\n[Runtime queue diagnostic] Draft remains unaccepted; no Session dispatched. '+str(exc)+' Read missing evidence or edit the existing draft; do not append a duplicate.'
            if component_mode:
                receipt=validate_component_selection(w,candidate,component_anchor)
                agent._fastrsi_component_receipt=receipt
                agent._fastrsi_component_lock=True
                record(log,event='native_components_consumed',session_id=candidate['session_id'],**receipt)
            queued=copy.deepcopy(candidate)
            from fastrsi.tool_origin import TOOL_ORIGIN
            record(log,event='queue_accepted',origin=TOOL_ORIGIN.get(),session_id=queued['session_id'],after_tool=name)
            if on_queued is not None:on_queued(queued)
            result += accepted_queue_receipt(queued['session_id'])
        return result
    agent.tools.execute=execute
    extra=''
    if component_mode:
        extra+=(' Registered component selection is enabled: choose target_ref and skillruntime_ref from the current compatible registries; '
                'these two references may differ from REQUEST.json.session. Keep the immutable task description and every other Session field unchanged. '
                'Only the same physical task, observations, VLA endpoint and registered OpenPI runtime contracts are allowed. '
                'After queue acceptance the component files are sealed until verified termination.')
    if case.get('session_flow_artifact'):
        from fastrsi.session_flow import execute_flow
        choice_callback=None
        if case.get('same_world_arm')=='choice':
            from same_world_decision import native_order_callback
            choice_callback=native_order_callback(agent,w,case,lambda **kw: record(log,**kw))
        with span('public_session_flow_planning',case=case['id']):
            flow=json.loads(Path(case['session_flow_artifact']).read_text())
            assert hashlib.sha256(Path(case['session_flow_artifact']).read_bytes()).hexdigest()==case['session_flow_sha256'],'Flow artifact changed'
            outcome=await execute_flow(agent.tools.execute,w,flow,lambda **kw: record(log,**kw),
                                       choose_steps=choice_callback)
        record(log,event='session_flow_outcome',outcome=outcome)
        if choice_callback is not None and outcome.get('status')!='QUEUED':
            agent.tools.execute=original_execute
            raise ValueError('Order decision failed before queue; do not replan the complete Session: '+str(outcome.get('reason')))
        if outcome.get('write_attempted') and queued is None:
            # An interrupted append may have written a valid draft. Let a fresh
            # native read validate/dispatch it, without a second side effect.
            try:
                draft_seen=any(s.get('session_id')==case['id'] for s in read_yaml_block(w/'SESSIONS.md').get('sessions',[]))
            except (yaml.YAMLError,ValueError,TypeError):
                draft_seen=False
        if queued is not None:
            agent.tools.execute=original_execute
            return 'Public deterministic flow queued the validated Session.',queued
        extra+=' Public flow handoff (actual completed operations; inspect current queue and resume the draft, never duplicate it): '+json.dumps(outcome,ensure_ascii=False)
    if case.get('require_discovery_reads'):
        extra+=' Actually read RUNTIME.md, TARGETS.md, SKILLRUNTIME.md, SESSIONS.md, LOG.md and REQUEST.json before writing; these are the original skill discovery files and the supplied request.'
    if case.get('workflow_macro'):
        extra+=' Use the compiled accepted-discovery procedure. The initial benchmarking skill read includes the current discovery files; inspect those live results and continue queue preparation without redundant reads unless errors or truncation need resolution.'
    elif case.get('workflow_artifact'):
        extra+=' For this matching LIBERO policy_loop request, use the active accepted-discovery skill: after reading the original benchmarking skill, call read_files once for the six current discovery files listed in accepted-discovery. Inspect all returned contents before queueing; use individual reads only for errors, pagination or additional evidence. This changes the read procedure only; all original capability checks and Session constraints still apply.'
    if case.get('workflow')=='batch_reads':
        extra+=' After reading the benchmarking skill, emit the independent read_file calls for those six files together in one assistant turn, before considering the Session. The native Agent supports several tool calls per turn. Keep all evidence and validation unchanged.'
    if case.get('planning_only'):
        extra+=' This is a dry planning probe. No Watchdog will dispatch this pending Session; report preparation only, not execution.'
    if case.get('instruction_experience') or case.get('session_intent'):
        with span('native_agent_planning',case=case['id']):
            experience_instruction=('Do not invent results or create extra sessions. Report when queued. '
                'Experience maintenance prescribed by the original skill is permitted using already verified outcomes; '
                'no additional reflection or consolidation step is required. '
                if native_maintenance_enabled(case) else
                'Do not modify experience, invent results, perform extra reflection, or create extra sessions. Report when queued.')
            prompt=(
                'Use the original benchmarking skill and actual tools to execute the requested LIBERO policy_loop task. '
                'Read skills/benchmarking/SKILL.md, the runtime registries and REQUEST.json, and the experience_file named there. Independent file reads may use the common read_files tool. '
                'REQUEST.json supplies the immutable user goal, benchmark identifiers and execution constraints, not a complete Session. '
                'Construct exactly one pending Session using an enabled matching target and SkillRuntime from current registries. '
                'Keep session_id, goal_id (null means no associated goal record), benchmark mapping, execution constraints and verification profile exactly as requested. '
                'Use instruction_decision.instruction exactly as task_description: this is a predeclared method decision or paid candidate test, not a new task goal. '
                'Use registered routing endpoints. Preserve all historical sessions and append only the new item with append_yaml_item. '
                'Use workspace-relative file paths. The append item contains only one Session; never include the document-level version or sessions keys inside it. '
                'Include the entire benchmark mapping from REQUEST.json. Copy the values inside execution_constraints exactly, rather than the RUNTIME example defaults. '
                'For a rejected saved draft, first follow the diagnostic: a missing read requires reading that file, not patching. '
                'Incorrect fields require patch_yaml_item matched by session_id and a patch of only those fields. Never append a saved draft again or rewrite YAML indentation. '
                +experience_instruction+extra)
            if not case.get('instruction_experience'):
                prompt=prompt.replace(', and the experience_file named there','').replace('Use instruction_decision.instruction exactly as task_description: this is a predeclared method decision or paid candidate test, not a new task goal. ','Use REQUEST.json.goal exactly as task_description. Copy routing_constraints exactly. ')
            response=await agent.process_direct(prompt,session_key=case.get('conversation_key','cli:'+case['id']))
    else:
        with span('native_agent_planning',case=case['id']):
            response=await agent.process_direct(
                'The files below already exist in your workspace. Actually use the registered file tools; do not simulate files or present a hypothetical example. '
                'First read skills/benchmarking/SKILL.md with read_file, then follow its capability-discovery steps. Use its policy_loop path for this one LIBERO simulation task: '+case['task_description']+'. '
                'Read REQUEST.json: its session object is the complete task-specific Session to copy, including its benchmark mapping. '
                'Read the runtime registries and existing queue before editing. Append exactly one pending Session based on REQUEST.json. '
                'Copy EVERY key and value from REQUEST.json.session, including all nested benchmark fields; do not drop fields or add guessed optional defaults from the generic RUNTIME example. '
                'Only task_description may be refined using relevant existing feedback while preserving the goal. '
                'Preserve existing Session history. For an empty queue, write_file may initialize the queue. '
                'For a nonempty queue, use append_yaml_item with path="SESSIONS.md", list_key="sessions", and item containing ONLY the new Session object. '
                'In the append call, serialize task_description as the final property inside item, after all its nested mappings; keep all field values unchanged. '
                'Never repeat old records in tool arguments. If a draft was saved but rejected, correct that draft with edit_file; do not append a duplicate. '
                'Do not invent feedback or create extra sessions. Report when queued.'+extra,
                session_key=case.get('conversation_key','cli:'+case['id']))
    planned=queued if queued is not None else validate_queued_session(case,w,spec,previous_sessions,log,component_anchor=component_anchor)
    agent.tools.execute=original_execute
    return response, planned


async def run_case(case, accepted):
    os.environ['FASTRSI_CURRENT_CASE']=case['id']
    folder=ROOT/case['id'];folder.mkdir();w,spec,previous_sessions=setup_workspace(folder,case,accepted)
    capability_receipt=None;capability_error=None
    if case.get('capability_interface'):
        from fastrsi.capability_interface import install as install_capability_interface
        capability_receipt=install_capability_interface(w,case)
        (folder/'CAPABILITY_INTERFACE.json').write_text(json.dumps(capability_receipt,indent=2))
    if case.get('workflow_artifact'):
        from fastrsi.workflow_reuse import install_workflow
        install_workflow(w,case['workflow_artifact'])
    old_ids={x['session_id'] for x in previous_sessions}
    evaluation=case['mode']=='evaluation'
    log=folder/'AGENT_EVENTS.jsonl';before=hashes(w)
    provider=measured_provider(folder,'agent',2048)
    verifier_provider=measured_provider(folder,'verifier',384)
    os.environ['FASTRSI_TOKEN_BUDGET']='384'
    verifier=SessionVerifier(workspace=w,provider=verifier_provider,model=os.environ.get('FASTRSI_AGENT_MODEL','qwen25vl7b'),max_replans=2,
        rgb_retention='all',timeout_s=300,service_port=PORTS['verifier'],
        service_provider_spec={'provider_name':'custom','model':os.environ.get('FASTRSI_AGENT_MODEL','qwen25vl7b'),'api_base':QWEN_API,'max_tokens':384,'temperature':0})
    supervisor=WatchdogSupervisor(w,verification_enabled=True,verification_settings={
        'local_service_url':f"http://127.0.0.1:{PORTS['verifier']}",'timeout_s':300,'max_replans_per_episode':2})
    from fastrsi.native_component_updates_v1 import requested as component_updates_requested
    agent=bind_agent(w,provider,log,evaluation=evaluation,verifier=verifier,max_iterations=10,read_files=case.get('read_files_tool',False),workflow_macro=case.get('workflow_macro',False),visible_reads=case.get('visible_reads',False),component_updates=component_updates_requested(case))
    row={'case':case,'status':'RUNNING','experience_before':before,'accepted_source':str(accepted) if accepted else None}
    started=time.perf_counter();outcomes=[];verifier_task=None;runtime_task=None
    restore_component_adapter=None
    restore_registered_consumer=None
    if component_updates_requested(case):
        from fastrsi.native_component_updates_v1 import install_prefix_consumer
        restore_registered_consumer=install_prefix_consumer(w, lambda **event: record(log, **event))
    if case.get('prerequisite_component_probe'):
        from fastrsi.prerequisite_component import install as install_component
        from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
        restore_component_adapter=install_component(OpenPIPi05Adapter,case,
            hashlib.sha256((CAMPAIGN/'MODEL_CONFIG.json').read_bytes()).hexdigest(),emit)
        row['component_probe_scope']='Primitive feasibility; original full-goal evaluator unchanged'
    restore_validity_adapter=None
    validity_observer=None
    if case.get('decision_validity_shadow'):
        from fastrsi.decision_validity_runtime import ValidityObserver, install
        from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
        validity_observer=ValidityObserver(case,hashlib.sha256((CAMPAIGN/'MODEL_CONFIG.json').read_bytes()).hexdigest())
        restore_validity_adapter=install(OpenPIPi05Adapter,validity_observer,emit)
    restore_stage_adapter=None
    if case.get('stage_flow_artifact'):
        if case.get('placement_recovery',False):
            from fastrsi.placement_recovery import load as load_flow, decision as flow_decision, policy_input as stage_input
        else:
            from fastrsi.staged_flow import load as load_flow, decision as flow_decision, policy_input as stage_input
        from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
        flow=load_flow(case['stage_flow_artifact'],case['stage_flow_sha256'])
        model_sha=hashlib.sha256((CAMPAIGN/'MODEL_CONFIG.json').read_bytes()).hexdigest()
        chosen=flow_decision(flow,case,model_sha,probe=case.get('stage_flow_probe',False))
        record(log,event='stage_flow_read',artifact=case['stage_flow_artifact'],sha256=case['stage_flow_sha256'],
               status=flow['status'],action=chosen,reader='runtime_stage_controller',session_id=case['id'])
        row['stage_flow_decision']={'action':chosen,'status':flow['status'],'sha256':case['stage_flow_sha256']}
        restore_stage_adapter=OpenPIPi05Adapter.to_policy_input
        def stage_adapter(adapter,observation,context):
            payload=restore_stage_adapter(adapter,observation,context)
            value,event=stage_input(payload,observation,context,case['id'],chosen)
            if event:emit({'event':'stage_flow_choice','case':case['id'],**event})
            return value
        OpenPIPi05Adapter.to_policy_input=stage_adapter
    restore_action_adapter=None
    window_controller=None
    selected_mode=None
    if case.get('action_window_artifact'):
        if case.get('stage_flow_artifact'):raise ValueError('Conflicting execution probes')
        from fastrsi.action_window import load as load_window, decision as window_decision, apply as window_apply
        from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
        state=load_window(case['action_window_artifact'],case['action_window_sha256'])
        model_sha=hashlib.sha256((CAMPAIGN/'MODEL_CONFIG.json').read_bytes()).hexdigest()
        chosen_window=window_decision(state,case,model_sha,probe=case.get('action_window_probe',False))
        record(log,event='action_window_read',artifact=case['action_window_artifact'],sha256=case['action_window_sha256'],
               status=state['status'],action=chosen_window,reader='runtime_action_controller',session_id=case['id'])
        row['action_window_decision']={'action':chosen_window,'status':state['status'],'sha256':case['action_window_sha256']}
        restore_action_adapter=OpenPIPi05Adapter.from_policy_output
        def action_adapter(adapter,payload,context):
            chunk=restore_action_adapter(adapter,payload,context)
            chunk,event=window_apply(chunk,context,case['id'],chosen_window)
            if event:emit({**event,'case':case['id']})
            return chunk
        OpenPIPi05Adapter.from_policy_output=action_adapter
    if case.get('feedback_window_mode'):
        if case.get('stage_flow_artifact') or case.get('action_window_artifact'):
            raise ValueError('Conflicting execution controllers')
        from fastrsi.feedback_window import FeedbackWindow, commitment_binding
        from PhyAgentOS.runtime.adapters.openpi.pi05_policy_adapter import OpenPIPi05Adapter
        selected_mode=case['feedback_window_mode']
        horizon_binding=commitment_binding(case, selected_mode)
        if horizon_binding is not None:
            row['horizon_commitment_binding']=dict(horizon_binding)
        if case.get('window_learning_artifact'):
            from fastrsi.window_learning import load as load_learning, decide as decide_learning
            with span('window_learning_read_select',case=case['id']):
                learned=load_learning(case['window_learning_artifact'],case['window_learning_sha256'])
                selection=decide_learning(learned,case['task_description'])
            selected_mode=selection['selected_mode']
            row['window_learning_decision']={**selection,'snapshot_sha256':case['window_learning_sha256']}
            record(log,event='window_learning_read',**row['window_learning_decision'],session_id=case['id'])
        if case.get('task_window_artifact'):
            if case.get('window_learning_artifact'):raise ValueError('Conflicting window learners')
            from fastrsi.task_window import load as load_task_window, select as select_task_window
            with span('task_window_read_select',case=case['id']):
                task_table=load_task_window(case['task_window_artifact'],case['task_window_sha256'])
                selection=select_task_window(task_table,case['task_description'])
            selected_mode=selection['selected_mode']
            row['task_window_decision']={**selection,'snapshot_sha256':case['task_window_sha256']}
            record(log,event='task_window_read',**row['task_window_decision'],session_id=case['id'])
        if selected_mode=='short_exit_once':
            if case.get('window_learning_artifact') or case.get('task_window_artifact'):
                raise ValueError('Experimental intervention is not a historical learned policy')
            from fastrsi.short_exit_probe import ShortExitOnce
            window_controller=ShortExitOnce(case)
        else:
            window_controller=FeedbackWindow(case,selected_mode)
        restore_action_adapter=OpenPIPi05Adapter.from_policy_output
        record(log,event='feedback_window_bound',mode=selected_mode,reader='runtime_action_controller',session_id=case['id'])
        if horizon_binding is not None:
            record(log,event='horizon_commitment_bound',session_id=case['id'],**horizon_binding)
        def feedback_adapter(adapter,payload,context):
            chunk=restore_action_adapter(adapter,payload,context)
            chunk,event=window_controller.apply(chunk,context)
            emit({**event,'case':case['id'],'runtime_session_id':context.get('session_id'),
                  'native_attempt':window_controller.case.get('native_replan_attempt',0),
                  'native_attempt_goal':context.get('task_description')})
            return chunk
        OpenPIPi05Adapter.from_policy_output=feedback_adapter
    from fastrsi.native_component_updates_v1 import requested as component_updates_requested
    component_mode=component_updates_requested(case)
    active_component_sessions=set()
    restore_verification_selector=(scope_component_verification(verifier.registry,active_component_sessions)
                                   if component_mode else None)
    original_verify=verifier.verify_session
    async def background_verify(session_id,*,source='tool'):
        if component_mode and session_id not in active_component_sessions:
            return dict(ok=False,code='out_of_scope',session_id=session_id)
        with span('semantic_verification',case=case['id'],source=source):
            outcome=await original_verify(session_id,source=source)
        if source=='auto':outcomes.append(outcome)
        return outcome
    verifier.verify_session=background_verify
    async def execute_runtime(initial_session_id):
        nonlocal window_controller
        from fastrsi.native_component_updates_v1 import requested as component_updates_requested
        component_mode=component_updates_requested(case)
        dispatch_id=initial_session_id
        dispatch_parent=None
        from fastrsi.resident_client import from_environment
        resident=from_environment()
        for attempt in range(3):
            if attempt and window_controller is not None:
                from fastrsi.native_window_attempt import rebind_native_replan
                with span('native_window_attempt_binding',case=case['id'],attempt=attempt):
                    window_controller,binding=rebind_native_replan(
                        w,case,selected_mode,outcomes[-1],attempt,window_controller.case['id'],spec,
                        previous_commitment=window_controller.commitment)
                row.setdefault('native_window_attempts',[]).append(binding)
                record(log,event='native_window_attempt_bound',case=case['id'],**binding)
            if attempt and resident is not None:
                request_id='watchdog-handoff-'+hashlib.sha256((resident.session_id+'|'+case['id']+'|'+str(attempt)).encode()).hexdigest()
                record(log,event='resident_policy_handoff_requested',case=case['id'],attempt=attempt,
                       resident_session_id=resident.session_id,request_id=request_id)
                try:
                    with span('resident_policy_handoff',case=case['id'],attempt=attempt):
                        receipt=await asyncio.to_thread(resident.handoff,request_id)
                except BaseException as exc:
                    record(log,event='resident_policy_handoff_failed',case=case['id'],attempt=attempt,error=repr(exc))
                    raise
                row.setdefault('resident_handoffs',[]).append(receipt)
                record(log,event='resident_policy_handoff',case=case['id'],attempt=attempt,control=receipt)
            with span('os_execution',case=case['id'],attempt=attempt):
                if component_mode:
                    ran=await asyncio.to_thread(dispatch_registered_session,supervisor,w,dispatch_id,
                        agent._fastrsi_component_receipt,expected_parent=dispatch_parent,active_ids=active_component_sessions)
                else:
                    ran=await asyncio.to_thread(supervisor.run_once)
            if not ran:break
            if component_mode or attempt==0:
                raise_unverifiable_runtime_failure(supervisor,w,dispatch_id,log)
            deadline=time.monotonic()+330
            while len(outcomes)<=attempt:
                if verifier_task.done():
                    raise RuntimeError('Background verifier exited: '+str(verifier_task.exception()))
                if time.monotonic()>deadline:raise TimeoutError('Background verification timeout')
                await asyncio.sleep(0.1)
            outcome=outcomes[-1]
            if component_mode and outcome.get('session_id')!=dispatch_id:
                raise ValueError('Verifier outcome is not for the dispatched Session')
            if outcome.get('status')!='replanned':
                if component_mode and outcome.get('code')=='verified':
                    agent._fastrsi_component_lock=False
                break
            if component_mode:
                child=outcome.get('replanned_session_id')
                if not isinstance(child,str) or not child or child==dispatch_id:
                    raise ValueError('Verified replan lacks a new explicit child Session')
                dispatch_parent,dispatch_id=dispatch_id,child
    def queued_callback(planned):
        nonlocal runtime_task
        record(log,event="session_dispatched",session_id=planned["session_id"])
        runtime_task=asyncio.create_task(execute_runtime(planned["session_id"]))
    try:
        await verifier.start()
        verifier_task=asyncio.create_task(verifier.run())
        response, planned = await plan_session(agent,case,w,spec,previous_sessions,log,on_queued=queued_callback)
        row.update(agent_response=response,planned_session=planned)
        (folder/'PLANNED_SESSION.json').write_text(json.dumps(planned,indent=2))
        assert runtime_task is not None,'No validated Session was dispatched'
        await runtime_task
        assert outcomes and all(o.get('code')=='verified' for o in outcomes),outcomes
        if os.environ.get('FASTRSI_CONSUME_TERMINAL')=='1':
            from fastrsi.terminal_results import consume_terminal
            with span('native_terminal_report',case=case['id']):
                row['terminal_report']=await consume_terminal(agent,case,w,log)
        if component_updates_requested(case):
            from fastrsi.native_component_updates_v1 import registry_hashes
            drafts=validate_component_drafts(w,spec,planned,agent._fastrsi_component_anchor)
            row['component_draft_structure']={'registry_sha256':registry_hashes(w),
                'fixed_contract_checks':drafts,
                'scope':'Registered factories and schemas only; not capability acceptance'}
        if case.get('require_clean_model_calls'):
            assert all((p.parent/'RESPONSE.json').exists() for p in folder.glob('*_call_*/REQUEST.json')), 'Model call failed inside native loop; not a clean baseline'
        row.update(status='PASSED',outcomes=outcomes,sessions=[s for s in read_yaml_block(w/'SESSIONS.md')['sessions'] if s['session_id'] not in old_ids])
        native=[json.loads(x) for x in (ROOT/'TARGET_COST_EVENTS.jsonl').read_text().splitlines()]
        ids={s['session_id'] for s in row['sessions']}
        native=[e for e in native if e.get('event')=='native_observation' and e.get('session_id') in ids]
        assert native,'Missing native scoring evidence'
        row['environment_success']=native[-1]['native_success']
        row['native_terminal_by_session']={sid:next(e for e in reversed(native) if e['session_id']==sid) for sid in ids}
    except BaseException:
        row.update(status='FAILED',error=traceback.format_exc());raise
    finally:
        if runtime_task is not None and not runtime_task.done():
            runtime_task.cancel()
            await asyncio.gather(runtime_task,return_exceptions=True)
        if restore_stage_adapter is not None:
            OpenPIPi05Adapter.to_policy_input=restore_stage_adapter
        if restore_action_adapter is not None:
            OpenPIPi05Adapter.from_policy_output=restore_action_adapter
        if restore_validity_adapter is not None:
            restore_validity_adapter()
            row['decision_validity_shadow']=validity_observer.summary()
        if restore_component_adapter is not None:
            restore_component_adapter()
        if restore_registered_consumer is not None:
            restore_registered_consumer()
        if restore_verification_selector is not None:
            verifier.registry.first_awaiting_verification=restore_verification_selector
        verifier.stop()
        if verifier_task is not None:
            verifier_task.cancel()
            await asyncio.gather(verifier_task,return_exceptions=True)
        await finish_agent(agent,log)
        await provider._client.close();await verifier_provider._client.close()
        if capability_receipt is not None:
            from fastrsi.capability_interface import verify as verify_capability_interface
            try:
                verify_capability_interface(w,capability_receipt)
            except Exception as exc:
                # Preserve the paid execution and its physical outcome before
                # aborting; a changed interface must never reach admission.
                capability_error=exc
                row.update(status='FAILED',capability_interface_error=repr(exc))
        row.update(seconds=time.perf_counter()-started,experience_after=hashes(w),outcomes=outcomes)
        # Preserve physical evidence even when verification or another chain stage fails.
        observed=[json.loads(x) for x in (ROOT/'TARGET_COST_EVENTS.jsonl').read_text().splitlines()] if (ROOT/'TARGET_COST_EVENTS.jsonl').exists() else []
        observed=[e for e in observed if e.get('event')=='native_observation' and str(e.get('session_id','')).startswith(case['id'])]
        row['environment_success']=observed[-1]['native_success'] if observed else None
        row['verification_status']=[o.get('code') for o in outcomes]
        row['chain_status']=row['status']
        if evaluation and hashes(w)!=before:
            row.update(status='FAILED',isolation_error='Evaluation mutated persistent learning state')
        row['chain_status']=row['status']
        row['evidence_layers']=evidence_layers(row['environment_success'],outcomes,row['chain_status'])
        (folder/'RESULT.json').write_text(json.dumps(row,indent=2,default=str))
        if capability_error is not None:
            raise RuntimeError('Capability interface changed; execution evidence was retained') from capability_error
    if evaluation:
        assert hashes(w)==before,'Evaluation changed fixed experience'
        return row,accepted
    if admission_mode()=='guarded_v1':
        with span('experience_admission',case=case['id']):
            next_source,decision=commit_experience(w,row,accepted,folder/'admission')
        row['experience_admission']={k:decision.get(k) for k in ['version','status','reasons','next_source']}
        (folder/'RESULT.json').write_text(json.dumps(row,indent=2,default=str))
        return row,next_source
    snapshot=folder/'accepted_experience'
    # Native lessons are retained as raw experience; acceptance metadata distinguishes
    # successful receipts from failure observations. Neither is labelled a verified skill.
    with span('experience_snapshot'):
        shutil.copytree(w,snapshot)
    (snapshot/'RECEIPT.json').write_text(json.dumps({'source_case':case['id'],'chain_status':row['status'],
        'environment_success':row['environment_success'],'hashes':hashes(snapshot),
        'interpretation':'native feedback, not a newly validated transferable robot skill'},indent=2))
    return row,snapshot


async def main():
    os.environ['FASTRSI_TRACE']=str(ROOT/'OS_COST_EVENTS.jsonl')
    cases=json.loads((CAMPAIGN/'ROSTER.json').read_text())
    metadata_expected={}
    for case in cases:
        if case.get('same_world_registration'):
            registration=json.loads(Path(case['same_world_registration']).read_text())
            metadata_expected[case['id']]=registration['policy_metadata_expected']
    metadata_recorded=set()
    original=OpenPIClientPolicyWrapper.infer
    def infer(self,*a,**kw):
        observation=a[0] if a else kw.get('observation',{})
        with span('policy_client'):
            current=os.environ.get('FASTRSI_CURRENT_CASE')
            if current in metadata_expected:
                expected=metadata_expected[current]
                if any(self.metadata.get(key)!=value for key,value in expected.items()):
                    raise ValueError('Same-world policy metadata differs from registration')
                identity=(current,id(self))
                if identity not in metadata_recorded:
                    emit({'event':'same_world_policy_metadata','case':current,'actual':self.metadata,'expected':expected})
                    metadata_recorded.add(identity)
            arrays={key:{'shape':list(value.shape),'dtype':str(value.dtype),
                         'sha256':hashlib.sha256(value.tobytes()).hexdigest()}
                    for key,value in observation.items() if hasattr(value,'tobytes') and hasattr(value,'shape')}
            state=observation.get('observation/state')
            emit({'event':'actual_policy_input','case':os.environ.get('FASTRSI_CURRENT_CASE'),
                  'prompt':observation.get('prompt'),'keys':sorted(observation),
                  'array_bindings':arrays,'robot_state':state.tolist() if hasattr(state,'tolist') else None})
            return original(self,*a,**kw)
    OpenPIClientPolicyWrapper.infer=infer
    old_bundle=ResultWriter.write_verification_bundle
    def bundle(self,*a,**kw):
        with span('evidence_preparation'):return old_bundle(self,*a,**kw)
    ResultWriter.write_verification_bundle=bundle
    initial_source=os.environ.get('FASTRSI_INITIAL_EXPERIENCE')
    report={'status':'RUNNING','experience_admission_mode':admission_mode(),'cases':[]};accepted=Path(initial_source) if initial_source else None;started=time.perf_counter()
    try:
        for case in cases:
            row,accepted=await run_case(case,accepted)
            if case.get('budget_checkpoint'):
                # Admission and Agent shutdown are complete. Allocation may inspect
                # development evidence but cannot dispatch or skip the next task.
                from continuous_budget_checkpoint import run_checkpoint
                try:
                    with span('development_budget_checkpoint'):
                        row['budget_checkpoint']=run_checkpoint(Path('/project'),CAMPAIGN,ROOT,row,accepted,time.perf_counter()-started)
                except Exception:
                    row['budget_checkpoint']={'status':'CHECKPOINT_FAILED','error':traceback.format_exc(),
                                              'actual_worker_dispatch':False}
                # Keep the source case RESULT immutable: checkpoint binds its SHA.
            report['cases'].append(row)
        report['status']='COMPLETED'
    except BaseException:
        report.update(status='FAILED',error=traceback.format_exc())
        report['cases']=[json.loads(p.read_text()) for p in sorted(ROOT.glob('*/RESULT.json'))]
        raise
    finally:
        report['seconds']=time.perf_counter()-started
        (ROOT/'RESULT.json').write_text(json.dumps(report,indent=2,default=str))
        print(json.dumps({'status':report['status'],'cases':len(report['cases']),'seconds':report['seconds']}),flush=True)

if __name__=='__main__':asyncio.run(main())
