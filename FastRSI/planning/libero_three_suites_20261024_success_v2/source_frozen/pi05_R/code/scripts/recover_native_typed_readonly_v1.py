"""Separate CPU consumption of an actually committed, partially failed teacher.

Never rewrites the teacher result, reruns its model, dispatches a Watchdog or
constructs a physical target. Public Session flow and original tools are shared.
"""
import argparse
import asyncio
import hashlib
import json
import os
import platform
from pathlib import Path
import time
import traceback

VERSION='native_typed_readonly_recovery_v1'
PARTIAL='PARTIAL_TYPED_COMMIT_WITH_SEPARATE_READONLY_RECEIPT'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def manifest(root):
    rows={}
    for p in sorted(Path(root).rglob('*')):
        if p.is_symlink():raise ValueError('Symlink input rejected')
        if p.is_file() and '__pycache__' not in p.parts:rows[p.relative_to(root).as_posix()]=sha(p)
    return rows

def verify_partial(campaign, teacher, launch, *, result_sha, launch_sha, registry_sha):
    import re
    import probe_native_goal_component_update_v1 as base
    from orchestrate_native_sampler_agent_arm_v1 import declaration
    campaign,teacher,launch=Path(campaign),Path(teacher),Path(launch)
    result_path=teacher/'RESULT.json';w=teacher/'proposal/workspace'
    if sha(result_path)!=result_sha or sha(launch)!=launch_sha:raise ValueError('Raw failed result/launch identity differs')
    result=read(result_path);start=read(launch);command=' '.join(start.get('argv',[]))
    if (start.get('returncode')!=0 or not re.fullmatch(r'[a-f0-9]{64}',start.get('stdout','').strip()) or
        'probe_native_goal_component_update_v1.py' not in command or '--allow-real-model-calls' not in command):
        raise ValueError('Actual teacher CLI launch missing')
    if (result.get('version')!=base.VERSION or result.get('status')!='FAILED' or result.get('new_session') is not None or
        'readonly_selection' not in result.get('error','') or 'plan_session' not in result.get('error','') or
        'Agent did not queue exactly one new session' not in result.get('error','')):
        raise ValueError('Failure is not the specific post-commit readonly queue failure')
    proposal=result.get('proposal',{});decl=declaration(w)
    if not decl or sha(w/'SKILLRUNTIME.md')!=registry_sha or proposal.get('registry_sha256')!=registry_sha or proposal.get('declaration')!=decl:
        raise ValueError('Actual saved proposal registry differs')
    required=set(base.REQUIRED_READS)|{base.STATE_REL,base.MANIFEST_REL}
    if set(proposal.get('complete_reads',[]))!=required:raise ValueError('Six complete teacher reads not recorded')
    transactions=[r for r in result.get('typed_component_transactions',[]) if r.get('candidate_id')=='dual_anchor_v1' and r.get('status')=='UNVALIDATED_DRAFT_REGISTERED']
    if len(transactions)!=1:raise ValueError('Expected one actual successful typed commit')
    transaction=transactions[0]
    if transaction.get('registry_after_sha256')!=registry_sha or transaction.get('declaration')!=decl or set(transaction.get('complete_reads',[]))!=required:
        raise ValueError('Typed transaction does not bind this declaration and reads')
    events_path=teacher/'proposal/AGENT_EVENTS.jsonl';events=[json.loads(x) for x in events_path.read_text().splitlines()]
    matched=[e for e in events if e.get('event')=='typed_component_transaction' and all(e.get(k)==v for k,v in transaction.items())]
    if len(matched)!=1:raise ValueError('Raw original tool transaction event missing')
    raw_calls={p.relative_to(teacher).as_posix():sha(p) for p in teacher.glob('proposal/*_call_*/[RQ]*.json')}
    if not any(k.endswith('/REQUEST.json') for k in raw_calls) or not any(k.endswith('/RESPONSE.json') for k in raw_calls):raise ValueError('Paid teacher raw calls absent')
    inputs=base.campaign_inputs(campaign);staged=read(w/base.MANIFEST_REL);draft=read(teacher/'proposal/DRAFT.json')
    if draft.get('source_manifest')!=staged:raise ValueError('Original staged manifest differs from paid proposal receipt')
    if staged['input_sha256']!={k:hashlib.sha256(v).hexdigest() for k,v in inputs.items()}:raise ValueError('Original campaign inputs changed')
    for rel,digest in staged['staged_sha256'].items():
        p=Path(rel)
        if p.is_absolute() or '..' in p.parts or sha(w/p)!=digest:raise ValueError('Staged immutable input changed')
    if sha(w/base.STATE_REL)!=proposal.get('current_component_state_sha256'):raise ValueError('Original pre-update state changed')
    return dict(version=PARTIAL,teacher_result_sha256=result_sha,teacher_launch_sha256=launch_sha,
        teacher_registry_sha256=registry_sha,source_workspace_manifest=manifest(w),declaration=decl,
        raw_transaction_events_sha256=sha(events_path),typed_transaction=transaction,raw_model_artifacts=raw_calls,
        campaign_inputs_sha256=staged['input_sha256'],inputs_unchanged=True,original_teacher_status='FAILED',
        teacher_model_calls=result.get('model_calls'),teacher_elapsed_seconds=result.get('elapsed_seconds'))

async def recover(campaign,teacher,launch,output,*,result_sha,launch_sha,registry_sha):
    campaign,teacher,output=Path(campaign),Path(teacher),Path(output)
    if output.exists():raise ValueError('Fresh separate recovery output required')
    output.mkdir(parents=True);tick=time.perf_counter();result=dict(version=VERSION,status='RUNNING',python=platform.python_version(),producer_source_sha256=sha(__file__),models=0,resets=0,actions=0,task_executed=False,learning_gain_claimed=False)
    teacher_before=manifest(teacher);agent=None
    try:
        evidence=verify_partial(campaign,teacher,launch,result_sha=result_sha,launch_sha=launch_sha,registry_sha=registry_sha)
        save(output/'PARTIAL_COMMIT_EVIDENCE.json',evidence)
        os.environ['FASTRSI_NATIVE_RUN']=str(output/'stream');os.environ['FASTRSI_EXPERIENCE_ADMISSION']='legacy_raw'
        import run_native_baseline as native
        import probe_native_goal_component_update_v1 as base
        from fastrsi.native_binding import bind_agent,finish_agent,hashes
        from fastrsi.session_flow import artifact
        from PhyAgentOS.providers.base import LLMProvider
        from PhyAgentOS.agent.memory import MemoryStore
        class NoModel(LLMProvider):
            def __init__(self):super().__init__();self.calls=0
            def get_default_model(self):return 'CPU_PUBLIC_FLOW_NO_MODEL'
            async def chat(self,*a,**kw):self.calls+=1;raise AssertionError('Readonly public flow attempted new model reasoning')
        folder=output/'selection';folder.mkdir();provider=NoModel();source=teacher/'proposal/workspace'
        case=dict(id='recovered_public_selection',mode='evaluation',suite='libero_90',task_id=64,init_state_id=29,max_steps=400,
            task_description=base.component.SOURCE_GOAL,development_suite_extension='libero90_stack_flow_v1',
            native_registry_lifecycle='incremental_v1',native_component_updates='registered_v1',registered_goal_component=True,
            registered_goal_component_physical_v1=True,native_experience_maintenance=False,session_intent=True,
            model_config=str(campaign/'MODEL_CONFIG.json'),model_config_sha256=sha(campaign/'MODEL_CONFIG.json'))
        w,spec,previous=native.setup_workspace(folder,case,source)
        if sha(w/'SKILLRUNTIME.md')!=registry_sha:raise ValueError('Inherited registry bytes differ')
        from probe_native_goal_learning_cycle_v1 import current_component_state
        state=current_component_state(w,spec,native)
        if state.get('component_is_registered') is not True or state.get('native_goal_component')!=evidence['declaration']:
            raise ValueError('New Session native schema does not see the saved component')
        save(w/base.STATE_REL,state);state_sha=sha(w/base.STATE_REL)
        flow=artifact((w/'skills/benchmarking/SKILL.md').read_bytes(),'common_original_benchmark_contract')
        save(output/'FLOW.json',flow);case.update(session_flow_artifact=str(output/'FLOW.json'),session_flow_sha256=sha(output/'FLOW.json'))
        save(output/'CASE.json',case);log=folder/'AGENT_EVENTS.jsonl';before=hashes(w)
        agent=bind_agent(w,provider,log,evaluation=True,component_updates=True,read_files=True)
        from fastrsi.paged_reads import read_complete
        from fastrsi.tool_origin import TOOL_ORIGIN
        pages=[];token=TOOL_ORIGIN.set('public_component_state_read')
        try:consumed=await read_complete(agent.tools.execute,base.STATE_REL,on_page=pages.append)
        finally:TOOL_ORIGIN.reset(token)
        if json.loads(consumed)!=state or sha(w/base.STATE_REL)!=state_sha:raise ValueError('Actual current-state tool consumption differs')
        native.record(log,event='public_component_state_consumed',component_path=base.STATE_REL,sha256=state_sha,component_is_registered=True,pages=pages)
        queued=[]
        with base.scoped_selection_validation(native,w,case):
            _,planned=await native.plan_session(agent,case,w,spec,previous,log,on_queued=queued.append)
            native.validate_queued_session(case,w,spec,previous,log,component_anchor=agent._fastrsi_component_anchor)
        from orchestrate_native_sampler_agent_arm_v1 import declaration
        if len(queued)!=1 or provider.calls or declaration(w)!=evidence['declaration']:raise ValueError('Original public queue did not consume the actual component')
        events=[json.loads(x) for x in log.read_text().splitlines()]
        outcomes=[e['outcome'] for e in events if e.get('event')=='session_flow_outcome']
        if not outcomes or outcomes[-1].get('status')!='QUEUED':raise ValueError('Shared public flow did not queue')
        await finish_agent(agent,log)
        async def background():
            await asyncio.sleep(0)
            MemoryStore(w).write_long_term('READONLY_RECOVERY_MUST_NOT_PERSIST')
            await agent.memory_consolidator.consolidate_messages([{'role':'user','content':'READONLY_RECOVERY_MUST_NOT_PERSIST'}])
        await asyncio.create_task(background())
        if hashes(w)!=before or provider.calls or not (w/'.FASTRSI_EVALUATION_CLOSED.json').exists():raise ValueError('End/background isolation failed')
        blocked=False
        try:(w/'SKILLRUNTIME.md').write_text('READONLY_RECOVERY_MUST_NOT_PERSIST')
        except PermissionError:blocked=True
        if not blocked:raise ValueError('Post-end registry write was not blocked')
        agent=None
        save(output/'PLANNED_SESSION.json',planned)
        result.update(status='PASSED',source_status=PARTIAL,registry_bytes_inherited=True,selected_declaration=evidence['declaration'],
            registry_sha256=registry_sha,teacher_result_sha256=result_sha,teacher_launch_sha256=launch_sha,
            partial_commit_evidence_sha256=sha(output/'PARTIAL_COMMIT_EVIDENCE.json'),queued=1,model_calls=0,
            public_flow_status='QUEUED',evaluation_updates_blocked=True,background_updates_blocked=True,
            source_workspace_manifest=evidence['source_workspace_manifest'],source_workspace=str(source),
            events_sha256=sha(log),planned_session_sha256=sha(output/'PLANNED_SESSION.json'),
            source_inputs_unchanged=True,empirical_admission=False,current_component_state_sha256=state_sha,
            current_component_state_registered=True,current_state_read_pages=pages)
    except BaseException:result.update(status='FAILED',error=traceback.format_exc())
    finally:
        if agent is not None:
            try:await finish_agent(agent,output/'selection/AGENT_EVENTS.jsonl')
            except BaseException:result['finish_error']=traceback.format_exc();result['status']='FAILED'
        result['teacher_outputs_unchanged']=manifest(teacher)==teacher_before
        if not result['teacher_outputs_unchanged']:result['status']='FAILED'
        result['seconds']=time.perf_counter()-tick
        save(output/'RESULT.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--campaign',type=Path,required=True);p.add_argument('--teacher-output',type=Path,required=True)
    p.add_argument('--teacher-launch',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    for field in ('result-sha','launch-sha','registry-sha'):p.add_argument('--'+field,required=True)
    a=p.parse_args();result=asyncio.run(recover(a.campaign,a.teacher_output,a.teacher_launch,a.output,result_sha=a.result_sha,launch_sha=a.launch_sha,registry_sha=a.registry_sha))
    print(json.dumps(result,indent=2));raise SystemExit(0 if result['status']=='PASSED' else 1)
if __name__=='__main__':main()
