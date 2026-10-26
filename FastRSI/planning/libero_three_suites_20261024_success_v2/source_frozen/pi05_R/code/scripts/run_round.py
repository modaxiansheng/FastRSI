"""Bounded full-OS round. Evaluation clones a fixed version and blocks experience writes."""
import asyncio
import copy
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from cost_trace import emit, span
from fastrsi.shadow import context_from_bundle, run_shadow
from fastrsi.versioning import initial_state, publish, digest, learned_budget, with_fixed_experience
from scripts.run_pi05_libero_real_e2e import _write_workspace, DEFAULT_TASK
from PhyAgentOS.runtime.watchdog.supervisor import WatchdogSupervisor
from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block, write_yaml_block
from PhyAgentOS.agent.session_verifier import SessionVerifier
from PhyAgentOS.providers.custom_provider import CustomProvider
from PhyAgentOS.runtime.policy.openpi.client import OpenPIClientPolicyWrapper
from PhyAgentOS.runtime.watchdog.result_writer import ResultWriter

ROOT=Path(os.environ['FASTRSI_ROUND_DIR'])
ARM=os.environ['FASTRSI_ARM']
ROUND=int(os.environ['FASTRSI_ROUND'])
SMOKE=os.environ.get('FASTRSI_SMOKE')=='1'
ARM_ROOT=ROOT.parent
MODE='development'
original_write=ResultWriter.write_lesson


def lesson_write(self,*args,**kwargs):
    if MODE=='evaluation':
        emit({'event':'persistent_update_blocked','operation':'write_lesson'})
        return None
    with span('experience_write'):
        return original_write(self,*args,**kwargs)


ResultWriter.write_lesson=lesson_write
original_infer=OpenPIClientPolicyWrapper.infer
def measured_infer(self,*args,**kwargs):
    with span('policy_client'):
        return original_infer(self,*args,**kwargs)
OpenPIClientPolicyWrapper.infer=measured_infer
original_bundle=ResultWriter.write_verification_bundle
def measured_bundle(self,*args,**kwargs):
    with span('evidence_preparation'):
        return original_bundle(self,*args,**kwargs)
ResultWriter.write_verification_bundle=measured_bundle


class AuditedVerifier(SessionVerifier):
    def _load_bundle(self,session):
        return with_fixed_experience(super()._load_bundle(session),STATE)

    async def _verify(self,session):
        bundle=self._load_bundle(session)
        context=context_from_bundle(bundle,self.workspace,STATE['skills'])
        with span('shadow_admissibility'):
            shadow=run_shadow(context)
        emit({'event':'decision_input','mode':MODE,'version':STATE['version'],
              'token_budget':STATE['token_budget'],'session_id':session.session_id,
              'lessons_sha256':__import__('hashlib').sha256(bundle.get('lessons_md','').encode()).hexdigest(),
              'lessons_present':bool(STATE['lessons']), 'shadow':shadow,
              'evidence_images':context['images']})
        if context['missing_required_fields']:
            raise ValueError('Incomplete decision input: '+str(context['missing_required_fields']))
        # Laya never changes the executed verdict. All arms retain the full verifier.
        return await super()._verify(session)


async def episode(init_id, mode, state, *, task_id=0, task_description=DEFAULT_TASK,
                  max_steps=220, case_id=None, suite='libero_spatial'):
    global MODE,STATE
    MODE=mode;STATE=copy.deepcopy(state)
    label=case_id or str(init_id)
    if not label or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in label):
        raise ValueError('Invalid case identifier')
    folder=ROOT/(mode+'_'+label);folder.mkdir()
    workspace=folder/'workspace'
    args=SimpleNamespace(policy_endpoint='openpi://127.0.0.1:49500',
        target_endpoint='targetws://127.0.0.1:49502',benchmark_name=suite,
        task_id=task_id,init_state_id=init_id,task_description=task_description,
        max_steps=max_steps,replan_every_steps=5)
    _write_workspace(workspace,args)
    if STATE.get('history_md'):
        (workspace/'LOG.md').write_text(STATE['history_md'])
    write_yaml_block(workspace/'LESSONS.md','Runtime Lessons',
                     {'version':'runtime_lessons_v1','lessons':STATE['lessons']})
    before=digest(workspace/'LESSONS.md')
    document=read_yaml_block(workspace/'SESSIONS.md')
    session=document['sessions'][0]
    session['session_id']=f'{ARM}_r{ROUND}_{mode}_i{label}'
    session['verification_profile']='recovery'
    session['timeouts']['execute_timeout_s']=1200
    write_yaml_block(workspace/'SESSIONS.md','Runtime Sessions',document)
    os.environ['FASTRSI_TOKEN_BUDGET']=str(STATE['token_budget'])
    provider=CustomProvider(api_base='http://127.0.0.1:49511/v1',default_model='qwen25vl7b',timeout_s=300)
    verifier=AuditedVerifier(workspace=workspace,provider=provider,model='qwen25vl7b',max_replans=2,
        rgb_retention='all',timeout_s=300,service_port=49520,
        service_provider_spec={'provider_name':'custom','model':'qwen25vl7b',
          'api_base':'http://127.0.0.1:49511/v1','max_tokens':STATE['token_budget'],'temperature':0})
    supervisor=WatchdogSupervisor(workspace,verification_enabled=True,
         verification_settings={'local_service_url':'http://127.0.0.1:49520',
             'timeout_s':300,'max_replans_per_episode':2})
    started=time.perf_counter();outcomes=[]
    try:
        await verifier.start()
        for attempt in range(3):
            with span('os_execution',arm=ARM,round=ROUND,mode=MODE,init_id=init_id,attempt=attempt):
                ran=await asyncio.to_thread(supervisor.run_once)
            if not ran:break
            pending=verifier.registry.first_awaiting_verification()
            if pending is None:raise RuntimeError('Missing verifiable session')
            with span('semantic_verification',session_id=pending.session_id):
                outcome=await verifier.verify_session(pending.session_id,source='auto')
            outcomes.append(outcome)
            if outcome.get('status')!='replanned':break
    finally:
        verifier.stop()
    sessions=read_yaml_block(workspace/'SESSIONS.md')['sessions']
    after=digest(workspace/'LESSONS.md')
    native_events=[json.loads(line) for line in (ROOT/'TARGET_COST_EVENTS.jsonl').read_text().splitlines()
                   if line.strip()]
    ids={s['session_id'] for s in sessions}
    native=[x for x in native_events if x.get('event')=='native_observation' and x.get('session_id') in ids]
    if not native:raise RuntimeError('Independent native outcome unavailable')
    chain_ok=bool(outcomes) and all(o.get('code')=='verified' for o in outcomes)
    row={'arm':ARM,'round':ROUND,'mode':MODE,'init_state_id':init_id,
         'suite':suite,'task_id':task_id,'task_description':task_description,'max_steps':max_steps,
         'input_version':STATE['version'],'input_token_budget':STATE['token_budget'],
         'environment_success':native[-1]['native_success'],
         'verification_status':[o.get('code') for o in outcomes],
         'chain_status':'passed' if chain_ok else 'failed',
         'seconds':time.perf_counter()-started,'sessions':sessions,'outcomes':outcomes,
         'native_terminal_by_session':{sid:next(x for x in reversed(native) if x['session_id']==sid)
             for sid in ids if any(x['session_id']==sid for x in native)},
         'lessons_sha_before':before,'lessons_sha_after':after,
         'evaluation_updates_blocked':mode=='evaluation' and before==after}
    (folder/'RESULT.json').write_text(json.dumps(row,ensure_ascii=False,indent=2,default=str))
    if mode=='evaluation' and before!=after:raise RuntimeError('Evaluation changed lesson snapshot')
    if not chain_ok:raise RuntimeError('Verification chain failed; stop this bounded campaign')
    accepted=[]
    if mode=='development':
        for lesson in read_yaml_block(workspace/'LESSONS.md').get('lessons',[]):
            metadata=lesson.get('metadata',{})
            if (lesson['session_id'] in ids and metadata.get('verifier_status')=='completed'
                and metadata.get('verdict')=='success' and row['environment_success']):
                accepted.append(lesson)
    return row,accepted,(workspace/'LOG.md').read_text() if (workspace/'LOG.md').exists() else ''


async def run():
    os.environ['FASTRSI_TRACE']=str(ROOT/'OS_COST_EVENTS.jsonl')
    old_path=ARM_ROOT/('VERSION_'+str(ROUND-1)+'.json')
    if ROUND==1 and not old_path.exists():publish(old_path,initial_state(),mode='development')
    old_hash=digest(old_path);state=json.loads(old_path.read_text());initial=copy.deepcopy(state)
    report={'status':'RUNNING','arm':ARM,'round':ROUND,'cases':[]}
    start=time.perf_counter()
    try:
        dev_ids=([0] if SMOKE else ([2,3] if ROUND==1 else [5,6]))
        eval_ids=([1] if SMOKE else ([4] if ROUND==1 else [7]))
        for init_id in dev_ids:
            row,accepted,history=await episode(init_id,'development',state)
            report['cases'].append(row)
            # Preserve original within-round OS lessons but promote only audited records.
            state['lessons'].extend(accepted)
            state['history_md']=history
        with span('round_update'):
            events=[json.loads(line) for line in (ROOT/'OS_COST_EVENTS.jsonl').read_text().splitlines()]
            sizes=[x['usage']['completion_tokens'] for x in events
                   if x.get('event')=='engine_completion' and x.get('ok') and x.get('usage',{}).get('completion_tokens')]
            state['accepted_completion_tokens'].extend(sizes)
            state['version']=ROUND
            proposed=learned_budget(state['accepted_completion_tokens'])
            state['token_budget']=proposed if ARM=='C' else 384
            state['update_receipts'].append({'parent_sha256':old_hash,'development_init_ids':dev_ids,
                'accepted_call_lengths':sizes,'previous_budget':initial['token_budget'],
                'proposed_budget':proposed,'published_budget':state['token_budget'],
                'method':'bounded token-envelope policy; fallback 384; not calibrated quality'})
            new_path=ARM_ROOT/('VERSION_'+str(ROUND)+'.json');publish(new_path,state,mode='development')
        new_hash=digest(new_path)
        for init_id in eval_ids:
            row,_,_=await episode(init_id,'evaluation',state)
            report['cases'].append(row)
            assert digest(new_path)==new_hash and digest(old_path)==old_hash
        report.update(status='PASSED',version_sha256=new_hash,persistent_versions_unchanged_by_eval=True)
    except Exception:
        report.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-start
        (ROOT/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
        print(json.dumps({'status':report['status'],'arm':ARM,'round':ROUND,
                          'cases':len(report['cases']),'seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':asyncio.run(run())
