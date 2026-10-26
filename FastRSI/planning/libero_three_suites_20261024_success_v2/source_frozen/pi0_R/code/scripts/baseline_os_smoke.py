"""Two real episodes through the frozen OS feedback path; no learned controller."""
import asyncio
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

from cost_trace import emit, span
from scripts.run_pi05_libero_real_e2e import _write_workspace, DEFAULT_TASK
from PhyAgentOS.runtime.watchdog.supervisor import WatchdogSupervisor
from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block, write_yaml_block
from PhyAgentOS.agent.session_verifier import SessionVerifier
from PhyAgentOS.providers.custom_provider import CustomProvider
from PhyAgentOS.providers.base import GenerationSettings
from PhyAgentOS.runtime.policy.openpi.client import OpenPIClientPolicyWrapper
from PhyAgentOS.runtime.watchdog.result_writer import ResultWriter

ROOT=Path('/project/runs/pilot_20261001/baseline_os_smoke_v1')
WORKSPACE=ROOT/'workspace'


def wrap_sync(cls,name,category):
    original=getattr(cls,name)
    def measured(self,*args,**kwargs):
        with span(category,method=name):
            return original(self,*args,**kwargs)
    setattr(cls,name,measured)


async def main():
    ROOT.mkdir(parents=True,exist_ok=False)
    os.environ['FASTRSI_TRACE']=str(ROOT/'OS_COST_EVENTS.jsonl')
    # This wrapper measures the same OS policy client without changing its output.
    # API presence is checked explicitly so instrumentation cannot silently vanish.
    wrap_sync(OpenPIClientPolicyWrapper,'infer','policy_client')
    wrap_sync(ResultWriter,'write_verification_bundle','evidence_preparation')
    wrap_sync(ResultWriter,'write_lesson','experience_write')
    provider=CustomProvider(api_base='http://127.0.0.1:49511/v1',default_model='qwen25vl7b',timeout_s=600)
    provider.generation=GenerationSettings(temperature=0,max_tokens=384)
    verifier=SessionVerifier(workspace=WORKSPACE,provider=provider,model='qwen25vl7b',
        max_replans=2,rgb_retention='all',timeout_s=600,service_port=49520,
        service_provider_spec={'provider_name':'custom','model':'qwen25vl7b',
                              'api_base':'http://127.0.0.1:49511/v1','max_tokens':384,'temperature':0})
    report={'scope':'original frozen OS feedback path smoke; not Table4 score or FastRSI comparison',
            'cases':[],'status':'RUNNING'}
    started=time.perf_counter()
    try:
        await verifier.start()
        for init_id in [0,1]:
            args=SimpleNamespace(policy_endpoint='openpi://127.0.0.1:49500',
                target_endpoint='targetws://127.0.0.1:49502',benchmark_name='libero_spatial',
                task_id=0,init_state_id=init_id,task_description=DEFAULT_TASK,max_steps=220,replan_every_steps=5)
            lessons=WORKSPACE/'LESSONS.md'
            before=lessons.read_text() if lessons.exists() else ''
            _write_workspace(WORKSPACE,args)
            document=read_yaml_block(WORKSPACE/'SESSIONS.md')
            document['sessions'][0]['verification_profile']='recovery'
            document['sessions'][0]['timeouts']['execute_timeout_s']=1200
            write_yaml_block(WORKSPACE/'SESSIONS.md','Runtime Sessions',document)
            supervisor=WatchdogSupervisor(WORKSPACE,verification_enabled=True,
                verification_settings={'local_service_url':'http://127.0.0.1:49520',
                                       'timeout_s':600,'max_replans_per_episode':2})
            for attempt in range(3):
                with span('os_execution',init_state_id=init_id,attempt=attempt):
                    ran=await asyncio.to_thread(supervisor.run_once)
                if not ran:break
                pending=verifier.registry.first_awaiting_verification()
                if pending is None:
                    raise RuntimeError('OS did not produce a verifiable completed session')
                bundle=verifier._load_bundle(pending)
                emit({'event':'feedback_input','session_id':pending.session_id,
                      'lessons_present':bool(bundle.get('lessons_md')),
                      'lessons_sha256':hashlib.sha256((bundle.get('lessons_md') or '').encode()).hexdigest()})
                with span('semantic_verification',session_id=pending.session_id):
                    outcome=await verifier.verify_session(pending.session_id,source='auto')
                emit({'event':'verification_outcome','session_id':pending.session_id,'outcome':outcome})
                if outcome.get('status')!='replanned':break
            sessions=read_yaml_block(WORKSPACE/'SESSIONS.md')
            after=lessons.read_text() if lessons.exists() else ''
            row={'init_state_id':init_id,'lessons_before_bytes':len(before.encode()),
                 'lessons_after_bytes':len(after.encode()),'sessions':sessions}
            report['cases'].append(row)
            (ROOT/f'CASE_{init_id}.json').write_text(json.dumps(row,ensure_ascii=False,indent=2,default=str))
        report['status']='EXECUTED_PENDING_NATIVE_AUDIT'
    except Exception:
        report.update(status='FAILED',error=traceback.format_exc())
        raise
    finally:
        verifier.stop()
        report['elapsed_seconds']=time.perf_counter()-started
        (ROOT/'RESULT.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str))
        print(json.dumps({'status':report['status'],'cases':len(report['cases']),
                          'elapsed_seconds':report['elapsed_seconds']}),flush=True)


if __name__=='__main__':
    asyncio.run(main())
