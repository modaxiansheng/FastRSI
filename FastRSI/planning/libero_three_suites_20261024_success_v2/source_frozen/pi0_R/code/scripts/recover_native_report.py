"""Resume original native conversation on a copy; no physical execution or re-verification."""
import asyncio,copy,hashlib,json,os,shutil,time
from pathlib import Path
from run_native_baseline import measured_provider
from fastrsi.native_binding import bind_agent,finish_agent,hashes
from fastrsi.terminal_results import consume_terminal
from fastrsi.experience_admission import commit_experience,tree_hashes,validate_snapshot
from PhyAgentOS.runtime.state_io.markdown_yaml import read_yaml_block

async def main():
    root=Path(os.environ['FASTRSI_NATIVE_RUN']);source=Path(os.environ['FASTRSI_REPORT_SOURCE'])
    if (source/'RESULT.json').exists():
        old=json.loads((source/'RESULT.json').read_text())
    else:
        case=next(c for c in json.loads((source.parent.parent/'ROSTER.json').read_text()) if c['id']==source.name)
        sessions=[s for s in read_yaml_block(source/'workspace/SESSIONS.md')['sessions'] if s['session_id']==case['id']]
        assert len(sessions)==1 and sessions[0]['status']=='succeeded'
        verification=sessions[0]['result']['metadata']['verification'];attempt=next(a for a in verification['attempts'] if a['attempt_id']==verification['applied_attempt_id'])
        assert attempt['mode']=='apply' and not attempt.get('error') and attempt['verdict']=='success'
        assert verification['rgb_retention']['status']=='retained' and not verification['rgb_retention']['errors']
        prior=source.parent/'coverage_s0_t0_i2/admission/ADMISSION.json';parent=json.loads(prior.read_text())['next_source']
        observed=[json.loads(x) for x in (source.parent/'TARGET_COST_EVENTS.jsonl').read_text().splitlines()]
        native=next(e for e in reversed(observed) if e.get('event')=='native_observation' and e.get('session_id')==case['id']);assert native['native_success'] is True
        old={'case':case,'status':'FAILED','chain_status':'FAILED','environment_success':native['native_success'],'accepted_source':parent,
             'sessions':sessions,'outcomes':[{'ok':True,'code':'verified','session_id':case['id'],'verdict':'success','provenance':'Derived from native persisted applied verdict/retention, not a new verification call'}],
             'error':'Owned runner stopped during repeated out-of-workspace report writes; no original case RESULT persisted',
             'source_row_provenance':'Reconstructed only in recovery artifact from saved native session and target receipts; original source untouched'}
    case=old['case']
    assert old['chain_status']=='FAILED'
    events=[json.loads(x) for x in (source/'AGENT_EVENTS.jsonl').read_text().splitlines()]
    assert sum(e.get('event')=='terminal_phase_tool_blocked' and e.get('name')=='write_file' for e in events)>=6, 'Recovery restricted to observed repeated report-path failure'
    assert old['environment_success'] is True and all(o.get('code')=='verified' and o.get('verdict')=='success' for o in old['outcomes'])
    original=source/'workspace';before=tree_hashes(original);w=root/'workspace';shutil.copytree(original,w)
    previous=Path(old['accepted_source']);validate_snapshot(previous)
    immutable=[p for p in [w/'SESSIONS.md',*(w/'artifacts/runtime'/case['id']).glob('*')] if p.is_file()]
    canonical={str(p.relative_to(w)):hashlib.sha256(p.read_bytes()).hexdigest() for p in immutable}
    log=root/'AGENT_EVENTS.jsonl';provider=measured_provider(root,'agent',2048);agent=bind_agent(w,provider,log,evaluation=False,read_files=case.get('read_files_tool',False),max_iterations=10)
    row=copy.deepcopy(old);start=time.time()
    try:
        row['terminal_report']=await consume_terminal(agent,case,w,log)
        row.update(status='PASSED',chain_status='PASSED',report_recovery={'source':str(source),'source_case_result_sha256':hashlib.sha256((source/'RESULT.json').read_bytes()).hexdigest() if (source/'RESULT.json').exists() else None,'same_conversation_key':'cli:'+case['id'],'robot_executions':0,'verification_calls':0,'interpretation':'Technical terminal recovery, not a new independent episode'})
    finally:
        await finish_agent(agent,log);await provider._client.close()
        assert tree_hashes(original)==before,'Original evidence changed'
        assert all(hashlib.sha256((w/p).read_bytes()).hexdigest()==h for p,h in canonical.items()),'Copied canonical result changed'
    row['sessions']=[s for s in read_yaml_block(w/'SESSIONS.md')['sessions'] if s['session_id']==case['id'] or s['session_id'].startswith(case['id']+'_replan')]
    observed=[json.loads(x) for x in (source.parent/'TARGET_COST_EVENTS.jsonl').read_text().splitlines()]
    row['native_terminal_by_session']={sid:next(e for e in reversed(observed) if e.get('event')=='native_observation' and e.get('session_id')==sid) for sid in [s['session_id'] for s in row['sessions']]}
    row['experience_after']=hashes(w)
    pointer,admission=commit_experience(w,row,previous,root/'admission');row['experience_admission']=admission;assert admission['status']=='ADMITTED'
    (root/'RESULT.json').write_text(json.dumps({'status':'COMPLETED','seconds':time.time()-start,'recovered_case':row,'source_unchanged':True,'canonical_result_hashes':canonical,'next_source':str(pointer)},indent=2,default=str))

if __name__=='__main__':asyncio.run(main())
