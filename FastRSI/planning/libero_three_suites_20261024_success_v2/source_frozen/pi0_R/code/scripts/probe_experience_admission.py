"""Replay real saved feedback through admission and the unchanged native Agent.

Provider is a scripted protocol fixture: proves input/IO isolation, not model
quality or new robot performance. No forced memory work in production.
"""
import asyncio,copy,hashlib,json,os,shutil
from pathlib import Path
os.environ['FASTRSI_EXPERIENCE_ADMISSION']='guarded_v1'
from run_native_baseline import setup_workspace
from fastrsi.experience_admission import commit_experience,validate_snapshot,tree_hashes
from fastrsi.native_binding import bind_agent,finish_agent,hashes
from PhyAgentOS.providers.base import LLMProvider,LLMResponse,ToolCallRequest
from PhyAgentOS.agent.memory import MemoryStore
from PhyAgentOS.bus.events import InboundMessage

class Reader(LLMProvider):
    def __init__(self):super().__init__();self.calls=[];self.n=0
    def get_default_model(self):return 'cpu-fixture'
    async def chat(self,messages,**kw):
        self.calls.append(copy.deepcopy(messages));self.n+=1
        if self.n<=3:return LLMResponse(content=None,tool_calls=[ToolCallRequest(id='call_'+str(self.n),name='read_file',arguments={'path':['LESSONS.md','SESSIONS.md','LOG.md'][self.n-1]})])
        return LLMResponse(content='Admitted history read.')

async def main():
    root=Path(os.environ['FASTRSI_NATIVE_RUN']);old=Path('/project/runs/native_long_v1_20261002/stream')
    sources=[old/'native_long_t0_i8',old/'native_long_t3_i8'];source_hashes={str(s):tree_hashes(s) for s in sources}
    good=json.loads((sources[0]/'RESULT.json').read_text());bad=json.loads((sources[1]/'RESULT.json').read_text());bad_lesson=bad['outcomes'][-1]['lesson']
    assert good['accepted_source'] is None
    accepted,first=commit_experience(sources[0]/'workspace',good,None,root/'recovered_clean')
    assert first['status']=='ADMITTED'
    # Replay binds the historical good parent to its new, byte-preserved sealed copy.
    # This is migration/replay metadata, not a new physical execution claim.
    raw_parent=bad['accepted_source'];bad['accepted_source']=str(accepted)
    next_source,decision=commit_experience(sources[1]/'workspace',bad,accepted,root/'conflict_replay')
    assert decision['status']=='QUARANTINED' and next_source==accepted
    assert any('runtime_semantic_conflict' in x for x in decision['reasons'])
    # Simulate every persistent sink in a contaminated candidate. None may pass.
    candidate=root/'all_sink_candidate';shutil.copytree(sources[1]/'workspace',candidate)
    marker='REJECTED_FEEDBACK_MUST_NOT_CROSS_SESSION'
    for rel in ['LESSONS.md','LOG.md','memory/MEMORY.md','sessions/injected.jsonl','skills/injected/SKILL.md','routing/policy.json','reports/injected.md']:
        p=candidate/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(marker)
    pointer,injected=commit_experience(candidate,bad,accepted,root/'all_sink_replay');assert pointer==accepted
    case={'id':'next_admission_session','mode':'development','suite':'libero_10','max_steps':520,
        'task_id':6,'init_state_id':8,'task_description':'put the white mug on the plate and put the chocolate pudding to the right of the plate'}
    folder=root/'new_session';folder.mkdir();workspace,_,history=setup_workspace(folder,case,pointer)
    assert [s['session_id'] for s in history]==['native_long_t0_i8']
    provider=Reader();log=folder/'AGENT_EVENTS.jsonl';agent=bind_agent(workspace,provider,log)
    await agent.process_direct('Read the admitted history without modifying it.',session_key='cli:new_admitted')
    attempted=await agent.tools.execute('read_file',{'path':str(sources[1]/'workspace/LESSONS.md')});assert attempted.startswith('Error')
    await agent._process_message(InboundMessage(channel='system',sender_id='probe',chat_id='cli:new_admitted',content='Summarize the admitted history only.'))
    await finish_agent(agent,log)
    rendered=' '.join(json.dumps(provider.calls,ensure_ascii=False).split())
    assert ' '.join(bad_lesson.split()) not in rendered and marker not in rendered
    assert 'native_long_t0_i8' in rendered and 'native_long_t3_i8' not in rendered
    # Actual guarded setup refuses the old contaminated accepted workspace.
    rejected=False
    try:setup_workspace(root/'legacy_import',case,sources[1]/'accepted_experience')
    except ValueError:rejected=True
    assert rejected
    # Evaluation: fixed experience readable; native /new and delayed memory paths stay closed.
    ef=root/'eval_session';ef.mkdir();ew,_,_=setup_workspace(ef,{**case,'id':'eval_admission','mode':'evaluation'},accepted)
    ep=Reader();el=ef/'EVENTS.jsonl';ea=bind_agent(ew,ep,el,evaluation=True);before=hashes(ew)
    await ea.process_direct('Read fixed admitted experience.',session_key='cli:eval')
    await ea.process_direct('/new',session_key='cli:eval')
    await ea._process_message(InboundMessage(channel='system',sender_id='probe',chat_id='cli:eval',content='Read only fixed history.'))
    await finish_agent(ea,el)
    MemoryStore(ew).write_long_term(marker)
    await ea.memory_consolidator.consolidate_messages([{'role':'user','content':marker}])
    assert hashes(ew)==before
    old_seal=validate_snapshot(accepted)
    same,no_update=commit_experience(ew,{'case':{'mode':'evaluation'}},accepted,root/'forbidden_eval_promotion')
    assert same==accepted and not (root/'forbidden_eval_promotion').exists()
    assert validate_snapshot(accepted)==old_seal
    assert all(tree_hashes(s)==source_hashes[str(s)] for s in sources)
    (folder/'ACTUAL_MODEL_INPUTS.json').write_text(json.dumps(provider.calls,ensure_ascii=False,indent=2))
    value={'status':'PASSED','scope':'Real saved feedback + unchanged native AgentLoop with scripted CPU provider; no GPU/model-quality/robot result claim',
        'accepted_source':str(accepted),'replay_original_parent':raw_parent,'replay_parent_rebound_to_sealed_identical_good_workspace':True,
        'conflict_status':decision['status'],'previous_source_retained':True,'all_persistent_sink_injection_blocked':True,
        'new_session_history_ids':[s['session_id'] for s in history],'actual_native_agent_input_excludes_conflicting_lesson':True,
        'actual_native_agent_input_keeps_good_experience':True,'outside_quarantined_read_blocked':True,
        'raw_legacy_import_rejected':True,'evaluation_no_promotion_or_memory_update':True,'raw_source_hashes_unchanged':True,
        'scripted_provider_calls':len(provider.calls),'evaluation_scripted_calls':len(ep.calls)}
    (root/'RESULT.json').write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)
if __name__=='__main__':asyncio.run(main())
