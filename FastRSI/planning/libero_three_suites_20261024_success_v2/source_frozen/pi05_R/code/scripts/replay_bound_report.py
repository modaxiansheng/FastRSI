"""Replay the saved malformed report call through the native AgentLoop (no model inference)."""
import asyncio,hashlib,json,re,shutil,sys
from pathlib import Path
from PhyAgentOS.providers.base import LLMProvider,LLMResponse,ToolCallRequest
from fastrsi.native_binding import bind_agent,finish_agent,hashes
from fastrsi.terminal_results import consume_terminal

class SavedReport(LLMProvider):
    def __init__(self,call):super().__init__();self.call=call;self.count=0
    def get_default_model(self):return 'saved-response-replay'
    async def chat(self,messages,**kwargs):
        self.count+=1
        if self.count==1:
            prompt=next(m['content'] for m in reversed(messages) if m['role']=='user')
            requests=json.loads(re.search(r'Canonical read requests: (\[.*?\])\. These ranges',prompt).group(1))
            return LLMResponse(content=None,tool_calls=[ToolCallRequest(id='read'+str(i),name='read_file',arguments=p) for i,p in enumerate(requests)])
        if self.count==2:
            return LLMResponse(content=None,tool_calls=[ToolCallRequest(id='saved-call',name=self.call['function']['name'],arguments=json.loads(self.call['function']['arguments']))])
        return LLMResponse(content='Report written to the bound destination.')

async def main():
    project=Path('/project');source=project/'runs/feedback_window_known_v1_20261003/pair0/R/stream'
    response=source/'service_calls/call_5a929799bfed4d75a873704bcfa6d9ad/RESPONSE.json'
    call=json.loads(response.read_text())['choices'][0]['message']['tool_calls'][0]
    out=Path(sys.argv[1]);out.mkdir(exist_ok=False);w=out/'workspace';w.mkdir()
    original=source/'net_t4_i31/workspace';shutil.copyfile(original/'SESSIONS.md',w/'SESSIONS.md')
    shutil.copytree(original/'artifacts/runtime/net_t4_i31',w/'artifacts/runtime/net_t4_i31')
    bad=Path(json.loads(call['function']['arguments'])['path']);assert not bad.exists()
    provider=SavedReport(call);log=out/'EVENTS.jsonl';agent=bind_agent(w,provider,log,evaluation=True)
    before=hashes(w)
    try:
        result=await consume_terminal(agent,{'id':'net_t4_i31','terminal_bind_report_destination':True},w,log)
    finally:await finish_agent(agent,log)
    assert provider.count==3 and result['status']=='PASSED' and before==hashes(w) and not bad.exists()
    events=[json.loads(s) for s in log.read_text().splitlines()]
    assert sum(e.get('event')=='terminal_report_destination_bound' for e in events)==1
    assert not any(e.get('event')=='terminal_phase_tool_blocked' for e in events)
    value={'status':'PASSED','scope':'Saved raw model tool call replay; no new real-model latency or method speedup claim',
           'response_path':str(response),'response_sha256':hashlib.sha256(response.read_bytes()).hexdigest(),
           'scripted_provider_calls':provider.count,'outside_path_created':bad.exists(),'experience_unchanged':before==hashes(w),'terminal':result}
    (out/'RESULT.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));print(json.dumps(value))

if __name__=='__main__':asyncio.run(main())
