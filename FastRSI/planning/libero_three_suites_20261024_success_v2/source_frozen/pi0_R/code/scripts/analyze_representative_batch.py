"""Native screening coverage/cost and admission evidence; no learning benefit claim."""
import argparse,collections,hashlib,json
from pathlib import Path
from summarize_native_binding import summarize
ROOT=Path(__file__).resolve().parents[1]
def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()] if p.exists() else []

def analyze(run):
    run=run.resolve()
    stream=run/'stream';events=lines(stream/'OS_COST_EVENTS.jsonl')
    if (stream/'RESULT.json').exists():result=read(stream/'RESULT.json')
    else:
        result={'status':read(run/'STATUS.json')['status'],'cases':[]}
        target=lines(stream/'TARGET_COST_EVENTS.jsonl')
        for c in read(run/'ROSTER.json'):
            saved=stream/c['id']/'RESULT.json'
            if saved.exists():result['cases'].append(read(saved));continue
            last=next((e for e in reversed(target) if e.get('event')=='native_observation' and e.get('session_id')==c['id']),None)
            result['cases'].append({'case':c,'chain_status':'INCOMPLETE_INTERRUPTED','environment_success':last.get('native_success') if last else None,'seconds':None,'outcomes':[],'native_terminal_by_session':{c['id']:last} if last else {},'source_note':'No original RESULT persisted: physical target record retained, semantic receipt assessed separately in post-mortem protocol check'})
    intervals=[]
    begins={e['span_id']:e for e in events if e.get('event')=='begin'}
    for e in events:
        if e.get('event')=='end' and e['span_id'] in begins:
            b=begins[e['span_id']];intervals.append({**b,'ended_ns':e['wall_ns'],'seconds':e['duration_seconds']})
    ended={e['span_id'] for e in events if e.get('event')=='end'}
    for sid,b in begins.items():
        if sid not in ended:intervals.append({**b,'ended_ns':round(read(run/'STATUS.json')['finished_at']*1e9),'seconds':None,'boundary':'Open span classified through observed shutdown; duration unknown'})
    cases=[];bounds=[]
    for row in result['cases']:
        case=row['case'];folder=stream/case['id'];agent=lines(folder/'AGENT_EVENTS.jsonl')
        times=[e['time'] for e in agent]
        if times:bounds.append((min(times),max(times) if row.get('chain_status')!='INCOMPLETE_INTERRUPTED' else read(run/'STATUS.json')['finished_at'],case['id']))
        w=folder/'workspace';images=list((w/'artifacts/runtime'/case['id']/'rgb').glob('initial_*.png'))
        calls=[e for e in agent if e.get('event')=='tool']
        admission=read(folder/'admission/ADMISSION.json') if (folder/'admission/ADMISSION.json').exists() else None
        cases.append({'id':case['id'],'task':case['task_id'],'init':case['init_state_id'],'goal':case['task_description'],
          'environment_success':row.get('environment_success'),'verification':[{'session_id':o.get('session_id'),'verdict':o.get('verdict'),'status':o.get('code')} for o in row.get('outcomes',[])],
          'chain_status':row.get('chain_status'),'report_status':row.get('terminal_report',{}).get('status'),'seconds':row['seconds'],
          'physical_attempts':len(row.get('native_terminal_by_session',{})) or len(row.get('outcomes',[])),'native_steps_by_session':{k:v['steps'] for k,v in row.get('native_terminal_by_session',{}).items()},
          'admission':admission,'accepted_source':row.get('accepted_source'),'experience_before':row.get('experience_before'),
          'tool_calls_by_name':dict(collections.Counter(e['name'] for e in calls)),
          'queue_protocol_rejections':[{'event':e['event'],'error':e.get('error'),'time':e.get('time')} for e in agent if e.get('event') in ('queue_not_accepted','queue_still_pending')],
          'terminal_tool_blocks':[{'time':e.get('time'),'name':e.get('name'),'reason':e.get('reason'),'expected_path':e.get('expected_path')} for e in agent if e.get('event')=='terminal_phase_tool_blocked'],
          'native_tools_used':any(e.get('event')=='queue_accepted' and e.get('origin','agent')=='agent' for e in agent),
          'initial_image_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in images},
          'actual_policy_prompts':sorted({e['prompt'] for e in events if e.get('event')=='actual_policy_input' and e.get('case')==case['id']}),
          'persistent_updates':'Native development rules, admitted snapshots only; no forced consolidation'})
    model=[]
    for p in sorted(stream.glob('service_calls/*/REQUEST.json')):
        body=read(p);rp=p.parent/'RECEIPT.json';r=read(rp) if rp.exists() else {};t=r.get('started_at');purpose='unknown';case=None
        if t is not None:
            case=next((c for lo,hi,c in bounds if lo<=t<=hi),None)
            containing=[i['name'] for i in intervals if i['wall_ns']<=t*1e9<=i['ended_ns']]
            tools={x.get('function',{}).get('name') for x in body.get('tools',[]) or []}
            if tools=={'save_memory'}:purpose='native_memory'
            elif 'semantic_verification' in containing:purpose='verification'
            elif 'native_terminal_report' in containing:purpose='report'
            elif 'native_agent_planning' in containing:purpose='planning'
        response=read(p.parent/'RESPONSE.json') if (p.parent/'RESPONSE.json').exists() else {}
        chosen=[c.get('function',{}).get('name') for c in (response.get('choices') or [{}])[0].get('message',{}).get('tool_calls',[]) or []]
        model.append({'receipt':str(rp.relative_to(ROOT)),'case':case,'purpose':purpose,'chosen_tool_names':chosen,'status':r.get('status'),'generation_seconds':r.get('generation_seconds'),'service_seconds':r.get('seconds'),'prompt_tokens':r.get('prompt_tokens'),'completion_tokens':r.get('completion_tokens')})
    purpose_summary={}
    for purpose in sorted({r['purpose'] for r in model}):
        subset=[r for r in model if r['purpose']==purpose]
        purpose_summary[purpose]={'calls':len(subset),**{k:sum(r.get(k) or 0 for r in subset) for k in ['generation_seconds','service_seconds','prompt_tokens','completion_tokens']}}
    return {'scope':'One bounded native S1 development batch, not ABC or algorithm efficacy','status':result['status'],'run':str(run.relative_to(ROOT)),
      'cases':cases,'model_calls':model,'purpose_summary':purpose_summary,'bill':summarize(run),
      'open_spans':[{k:i.get(k) for k in ['name','span_id','case','boundary']} for i in intervals if i['seconds'] is None],
      'stage_seconds':dict(collections.Counter({name:sum(i['seconds'] for i in intervals if i['name']==name and i['seconds'] is not None) for name in {i['name'] for i in intervals}})),
      'unknowns':['No counterfactual cost measured','No causal effect of experience on decisions established by admission/read alone','Human/local/network cost unmeasured','Device capacity-time is not actual utilization'],
      'no_claim':['2pp noninferiority','30% reduction','learning benefit','representative suite quality from this partial batch']}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('--out',required=True);a=ap.parse_args();p=ROOT/a.run;assert p.resolve().is_relative_to((ROOT/'runs').resolve());value=analyze(p)
    out=ROOT/a.out;out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(value,f,ensure_ascii=False,indent=2)
    print(json.dumps({'status':value['status'],'cases':[{k:c[k] for k in ['id','environment_success','verification','chain_status']} for c in value['cases']],'purpose_summary':value['purpose_summary'],'bill_seconds':value['bill']['full_wall_seconds']},ensure_ascii=False,indent=2))
