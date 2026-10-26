"""Attribute recorded prompt text volume; no inference or token-savings estimate."""
import argparse,collections,hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def load(p):return json.loads(p.read_text())
def text_content(value):
    if isinstance(value,str):return value
    if isinstance(value,list):return '\n'.join(x.get('text','') for x in value if isinstance(x,dict) and x.get('type')=='text')
    return ''

def profile(body,case):
    calls={};volumes=collections.Counter();seen=set();duplicates=[];pages=[]
    for m in body.get('messages',[]):
        for call in m.get('tool_calls',[]) or []:
            fn=call.get('function',{});raw=fn.get('arguments',{})
            try:args=json.loads(raw) if isinstance(raw,str) else raw
            except (ValueError,TypeError):args={}
            calls[call.get('id')]={'name':fn.get('name'),'args':args if isinstance(args,dict) else {}}
            volumes['assistant_tool_arguments']+=len(raw) if isinstance(raw,str) else len(json.dumps(raw,ensure_ascii=False))
        content=text_content(m.get('content'));role=m.get('role','unknown');bucket=role
        if role=='tool':
            call=calls.get(m.get('tool_call_id'),{});name=call.get('name','unbound');args=call.get('args',{})
            path=args.get('path','');basename=Path(path).name if isinstance(path,str) else ''
            bucket='tool:'+name+(':'+basename if basename else '')
            fingerprint=(bucket,hashlib.sha256(content.encode()).hexdigest())
            if fingerprint in seen:duplicates.append({'bucket':bucket,'chars':len(content),'text_sha256':fingerprint[1]})
            seen.add(fingerprint)
            if name=='read_file':
                marker=re.search(r'\(Showing lines (\d+)-(\d+) of (\d+)\. Use offset=(\d+) to continue\.\)',content)
                pages.append({'file':basename,'offset':args.get('offset',1),'limit':args.get('limit'),
                  'chars':len(content),'has_current_session_id':bool(case and case in content),
                  'continuation':dict(zip(['start','end','total','next'],map(int,marker.groups()))) if marker else None})
        volumes[bucket]+=len(content)
    return {'text_chars_by_source':dict(volumes),'text_chars_total':sum(volumes.values()),
      'exact_duplicate_tool_text_in_same_request':duplicates,'file_read_pages_in_request':pages}

def audit(analysis_path):
    analysis=load(analysis_path);rows=[]
    for call in analysis['model_calls']:
        receipt=ROOT/call['receipt'];request=receipt.parent/'REQUEST.json';body=load(request)
        rows.append({**{k:call.get(k) for k in ['case','purpose','status','prompt_tokens','completion_tokens','generation_seconds']},
          'request':str(request.relative_to(ROOT)),'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),
          **profile(body,call.get('case'))})
    phases={}
    for phase in sorted({r['purpose'] for r in rows}):
        subset=[r for r in rows if r['purpose']==phase];volumes=collections.Counter()
        for row in subset:volumes.update(row['text_chars_by_source'])
        phases[phase]={'calls':len(subset),'measured_prompt_tokens':sum(r.get('prompt_tokens') or 0 for r in subset),
          'peak_prompt_tokens':max((r.get('prompt_tokens') or 0 for r in subset),default=0),
          'repeated_text_chars_over_requests':dict(volumes),'total_text_chars_over_requests':sum(volumes.values())}
    return {'run':analysis['run'],'analysis':str(analysis_path.relative_to(ROOT)),
      'scope':'Read-only attribution of saved actual requests; calls and tokens include failures.',
      'limitations':['Character allocation is not tokenizer-level attribution; images and tool schemas excluded from character counts.',
        'Repeated context across requests is not by itself redundant work; fixed native obligations and evidence dependencies remain.',
        'No payload reduction, model rerun, decision quality comparison or learning benefit measured.'],
      'phases':phases,'requests':rows}

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('analysis');ap.add_argument('--out',required=True);args=ap.parse_args()
    path=(ROOT/args.analysis).resolve();assert path.is_relative_to(ROOT)
    result=audit(path);out=ROOT/args.out
    with out.open('x') as f:json.dump(result,f,ensure_ascii=False,indent=2)
    print(json.dumps(result['phases'],ensure_ascii=False,indent=2))
