"""Read-only upper-bound audit of current terminal record exposure in old traces."""
import hashlib,json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def audit(case):
    log=case/'AGENT_EVENTS.jsonl';events=[json.loads(x) for x in log.read_text().splitlines()]
    terminals=[e for e in events if e['event']=='terminal_results_consumed']
    ws=case/'workspace';lines=(ws/'SESSIONS.md').read_text().splitlines();sid=case.name
    starts=[(n,l.split(': ',1)[1]) for n,l in enumerate(lines,1) if l.startswith('- session_id: ')]
    matches=[(i,n) for i,(n,s) in enumerate(starts) if s==sid or s.startswith(sid+'_replan')]
    assert matches,case
    needed=set()
    for i,start in matches:
        end=starts[i+1][0]-1 if i+1<len(starts) else next(n for n,l in enumerate(lines[start:],start+1) if l=='```')-1
        needed.update(range(start,end+1))
    covered=set();read_text=''
    for e in events:
        if e.get('event')=='tool' and e.get('name')=='read_file' and Path(e['params']['path']).name=='SESSIONS.md':
            read_text+='\n'+e['result']
            for line in e['result'].splitlines():
                m=re.fullmatch(r'(\d+)\| (.*)',line)
                if m:
                    n=int(m[1])
                    if 1<=n<=len(lines) and m[2]==lines[n-1]:covered.add(n)
    required_refs=[str(p.relative_to(ws)) for p in (ws/'artifacts/runtime').glob(sid+'*/episode.json')]
    report=(ws/terminals[-1]['report']).read_text() if terminals else ''
    return {'case':sid,'path':str(case.relative_to(ROOT)),
      'source_log_sha256':hashlib.sha256(log.read_bytes()).hexdigest(),
      'original_terminal_receipt':bool(terminals),'current_id_in_any_sessions_read':sid in read_text,
      'canonical_current_lines':len(needed),'canonical_lines_never_observed':len(needed-covered),
      'missing_required_report_references':[p for p in required_refs if p not in report],
      'audit':'READ_GAP' if needed-covered else 'CONTENT_OBSERVED_NOT_RETROACTIVE_STRICT_PASS'}

if __name__=='__main__':
    rows=[audit(p) for p in sorted((ROOT/'runs/representative_native_v1_20261002').glob('batch_*/stream/coverage_*')) if (p/'AGENT_EVENTS.jsonl').exists()]
    result={'scope':'All ten original S1 attempts; existing receipts unchanged',
      'method':'Upper bound over all saved reads, including planning: absence proves missing exposure; presence does not retroactively establish fresh terminal read acceptance.',
      'rows':rows,'read_gaps':sum(r['audit']=='READ_GAP' for r in rows),
      'learning_claim':False}
    out=ROOT/'planning/representative_coverage_v1/ALL_TEN_TERMINAL_REAUDIT.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result,indent=2))
