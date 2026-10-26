"""Audit immutable feedback and image completeness; never rewrite a native run."""
import argparse,base64,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from fastrsi.evidence_layers import evidence_layers

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('run_id');ap.add_argument('--out',required=True);a=ap.parse_args()
    run=ROOT/'runs'/a.run_id;stream=run/'stream';roster=json.loads((run/'ROSTER.json').read_text());rows=[]
    service=[]
    for p in stream.glob('service_calls/*/REQUEST.json'):
        d=json.loads(p.read_text())
        if d.get('max_tokens')!=384 or d.get('tools'):continue
        images=[]
        for m in d['messages']:
            if isinstance(m.get('content'),list):
                for item in m['content']:
                    if item.get('type')=='image_url':images.append(hashlib.sha256(base64.b64decode(item['image_url']['url'].split(',',1)[1])).hexdigest())
        service.append((p,d,images))
    for c in roster:
        folder=stream/c['id'];result_path=folder/'RESULT.json';r=json.loads(result_path.read_text())
        layer=evidence_layers(r['environment_success'],r['outcomes'],r['chain_status'])
        entry={'case_id':c['id'],'source_result':str(result_path.relative_to(ROOT)),'source_result_sha256':sha(result_path),**layer}
        if layer['native_vs_final_semantic_agreement'] is False:
            w=folder/'workspace';b=w/'artifacts/runtime'/c['id']/'verification_bundle.json';bundle=json.loads(b.read_text())
            paths=bundle['initial_rgb_paths']+bundle['final_rgb_paths'];expected=[sha(w/p) for p in paths]
            matches=[(p,d,images) for p,d,images in service if images==expected];assert len(matches)==1
            p,d,images=matches[0];receipt=json.loads((p.parent/'RECEIPT.json').read_text())
            lesson=r['outcomes'][-1]['lesson'];exposures=[]
            for later in roster[roster.index(c)+1:]:
                q=stream/later['id']/'agent_call_1/REQUEST.json';body=json.loads(q.read_text())
                if any(' '.join(lesson.split()) in ' '.join(str(m.get('content','')).split()) for m in body['messages']):exposures.append({'case_id':later['id'],'request':str(q.relative_to(ROOT)),'sha256':sha(q),'matching':'Whitespace normalized only; YAML wraps the exact lesson across lines'})
            entry.update({'raw_verifier_request':str(p.relative_to(ROOT)),'raw_request_sha256':sha(p),
                'image_integrity':{'expected_paths':paths,'actual_image_sha256':images,'count':len(images),'all_bytes_and_order_match':True},
                'raw_request_task_present':c['task_description'] in json.dumps(d),
                'model_protocol':{k:receipt.get(k) for k in ['status','generation_finish_reason','ended_with_eos','reached_token_limit','completion_tokens']},
                'conflicting_lesson':lesson,'propagated_to_actual_next_session_requests':exposures,
                'inspection':'Final front view does not support the stated visible bowl on countertop; wrist view mostly observes tabletop. Occlusion-related hallucination is a hypothesis, not a proved general cause.',
                'admission':'Do not use as accepted FastRSI training label/skill. This sidecar quarantines the source; original native baseline feedback remains untouched.'})
        rows.append(entry)
    report={'run_id':a.run_id,'protocol_completed':sum(r['verification_protocol_completed'] for r in rows),
        'native_successes':sum(r['native_success'] is True for r in rows),
        'semantic_successes':sum(r['semantic_verdicts'][-1]=='success' for r in rows),
        'disagreements':sum(r['native_vs_final_semantic_agreement'] is False for r in rows),'cases':rows,
        'not_claimed':'No learned corrective rule or cost saving. Native benchmark truth is used post hoc for audit, not as a new router input.'}
    p=ROOT/a.out
    with p.open('x') as f:json.dump(report,f,ensure_ascii=False,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='cases'},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
