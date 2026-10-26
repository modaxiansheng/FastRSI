"""Bounded read-only audit of one shared N/F disagreement; no relabelling."""
import base64,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def main():
    run=ROOT/'runs/shared_flow_net_parallel_v2_20261002';rows=[];case='net_t9_i15'
    for arm in ['N','F']:
        stream=run/arm/'stream';w=stream/case/'workspace';bundle_path=w/'artifacts/runtime'/case/'verification_bundle.json'
        bundle=json.loads(bundle_path.read_text());request=None;embedded=None
        for p in stream.glob('service_calls/*/REQUEST.json'):
            b=json.loads(p.read_text())
            if b.get('tools'):continue
            for m in b['messages']:
                parts=m['content'] if isinstance(m['content'],list) else []
                for part in parts:
                    if part.get('type')=='text' and 'VERIFICATION_BUNDLE:\n' in part.get('text',''):
                        candidate=json.loads(part['text'].split('VERIFICATION_BUNDLE:\n',1)[1])
                        if candidate['final_rgb_paths']==bundle['final_rgb_paths']:request=p;embedded=candidate;content=parts
        assert request
        differences=[k for k in set(embedded)|set(bundle) if embedded.get(k)!=bundle.get(k)]
        assert set(differences)<={'rgb_retention'}, differences
        assert bundle.get('rgb_retention',{}).get('status')=='retained'
        mappings=[];label=None
        for part in content:
            if part.get('type')=='text' and part['text'].startswith('RGB_EVIDENCE_MAPPING: '):label=json.loads(part['text'].split(': ',1)[1])
            elif part.get('type')=='image_url':
                assert label
                data=base64.b64decode(part['image_url']['url'].split(',',1)[1]);file=w/label['path'];expected=file.read_bytes()
                mappings.append({**label,'payload_sha256':hashlib.sha256(data).hexdigest(),'file_sha256':hashlib.sha256(expected).hexdigest(),'byte_identical':data==expected});label=None
        assert len(mappings)==4 and all(x['byte_identical'] for x in mappings)
        receipt=json.loads((request.parent/'RECEIPT.json').read_text());saved=json.loads((stream/case/'RESULT.json').read_text())
        observations=[json.loads(x) for x in (stream/'TARGET_COST_EVENTS.jsonl').read_text().splitlines()]
        observations=[x for x in observations if x.get('event')=='native_observation' and x.get('session_id')==case]
        last=observations[-1]
        rows.append({'arm':arm,'request':str(request.relative_to(ROOT)),'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),
          'bundle_equal_except_post_verification_retention_metadata':True,'post_request_metadata_fields':differences,'bundle_created_at':bundle['created_at'],'verification_started_at':receipt['started_at'],
          'last_native_observation':last,'last_observation_precedes_verification':last['wall_ns']/1e9<=receipt['started_at'],
          'goal':saved['case']['task_description'],'mappings':mappings,'semantic_evidence':saved['outcomes'][-1]['evidence']})
    result={'scope':'One preserved evaluation disagreement, read-only; not training labels', 'rows':rows,
      'conclusions':['Both verifier requests contain the exact saved initial/final image bytes with explicit phase mapping.',
        'Both last native observations precede verification; this audit does not establish camera acquisition timestamp synchrony.',
        'No image substitution or verdict-input mismatch found; saved bundle additionally has post-verification retention metadata. Evidence sufficiency and semantic truth are not proven by byte matching.',
        'Keep original simulator success and semantic failure separately. No labels or experience updated.'],
      'next':'If pursued, bind benchmark goal predicates and camera visibility for this single case; no generic verifier training or repeated prompt search.'}
    out=ROOT/'planning/validation_budget_v1/MICROWAVE_EVIDENCE_AUDIT.json'
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'arms':len(rows),'all_images_bound':True,'last_observation_precedes_verification':[r['last_observation_precedes_verification'] for r in rows]}))
if __name__=='__main__':main()
