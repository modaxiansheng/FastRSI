"""Audit immutable M2 verifier requests; never query a model or replace labels."""
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'm2_transfer_v1_20261005'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    rows = []
    for initial in (2, 3):
        for arm in ('N', 'F'):
            stream = ROOT / 'runs' / NAME / f'init{initial}' / arm / 'stream'
            matches = [p for p in stream.glob('service_calls/*/REQUEST.json')
                       if 'Verify the Agent session' in str(read(p)['messages'][0])]
            assert len(matches) == 1
            path = matches[0]
            req = read(path)
            receipt = read(path.parent / 'RECEIPT.json')
            response = read(path.parent / 'RESPONSE.json')
            raw = read(path.parent / 'RAW.txt')
            assert json.loads(response['choices'][0]['message']['content']) == raw
            assert receipt['generation_finish_reason'] == 'eos' and not receipt['reached_token_limit']
            pieces = [p for m in req['messages'] if isinstance(m['content'], list) for p in m['content']]
            texts = [p['text'] for p in pieces if p['type'] == 'text' and 'VERIFICATION_BUNDLE:' in p['text']]
            assert len(texts) == 1
            bundle = json.JSONDecoder().raw_decode(texts[0].split('VERIFICATION_BUNDLE:', 1)[1].lstrip())[0]
            case = f'net_t64_i{initial}'
            workspace = stream / case / 'workspace'
            trace = [json.loads(line) for line in (stream / 'stage_trace' / case / 'TRACE.jsonl').read_text().splitlines()]
            images = []
            for idx, item in enumerate(pieces):
                if item['type'] != 'image_url': continue
                prior = pieces[idx - 1]
                mapping = json.loads(prior['text'].split('RGB_EVIDENCE_MAPPING:', 1)[1])
                relative = Path(mapping['path'])
                assert not relative.is_absolute() and '..' not in relative.parts
                content = base64.b64decode(item['image_url']['url'].split(',', 1)[1], validate=True)
                actual = workspace / relative
                assert content == actual.read_bytes()
                phase = mapping['phase']
                assert phase in ('initial', 'final') and mapping['path'] in bundle[f'{phase}_rgb_paths']
                camera = 'agentview_image' if 'rgb_agentview_image' in relative.name else 'robot0_eye_in_hand_image'
                boundary = trace[0 if phase == 'initial' else -1]
                source = stream / 'stage_trace' / case / f'{boundary["step"]:04d}_{camera}.png'
                assert content == source.read_bytes()
                images.append(dict(**mapping, step=boundary['step'], sha256=sha(actual), bytes_match_workspace_and_trace=True))
            assert len(images) == 4 and [r['phase'] for r in images] == ['initial','initial','final','final']
            analysis = read(ROOT / 'planning' / NAME / f'INIT{initial}_ANALYSIS.json')
            result = next(r for r in analysis['rows'] if r['arm'] == arm)
            assert result['semantic'] == [raw['verdict']]
            rows.append(dict(init_state_id=initial, arm=arm, request=str(path.relative_to(ROOT)),
                request_options={k:v for k,v in req.items() if k != 'messages'}, images=images,
                decoding=receipt['decoding'], prompt_tokens=receipt['prompt_tokens'], completion_tokens=receipt['completion_tokens'],
                raw_verdict=raw['verdict'], evidence=raw.get('evidence'), failure_reason=raw.get('failure_reason'),
                raw_api_and_final_label_match=True, no_truncation=True,
                environment_success=result['environment_success'], admission=result['admission'],
                service_identity=read(stream/'service_calls/SERVICE_IDENTITY.json'),
                source_sha256={str(p.relative_to(ROOT)):sha(p) for p in
                              [path,path.parent/'RESPONSE.json',path.parent/'RAW.txt',path.parent/'RECEIPT.json']}))
    identities = [r['service_identity'] for r in rows]
    same = all(all(i[k] == identities[0][k] for k in ('config_sha256','chat_template_sha256','model_class','transformers','torch')) for i in identities)
    assert same
    report = dict(status='PASSED_REQUEST_INTEGRITY', rows=rows, same_model_config_template_runtime=True,
        conclusions=['All 16 sent images exactly match mapped initial/final workspace AND trace bytes.',
            'Raw generation, API response and parsed semantic label agree; no truncation.',
            'F init2/init3 native success conflicts with generated visual verdict. Both remain quarantined.',
            'Request integrity does not establish semantic correctness. Whole inputs differ across arms; no same-input contradiction claim.',
            'No new model calls, re-labels, accepted experience or training performed by this audit.'])
    dest = ROOT/'planning'/NAME/'VERIFIER_REQUEST_AUDIT.json'
    with dest.open('x') as f: json.dump(report,f,indent=2,ensure_ascii=False)
    print(json.dumps({'status':report['status'],'rows':len(rows),'images':sum(len(r['images']) for r in rows)}))
if __name__ == '__main__': main()
