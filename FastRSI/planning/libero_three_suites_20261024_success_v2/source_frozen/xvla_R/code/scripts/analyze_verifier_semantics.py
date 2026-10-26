"""Bind every saved-input comparison; agreement is not general verifier accuracy."""
import argparse
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='verifier_semantics_v2_20261003')
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    run = root / 'runs' / args.name
    status, protocol, result = (read(run / p) for p in ('STATUS.json', 'PROTOCOL.json', 'stream/RESULT.json'))
    assert status['status'] == result['status'] == 'COMPLETED'
    assert len(result['cases']) == 16
    cases = []; response_ids = []
    for spec in protocol['cases']:
        source = root / spec['source']
        assert sha(source) == spec['sha256'] == sha(run / 'inputs' / (spec['name'] + '.json'))
        archived = read(source.parent / 'RESPONSE.json')
        archive_tokens = read(source.parent / 'OUTPUT_TOKEN_IDS.json')
        rows = {}
        for variant in ('original', 'adapted'):
            row = next(r for r in result['cases'] if r['name'] == spec['name'] and r['variant'] == variant)
            assert row['status'] == 'PASSED'
            folder = run / 'stream' / (spec['name'] + '_' + variant)
            request = read(folder / 'REQUEST_BYTES.json')
            response = read(folder / 'HTTP_RESPONSE.json')
            receipt = read(folder / 'RECEIPT.json')
            response_ids.append(response['id'])
            assert request == read(folder / 'REQUEST.json')
            before = read(source)
            a = before['messages'][1]['content']; b = request['messages'][1]['content']
            if variant == 'original':
                assert sha(folder / 'REQUEST_BYTES.json') == spec['sha256']
            else:
                assert row['adaptation']['applied']
            images = lambda content: [p for p in content if p.get('type') == 'image_url']
            assert images(a) == images(b)
            assert a[0]['text'].split('VERIFICATION_BUNDLE:\n', 1)[1] == b[0]['text'].split('VERIFICATION_BUNDLE:\n', 1)[1]
            request['messages'][1]['content'] = a
            assert request == before
            rows[variant] = {'verdict': row['normalized']['verdict'], 'normalized': row['normalized'],
                'runtime_agreement': row['runtime_agreement'],
                'http_seconds': row['replay']['http_seconds'],
                'generation_seconds': receipt['generation_seconds'], 'service_seconds': receipt['seconds'],
                'prompt_tokens': receipt['prompt_tokens'], 'completion_tokens': receipt['completion_tokens'],
                'input_sha256': sha(folder / 'REQUEST_BYTES.json'),
                'response_sha256': sha(folder / 'HTTP_RESPONSE.json'),
                'original_archive_content_equal': response['choices'][0]['message'] == archived['choices'][0]['message'] if variant == 'original' else None,
                'original_archive_token_ids_equal': read(folder / 'OUTPUT_TOKEN_IDS.json') == archive_tokens if variant == 'original' else None}
        cases.append({'name': spec['name'], 'task_id': spec['case']['task_id'], 'init_state_id': spec['case']['init_state_id'],
                      'archived_verdict': spec['archived_verdict'], 'measured_environment_success': spec['measured_environment_success'],
                      'original': rows['original'], 'adapted': rows['adapted']})
    assert len(set(response_ids)) == 16
    assert len(list((run / 'stream/service_calls').glob('*/REQUEST.json'))) == 16
    summary = {'run': args.name, 'case_count': len(cases), 'model_calls': 16, 'robot_episodes': 0,
        'unchanged_source_and_images': True, 'all_original_content_reproduced': all(c['original']['original_archive_content_equal'] for c in cases),
        'all_original_tokens_reproduced': all(c['original']['original_archive_token_ids_equal'] for c in cases),
        'runtime_agreement': {v: sum(c[v]['runtime_agreement'] is True for c in cases) for v in ('original', 'adapted')},
        'costs': {v: {k: sum(c[v][k] for c in cases) for k in ('http_seconds', 'generation_seconds', 'service_seconds', 'prompt_tokens', 'completion_tokens')} for v in ('original', 'adapted')},
        'full_wall_seconds': status['full_wall_seconds'], 'model_startup_seconds': status['model_startup_seconds'],
        'replay_seconds': status['replay_seconds'], 'cases': cases,
        'default_enabled': False, 'old_experience_relabelled': False, 'learning_benefit': None,
        'scope': 'In-sample saved development requests. Environment-agreement counts are not representative semantic accuracy, learning gain, end-to-end speedup, or new robot outcomes.'}
    out = root / 'planning' / args.name / 'ANALYSIS.json'
    with out.open('x') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps({k:v for k,v in summary.items() if k != 'cases'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
