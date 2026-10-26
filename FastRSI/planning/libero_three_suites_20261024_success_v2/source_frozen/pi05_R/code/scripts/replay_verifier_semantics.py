"""One original/adapted request pair per declared saved case; no robot or learning."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import traceback

from fastrsi.libero_verification_semantics import adapt_libero_verification_content
from fastrsi.contracts import json_object_text
from PhyAgentOS.verification.service import _normalize
from replay_resident_qwen import replay_one


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(body, variant):
    result = copy.deepcopy(body)
    users = [m for m in result['messages'] if m.get('role') == 'user']
    if len(users) != 1 or not isinstance(users[0]['content'], list):
        raise ValueError('Expected exactly one multimodal user input')
    before = copy.deepcopy(users[0]['content'])
    mapped, receipt = adapt_libero_verification_content(
        before, mode='libero_goal_v1' if variant == 'adapted' else 'off')
    if variant == 'adapted' and not receipt.get('applied'):
        raise ValueError('Declared input did not match adaptation binding')
    if variant == 'original' and mapped != before:
        raise ValueError('Disabled adapter changed original evidence')
    images = lambda parts: [p for p in parts if p.get('type') == 'image_url']
    labels = lambda parts: [p for p in parts if p.get('text', '').startswith('RGB_EVIDENCE_MAPPING:')]
    assert images(mapped) == images(before) and labels(mapped) == labels(before)
    # No system prompt, tools, sampling, timeout or model change in either arm.
    users[0]['content'] = mapped
    check = copy.deepcopy(result)
    next(m for m in check['messages'] if m.get('role') == 'user')['content'] = before
    assert check == body
    return result, receipt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    run = args.run
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    assert len(protocol['cases']) == 8 and protocol['maximum_model_calls'] == 16
    stream = run / 'stream'
    service = {'url': 'http://127.0.0.1:49510/v1/chat/completions',
               'log_dir': str(stream / 'service_calls')}
    started = time.time()
    deadline = time.monotonic() + max(1, protocol['deadline'] - time.time() - 25)
    rows = []
    result = {'status': 'RUNNING', 'cases': rows, 'started_at': started,
              'scope': 'Saved development evidence; no robot execution, memory update or quality generalization'}
    try:
        for spec in protocol['cases']:
            source = run / 'inputs' / (spec['name'] + '.json')
            data = source.read_bytes()
            assert sha(data) == spec['sha256']
            original = json.loads(data)
            for variant in spec['order']:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Declared replay deadline')
                folder = stream / (spec['name'] + '_' + variant)
                row = {'name': spec['name'], 'variant': variant, 'source_sha256': spec['sha256'],
                       'archived_verdict': spec.get('archived_verdict'),
                       'bundle_runtime_success': spec['bundle_runtime_success']}
                try:
                    body, adaptation = prepare(original, variant)
                    row['adaptation'] = adaptation
                    wire = data if variant == 'original' else json.dumps(body, ensure_ascii=False).encode()
                    reply = replay_one(service, wire, folder, timeout=180, deadline=deadline)
                    row['replay'] = reply
                    response = json.loads((folder / 'HTTP_RESPONSE.json').read_text())
                    content = response['choices'][0]['message']['content']
                    normalized = _normalize(json.loads(json_object_text(content)))
                    row['normalized'] = normalized
                    row['status'] = 'PASSED' if normalized['verifier_status'] == 'completed' else 'INVALID_VERDICT'
                    verdict = normalized.get('verdict')
                    row['runtime_agreement'] = (verdict == 'success') == spec['bundle_runtime_success'] if verdict in ('success', 'failure') else None
                except Exception:
                    row.update(status='FAILED', error=traceback.format_exc())
                rows.append(row)
                folder.mkdir(exist_ok=True)
                (folder / 'EVALUATION.json').write_text(json.dumps(row, ensure_ascii=False, indent=2))
                (stream / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        result['status'] = 'COMPLETED' if len(rows) == 16 and all(r['status'] == 'PASSED' for r in rows) else 'COMPLETED_WITH_ERRORS'
    except BaseException:
        result.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        result.update(finished_at=time.time(), seconds=time.time()-started)
        (stream / 'RESULT.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
