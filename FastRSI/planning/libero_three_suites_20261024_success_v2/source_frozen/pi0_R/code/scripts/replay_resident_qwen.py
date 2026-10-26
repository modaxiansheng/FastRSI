"""Bounded real-request residency probe; never executes returned tools.

Posts immutable saved request bytes in A -> B -> unrelated B -> repeated B order.
Two newly loaded services and their readable log directories are supplied by the
orchestrator. This checks limited sequential request independence, not concurrent
service safety, long-run isolation, latency benefit, or benchmark correctness.
"""
import argparse
import copy
import hashlib
import json
import re
import shutil
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

SETTING_FIELDS = ('decoding', 'requested_max_tokens', 'wire_format',
                  'memory_generated_repetition_penalty', 'memory_prompt_penalized')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def normalize_response(response):
    """Remove only transport/provider-generated identities; preserve arguments.

    In particular, IDs/timestamps *inside* content or function arguments are
    model output and must not be removed. Token and RAW equality are additional,
    independent requirements, so formatting is not silently normalized away.
    """
    result = copy.deepcopy(response)
    result.pop('id', None)
    result.pop('created', None)
    for choice in result.get('choices', []):
        for call in choice.get('message', {}).get('tool_calls', []):
            call.pop('id', None)
    return result


def validate_output(response, receipt, tokens):
    if response.get('error') or receipt.get('status') != 'OK':
        raise ValueError('Service error or unsuccessful receipt')
    if not re.fullmatch(r'call_[0-9a-f]{32}', response.get('id', '')):
        raise ValueError('Invalid service call identity')
    if receipt.get('generation_finish_reason') != 'eos' or not receipt.get('ended_with_eos'):
        raise ValueError('Incomplete or truncated generation')
    if not tokens or any(type(token) is not int for token in tokens):
        raise ValueError('Missing or invalid output token IDs')
    if tokens[-1] not in receipt.get('eos_token_ids', []):
        raise ValueError('EOS receipt disagrees with actual last token')
    choices = response.get('choices', [])
    if len(choices) != 1 or choices[0].get('finish_reason') not in ('stop', 'tool_calls'):
        raise ValueError('Invalid or truncated API result')
    if choices[0]['finish_reason'] != receipt.get('api_finish_reason'):
        raise ValueError('API result and receipt disagree')
    if choices[0].get('message', {}).get('role') != 'assistant':
        raise ValueError('Missing assistant message')
    if response.get('usage', {}).get('completion_tokens') != len(tokens):
        raise ValueError('Token count disagrees with API result')
    if receipt.get('completion_tokens') != len(tokens):
        raise ValueError('Token count disagrees with receipt')


def compare(left, right):
    fields = ('normalized_response', 'output_token_ids', 'raw_sha256',
              'model_prompt_sha256', 'wire_tools_sha256', 'request_sha256',
              *SETTING_FIELDS)
    checks = {field: left[field] == right[field] for field in fields}
    return {'passed': all(checks.values()), 'checks': checks}


def request_binding(item):
    path = Path(item['request'])
    data = path.read_bytes()
    if sha(data) != item['sha256']:
        raise ValueError('Saved request SHA mismatch: ' + str(path))
    body = json.loads(data)
    if not isinstance(body.get('messages'), list) or not body['messages']:
        raise ValueError('Saved request has no complete message list')
    # Do not change histories, paths, inline images, schemas, decoder, or budget.
    receipt_path = path.parent/'RECEIPT.json'
    receipt = json.loads(receipt_path.read_text()) if receipt_path.is_file() else {}
    return data, {
        'name': item['name'], 'request': str(path), 'sha256': sha(data),
        'bytes': len(data), 'messages': len(body['messages']),
        'roles': [message['role'] for message in body['messages']],
        'tool_schemas': len(body.get('tools', [])),
        'source_receipt_sha256': sha(receipt_path.read_bytes()) if receipt_path.is_file() else None,
        'expected_service_settings': {field: receipt[field] for field in SETTING_FIELDS if field in receipt},
        'inline_image_parts': sum(
            part.get('type') == 'image_url'
            for message in body['messages']
            for part in (message['content'] if isinstance(message.get('content'), list) else [])
        ),
    }


def wait_evidence(folder, deadline):
    required = ('REQUEST.json', 'RESPONSE.json', 'RECEIPT.json', 'RAW.txt',
                'OUTPUT_TOKEN_IDS.json', 'MODEL_PROMPT.txt', 'WIRE_TOOLS.json')
    # FastAPI finally flushes receipt before sending, but tolerate filesystem lag.
    until = min(deadline, time.monotonic() + 3)
    while not all((folder / name).is_file() for name in required):
        if time.monotonic() >= until:
            raise ValueError('Incomplete service evidence: ' + str(folder))
        time.sleep(0.05)
    return required


def replay_one(service, data, destination, timeout, deadline):
    destination.mkdir(exist_ok=False)
    (destination / 'REQUEST_BYTES.json').write_bytes(data)
    started = time.monotonic()
    request = urllib.request.Request(service['url'], data=data,
                                     headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urllib.request.urlopen(request, timeout=min(timeout, deadline-started)) as reply:
            response_bytes = reply.read()
            http_status = reply.status
    except urllib.error.HTTPError as exc:
        (destination / 'HTTP_ERROR_BODY').write_bytes(exc.read())
        save(destination / 'HTTP_ERROR.json', {'status': exc.code, 'seconds': time.monotonic()-started})
        raise
    elapsed = time.monotonic() - started
    (destination / 'HTTP_RESPONSE.json').write_bytes(response_bytes)
    response = json.loads(response_bytes)
    if not re.fullmatch(r'call_[0-9a-f]{32}', response.get('id', '')):
        raise ValueError('Response missing valid service log identity')
    call = Path(service['log_dir']) / response['id']
    required = wait_evidence(call, deadline)
    receipt = json.loads((call / 'RECEIPT.json').read_text())
    tokens = json.loads((call / 'OUTPUT_TOKEN_IDS.json').read_text())
    validate_output(response, receipt, tokens)
    if json.loads((call / 'REQUEST.json').read_text()) != json.loads(data):
        raise ValueError('Service did not receive exact saved request values')
    if json.loads((call / 'RESPONSE.json').read_text()) != response:
        raise ValueError('HTTP response differs from recorded response')
    bound = {}
    for name in required:
        value = (call / name).read_bytes()
        bound[name] = {'sha256': sha(value), 'bytes': len(value)}
        shutil.copyfile(call / name, destination / name)
    result = {
        'call_id': response['id'], 'http_status': http_status,
        'http_seconds': elapsed, 'service_generation_seconds': receipt.get('generation_seconds'),
        'service_log_dir': str(call), 'artifacts': bound,
        'request_sha256': sha(data), 'normalized_response': normalize_response(response),
        'output_token_ids': tokens, 'raw_sha256': bound['RAW.txt']['sha256'],
        'model_prompt_sha256': bound['MODEL_PROMPT.txt']['sha256'],
        'wire_tools_sha256': bound['WIRE_TOOLS.json']['sha256'],
        **{field: receipt.get(field) for field in SETTING_FIELDS},
    }
    save(destination / 'RESULT.json', result)
    return result


def run(protocol, output):
    config = protocol.get('qwen_replay', protocol)
    probes = config['probes']; services = config['services']
    if not 1 <= len(probes) <= 3:
        raise ValueError('Expected one to three predeclared probe requests')
    if len({item['name'] for item in probes}) != len(probes):
        raise ValueError('Probe names must be unique')
    for item in [*probes, config['exposure']]:
        if not re.fullmatch(r'[A-Za-z0-9_-]+', item['name']):
            raise ValueError('Unsafe probe name')
    calls = len(probes)*3+1
    if calls > min(10, config.get('max_calls', 10)):
        raise ValueError('Call cap exceeded')
    if services['A']['url'] == services['B']['url'] or services['A']['log_dir'] == services['B']['log_dir']:
        raise ValueError('A and B must be independent service endpoints and logs')
    bound = [request_binding(item) for item in probes]
    exposure = request_binding(config['exposure'])
    for _, binding in [*bound, exposure]:
        if set(binding['expected_service_settings']) != set(SETTING_FIELDS):
            raise ValueError('Saved receipt lacks bound generation settings: ' + binding['request'])
    output.mkdir(exist_ok=False, parents=True)
    save(output / 'PROTOCOL.json', protocol)
    result = {'status': 'RUNNING', 'expected_calls': calls, 'calls_completed': 0,
              'bindings': [binding for _, binding in bound], 'exposure_binding': exposure[1],
              'tools_executed': 0, 'persistent_experience_updates': 0,
              'scope': 'Sequential saved-request equivalence only; no speed or quality claim',
              'rows': [], 'comparisons': []}
    start = time.monotonic(); deadline = start + config.get('maximum_seconds', 700)
    evidence = {}
    try:
        for key in ('A', 'B'):
            log = Path(services[key]['log_dir'])
            prior = sorted(path.name for path in log.glob('call_*'))
            if config.get('require_fresh_services', True) and prior:
                raise ValueError('Service already has model calls: ' + key)
            identity = log / 'SERVICE_IDENTITY.json'
            if not identity.is_file():
                raise ValueError('Missing service identity: ' + key)
            shutil.copyfile(identity, output / ('SERVICE_IDENTITY_' + key + '.json'))
            result['service_' + key] = {'prior_calls': prior, 'identity': json.loads(identity.read_text())}
        identity_keys = ('model_name', 'model_class', 'model_type', 'transformers', 'torch',
                         'native_template', 'config_sha256', 'chat_template_sha256')
        result['service_identity_matches'] = {
            key: result['service_A']['identity'].get(key) == result['service_B']['identity'].get(key)
            for key in identity_keys}
        if not all(result['service_identity_matches'].values()):
            raise ValueError('Service model/protocol identities differ')
        jobs = [('A', 'before', item) for item in bound]
        jobs += [('B', 'before', item) for item in bound]
        jobs += [('B', 'exposure', exposure)]
        jobs += [('B', 'after', item) for item in bound]
        seen_ids = set()
        for index, (service, phase, (data, binding)) in enumerate(jobs):
            if time.monotonic() >= deadline:
                raise TimeoutError('Replay wall deadline reached')
            name = f'{index:02d}_{service}_{phase}_{binding["name"]}'
            row = replay_one(services[service], data, output/name,
                             config.get('request_timeout_seconds', 100), deadline)
            if any(row[field] != expected for field, expected in binding['expected_service_settings'].items()):
                raise ValueError('Replay differs from bound original generation settings: ' + name)
            if row['call_id'] in seen_ids:
                raise ValueError('Service reused a call identity')
            seen_ids.add(row['call_id'])
            evidence[(service, phase, binding['name'])] = row
            result['rows'].append({'job': name, 'service': service, 'phase': phase,
                                   'name': binding['name'], 'call_id': row['call_id'],
                                   'http_seconds': row['http_seconds']})
            result['calls_completed'] += 1
            save(output/'RESULT.json', result)
        for _, binding in bound:
            name = binding['name']
            for label, left, right in (
                ('independent_service_baseline', ('A', 'before'), ('B', 'before')),
                ('resident_exposure', ('B', 'before'), ('B', 'after')),
                ('resident_vs_independent', ('A', 'before'), ('B', 'after'))):
                result['comparisons'].append(dict(name=name, comparison=label,
                    **compare(evidence[(*left, name)], evidence[(*right, name)])))
        for item in [*probes, config['exposure']]:
            request_binding(item)
        result['source_requests_unchanged'] = True
        result['status'] = 'COMPLETED'
        result['equivalence_passed'] = all(row['passed'] for row in result['comparisons'])
    except BaseException:
        result.update(status='FAILED', error=traceback.format_exc(), equivalence_passed=False)
        raise
    finally:
        result['wall_seconds'] = time.monotonic()-start
        save(output/'RESULT.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    summary = run(json.loads(args.protocol.read_text()), args.output)
    print(json.dumps({key: summary[key] for key in ('status', 'calls_completed', 'equivalence_passed', 'wall_seconds')}))
    if not summary['equivalence_passed']:
        raise SystemExit(1)
