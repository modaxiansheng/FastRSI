"""Six saved verifier calls: original bytes versus static entity identity only.

No service is started, robot action executed, or experience written. The frozen
inputs retain pre-existing runtime fields in BOTH arms; this diagnostic does
not measure independence from those fields or establish general visual quality.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
import traceback


TASKS = {
    9: {
        'task_description': 'put the yellow and white mug in the microwave and close it',
        'entities': [
            {'phrase': 'the yellow and white mug', 'object_id': 'white_yellow_mug_1',
             'type': 'white_yellow_mug', 'count': 1},
            {'phrase': 'the microwave', 'object_id': 'microwave_1',
             'type': 'microwave', 'count': 1},
        ],
        'disambiguation': 'The words yellow and white describe one mug, not two mugs.',
        'source': 'KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it.bddl',
    },
    2: {
        'task_description': 'turn on the stove and put the moka pot on it',
        'entities': [
            {'phrase': 'the moka pot', 'object_id': 'moka_pot_1', 'type': 'moka_pot', 'count': 1},
            {'phrase': 'the stove', 'object_id': 'flat_stove_1', 'type': 'flat_stove', 'count': 1},
        ],
        'disambiguation': 'The pronoun it refers to the stove.',
        'source': 'KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it.bddl',
    },
}
CASES = ((9, 'N'), (9, 'F'), (2, 'N'))
IDENTITY_KEYS = ('model_name', 'model_class', 'model_type', 'transformers', 'torch',
                 'native_template', 'config_sha256', 'chat_template_sha256')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2))


def static_binding(task):
    definition = TASKS[task]
    return {
        'version': 'static_entity_identity_v1',
        'task_description': definition['task_description'],
        'entities': copy.deepcopy(definition['entities']),
        'disambiguation': definition['disambiguation'],
        'scope': 'Entity identity and count from the task specification only; no claim about current location, grasp, door state, or completion.',
    }


def prepare(body, task, variant, binding):
    if binding != static_binding(task):
        raise ValueError('Static entity binding changed or contains additional information')
    if variant not in ('original', 'identity'):
        raise ValueError('Unknown variant')
    result = copy.deepcopy(body)
    users = [m for m in result.get('messages', []) if m.get('role') == 'user']
    if len(users) != 1 or not isinstance(users[0].get('content'), list) or result.get('tools'):
        raise ValueError('Expected one saved multimodal verifier request, without tools')
    before = copy.deepcopy(users[0]['content'])
    texts = [p['text'] for p in before if p.get('type') == 'text' and 'VERIFICATION_BUNDLE:\n' in p.get('text', '')]
    if len(texts) != 1:
        raise ValueError('Missing unmodified verification bundle')
    bundle = json.loads(texts[0].split('VERIFICATION_BUNDLE:\n', 1)[1])
    if bundle['session']['task_description'] != TASKS[task]['task_description']:
        raise ValueError('Saved request task does not match static entity identity')
    if variant == 'identity':
        users[0]['content'].append({'type': 'text', 'text': 'STATIC_TASK_ENTITY_IDENTITY: ' + json.dumps(binding, ensure_ascii=False, sort_keys=True)})
    # Exact removal restores all source messages, images, system prompt and settings.
    check = copy.deepcopy(result)
    next(m for m in check['messages'] if m['role'] == 'user')['content'] = before
    if check != body:
        raise AssertionError('Unexpected modification outside the appended static binding')
    return result


def declared_sections(text):
    """Read only the static declarations needed for the entity adapter.

    BDDL files are archived in full for source provenance but regions, initial
    states and goal predicates are never sent to the verifier by this adapter.
    """
    language = re.search(r'\(:language\s+([^()]+)\)', text)
    if not language:
        raise ValueError('Missing BDDL language')
    sections = {}
    for key in ('fixtures', 'objects', 'obj_of_interest'):
        match = re.search(r'\(:' + key + r'\s+([^()]+)\)', text)
        if not match:
            raise ValueError('Missing BDDL declaration: ' + key)
        sections[key] = match.group(1).split()
    return language.group(1).strip(), sections


def freeze_inputs(project, output):
    """Requires static_sources/*.bddl already byte-copied from frozen LIBERO."""
    if (output / 'PROTOCOL.json').exists() or (output / 'inputs').exists():
        raise FileExistsError('Will not replace frozen entity diagnostic inputs')
    inputs = output / 'inputs'
    inputs.mkdir()
    rows = []
    for task, arm in CASES:
        name = f'task{task}_{arm}'
        definition = TASKS[task]
        bddl = output / 'static_sources' / definition['source']
        language, sections = declared_sections(bddl.read_text())
        if language != definition['task_description']:
            raise ValueError('Static task language mismatch')
        declaration = sections['fixtures'] + sections['objects']
        for entity in definition['entities']:
            token = entity['object_id']
            if declaration.count(token) != 1 or sections['obj_of_interest'].count(token) != 1:
                raise ValueError('Entity must occur exactly once in static declarations and task interests')
            index = declaration.index(token)
            if declaration[index+1:index+3] != ['-', entity['type']]:
                raise ValueError('Static entity class mismatch')
        source = project / 'runs/procedure_probe_v1_20261004' / f'task{task}' / arm / 'stream/service_calls'
        matches = []
        for path in source.glob('*/REQUEST.json'):
            body = json.loads(path.read_bytes())
            if not body.get('tools') and any('VERIFICATION_BUNDLE:' in p.get('text', '')
                    for m in body.get('messages', []) if isinstance(m.get('content'), list)
                    for p in m['content']):
                matches.append(path)
        if len(matches) != 1:
            raise ValueError('Expected exactly one verifier request for ' + name)
        original = matches[0]
        folder = inputs / name
        folder.mkdir()
        artifacts = {}
        for filename in ('REQUEST.json', 'RESPONSE.json', 'RECEIPT.json', 'RAW.txt', 'OUTPUT_TOKEN_IDS.json'):
            blob = (original.parent / filename).read_bytes()
            (folder / filename).write_bytes(blob)
            artifacts[filename] = {'sha256': sha(blob), 'bytes': len(blob)}
        identity = original.parent.parent / 'SERVICE_IDENTITY.json'
        blob = identity.read_bytes()
        (folder / 'SERVICE_IDENTITY.json').write_bytes(blob)
        artifacts['SERVICE_IDENTITY.json'] = {'sha256': sha(blob), 'bytes': len(blob)}
        binding = static_binding(task)
        save(folder / 'STATIC_BINDING.json', binding)
        artifacts['STATIC_BINDING.json'] = {'sha256': sha((folder / 'STATIC_BINDING.json').read_bytes())}
        body = json.loads(original.read_bytes())
        prepare(body, task, 'identity', binding)
        rows.append({'name': name, 'task': task, 'arm': arm,
                     'order': ['identity', 'original'] if arm == 'F' else ['original', 'identity'],
                     'source_request': str(original.relative_to(project)), 'artifacts': artifacts,
                     'static_source': {'file': 'static_sources/' + bddl.name, 'sha256': sha(bddl.read_bytes()),
                                       'extracted_sections': ['language', 'fixtures', 'objects', 'obj_of_interest']}})
    protocol = {'version': 1, 'cases': rows, 'maximum_model_calls': 6, 'maximum_seconds': 450,
                'request_timeout_seconds': 90, 'robot_calls': 0, 'experience_updates': 0,
                'source_scope': 'Previously exposed development cases, selected for a known entity ambiguity plus one successful control; not a test set.',
                'input_scope': 'Original requests include pre-existing runtime success/reward fields. Both variants retain them verbatim. Only the identity variant appends task-static entity identities/counts. No replay state or hidden outcome is added.',
                'interpretation': 'Request-interpretation diagnostic only. No independent visual accuracy, robot performance, experience admission or learning claim.'}
    save(output / 'PROTOCOL.json', protocol)
    files = {str(p.relative_to(output)): sha(p.read_bytes()) for p in sorted(output.rglob('*')) if p.is_file()}
    save(output / 'INPUT_MANIFEST.json', {'files': files, 'maximum_model_calls': 6})


def run(root, url):
    from replay_resident_qwen import replay_one, SETTING_FIELDS
    from fastrsi.contracts import json_object_text
    from PhyAgentOS.verification.service import _normalize
    protocol = json.loads((root / 'PROTOCOL.json').read_text())
    manifest = json.loads((root / 'INPUT_MANIFEST.json').read_text())
    for name, expected in manifest['files'].items():
        if sha((root / name).read_bytes()) != expected:
            raise ValueError('Frozen input SHA mismatch: ' + name)
    if [(x['task'], x['arm']) for x in protocol['cases']] != list(CASES) or protocol['maximum_model_calls'] != 6:
        raise ValueError('Unexpected diagnostic cases or call budget')
    stream = root / 'stream'
    stream.mkdir(exist_ok=True)
    if (stream / 'RESULT.json').exists():
        raise FileExistsError('Diagnostic already attempted; never overwrite or repeat automatically')
    service = {'url': url, 'log_dir': str(stream / 'service_calls')}
    current_identity = json.loads((Path(service['log_dir']) / 'SERVICE_IDENTITY.json').read_text())
    started = time.monotonic()
    deadline = started + protocol['maximum_seconds']
    result = {'status': 'RUNNING', 'rows': [], 'maximum_model_calls': 6, 'calls_attempted': 0,
              'robot_calls': 0, 'persistent_learning_updates': 0,
              'scope': protocol['interpretation'], 'input_scope': protocol['input_scope']}
    save(stream / 'RESULT.json', result)
    try:
        for spec in protocol['cases']:
            folder = root / 'inputs' / spec['name']
            identity = json.loads((folder / 'SERVICE_IDENTITY.json').read_text())
            mismatched = [k for k in IDENTITY_KEYS if identity.get(k) != current_identity.get(k)]
            if mismatched:
                raise ValueError('Service differs from frozen request source: ' + str(mismatched))
            source = (folder / 'REQUEST.json').read_bytes()
            body = json.loads(source)
            source_receipt = json.loads((folder / 'RECEIPT.json').read_text())
            binding = json.loads((folder / 'STATIC_BINDING.json').read_text())
            archived = json.loads((folder / 'RESPONSE.json').read_text())
            for variant in spec['order']:
                if time.monotonic() >= deadline:
                    raise TimeoutError('Bounded saved-request diagnostic deadline')
                destination = stream / (spec['name'] + '_' + variant)
                row = {'name': spec['name'], 'variant': variant,
                       'source_request_sha256': sha(source),
                       'archived_message': archived['choices'][0]['message'],
                       'images_and_original_text_unchanged': True,
                       'only_added_information': 'static entity identity/count' if variant == 'identity' else None}
                tick = time.monotonic()
                try:
                    prepared = prepare(body, spec['task'], variant, binding)
                    wire = source if variant == 'original' else json.dumps(prepared, ensure_ascii=False).encode()
                    result['calls_attempted'] += 1
                    reply = replay_one(service, wire, destination, protocol['request_timeout_seconds'], deadline)
                    row['replay'] = reply
                    if any(reply.get(k) != source_receipt.get(k) for k in SETTING_FIELDS):
                        raise ValueError('Model decoding/settings changed versus source request')
                    response = json.loads((destination / 'HTTP_RESPONSE.json').read_text())
                    row['normalized'] = _normalize(json.loads(json_object_text(response['choices'][0]['message']['content'])))
                    row['status'] = 'PASSED' if row['normalized']['verifier_status'] == 'completed' else 'INVALID_VERDICT'
                except Exception:
                    row.update(status='FAILED', error=traceback.format_exc())
                row['complete_diagnostic_seconds'] = time.monotonic() - tick
                result['rows'].append(row)
                destination.mkdir(exist_ok=True)
                save(destination / 'EVALUATION.json', row)
                save(stream / 'RESULT.json', result)
        result['status'] = 'COMPLETED' if len(result['rows']) == 6 and all(x['status'] == 'PASSED' for x in result['rows']) else 'COMPLETED_WITH_ERRORS'
    except BaseException:
        result.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        result['seconds'] = time.monotonic() - started
        result['source_files_unchanged'] = all(sha((root / name).read_bytes()) == expected for name, expected in manifest['files'].items())
        save(stream / 'RESULT.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--prepare-project', type=Path)
    parser.add_argument('--url', default='http://127.0.0.1:49510/v1/chat/completions')
    arguments = parser.parse_args()
    if arguments.prepare_project:
        freeze_inputs(arguments.prepare_project, arguments.run)
    else:
        run(arguments.run, arguments.url)
