"""Bind paid teacher proposals to exact offline replay, never auto-admit labels."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(root, output):
    name = 'procedure_probe_v1_20261004'
    run = root / 'runs' / name
    planning = root / 'planning' / name
    assert read(run / 'STATUS.json')['status'] == 'COMPLETED'
    assert read(planning / 'replay_v1/STATUS.json')['status'] == 'COMPLETED'
    records = []
    for task, init in [(2, 9), (9, 15)]:
        stream = run / f'task{task}/F/stream'
        sid = f'net_t{task}_i{init}'
        trace = stream / 'stage_trace' / sid / 'TRACE.jsonl'
        replay_path = planning / f'replay_v1/task{task}/F/REPLAY.json'
        replay = read(replay_path)
        assert replay['status'] == 'EXACT_IMAGE_REPLAY'
        assert replay['source_trace_sha256'] == sha(trace)
        observed = {r['step']: r for r in replay['rows']}
        mapping = ({'stove_on': 'native_stove_on', 'pot_on_stove': 'native_pot_on_stove'}
                   if task == 2 else {'mug_inside': 'native_mug_inside', 'door_closed': 'native_microwave_closed'})
        for folder in sorted((stream / sid / 'procedure_calls').iterdir(), key=lambda p: int(p.name)):
            inputs = read(folder / 'INPUT.json')
            request = read(folder / 'REQUEST.json')
            result = read(folder / 'RESULT.json')
            assert result['error'] is None
            step = result['step']
            row = observed[step]
            # Validate all request images, including previous-check images, against saved pixels.
            urls = [part['image_url']['url'] for part in request['messages'][1]['content'] if part['type'] == 'image_url']
            assert len(urls) == len(inputs['images'])
            for url, item in zip(urls, inputs['images']):
                data = base64.b64decode(url.split(',', 1)[1], validate=True)
                assert data == (folder / item['file']).read_bytes()
                assert hashlib.sha256(data).hexdigest() == item['sha256']
            fields = {}
            for key, native in mapping.items():
                proposal = result['facts'][key]
                fields[key] = dict(proposal=proposal, native_predicate=row[native],
                    comparison='UNKNOWN' if proposal == 'unknown' else
                    ('AGREES' if (proposal == 'true') == row[native] else 'DISAGREES'),
                    scope='Benchmark-predicate agreement; natural-language fact and geometry are not fully synonymous')
            if task == 9:
                fields['mug_released'] = dict(proposal=result['facts']['mug_released'],
                    native_grasp=row['native_grasp'], contact=row['gripper_contacts_object'],
                    comparison='CONTRADICTED_BY_NATIVE_GRASP' if row['native_grasp'] and result['facts']['mug_released'] == 'true'
                    else 'NOT_ESTABLISHED',
                    scope='Absence of native two-finger grasp does not establish stable release or safe closure')
            bindings = {str(p.relative_to(root)): sha(p) for p in
                        [folder / 'INPUT.json', folder / 'REQUEST.json', folder / 'RESPONSE.json',
                         folder / 'RESULT.json', replay_path, trace]}
            records.append(dict(task=task, init=init, step=step,
                online_input=dict(request_path=str((folder / 'REQUEST.json').relative_to(root)),
                                  request_sha256=sha(folder / 'REQUEST.json'),
                                  saved_images_match_request=True),
                teacher=result['facts'], offline_evidence=fields, source_sha256=bindings,
                learning_target=None, admission='DIAGNOSTIC_ONLY',
                reason='No observed comparison of alternate actions at this exact decision. '
                       'Offline predicates are not online features; two episodes do not establish transfer.'))
    comparisons = [v['comparison'] for r in records for k, v in r['offline_evidence'].items() if k != 'mug_released']
    result = dict(status='AUDITED', teacher_decisions=len(records), independent_initial_states=2,
        native_predicate_agreements=comparisons.count('AGREES'),
        native_predicate_disagreements=comparisons.count('DISAGREES'),
        release_contradictions=sum(r['offline_evidence'].get('mug_released', {}).get('comparison') == 'CONTRADICTED_BY_NATIVE_GRASP' for r in records),
        original_state_snapshots_available=False, trained=False, decision_training_labels=0,
        scope='Four decisions in two development instances, not held-out model accuracy or learned gain',
        candidates=[dict(task=t, candidate='remaining_subgoal', version='1', status='NOT_APPROVED_FOR_REUSE',
                         reason='Unreliable execution preconditions and no measured reduction in VLA calls or elapsed cost') for t in (2, 9)],
        records=records)
    with output.open('x') as handle:
        json.dump(result, handle, indent=2)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = audit(args.root, args.output)
    print(json.dumps({k: v for k, v in result.items() if k not in ('records', 'candidates')}))
