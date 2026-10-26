"""Analyze seen-point history diagnostics; never produce training labels."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    run = root / 'runs/drawer_memory_v1_20261004'
    inputs = root / 'planning/drawer_memory_v1_20261004/inputs'
    native = json.loads((root / 'runs/drawer_facts_v2_20261004/NATIVE_LABELS.json').read_text())
    assert native['status'] == 'EXACT_IMAGE_REPLAY'
    labels = {row['step']: row for row in native['rows']}
    records = {}
    for entry in json.loads((inputs / 'MANIFEST.json').read_text())['records']:
        raw = (inputs / entry['file']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == entry['sha256']
        record = json.loads(raw)
        timeline = record['request']['state']['observation_timeline']
        steps = [int(frame['observation_id'].split('_')[-1]) for frame in timeline]
        assert steps == sorted(set(steps)) and steps[-1] == record['current_step']
        for frame, step in zip(timeline, steps):
            for index in frame['image_indices']:
                image = record['images'][index]
                assert labels[step]['images'][Path(image['path']).name] == image['sha256']
        records[entry['file']] = record
    qwen = json.loads((run / 'qwen/RESULT.json').read_text())
    clef = json.loads((run / 'replay/RESULT.json').read_text())
    assert qwen['status'] == clef['status'] == 'COMPLETED'
    models = {
        'qwen': {r['file']: (r['facts'], r['seconds']) for r in qwen['rows']},
        'clef': {r['file']: ({k: v['choice'] for k, v in r['response']['answers'].items()}, r['total_seconds']) for r in clef['calls']},
    }
    results = []
    for model, predictions in models.items():
        assert set(predictions) == set(records)
        for name, (facts, seconds) in predictions.items():
            record = records[name]
            truth = labels[record['current_step']]['facts']
            assert set(facts) == set(truth)
            assert all(v in ('true', 'false', 'unknown') for v in facts.values())
            results.append(dict(model=model, file=name, step=record['current_step'],
                variant=record['variant'], predicted=facts, native_reference=truth,
                reference_agreement=sum(v != 'unknown' and (v == 'true') == truth[k] for k,v in facts.items()),
                reference_disagreement=sum(v != 'unknown' and (v == 'true') != truth[k] for k,v in facts.items()),
                unknown=sum(v == 'unknown' for v in facts.values()), seconds=seconds,
                cold_first_call=name == '000.json'))
    pairs = []
    for model in models:
        for step in (160, 200):
            pair = {r['variant']: r for r in results if r['model'] == model and r['step'] == step}
            pairs.append(dict(model=model, step=step, changed_facts={k: [pair['current_only']['predicted'][k], pair['recent_history']['predicted'][k]]
                for k in truth if pair['current_only']['predicted'][k] != pair['recent_history']['predicted'][k]}))
    analysis = dict(status='ANALYZED', rows=results, pairs=pairs,
        scope='Two previously seen points from one development episode; no unseen evaluation or learning.',
        label_limit='Native geometric containment is not release/stability/clearance. Open threshold differs from loose visual accessibility. No closed-positive examples.',
        decision='Keep bounded history as a candidate representation; reject present outputs as skill authority or distillation labels.',
        skill_updates=0, training_labels=0, learning_gain_proven=False,
        call_seconds={model: sum(r['seconds'] for r in results if r['model'] == model) for model in models},
        wall_seconds=json.loads((run / 'STATUS.json').read_text())['seconds'])
    (run / 'ANALYSIS.json').write_text(json.dumps(analysis, indent=2) + '\n')
    print(json.dumps(dict(pairs=pairs, call_seconds=analysis['call_seconds']), indent=2))


if __name__ == '__main__':
    main()
