"""Register sources before new labels; prepare legal images separately from truth."""
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAME = 'object_stage_memory_v2_20261006'
SOURCES = [
    dict(init=26, role='seed', stream='runs/task4_informative_pair_dev_187_v8_20261006/native10a/init26/F/stream'),
    dict(init=31, role='update', stream='runs/placement_recovery_v1_20261003/pair0/N/stream'),
    dict(init=32, role='probe', stream='runs/task4_corrected28_trial_dev_187_v1_20261006/native_control/init32/F/stream'),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    with path.open('x') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)


def register():
    run = ROOT / 'runs' / NAME
    run.mkdir(exist_ok=False)
    sources = []
    for source in SOURCES:
        p = ROOT / source['stream']
        result = json.loads((p / 'RESULT.json').read_text())
        case = result['cases'][0]['case']
        if case['mode'] != 'development' or case['task_id'] != 4 or case['init_state_id'] != source['init']:
            raise ValueError('Development source identity changed')
        trace = p / 'stage_trace' / case['id'] / 'TRACE.jsonl'
        sources.append(dict(**source, trace=str(trace.relative_to(ROOT)),
            trace_sha256=sha(trace), result_sha256=sha(p / 'RESULT.json'),
            launch_sha256=sha(p / 'LAUNCH_target.json')))
    predecessor = ROOT / 'runs/object_stage_memory_v1_20261006'
    failed = json.loads((predecessor / 'replays/STATUS.json').read_text())
    if failed['status'] != 'FAILED':
        raise ValueError('Expected retained failed annotation attempt')
    save(run / 'REGISTRATION.json', dict(version=2, name=NAME, sources=sources,
        predecessor=dict(path=str(predecessor.relative_to(ROOT)),
            registration_sha256=sha(predecessor / 'REGISTRATION.json'),
            status_sha256=sha(predecessor / 'replays/STATUS.json'),
            parallel_cpu_replay_seconds=failed['seconds'],
            cause='Static task object binding used object-name heuristics; repaired exact task definition binding.'),
        registered_at=time.time(), sampled_steps='reset, every recorded boundary divisible by 30, and terminal; previous is actual preceding boundary',
        online_hidden_facts=False, scope='Retrospective development stage supervision, not final test or policy-value labels',
        model_revision='17f0b0ad64efb65d273590632833508766b2aae6',
        fit=dict(seed=17, learning_rate=.02, iterations=400, weight_decay=.01), threshold=.95,
        maximum_model_inputs=80, default_changed=False))
    print(str(run))


def inputs():
    sys.path.insert(0, str(ROOT / 'src'))
    from fastrsi.object_stage_memory_v1 import request
    from fastrsi.decision_corpus import png_pixels
    run = ROOT / 'runs' / NAME
    reg = json.loads((run / 'REGISTRATION.json').read_text())
    out = run / 'inputs'
    out.mkdir(exist_ok=False)
    records, labels, probes, manifest = [], [], [], {}
    for source in reg['sources']:
        replay_path = run / 'replays' / ('init' + str(source['init'])) / 'REPLAY.json'
        replay = json.loads(replay_path.read_text())
        if (replay['status'] != 'EXACT_IMAGE_REPLAY' or replay['source_trace_sha256'] != source['trace_sha256']
                or replay['source_result_sha256'] != source['result_sha256']):
            raise ValueError('Unverified exact developmental supervision')
        manifest[str(replay_path.relative_to(ROOT))] = sha(replay_path)
        folder = ROOT / source['trace']
        group = 'task4/development/init' + str(source['init'])
        for index, current in enumerate(replay['rows']):
            if index not in (0, len(replay['rows']) - 1) and current['step'] % 30:
                continue
            previous = replay['rows'][max(0, index - 1)]
            images = []
            for boundary in (previous, current):
                for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
                    name = f'{boundary["step"]:04d}_{camera}.png'
                    path = folder.parent / name
                    raw = path.read_bytes()
                    h = hashlib.sha256(raw).hexdigest()
                    if h != boundary['images'][name]:
                        raise ValueError('Original policy image changed')
                    pixels, shape = png_pixels(raw)
                    rel = str(path.relative_to(ROOT))
                    images.append(dict(path=rel, sha256=h,
                        pixel_sha256=hashlib.sha256(pixels).hexdigest()))
                    manifest[rel] = h
            for subject in ('white_mug', 'yellow_white_mug'):
                identity = f'i{source["init"]}-s{current["step"]}-{subject}'
                records.append(dict(id=identity, group=group, role=source['role'], subject=subject,
                    images=images, request=request(subject, current['robot8'], previous['robot8'])))
                target = dict(id=identity, group=group, role=source['role'], subject=subject,
                    eligible_to_skip=current['offline_supervision'][subject]['eligible_to_skip'],
                    label_scope='native_subgoal_plus_release_two_observed_boundaries')
                (probes if source['role'] == 'probe' else labels).append(target)
    if len(records) > reg['maximum_model_inputs']:
        raise ValueError('Registered extraction budget exceeded')
    for filename, value in [('INPUTS.json', records), ('SUPERVISION.json', labels),
                            ('PROBE_SUPERVISION.json', probes), ('SOURCE_MANIFEST.json', manifest),
                            ('PROTOCOL.json', {k: reg[k] for k in ('model_revision', 'fit', 'threshold')})]:
        save(out / filename, value)
    save(out / 'SUMMARY.json', dict(rows=len(records), groups=len(reg['sources']),
        training_rows=len(labels), probe_rows=len(probes),
        label_positive_counts={role: sum(x['eligible_to_skip'] for x in labels + probes if x['role'] == role)
                               for role in ('seed', 'update', 'probe')},
        scope='Existing developmental actions only; adjacent frames are not independent contexts.'))
    print((out / 'SUMMARY.json').read_text())


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ('register', 'inputs'):
        raise SystemExit('Expected register or inputs')
    register() if sys.argv[1] == 'register' else inputs()
