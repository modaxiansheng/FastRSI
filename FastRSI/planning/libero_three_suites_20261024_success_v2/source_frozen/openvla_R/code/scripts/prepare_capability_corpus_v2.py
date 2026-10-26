"""Build grouped behavior data from an explicit development-only allowlist.

No inferred branch values: targets are only commands actually submitted by
the original policy and fully consumed. Outcome labels never enter inputs.
"""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAME = 'capability_corpus_v3_20261006'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, obj):
    with p.open('x') as f: json.dump(obj, f, indent=2, allow_nan=False)


def main():
    start = time.perf_counter()
    path = ROOT/'planning/capability_corpus_v2_20261006/REGISTRATION_v2.json'
    recipe = json.loads(path.read_text())
    train, probe = set(recipe['expanded_train_groups']), set(recipe['diagnostic_groups'])
    assert not train & probe
    assert not (train | probe) & set(recipe['forbidden_evaluation_groups'])
    assert len(recipe['source_streams']) == len(train | probe)
    run = ROOT/'runs'/NAME
    inputs = run/'inputs'
    inputs.mkdir(parents=True, exist_ok=False)
    bindings = {str(path.relative_to(ROOT)): sha(path)}
    samples, labels, audit = [], {}, []
    for source in recipe['source_streams']:
        group = source['init']
        stream = ROOT/source['stream']
        trace_path = ROOT/source['trace']
        result_path, cost_path = stream/'RESULT.json', stream/'OS_COST_EVENTS.jsonl'
        result = json.loads(result_path.read_text())['cases'][0]
        case = result['case']
        assert case['mode'] == 'development' and case['suite'] == 'libero_10'
        assert case['task_id'] == 4 and case['init_state_id'] == group
        assert result['environment_success'] is True and result['chain_status'] == 'PASSED'
        assert result['experience_admission']['status'] == 'ADMITTED'
        assert case['feedback_window_mode'] == 'fixed10'
        traces = list((stream/'stage_trace').glob('*/TRACE.jsonl'))
        assert traces == [trace_path], 'No reset/retry pooling'
        trace = [json.loads(l) for l in trace_path.read_text().splitlines()]
        events = [json.loads(l) for l in cost_path.read_text().splitlines()]
        actual = [e for e in events if e['event'] == 'actual_policy_input']
        windows = [e for e in events if e['event'] == 'feedback_window_choice']
        assert len(actual) == len(windows) == len(trace)-1
        assert trace[0]['method'] == 'reset' and trace[0]['step'] == 0
        for i, window in enumerate(windows):
            assert window['step'] == trace[i]['step']
            assert window['original_steps'] == window['submitted_steps'] == 10
            assert np.array_equal(np.asarray(window['predicted_actions'], np.float32),
                                  np.asarray(trace[i+1]['actions'], np.float32))
            assert actual[i]['prompt'] == case['task_description']
        closing = active = False
        group_ids = []
        for i in range(1, len(trace)-1):
            consumed_prior = trace[i]['step']-trace[i-1]['step']
            prior = np.asarray(trace[i]['actions'], np.float32)[:consumed_prior, 6]
            if closing and len(prior) == 10 and np.all(prior > 0): active = True
            if active and np.any(prior <= 0): break
            if np.any(prior <= 0) and np.any(prior > 0) and prior[-1] > 0: closing = True
            if not active or trace[i+1]['step']-trace[i]['step'] != 10: continue
            # All conditions above are from the observed prefix. Future original
            # commands below are labels only, including release negatives.
            event, arrays = actual[i], {}
            for name, camera in [('observation/image','agentview_image'),
                                 ('observation/wrist_image','robot0_eye_in_hand_image')]:
                image = trace_path.parent/f'{trace[i]["step"]:04d}_{camera}.png'
                digest = sha(image)
                assert digest == trace[i]['images'][image.name]
                bindings[str(image.relative_to(ROOT))] = digest
                arrays[name] = np.asarray(Image.open(image)).copy()
            arrays['observation/state'] = np.asarray(event['robot_state'], np.float32)
            for name, array in arrays.items():
                assert hashlib.sha256(array.tobytes()).hexdigest() == event['array_bindings'][name]['sha256']
            action = np.asarray(trace[i+1]['actions'], np.float32)
            assert action.shape == (10,7) and np.isfinite(action).all()
            identity = f'init{group}_step{trace[i]["step"]:03d}'
            assert identity not in labels
            np.savez(inputs/(identity+'.npz'), **arrays)
            samples.append(dict(id=identity, file=identity+'.npz', group=group,
                                step=trace[i]['step'], prompt=event['prompt'], source=source['stream']))
            labels[identity] = dict(actions=action.tolist(), closed_only=bool(np.all(action[:,6] > 0)),
                group=group, split='train' if group in train else 'diagnostic',
                scope='Consumed original behavior; not causal operation value or unique correct action')
            group_ids.append(identity)
        for p in [trace_path, cost_path, result_path]: bindings[str(p.relative_to(ROOT))] = sha(p)
        audit.append(dict(group=group, samples=len(group_ids), closed=sum(labels[k]['closed_only'] for k in group_ids),
                          actual_policy_calls=len(actual), source=source['stream'], all_submitted_actions_original=True))
    assert len(samples) <= recipe['bounds']['prefix_calls_max']
    assert {r['group'] for r in samples} == train | probe
    old = json.loads((ROOT/'runs/contextual_capability_features_v1_20261006/inputs/REGISTRATION.json').read_text())
    reg = dict(status='REGISTERED', registered_at=time.time(), samples=samples,
        input_manifest={p.name:sha(p) for p in inputs.iterdir()}, historical_sources=bindings,
        bound_module_sha256=old['bound_module_sha256'], recipe_sha256=sha(path),
        expected_robot_steps=0, expected_decoder_calls=0,
        preparation_seconds=time.perf_counter()-start, evaluation_inputs_read=False)
    save(inputs/'REGISTRATION.json', reg)
    save(run/'TRAIN_LABELS.json', {k:v for k,v in labels.items() if v['split']=='train'})
    save(run/'DIAGNOSTIC_LABELS.json', {k:v for k,v in labels.items() if v['split']=='diagnostic'})
    save(run/'SOURCE_AUDIT.json', audit)
    print(json.dumps(dict(samples=len(samples), groups=audit, seconds=reg['preparation_seconds'])))


if __name__ == '__main__': main()
