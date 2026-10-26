"""Reconstruct exact historical policy requests from lossless diagnostic images.

This prepares inputs for a service reset equivalence check. The selected nodes
are not a contiguous episode, and replay does not establish robot outcomes or
historical output reproduction. No image conversion, interpolation or hidden
simulator state is allowed. Source files are read-only.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NAME = 'resident_services_v1_20261003'
SPECS = [
    ('mug31_step0', 'pair0', 'net_t4_i31', 0),
    ('mug31_step110', 'pair0', 'net_t4_i31', 110),
    ('drawer38_step140', 'pair1', 'net_t3_i38', 140),
]
IMAGE_KEYS = {'observation/image': 'agentview_image',
              'observation/wrist_image': 'robot0_eye_in_hand_image'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def binding(array):
    return {'shape': list(array.shape), 'dtype': str(array.dtype),
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def check_array(array, expected):
    actual = binding(array)
    if actual != expected:
        raise ValueError(f'Array differs from original policy request: {actual} != {expected}')
    return actual


def exact_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def reconstruct(stream, case, step):
    events_path = stream / 'OS_COST_EVENTS.jsonl'
    events = exact_rows(events_path)
    inputs = [e for e in events if e.get('event') == 'actual_policy_input']
    choices = [e for e in events if e.get('event') == 'feedback_window_choice']
    if len(inputs) != len(choices):
        raise ValueError('Input/choice count mismatch')
    # Check the event ordering, not only separate-list lengths.
    relevant = [e for e in events if e.get('event') in ('actual_policy_input', 'feedback_window_choice')]
    if len(relevant) != 2 * len(inputs) or any(e['event'] != ('actual_policy_input' if i % 2 == 0 else 'feedback_window_choice') for i, e in enumerate(relevant)):
        raise ValueError('Input/choice order is not strictly paired')
    indices = [i for i, e in enumerate(choices) if e.get('step') == step]
    if len(indices) != 1:
        raise ValueError(f'Expected one choice at step {step}; got {indices}')
    index = indices[0]
    inp, choice = inputs[index], choices[index]
    if inp.get('case') != case or choice.get('source_observation_id') != f'libero_obs_{step}':
        raise ValueError('Case/observation identity mismatch')
    expected_keys = sorted([*IMAGE_KEYS, 'observation/state', 'prompt'])
    if inp.get('keys') != expected_keys or sorted(inp['array_bindings']) != sorted(expected_keys[:-1]):
        raise ValueError('Unexpected request keys')
    folder = stream / 'stage_trace' / case
    trace_path = folder / 'TRACE.jsonl'
    traces = [r for r in exact_rows(trace_path) if r.get('step') == step and r.get('observation_id') == choice['source_observation_id'] and r.get('session_id') == case]
    if len(traces) != 1:
        raise ValueError('Missing or ambiguous diagnostic observation')
    sources = {str(events_path): sha(events_path), str(trace_path): sha(trace_path)}
    arrays = {}
    for key, image_name in IMAGE_KEYS.items():
        name = f'{step:04d}_{image_name}.png'
        path = folder / name
        digest = sha(path)
        if traces[0]['images'].get(name) != digest:
            raise ValueError('Diagnostic PNG source SHA mismatch')
        sources[str(path)] = digest
        with Image.open(path) as image:
            if image.mode != 'RGB':
                raise ValueError('Unexpected PNG mode; conversion forbidden')
            arrays[key] = np.array(image)
        check_array(arrays[key], inp['array_bindings'][key])
    state_binding = inp['array_bindings']['observation/state']
    arrays['observation/state'] = np.asarray(inp['robot_state'], dtype=np.dtype(state_binding['dtype']))
    check_array(arrays['observation/state'], state_binding)
    if not isinstance(inp.get('prompt'), str) or not inp['prompt']:
        raise ValueError('Missing prompt')
    arrays['prompt'] = np.asarray(inp['prompt'])
    return arrays, {'case': case, 'step': step, 'historical_choice_index': index,
                    'source_observation_id': choice['source_observation_id'],
                    'prompt': inp['prompt'], 'array_bindings': inp['array_bindings'],
                    'source_sha256': sources, 'status': 'EXACT_INPUT_RECONSTRUCTION'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.root / 'planning' / NAME / 'inputs'
    output.mkdir(parents=True, exist_ok=False)
    run = args.root / 'runs/short_exit_v1_20261003'
    report = {'version': 'exact_policy_input_reconstruction_v1', 'status': 'PREPARING',
              'scope': 'Service lifecycle replay inputs only; not a contiguous episode or a robot trial.',
              'source_run': str(run), 'nodes': [], 'inputs': []}
    try:
        for label, pair, case, step in SPECS:
            arrays, receipt = reconstruct(run / pair / 'R/stream', case, step)
            path = output / (label + '.npz')
            with path.open('xb') as f:
                np.savez(f, **arrays)
            with np.load(path, allow_pickle=False) as saved:
                for key in receipt['array_bindings']:
                    check_array(saved[key], receipt['array_bindings'][key])
                if str(saved['prompt']) != receipt['prompt']:
                    raise ValueError('Saved prompt changed')
            receipt.update(id=label, npz_file=path.name, npz_sha256=sha(path))
            report['nodes'].append(receipt)
            report['inputs'].append({'name': label, 'path': '/project/runs/' + NAME + '/inputs/' + path.name,
                                     'sha256': receipt['npz_sha256'], 'source_array_bindings': receipt['array_bindings']})
        for r in report['nodes']:
            for path, digest in r['source_sha256'].items():
                if sha(Path(path)) != digest:
                    raise ValueError('Source changed during preparation')
        report['status'] = 'PASSED'
    except Exception as exc:
        report.update(status='FAILED', error=repr(exc))
        raise
    finally:
        (output / 'MANIFEST.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'nodes': len(report['nodes']), 'manifest': str(output / 'MANIFEST.json')}))


if __name__ == '__main__':
    main()
