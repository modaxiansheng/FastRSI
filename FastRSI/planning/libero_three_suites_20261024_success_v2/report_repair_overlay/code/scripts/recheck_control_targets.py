"""Recheck saved controller targets without another simulator rollout."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastrsi.retarget_geometry import rotation_vector


def recheck(path):
    raw = path.read_bytes()
    value = json.loads(raw)
    if value.get('status') != 'EXACT_REPLAY' and not (
            value.get('status') == 'FAILED' and value.get('inversion_check') is False and
            value.get('error', '').startswith('AssertionError')):
        raise ValueError('Capture did not complete its simulator/image checks')
    if value['controls_count'] != len(value['controls']) or len(value['controls']) != value['boundaries'][-1]['step']:
        raise ValueError('Incomplete capture')
    c = value['controller']
    scale = (np.array(c['output_max'])-c['output_min'])/(np.array(c['input_max'])-c['input_min'])
    errors = []
    for index, row in enumerate(value['controls']):
        if row['step'] != index:
            raise ValueError('Missing control step')
        delta = np.r_[np.array(row['target_pos'])-row['eef_pos'],
                      rotation_vector(np.array(row['target_rot']) @ np.array(row['eef_rot']).T)]
        recovered = (delta-(np.array(c['output_max'])+c['output_min'])/2)/scale+(np.array(c['input_max'])+c['input_min'])/2
        n = 3 if row['zero_ori_preserves_goal'] else 6
        errors.append(float(np.max(np.abs(recovered[:n]-row['clipped_action'][:n]))))
    if not errors or max(errors) >= 1e-4:
        raise ValueError('Controller inversion still differs')
    value.update(status='EXACT_REPLAY_RECHECKED', parent_capture_sha256=hashlib.sha256(raw).hexdigest(),
        recheck_max_error=max(errors), recheck_scope='Numerical inversion only; no new simulation. Original capture and any failure remain immutable.')
    return value


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('output', type=Path)
    a = p.parse_args()
    value = recheck(a.capture)
    with a.output.open('x') as f:
        json.dump(value, f, indent=2)
    print(value['recheck_max_error'])
