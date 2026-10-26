"""Export three predetermined demonstrations from a SHA-verified raw HDF5."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def plain(value):
    if isinstance(value, bytes):
        return value.decode('utf8')
    return value.tolist() if hasattr(value, 'tolist') else value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--candidate', type=Path, required=True)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--deps', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--demos', type=int, nargs='+', default=[0, 25, 49])
    args = ap.parse_args()
    sys.path.insert(0, str(args.deps.resolve()))
    import h5py
    import numpy as np
    start = time.perf_counter()
    candidate = json.loads(args.candidate.read_text())
    item = candidate['matches'][0]
    assert args.data.stat().st_size == item['size']
    assert sha(args.data) == item['lfs']['oid'], 'Raw source SHA mismatch'
    args.output.mkdir(parents=True, exist_ok=False)
    result = dict(status='EXPORTED_NOT_YET_REPLAYED', source_sha256=item['lfs']['oid'],
                  source_revision=candidate['revision'], source_path=item['path'],
                  scope='Offline development qualification; not held-out evaluation or admitted learning',
                  sampling_rule='Explicit IDs fixed before replay: ' + ','.join(map(str, args.demos)),
                  model_calls=0, training_steps=0, robot_actions=0, episodes=[])
    with h5py.File(args.data, 'r') as file:
        data = file['data']
        result['data_attrs'] = {k: plain(v) for k, v in data.attrs.items()}
        metadata = []
        for name in sorted(data, key=lambda n: int(n.split('_')[-1])):
            demo = data[name]
            actions = demo['actions'][:]
            metadata.append(dict(name=name, actions_shape=list(actions.shape),
                actions_dtype=str(actions.dtype), finite=bool(np.isfinite(actions).all()),
                action_abs_max=float(np.max(np.abs(actions))),
                states_shape=list(demo['states'].shape),
                observation_shapes={k: list(v.shape) for k, v in demo['obs'].items()},
                declared_samples=int(demo.attrs['num_samples'])))
        result['all_episode_metadata'] = metadata
        assert len(metadata) == 50 and all(m['finite'] and m['actions_shape'][1] == 7 for m in metadata)
        assert len(set(args.demos)) == len(args.demos) and all(0 <= n < 50 for n in args.demos)
        for name in (f'demo_{n}' for n in args.demos):
            demo = data[name]
            folder = args.output / name
            folder.mkdir()
            arrays = dict(actions=demo['actions'][:], states=demo['states'][:],
                          declared_init_state=np.asarray(demo.attrs['init_state']))
            for key in ('ee_pos', 'ee_ori', 'ee_states', 'gripper_states', 'joint_states'):
                arrays[key] = demo['obs'][key][:]
            # Samples are exact source pixels. No rotation or image resizing here.
            indices = np.array([0, 1, len(arrays['actions']) - 1])
            arrays['image_indices'] = indices
            for key in ('agentview_rgb', 'eye_in_hand_rgb'):
                arrays[key] = demo['obs'][key][indices]
            np.savez_compressed(folder / 'replay.npz', **arrays)
            (folder / 'model.xml').write_text(plain(demo.attrs['model_file']))
            entry = dict(name=name, action_count=len(arrays['actions']),
                initial_attr_vs_first_saved_state_max_abs=float(np.max(np.abs(
                    arrays['declared_init_state'] - arrays['states'][0]))),
                files={p.name: dict(bytes=p.stat().st_size, sha256=sha(p)) for p in folder.iterdir()})
            (folder / 'MANIFEST.json').write_text(json.dumps(entry, indent=2) + '\n')
            result['episodes'].append(entry)
    result.update(elapsed_seconds=time.perf_counter() - start,
                  total_actions=sum(m['actions_shape'][0] for m in metadata))
    (args.output / 'EXPORT.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('data_attrs', 'all_episode_metadata')}))


if __name__ == '__main__':
    main()
