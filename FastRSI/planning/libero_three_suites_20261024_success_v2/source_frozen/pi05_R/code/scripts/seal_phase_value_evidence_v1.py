"""Copy this completed phase's run evidence to shared planning and hash it.

No launches, model/default changes, or deletion. Tokenizer caches and bytecode
are shared dependencies rather than experimental evidence and are excluded.
"""
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

P = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
O = P / 'planning/phase_goal_v1_20261010'
EXCLUDED = {'tokenizer_cache', '__pycache__'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    O.mkdir(exist_ok=True)
    names = ['phase_iql_training_v1_20261010', 'phase_iql_feedback_v1_20261010']
    names += [f'phase_iql_rollout_gpu{i}_v1_20261010' for i in range(4)]
    names += [f'phase_iql_{batch}_gpu{i}_v1_20261010'
              for batch in ('adaptive', 'repeat') for i in range(8)]
    copied = []
    for name in names:
        source, target = P / 'runs' / name, P / 'planning' / name
        # An interrupted copy may be resumed, but any existing file must agree.
        for f in sorted(source.rglob('*')):
            relative = f.relative_to(source)
            if EXCLUDED.intersection(relative.parts) or not f.is_file():
                continue
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                assert sha(f) == sha(destination), str(destination)
            else:
                shutil.copy2(f, destination)
            copied.append({'source': str(f.relative_to(P)),
                           'snapshot': str(destination.relative_to(P))})
    roots = names + ['phase_value_dataset_v1_20261010',
                     'phase_value_features_v1_20261010']
    roots += [f'phase_iql_{batch}{suffix}_v1_20261010'
              for batch in ('rollout', 'adaptive', 'repeat')
              for suffix in ('', '_report')]
    files = []
    for name in roots:
        root = P / 'planning' / name
        for f in sorted(root.rglob('*')):
            if f.is_file() and not EXCLUDED.intersection(f.relative_to(root).parts):
                files.append({'path': str(f.relative_to(P)), 'bytes': f.stat().st_size,
                              'sha256': sha(f)})
    default = P / 'CURRENT_NATIVE_ENTRY.json'
    files.append({'path': str(default.relative_to(P)), 'bytes': default.stat().st_size,
                  'sha256': sha(default)})
    names_in_docker = subprocess.check_output(
        ['docker', 'ps', '-a', '--format', '{{.Names}}'], text=True).splitlines()
    owned = [n for n in names_in_docker if n.startswith(
        ('yuchen-fastrsi-phase-value-', 'yuchen-fastrsi-phase-iql-'))]
    containers = []
    for name in sorted(owned):
        detail = json.loads(subprocess.check_output(['docker', 'inspect', name]))[0]
        state = detail['State']
        assert not state['Running'] and state['Pid'] == 0, name
        containers.append({'name': name, 'image': detail['Image'], 'state': state})
    receipt = dict(status='SEALED_COMPLETED_PHASE_EVIDENCE',
                   files=len(files), bytes=sum(f['bytes'] for f in files),
                   excluded=sorted(EXCLUDED), entries=files,
                   run_snapshot_files=copied, containers=containers,
                   default_sha256=sha(default),
                   complete_agent_comparison=False, goal_complete=False)
    target = O / 'EVIDENCE_MANIFEST.json'
    assert not target.exists(), 'Do not overwrite a sealed receipt'
    target.write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ('entries', 'run_snapshot_files', 'containers')}, indent=2))
    print('exited_owned_containers', len(containers), flush=True)


if __name__ == '__main__':
    main()
