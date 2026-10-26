"""One root-owned targeted hash pass; no inference/environment/device calls.

Reuse the original sealed archive manifest. Fresh-hash only named checkpoints
that will actually run, plus explicit source/config/identity files in the plan.
Never traverse a whole venv, evaluation tree, or the112GB archive.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

ARCHIVE = Path('/mnt/data/task/yuchen/FastPhyagentos/frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1')
SUMS_SHA = '7039833004926d4d24c8cd4b68bb33c71c82acaee78ac20fa1b555f1ae456df3'
CHECKPOINTS = {'pi05_libero_pytorch_fp32', 'pi0_libero_pytorch_fp32', 'xvla_libero',
               'openvla_object', 'openvla_spatial', 'openvla_goal', 'openvla_libero10'}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while True:
            chunk = stream.read(8 * 1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def pin(item):
    path = Path(item['path'])
    assert path.is_file() and sha(path) == item['sha256'], str(path)
    return dict(item, bytes=path.stat().st_size)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    started = time.time()
    plan = json.loads(args.plan.read_text())
    sums = ARCHIVE / 'SHA256SUMS'
    assert sha(sums) == SUMS_SHA
    names = plan['checkpoints']
    assert names and len(names) == len(set(names)) and set(names) <= CHECKPOINTS
    expected = {}
    for line in sums.read_text().splitlines():
        digest, relative = line.split(None, 1)
        relative = relative.lstrip('*')
        if relative.startswith('./'):
            relative = relative[2:]
        if any(relative.startswith('weights/' + name + '/') for name in names):
            path = Path(relative)
            assert not path.is_absolute() and '..' not in path.parts
            expected[relative] = digest
    identities = {}
    for name in names:
        selected = {p: d for p, d in expected.items() if p.startswith('weights/' + name + '/')}
        assert selected, ('Checkpoint absent from sealed archive manifest', name)
        # Predeclare all targets before reading the first weight shard.
        hashes, byte_count = {}, 0
        for relative, digest in sorted(selected.items()):
            path = ARCHIVE / relative
            assert path.is_file() and sha(path) == digest, relative
            hashes[relative] = digest
            byte_count += path.stat().st_size
        identities[name] = dict(checkpoint_path=str(ARCHIVE / 'weights' / name),
                                original_sums_sha256=SUMS_SHA,
                                actual_file_sha256=hashes, fresh_sha_verified=True,
                                files=len(hashes), bytes=byte_count)
    manifests = [pin(item) for item in plan['pinned_manifests']]
    used = [pin(item) for item in plan['used_source_and_config_files']]
    dependencies = []
    for item in plan['dependency_identities']:
        assert item['name'] and item['scope'] and item['binding_files']
        dependencies.append(dict(item, binding_files=[pin(p) for p in item['binding_files']]))
    assert used and dependencies and plan['applicable_groups']
    receipt = dict(status='TARGETED_ACTUAL_RUNTIME_INPUT_IDENTITIES_BOUND', accepted=True,
                   host_start_unix=started, host_end_unix=time.time(),
                   root_plan_path=str(args.plan.resolve()), root_plan_sha256=sha(args.plan),
                   actual_used_inputs_only=True, frozen_weight_identity_proved=True,
                   applicable_groups=plan['applicable_groups'],
                   checkpoint_identities=identities,
                   pinned_manifests=[dict(path=str(sums), sha256=SUMS_SHA)] + manifests,
                   used_source_and_config_files=used, dependency_identities=dependencies,
                   dependency_identity_boundary=plan['dependency_identity_boundary'],
                   unmeasured_scope=plan.get('unmeasured_scope', []),
                   full_venv_or_archive_rehash=False, model_or_environment_calls=0,
                   policy_or_vision_calls=0, gpu_operations=0, full_evaluation='PAUSED')
    with args.output.open('x') as stream:
        json.dump(receipt, stream, indent=2)
        stream.write('\n')
    print(json.dumps(dict(status=receipt['status'], receipt_sha256=sha(args.output),
                         checkpoint_names=names, checkpoint_bytes=sum(v['bytes'] for v in identities.values()),
                         seconds=time.time() - started)))


if __name__ == '__main__':
    main()
