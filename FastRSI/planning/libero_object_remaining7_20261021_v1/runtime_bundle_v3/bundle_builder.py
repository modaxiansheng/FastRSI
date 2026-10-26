"""Byte-copy the proven Object runtime plus narrowly adapted metadata/controller.

Default is a local metadata/source-byte audit. --assemble copies a NEW host bundle
only; it never starts containers, models, generation, physical tasks or training.
Missing local binary snapshots stay missing, rather than becoming fake inputs.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def audit(manifest, local=True):
    good, missing, changed = [], [], []
    for item in manifest['files']:
        path = Path(item['local_path'] if local else item['host_source_path'])
        if not path.is_file():
            missing.append(dict(path=str(path), destination=item['destination'],
                                required_sha256=item['sha256']))
        elif sha(path) != item['sha256']:
            changed.append(dict(path=str(path), destination=item['destination'],
                                expected=item['sha256'], actual=sha(path)))
        else:
            good.append(item['destination'])
    for name, digest in manifest['package_files'].items():
        path = ROOT / name
        if not path.is_file():
            missing.append(dict(path=str(path), destination=name, required_sha256=digest))
        elif sha(path) != digest:
            changed.append(dict(path=str(path), expected=digest, actual=sha(path)))
    return dict(status=('SOURCE_BYTES_VERIFIED_EXTERNAL_BINDINGS_STILL_REQUIRED'
                        if not missing and not changed else 'SOURCE_SNAPSHOT_INCOMPLETE_OR_CHANGED'),
                checked_at_unix=time.time(), local_metadata_only=local,
                verified_files=len(good), missing_files=missing, changed_files=changed,
                launch_ready=False, external_host_inputs_not_verified=True,
                models_started=0, environments_started=0, physical_attempts=0,
                full_evaluation='PAUSED')


def assemble(destination, manifest, report, local=False):
    assert not report['missing_files'] and not report['changed_files']
    target = Path(destination).resolve()
    assert target.is_absolute() and not target.exists() and target.parent.is_dir()
    for prefix in manifest['forbidden_existing_project_roots']:
        p = Path(prefix)
        assert target != p and p not in target.parents
    sources = []
    for item in manifest['files']:
        rel = Path(item['destination'])
        assert not rel.is_absolute() and '..' not in rel.parts
        source = Path(item['local_path'] if local else item['host_source_path'])
        sources.append((rel, source, item['sha256']))
    for name, digest in manifest['package_files'].items():
        sources.append((Path(name), ROOT / name, digest))
    assert len(sources) == len({str(row[0]) for row in sources})
    assert all(sha(source) == digest for _, source, digest in sources)
    target.mkdir()
    written = {}
    for rel, source, digest in sources:
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        with source.open('rb') as inp, dest.open('xb') as out:
            shutil.copyfileobj(inp, out)
        assert sha(dest) == digest
        written[str(rel)] = digest
    with (ROOT / 'BUILD_INPUT_MANIFEST.json').open('rb') as inp, \
            (target / 'BUILD_INPUT_MANIFEST.json').open('xb') as out:
        shutil.copyfileobj(inp, out)
    save(target / 'BUNDLE_BYTE_COPY_RECEIPT.json', dict(
        time=time.time(), source_manifest_sha256=sha(ROOT / 'BUILD_INPUT_MANIFEST.json'),
        files=written, local_metadata_only=local, historical_sources_modified=False,
        launch_ready=False, external_runtime_admission_pending=True,
        normal_physical_attempts=0, models_started=0, environments_started=0,
        full_evaluation='PAUSED'))
    save(target / 'ASSEMBLY_READINESS.json', report)
    return dict(status='NEW_BYTE_COPY_BUNDLE_ASSEMBLED_NOT_LAUNCHED', path=str(target),
                source_files=len(written), launch_ready=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--host-sources', action='store_true')
    parser.add_argument('--assemble', type=Path)
    args = parser.parse_args()
    assert not (args.audit_only and args.assemble)
    manifest = read(ROOT / 'BUILD_INPUT_MANIFEST.json')
    report = audit(manifest, local=not args.host_sources)
    result = (assemble(args.assemble, manifest, report, local=not args.host_sources)
              if args.assemble else report)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
