"""Verify explicit immutable sources and byte-copy a NEW bundle; never dispatch.

Default and --audit-only are read-only. --assemble requires a new absolute
destination. No subprocess, network, Docker, environment or model imports.
External weights/dependency directories remain explicit prerequisites; assembly
does not turn missing external identities into a launch-ready assertion.
"""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    with Path(path).open('x') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write('\n')


def check(manifest, local=False):
    verified, missing, mismatched = [], [], []
    for entry in manifest['files']:
        source = Path(entry['local_path'] if local else entry['host_source_path'])
        if not source.is_file():
            missing.append(dict(path=str(source), destination=entry['destination']))
        elif sha(source) != entry['sha256']:
            mismatched.append(dict(path=str(source), expected=entry['sha256'], actual=sha(source)))
        else:
            verified.append(entry['destination'])
    for name, expected in manifest['package_files'].items():
        path = ROOT / name
        if not path.is_file():
            missing.append(dict(path=str(path), destination=name))
        elif sha(path) != expected:
            mismatched.append(dict(path=str(path), expected=expected, actual=sha(path)))
    external = []
    for item in manifest['external_prerequisites']:
        p = Path(item['host_source_path'])
        external.append(dict(item, exists=p.exists(),
                             current_content_identity_verified=False))
    return dict(status='BYTE_SOURCES_VERIFIED_EXTERNAL_RUNTIME_ADMISSION_PENDING'
                if not missing and not mismatched else 'SOURCE_BYTES_MISSING_OR_MISMATCHED',
                checked_at_unix=time.time(), mode='LOCAL_METADATA_COPY' if local else 'EXACT_HOST_PATHS',
                verified_file_count=len(verified), missing_files=missing,
                mismatched_files=mismatched, external_prerequisites=external,
                launch_ready=False, models_started=0, environments_started=0,
                full_evaluation='PAUSED')


def assemble(destination, manifest, report, local=False):
    assert not report['missing_files'] and not report['mismatched_files'], report['status']
    target = Path(destination).resolve()
    assert target.is_absolute() and not target.exists(), 'Use a new absolute bundle directory'
    assert target.parent.is_dir(), 'Parent must already exist'
    forbidden = [Path(p) for p in manifest['forbidden_existing_project_roots']]
    assert not any(target == p or p in target.parents for p in forbidden)
    assert not any((ROOT / name).exists() for name in ['confirmation_generated', 'campaigns'])
    target.mkdir()
    written = {}
    # Validate every source before the first mutation; each destination is unique.
    for entry in manifest['files']:
        rel = Path(entry['destination'])
        assert not rel.is_absolute() and '..' not in rel.parts
        source = Path(entry['local_path'] if local else entry['host_source_path'])
        dest = target / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        assert not dest.exists()
        with source.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        assert sha(dest) == entry['sha256']
        written[str(rel)] = entry['sha256']
    for name in manifest['package_files']:
        source, dest = ROOT / name, target / name
        assert sha(source) == manifest['package_files'][name]
        assert not dest.exists()
        with source.open('rb') as src, dest.open('xb') as dst:
            shutil.copyfileobj(src, dst)
        written[name] = sha(dest)
    shutil.copyfile(ROOT / 'BUILD_INPUT_MANIFEST.json', target / 'BUILD_INPUT_MANIFEST.json')
    written['BUILD_INPUT_MANIFEST.json'] = sha(target / 'BUILD_INPUT_MANIFEST.json')
    for name in ('assets', 'upstream', 'runtime_vendor', 'libero-config', 'generated'):
        (target / name).mkdir(exist_ok=False)
    save(target / 'BUNDLE_BYTE_COPY_RECEIPT.json', dict(
        time=time.time(), source_manifest_sha256=sha(ROOT / 'BUILD_INPUT_MANIFEST.json'),
        files=written, no_symlinks=True, local_metadata_only=local,
        historical_sources_modified=False, source_tree_roots_are_not_mutable_pointers=True,
        launch_ready=False, external_runtime_admission_pending=True,
        state_generation_started=False, models_started=0, environments_started=0,
        full_evaluation='PAUSED'))
    save(target / 'ASSEMBLY_READINESS.json', report)
    return dict(status='NEW_BYTE_COPY_BUNDLE_ASSEMBLED_NOT_LAUNCHED', path=str(target),
                files=len(written), launch_ready=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--local-metadata', action='store_true')
    parser.add_argument('--assemble', type=Path)
    args = parser.parse_args()
    assert not (args.audit_only and args.assemble)
    manifest = read(ROOT / 'BUILD_INPUT_MANIFEST.json')
    assert manifest['states_generated'] == 0 and manifest['execution_allowed'] is False
    destinations = [entry['destination'] for entry in manifest['files']]
    assert len(destinations) == len(set(destinations))
    report = check(manifest, args.local_metadata)
    if args.assemble:
        result = assemble(args.assemble, manifest, report, args.local_metadata)
        print(json.dumps(result, indent=2))
    else:
        print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
