"""Meaningful metadata/entrypoint checks; no model, environment or subprocess."""
from pathlib import Path
import argparse
import ast
import contextlib
import importlib
import io
import json
import sys
import time
from bundle_builder import check
from coordinator_contract import groups
from row_factory import proposal, MODELS, BRANCHES, condition_row, branch_row, capability

ROOT = Path(__file__).resolve().parent


def main():
    started = time.time()
    for path in ROOT.glob('*.py'):
        ast.parse(path.read_text(), feature_version=(3, 8))
    manifest = json.loads((ROOT / 'BUILD_INPUT_MANIFEST.json').read_text())
    checked = check(manifest, local=True)
    assert not checked['missing_files'] and not checked['mismatched_files']
    _, design = proposal()
    selected = groups()
    assert len(selected) == 16 and sum(g['conditions'] for g in selected) == 48
    assert sum(g['normal_attempts_max'] for g in selected) == 288
    counts = {branch: 0 for branch in BRANCHES}
    for item in design:
        state = {k: item[k] for k in ['identity_id', 'suite', 'task_id', 'variant',
                                      'max_steps', 'language']}
        state.update(state_path='states/' + item['identity_id'] + '.npy')
        for key in ['initial_state_sha256', 'physical_state_sha256', 'asset_sha256',
                    'source_receipt_sha256']:
            state[key] = '1' * 64
        for model in MODELS:
            condition = condition_row(state, model, '187', 0)
            for branch in BRANCHES:
                row = branch_row(condition, branch, {'python': 'metadata-check'},
                                 '2' * 64, '3' * 64)
                assert row['skill'] == item['skill_by_branch'][branch]
                assert row['after_baseline_rng_hashes'] == {'python': 'metadata-check'}
                assert row['baseline_reset_state_sha256'] == '2' * 64
                assert row['baseline_first_input_sha256'] == '3' * 64
                counts[branch] += row['skill'] is not None
    assert counts == {'ordinary': 0, 'frozen': 0, 'candidate': 20}
    assert all(capability('libero_object', task, 'frozen') is None
               for task in [1, 2, 4, 6, 7, 9])
    assert capability('libero_object', 4, 'candidate') == 'newtask_07_prefix_v1'
    assert capability('libero_object', 2, 'candidate') is None
    # Exercise the actual coordinator CLI with all subprocess/network entrypoints
    # disabled. Its default path must remain a genuinely metadata-only plan.
    entry = importlib.import_module('isolated_coordinator')
    import subprocess
    import urllib.request
    def forbidden(*args, **kwargs):
        raise AssertionError('Plan attempted a subprocess or network operation')
    old_run, old_popen, old_url = subprocess.run, subprocess.Popen, urllib.request.urlopen
    old_argv = sys.argv
    stream = io.StringIO()
    try:
        subprocess.run = subprocess.Popen = urllib.request.urlopen = forbidden
        sys.argv = ['isolated_coordinator.py', '--plan']
        with contextlib.redirect_stdout(stream):
            entry.main()
    finally:
        subprocess.run, subprocess.Popen, urllib.request.urlopen = old_run, old_popen, old_url
        sys.argv = old_argv
    plan = json.loads(stream.getvalue())
    assert plan['conditions'] == 48 and plan['normal_physical_attempts_max'] == 288
    print(json.dumps(dict(status='PYTHON38_METADATA_AND_REAL_PLAN_ENTRY_PASSED',
         time=time.time(), seconds=time.time() - started,
         source_files_verified=checked['verified_file_count'],
         package_files_verified=len(manifest['package_files']),
         groups=16, conditions=48, maximum_normal_attempts=288,
         supported_branch_conditions=counts, no_subprocess_or_network=True,
         environment_or_model_calls=0, gpu_operations=0), indent=2))


if __name__ == '__main__':
    main()
