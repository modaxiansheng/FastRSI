"""Validate all original banks in the actual simulation image, without physics."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path('/eval')
rows = json.loads((ROOT / 'ASSET_INDEX_FINAL.json').read_text())
assert len(rows) == 30
for p in (ROOT / 'runtime').glob('*.py'):
    ast.parse(p.read_text())
ast.parse((ROOT / 'run_lane.py').read_text())
from libero.libero import benchmark
import torch

out = []
for suite in ['libero_spatial', 'libero_object', 'libero_goal']:
    bank = benchmark.get_benchmark_dict()[suite](task_order_index=0)
    selected = sorted((r for r in rows if r['suite'] == suite), key=lambda x: x['task_id'])
    assert bank.get_num_tasks() == len(selected) == 10
    for r in selected:
        task = bank.get_task(r['task_id'])
        assert task.name == r['task_name'] and task.language == r['language']
        assert task.problem_folder == suite
        for field, digest in [('bddl_rel', 'bddl_sha256'), ('init_rel', 'init_sha256')]:
            assert hashlib.sha256((ROOT / 'assets' / r[field]).read_bytes()).hexdigest() == r[digest]
        states = bank.get_task_init_states(r['task_id'])
        assert len(states) >= 50
        assert str(bank.get_task_bddl_file_path(r['task_id'])).startswith('/eval/assets/')
        out.append(dict(suite=suite, task_id=r['task_id'], initial_states=len(states),
                        official_suite=task.problem_folder, name=task.name))
print(json.dumps(dict(status='PASS_ORIGINAL_BANKS_NO_ENV_NO_MODEL', rows=out,
                      model_loads=0, environments=0, actions=0)))
