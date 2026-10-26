"""Four CPU workers collect pre-action samples; no model calls or training."""
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def main():
    project = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
    assets = project.parent / 'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
    run = project / 'runs/bowl_demo_collection_v1_20261009'
    lock = (project / 'runs/full_os_v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not subprocess.check_output(['docker', 'ps', '--format', '{{.Names}}'], text=True).strip()
    run.mkdir(exist_ok=False)
    (run / 'code').mkdir()
    for name in ('replay_public_bowl_demonstration_v1.py', 'replay_stack_progress.py', 'stage_trace.py'):
        shutil.copy2(project / 'scripts' / name, run / 'code' / name)
    demos = [0, 2, 7, 12, 17, 22, 25, 32, 37, 42]
    validation = [25, 42]
    protocol = dict(demos=demos, train_demos=[n for n in demos if n not in validation],
        validation_demos=validation, stride=10, action_horizon=10, collection_workers=4,
        scope='Development data acquisition; 0 and 25 repeated to save aligned data, not new independent evidence',
        code_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (run/'code').iterdir()})
    (run / 'PROTOCOL.json').write_text(json.dumps(protocol, indent=2))
    started = time.perf_counter()
    def worker(index):
        demo = f'demo_{index}'
        receipt = run / (demo + '_host')
        receipt.mkdir()
        name = 'yuchen-fastrsi-bowl-collect-v1-' + demo + '-20261009'
        argv = ['docker', 'run', '--name', name, '--network', 'none', '--cpus', '4', '--memory', '12g',
            '--pids-limit', '1024', '--shm-size', '2g', '--entrypoint', 'python',
            '-e', 'PYTHONPATH=/workspace/openpi/third_party/libero:/collection/code',
            '-e', 'LIBERO_CONFIG_PATH=/libero-config', '-e', 'MUJOCO_GL=egl', '-e', 'LIBGL_ALWAYS_SOFTWARE=1',
            '-v', str(project) + ':/project', '-v', str(run) + ':/collection',
            '-v', str(assets/'dependencies/openpi') + ':/workspace/openpi:ro',
            '-v', str(assets/'dependencies/PhyAgentOS-core') + ':/core:ro',
            '-v', str(project/'configs/libero') + ':/libero-config:ro',
            'sha256:3171fda2362d773433482f79c4fd53f6f428413efbf41c3fdfb43d8c799d6c95',
            '/collection/code/replay_public_bowl_demonstration_v1.py',
            '--demo', '/project/planning/bowl_behavior_adaptation_v1_20261009/exports/' + demo,
            '--output', '/collection/' + demo, '--collect-stride', '10']
        (receipt / 'LAUNCH.json').write_text(json.dumps(argv, indent=2))
        start = time.perf_counter()
        with (receipt / 'LOG.txt').open('w') as log:
            try:
                p = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=420)
                code = p.returncode
            except subprocess.TimeoutExpired:
                code = 124
                subprocess.run(['docker', 'stop', '-t', '10', name], capture_output=True, timeout=40)
        info = json.loads(subprocess.check_output(['docker', 'inspect', name]))
        (receipt / 'CONTAINER.json').write_text(json.dumps(info, indent=2))
        row = dict(demo=demo, host_exit=code, host_seconds=time.perf_counter()-start,
                   running=info[0]['State']['Running'], role='validation' if index in validation else 'train')
        result_path = run / demo / 'RESULT.json'
        if result_path.exists():
            r = json.loads(result_path.read_text())
            row.update(native_success=r.get('native_success'), executed_actions=r.get('executed_actions'),
                result_status=r['status'], samples=r.get('samples', []),
                result_sha256=hashlib.sha256(result_path.read_bytes()).hexdigest())
        (receipt / 'HOST.json').write_text(json.dumps(row, indent=2))
        print(json.dumps({k:v for k,v in row.items() if k!='samples'}), flush=True)
        return row
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, demos))
    samples = []
    for row in results:
        if row['host_exit'] == 0 and row.get('native_success') and not row['running']:
            for sample in row['samples']:
                if sample['eligible']:
                    samples.append(dict(sample, demo=row['demo'], role=row['role'],
                        path=str(run / row['demo'] / sample['path'])))
    dataset = dict(status='COLLECTED', protocol=protocol, episodes=results, samples=samples,
                   acquisition_wall_seconds=time.perf_counter()-started,
                   worker_wall_seconds_sum=sum(r['host_seconds'] for r in results),
                   costs_scope='Parallel acquisition wall and worker sum kept separate; all failures retained')
    (run / 'DATASET.json').write_text(json.dumps(dataset, indent=2))
    assert not any(r['running'] for r in results)
    assert any(s['role']=='train' for s in samples) and any(s['role']=='validation' for s in samples)
    print(json.dumps(dict(status=dataset['status'], samples=len(samples),
                         successful_demos=sum(bool(r.get('native_success')) for r in results),
                         seconds=dataset['acquisition_wall_seconds'])), flush=True)


if __name__ == '__main__':
    main()
