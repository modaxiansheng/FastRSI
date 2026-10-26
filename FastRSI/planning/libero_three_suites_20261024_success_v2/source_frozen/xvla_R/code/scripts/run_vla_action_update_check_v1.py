"""Launch one bounded, isolated HCU update check after a fresh resource check."""
import fcntl
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time


def main():
    project = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
    assets = project.parent / 'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
    run = project / 'runs/vla_action_update_check_v1_20261009'
    lock = (project / 'runs/full_os_v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.mkdir(exist_ok=False)
    def save(name, obj):
        (run / name).write_text(json.dumps(obj, indent=2) + '\n')
    def capture(argv):
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    preflight = dict(time=time.time(),
        hcu=capture(['bash', '-lc', 'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse']),
        containers=capture(['docker', 'ps', '--format', '{{.ID}} {{.Names}}']),
        holders=capture(['bash', '-lc', 'fuser -v /dev/kfd /dev/mkfd /dev/dri/render* 2>&1 || true']))
    save('PREFLIGHT.json', preflight)
    memory = re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)', preflight['hcu']['stdout'])
    assert len(memory) == 8 and all(float(v) == 0 for _, v in memory)
    assert not preflight['containers']['stdout'].strip()
    before = hashlib.sha256((project / 'CURRENT_NATIVE_ENTRY.json').read_bytes()).hexdigest()
    shutil.copytree(project / 'models/openpi_data', run / 'tokenizer_cache')
    script = run / 'check_vla_action_update_v1.py'
    shutil.copy2(project / 'scripts/check_vla_action_update_v1.py', script)
    name = 'yuchen-fastrsi-vla-action-update-check-v1-20261009'
    image = 'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'
    argv = ['docker', 'run', '--name', name, '--network', 'none', '--cpus', '8', '--memory', '96g',
        '--pids-limit', '1024', '--shm-size', '2g', '--device', '/dev/kfd', '--device', '/dev/mkfd',
        '--device', '/dev/dri', '--group-add', 'video', '--group-add', 'render',
        '-e', 'HIP_VISIBLE_DEVICES=0', '-e', 'CUDA_VISIBLE_DEVICES=0',
        '-e', 'PYTHONDONTWRITEBYTECODE=1', '-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4',
        '-e', 'OPENPI_DATA_HOME=/workspace/openpi_data',
        '-e', 'PYTHONPATH=/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/workspace/PhyAgentOS-core',
        '-v', str(project) + ':/project:rw',
        '-v', str(assets / 'dependencies/openpi') + ':/workspace/openpi:ro',
        '-v', str(assets / 'dependencies/PhyAgentOS-core') + ':/workspace/PhyAgentOS-core:ro',
        '-v', str(assets / 'weights/pi05_libero_pytorch_fp32') + ':/model:ro',
        '-v', str(run / 'tokenizer_cache') + ':/workspace/openpi_data:rw',
        '-v', '/opt/hyhal:/opt/hyhal:ro', '-v', '/etc/hfm:/etc/hfm:ro',
        image, 'timeout', '900', 'bash', '-lc',
        'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec python '
        '/project/runs/vla_action_update_check_v1_20261009/check_vla_action_update_v1.py '
        '--replay /project/runs/public_bowl_replay_v3_20261009/demo_0 '
        '--host-receipt /project/planning/public_bowl_replay_v3_20261009/receipts/demo_0_HOST.json '
        '--export /project/planning/public_bowl_replay_v1_20261009/exports/demo_0 '
        '--checkpoint /model --output /project/runs/vla_action_update_check_v1_20261009/check']
    save('LAUNCH.json', dict(argv=argv, device=0, script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),
                            native_entry_before_sha256=before, lifetime_limit_seconds=900))
    started = time.perf_counter()
    with (run / 'MODEL.log').open('w') as output:
        try:
            result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=960)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
            capture(['docker', 'stop', '-t', '10', name])
    inspect = capture(['docker', 'inspect', name])
    save('CONTAINER.json', json.loads(inspect['stdout']) if inspect['returncode'] == 0 else inspect)
    after = hashlib.sha256((project / 'CURRENT_NATIVE_ENTRY.json').read_bytes()).hexdigest()
    save('HOST_RESULT.json', dict(exit_code=exit_code, full_wall_seconds=time.perf_counter()-started,
        native_entry_after_sha256=after, default_entry_unchanged=before == after))
    assert exit_code == 0 and before == after, 'Check failed; preserve logs, do not enable update'
    print('One-step update check passed; candidate remains isolated')


if __name__ == '__main__':
    main()
