"""One fresh device, fixed 400 native-flow updates of independent expert Q/V rank-four adapters."""
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
    run = project / 'runs/bowl_expert_lora400_v1_20261009'
    lock = (project / 'runs/full_os_v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.mkdir(exist_ok=False)
    def save(name, data):
        (run / name).write_text(json.dumps(data, indent=2) + '\n')
    def capture(argv):
        p = subprocess.run(argv, capture_output=True, text=True, timeout=30)
        return dict(argv=argv, returncode=p.returncode, stdout=p.stdout, stderr=p.stderr)
    pre = dict(time=time.time(), hcu=capture(['bash', '-lc',
        'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && hy-smi --showuse --showmemuse']),
        containers=capture(['docker', 'ps', '--format', '{{.ID}} {{.Names}}']),
        holders=capture(['bash', '-lc', 'fuser -v /dev/kfd /dev/mkfd /dev/dri/render*']))
    save('PREFLIGHT.json', pre)
    memory = re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)', pre['hcu']['stdout'])
    assert pre['hcu']['returncode'] == 0 and len(memory) == 8 and all(float(v) == 0 for _, v in memory)
    assert pre['containers']['returncode'] == 0 and not pre['containers']['stdout'].strip()
    holder_ids = [int(x) for x in pre['holders']['stdout'].split()]
    holder_details = []
    for pid in holder_ids:
        comm = (Path('/proc') / str(pid) / 'comm').read_text().strip()
        executable = str((Path('/proc') / str(pid) / 'exe').resolve(strict=True))
        holder_details.append(dict(pid=pid, comm=comm, executable=executable))
        assert comm == 'hymgr' and executable == '/usr/local/hyhal/bin/hymgr', holder_details
    save('HOLDER_IDENTITIES.json', dict(holders=holder_details, allowed='Verified host GPU management daemon only'))
    assert pre['holders']['returncode'] in (0, 1)
    native_before = hashlib.sha256((project / 'CURRENT_NATIVE_ENTRY.json').read_bytes()).hexdigest()
    shutil.copytree(project / 'models/openpi_data', run / 'tokenizer_cache')
    scripts = ['train_bowl_expert_lora_v1.py', 'bowl_expert_lora_v1.py', 'audit_bowl_action_pipeline_v1.py']
    for filename in scripts:
        shutil.copy2(project / 'scripts' / filename, run / filename)
    source_dir = run / 'frozen_sources'
    source_dir.mkdir()
    for rel in ['openpi/src/openpi/training/config.py', 'openpi/src/openpi/transforms.py',
                'openpi/src/openpi/policies/libero_policy.py', 'openpi/src/openpi/policies/policy.py',
                'openpi/src/openpi/models_pytorch/pi0_pytorch.py',
                'PhyAgentOS-core/PhyAgentOS/runtime/policy/openpi/native_openpi_server.py',
                'PhyAgentOS-core/PhyAgentOS/runtime/adapters/openpi/base_openpi_adapter.py',
                'PhyAgentOS-core/PhyAgentOS/runtime/targets/remote/libero/server.py']:
        destination = source_dir / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(assets / 'dependencies' / rel, destination)
    save('PROTOCOL.json', dict(stages=['A', 'LORA400'], sampling_steps=10, training_steps=400,
        robot_executions=0, fresh_python_process_per_stage=False, same_native_base_loader=True,
        explicit_noise=True, default_entry_before_sha256=native_before,
        scripts_sha256={f: hashlib.sha256((run / f).read_bytes()).hexdigest() for f in scripts}))
    name = 'yuchen-fastrsi-expert-lora400-v1-20261009'
    argv = ['docker', 'run', '--name', name, '--network', 'none', '--cpus', '8', '--memory', '64g',
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
        'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81',
        'timeout', '1500', 'bash', '-lc',
        'set -e; source /opt/dtk/env.sh; source /opt/hyhal/env.sh; '
        'exec python /project/runs/bowl_expert_lora400_v1_20261009/'
        'train_bowl_expert_lora_v1.py '
        '--output /project/runs/bowl_expert_lora400_v1_20261009']
    save('LAUNCH.json', dict(argv=argv, device=0, lifetime_limit_seconds=1500))
    started = time.perf_counter()
    with (run / 'MODEL.log').open('w') as output:
        try:
            result = subprocess.run(argv, stdout=output, stderr=subprocess.STDOUT, timeout=1560)
            exit_code = result.returncode
        except subprocess.TimeoutExpired:
            exit_code = 124
            capture(['docker', 'stop', '-t', '10', name])
    inspect = capture(['docker', 'inspect', name])
    save('CONTAINER.json', json.loads(inspect['stdout']) if inspect['returncode'] == 0 else inspect)
    after = hashlib.sha256((project / 'CURRENT_NATIVE_ENTRY.json').read_bytes()).hexdigest()
    save('HOST_RESULT.json', dict(exit_code=exit_code, full_wall_seconds=time.perf_counter()-started,
        default_entry_unchanged=native_before == after, native_entry_after_sha256=after))
    assert exit_code == 0 and native_before == after
    print('Expert rank-four training completed; source closed-loop still required')


if __name__ == '__main__':
    main()
