"""Launch only our bounded pilot model services on gpu-187 after a fresh gate."""
import json
import socket
import subprocess
import time
from pathlib import Path

ROOT = Path('/mnt/data/task/yuchen/FastPhyagentos')
PROJECT = ROOT / 'FastRSI'
RUN = PROJECT / 'runs/pilot_20261001'
ASSETS = ROOT / 'frozen_archive/extra-freeze-20260929T145255Z/frozen/assets-20260929-v1'
IMAGE = 'sha256:c98dca2210774e2311a0780146956ef3df3c6f0bb099b6bc671ab31befc42a81'


def capture(argv, timeout=30):
    p = subprocess.run(argv, text=True, capture_output=True, timeout=timeout)
    return {'argv': argv, 'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    gate = capture(['bash', '-lc', 'source /opt/hyhal/env.sh; hy-smi --showuse --showmemuse'])
    processes = capture(['bash', '-lc', 'fuser -v /dev/kfd /dev/mkfd 2>&1 || true'])
    containers = capture(['docker', 'ps', '--format', '{{.Names}}'])
    (RUN / 'SERVICE_PREFLIGHT.json').write_text(json.dumps(
        {'time': time.time(), 'hcu': gate, 'holders': processes, 'containers': containers}, indent=2))
    import re
    mem = re.findall(r'HCU\[(\d+)\].*?HCU memory use \(%\):\s*([0-9.]+)', gate['stdout'])
    assert gate['returncode'] == 0 and len(mem) == 8, 'HCU gate unavailable'
    assert all(float(v) == 0 for _, v in mem), 'Reinspect active devices before allocating'
    assert not containers['stdout'].strip(), 'Reinspect existing workload or duplicate services'
    for port in (49500, 49510):
        with socket.socket() as sock:
            assert sock.connect_ex(('127.0.0.1', port)) != 0, f'Port {port} is already active'
    for role, gpu, port in [('pi05', 0, 49500), ('qwen', 1, 49510)]:
        name = f'yuchen-fastrsi-{role}-20261001-v1'
        argv = ['docker', 'run', '-d', '--name', name, '--network', 'host',
                '--cpus', '8', '--memory', '64g', '--pids-limit', '1024', '--shm-size', '2g',
                '--device', '/dev/kfd', '--device', '/dev/mkfd', '--device', '/dev/dri',
                '--group-add', 'video', '--group-add', 'render']
        env = {'HIP_VISIBLE_DEVICES': str(gpu), 'CUDA_VISIBLE_DEVICES': str(gpu),
               'PYTHONDONTWRITEBYTECODE': '1', 'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4',
               'OPENPI_DATA_HOME': '/workspace/openpi_data',
               'PYTHONPATH': '/workspace/openpi/src:/workspace/openpi/packages/openpi-client/src:/workspace/PhyAgentOS-core'}
        for key, value in env.items():
            argv += ['-e', f'{key}={value}']
        mounts = [(str(ASSETS/'dependencies/openpi'), '/workspace/openpi', 'ro'),
                  (str(ASSETS/'dependencies/openpi_data'), '/workspace/openpi_data', 'ro'),
                  (str(ASSETS/'dependencies/PhyAgentOS-core'), '/workspace/PhyAgentOS-core', 'ro'),
                  ('/opt/hyhal', '/opt/hyhal', 'ro'), ('/etc/hfm', '/etc/hfm', 'ro'),
                  (str(RUN), '/pilot', 'rw')]
        if role == 'pi05':
            mounts += [(str(ASSETS/'weights/pi05_libero_pytorch_fp32'), '/model', 'ro')]
            command = ('python -m PhyAgentOS.runtime.policy.openpi.native_openpi_server '
                       '--policy-config pi05_libero --checkpoint-dir /model --pytorch-device cuda:0 '
                       '--disable-torch-compile --torch-seed 7 --host 127.0.0.1 --port 49500')
        else:
            mounts += [(str(ASSETS/'weights/qwen25vl7b'), '/model', 'ro'),
                       (str(ROOT/'evaluation/table4-full-20260929-v1/bundle'), '/frozen-bundle', 'ro')]
            command = ('python -c "import runpy,uvicorn; '
                       "m=runpy.run_path('/frozen-bundle/qwen_server.py'); "
                       "uvicorn.run(m['app'],host='127.0.0.1',port=49510)\"")
        for src, dst, mode in mounts:
            argv += ['-v', f'{src}:{dst}:{mode}']
        argv += [IMAGE, 'timeout', '3600', 'bash', '-lc',
                 'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec ' + command]
        started = time.time()
        result = capture(argv, timeout=60)
        receipt = {'started_at': started, 'image': IMAGE, 'role': role, 'gpu': gpu,
                   'port': port, 'name': name, 'lifetime_limit_seconds': 3600, **result}
        with (RUN/f'LAUNCH_{role}.json').open('x') as f:
            json.dump(receipt, f, indent=2)
        print(json.dumps({'role':role,'name':name,'returncode':result['returncode'],
                          'container':result['stdout'].strip(),'stderr':result['stderr']}), flush=True)
        if result['returncode']:
            raise SystemExit(result['stderr'])


if __name__ == '__main__':
    main()
