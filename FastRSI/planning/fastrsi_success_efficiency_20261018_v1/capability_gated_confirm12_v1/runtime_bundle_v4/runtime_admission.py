"""Fresh read-only host checks immediately before the owned policy launch.

The root-reviewed authorization supplies the exact existing collector hash and
any exact infrastructure-holder identities. Unknown holders remain blocking.
This module never starts or stops another task's process/container.
"""
from pathlib import Path
import json
import os
import re
import socket
import subprocess
import time
from coordinator_contract import read, sha

COLLECTOR = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning/liberopro_eval_20261012_v1/technical_history/expand107108_shared_case_v1/audit_node.py')


def ident(pid):
    proc = Path('/proc') / str(pid)
    return dict(pid=pid,
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start=(proc / 'stat').read_text().split(') ', 1)[1].split()[19],
                argv=(proc / 'cmdline').read_bytes().decode().split('\0')[:-1])


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)


def audit(root, output, approval, node, gpu):
    assert node in ['187', '188'] and gpu in range(8)
    assert approval['resource_collector']['path'] == str(COLLECTOR)
    assert COLLECTOR.is_file() and sha(COLLECTOR) == approval['resource_collector']['sha256']
    assert approval['reservation_inventory_complete'] is True
    for item in approval['reservation_bindings']:
        assert sha(item['path']) == item['sha256'], 'Resource claim changed after root review'
    started = time.time()
    process = subprocess.run(['python3', str(COLLECTOR)], capture_output=True,
                             text=True, timeout=170)
    save(output / 'RESOURCE_COLLECTOR_PROCESS.json', dict(
        start=started, end=time.time(), argv=['python3', str(COLLECTOR)],
        collector_sha256=sha(COLLECTOR), returncode=process.returncode,
        stdout=process.stdout, stderr=process.stderr))
    assert process.returncode == 0, process.stderr
    report = json.loads(process.stdout)
    save(output / 'FRESH_RESOURCE_REPORT.json', report)
    for key in ['hcu', 'holders', 'held_locks', 'ports', 'storage', 'mounts', 'kernel']:
        assert report[key]['code'] == 0, (key, report[key])
    assert not report['kernel']['stdout'].strip(), 'Unresolved driver/kernel diagnostic'
    owned = set()
    for path in (root / 'campaigns').rglob('LAUNCH_*.json'):
        value = read(path)
        if value.get('returncode') == 0:
            owned.add(value['container_id'])
    allowed_pids = set()
    for container in report['containers']:
        assert any(cid.startswith(container['id']) for cid in owned), \
            ('Unregistered container; do not stop it', container['name'])
        for line in container['top']['stdout'].splitlines()[1:]:
            if line.split() and line.split()[0].isdigit():
                allowed_pids.add(int(line.split()[0]))
        if container['devices'] or container['device_requests'] or container['privileged']:
            for value in container['visible']:
                if value.startswith(('HIP_VISIBLE_DEVICES=', 'CUDA_VISIBLE_DEVICES=')):
                    visible = {int(v) for v in value.split('=', 1)[1].split(',')}
                    assert gpu not in visible, ('Requested card already occupied', container['name'])
    infrastructure = approval['exact_infrastructure_holders'].get(node, [])
    for expected in infrastructure:
        assert ident(expected['pid']) == expected
        allowed_pids.add(expected['pid'])
    for line in report['holders']['stdout'].splitlines():
        if not line.strip() or 'USER' in line or 'hymgr' in line:
            continue
        pids = [int(value) for value in line.split() if value.isdigit()]
        assert pids and all(pid in allowed_pids for pid in pids), \
            ('Unknown device holder; do not stop it', line)
    assert re.search(r'HCU\[' + str(gpu) + r'\].*HCU use \(%\): 0(?:\.0)?\b',
                     report['hcu']['stdout'])
    assert re.search(r'HCU\[' + str(gpu) + r'\].*HCU memory use \(%\): 0\b',
                     report['hcu']['stdout'])
    leases = '/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning/liberopro_eval_20261012_v1/leases'
    held = json.loads(report['held_locks']['stdout'])['locks']
    assert not any((item.get('path') or '').startswith(leases) for item in held)
    ram = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))) * 1024
    assert ram > 196 * 1024 ** 3
    for directory, minimum in [(root, 100), (Path('/var/lib/docker'), 50)]:
        stat = os.statvfs(str(directory))
        assert stat.f_bavail * stat.f_frsize > minimum * 1024 ** 3
    port = 31000 + 100 * gpu
    with socket.socket() as connection:
        assert connection.connect_ex(('127.0.0.1', port)) != 0, ('Occupied policy port', port)
    for image in approval['images']:
        query = subprocess.run(['docker', 'image', 'inspect', image],
                               capture_output=True, text=True, timeout=30)
        assert query.returncode == 0 and json.loads(query.stdout)[0]['Id'] == image
    result = dict(status='FRESH_IDLE_CARD_AND_RESOURCE_BINDINGS_VERIFIED',
                  node=node, gpu=gpu, time=time.time(), ram_available=ram,
                  report_sha256=sha(output / 'FRESH_RESOURCE_REPORT.json'),
                  reservation_bindings=approval['reservation_bindings'],
                  unknown_holders_allowed=False, other_tasks_modified=False)
    save(output / 'ADMISSION_BEFORE_POLICY.json', result)
    return result
