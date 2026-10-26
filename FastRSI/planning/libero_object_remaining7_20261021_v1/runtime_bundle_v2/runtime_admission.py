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
LEASE_ROOT = '/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning/liberopro_eval_20261012_v1/leases'


class ResourceUnavailable(RuntimeError):
    """A resource is occupied/unexplained; leave it untouched, do not call it model failure."""


def ident(pid):
    proc = Path('/proc') / str(pid)
    return dict(pid=pid,
                boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                start=(proc / 'stat').read_text().split(') ', 1)[1].split()[19],
                argv=(proc / 'cmdline').read_bytes().decode().split('\0')[:-1])


def save(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2)


def validate_occupancy(root, report, approval, node, gpu):
    registered = {}
    for path in (root / 'campaigns').rglob('LAUNCH_*.json'):
        value = read(path)
        if value.get('returncode') == 0:
            registered[value['container_id']] = path.parent
    allowed_pids = set()
    for container in report['containers']:
        matches = [cid for cid in registered if cid.startswith(container['id'])]
        if len(matches) != 1:
            raise ResourceUnavailable(('Unregistered container; leave unrelated task untouched', container['name']))
        cid = matches[0]
        result = subprocess.run(['docker', 'inspect', cid], capture_output=True,
                                text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        actual = json.loads(result.stdout)[0]
        assert actual['Id'] == cid
        # A historical LAUNCH is not live-holder evidence.
        if actual['State']['Running'] is False:
            continue
        identity = read(registered[cid] / 'IDENTITY.json')
        assert identity['node'] == node
        assert ident(identity['process']['pid']) == identity['process']
        for line in container['top']['stdout'].splitlines()[1:]:
            if line.split() and line.split()[0].isdigit():
                allowed_pids.add(int(line.split()[0]))
        actual_host = actual['HostConfig']
        if actual_host['Devices'] or actual_host['DeviceRequests'] or actual_host['Privileged']:
            visible_rows = [value for value in actual['Config']['Env']
                            if value.startswith(('HIP_VISIBLE_DEVICES=', 'CUDA_VISIBLE_DEVICES='))]
            assert visible_rows, 'Live registered device holder lacks actual visibility binding'
            for value in visible_rows:
                visible = {int(v) for v in value.split('=', 1)[1].split(',')}
                if gpu in visible:
                    raise ResourceUnavailable(('Requested card already occupied', container['name']))
    for expected in approval['exact_infrastructure_holders'].get(node, []):
        assert ident(expected['pid']) == expected
        allowed_pids.add(expected['pid'])
    for line in report['holders']['stdout'].splitlines():
        if not line.strip() or line.strip().startswith('USER '):
            continue
        pids = [int(value) for value in line.split() if value.isdigit()]
        if not pids or not all(pid in allowed_pids for pid in pids):
            raise ResourceUnavailable(('Unknown device holder; do not stop it', line))
    leases = LEASE_ROOT
    held = json.loads(report['held_locks']['stdout'])['locks']
    held_paths = {item.get('path') for item in held}
    permitted_parallel = set()
    for item in approval.get('exact_parallel_reservations', []):
        if item['path'] not in held_paths:
            # A closed predecessor's metadata is not an active reservation.
            continue
        assert item['project_root'] == str(root.resolve())
        assert item['node'] == node and type(item['gpu']) is int
        assert item['gpu'] in range(8) and item['gpu'] != gpu
        assert sha(item['path']) == item['sha256']
        assert ident(item['owner']['pid']) == item['owner']
        receipt = Path(item['owner_receipt']['path']).resolve()
        assert root.resolve() in receipt.parents
        assert sha(receipt) == item['owner_receipt']['sha256']
        owner = read(receipt)
        assert owner['process'] == item['owner']
        assert owner['node'] == node and owner['gpu'] == item['gpu']
        permitted_parallel.add(item['path'])
    for item in held:
        path = item.get('path') or ''
        if path.startswith(leases) and path not in permitted_parallel:
            raise ResourceUnavailable(('Unexplained or target-card held reservation; leave untouched', path))
    return dict(explained_live_pids=sorted(allowed_pids),
                permitted_parallel_reservations=sorted(permitted_parallel))


def validate_target_card(report, gpu):
    compute = re.search(r'HCU\[' + str(gpu) + r'\].*HCU use \(%\): 0(?:\.0+)?(?=\s|$)',
                        report['hcu']['stdout'])
    memory = re.search(r'HCU\[' + str(gpu) + r'\].*HCU memory use \(%\): 0(?:\.0+)?(?=\s|$)',
                       report['hcu']['stdout'])
    if not compute or not memory:
        raise ResourceUnavailable(('Requested card is not freshly compute/memory idle', gpu))


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
    explained = validate_occupancy(root, report, approval, node, gpu)
    validate_target_card(report, gpu)
    ram = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                   if line.startswith('MemAvailable:'))) * 1024
    assert ram > 196 * 1024 ** 3
    for directory, minimum in [(root, 100), (Path('/var/lib/docker'), 50)]:
        stat = os.statvfs(str(directory))
        assert stat.f_bavail * stat.f_frsize > minimum * 1024 ** 3
    port = 31000 + 100 * gpu
    with socket.socket() as connection:
        if connection.connect_ex(('127.0.0.1', port)) == 0:
            raise ResourceUnavailable(('Occupied policy port; leave unrelated task untouched', port))
    for image in approval['images']:
        query = subprocess.run(['docker', 'image', 'inspect', image],
                               capture_output=True, text=True, timeout=30)
        assert query.returncode == 0 and json.loads(query.stdout)[0]['Id'] == image
    result = dict(status='FRESH_IDLE_CARD_AND_RESOURCE_BINDINGS_VERIFIED',
                  node=node, gpu=gpu, time=time.time(), ram_available=ram,
                  report_sha256=sha(output / 'FRESH_RESOURCE_REPORT.json'),
                  reservation_bindings=approval['reservation_bindings'],
                  explained_occupancy=explained,
                  unknown_holders_allowed=False, other_tasks_modified=False)
    save(output / 'ADMISSION_BEFORE_POLICY.json', result)
    return result
