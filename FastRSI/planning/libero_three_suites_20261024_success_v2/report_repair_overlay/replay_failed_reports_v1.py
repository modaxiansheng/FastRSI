"""One-owner neural replay of two archived report requests; no robot execution."""
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

ROOT = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI/planning/libero500_remediation_20261011_v1/joint_full3430_v4')
PROJECT = Path('/mnt/data/task/yuchen/FastPhyagentos/FastRSI')
CAND = ROOT / 'report_loop_candidate_v1'
OUT = CAND / 'replay2'
FROZEN = ROOT / 'system_joint_v3_pi05_P'
GATE = PROJECT / 'planning/libero500_pr_20261011_v2/controller/strict_gate_v8_frozen.py'
PORT = 27510


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def command(argv, timeout=30):
    p = subprocess.run(argv, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       timeout=timeout, check=False)
    return {'returncode': p.returncode, 'stdout': p.stdout, 'stderr': p.stderr}


def stop_on_signal(signum, _frame):
    raise KeyboardInterrupt('signal ' + str(signum))


def replace_after(argv, flag, value):
    argv[argv.index(flag) + 1] = value


def main():
    signal.signal(signal.SIGTERM, stop_on_signal)
    signal.signal(signal.SIGINT, stop_on_signal)
    addresses = json.loads(subprocess.check_output(['ip', '-j', 'address']))
    assert any(a.get('local') == '10.232.1.187' for row in addresses
               for a in row.get('addr_info', [])), 'Wrong host'
    assert CAND.is_dir() and OUT.is_dir() and FROZEN.is_dir()
    assert digest(CAND / 'code/scripts/qwen_agent_server.py') != digest(FROZEN / 'code/scripts/qwen_agent_server.py')
    assert digest(CAND / 'code/src/fastrsi/terminal_results.py') != digest(FROZEN / 'code/src/fastrsi/terminal_results.py')
    identity = uuid.uuid4().hex
    receipt = {'status': 'CHECKING', 'identity': identity, 'node': '187',
               'source_sha256': {f: digest(CAND / f) for f in
                                 ('code/scripts/qwen_agent_server.py',
                                  'code/src/fastrsi/terminal_results.py')},
               'original_request_sha256': {}, 'started_at': time.time(),
               'scope': 'Archived terminal-report requests only; no robot execution or result promotion'}
    cid = None
    handles = []
    try:
        lock_paths = [PROJECT / 'runs/full_os_v1.lock',
                      PROJECT / 'planning/multi_vla_efficiency_20261010_v1/parallel_group_coordinator.lock',
                      PROJECT / 'runs/system_efficiency_node187.lock',
                      PROJECT / 'planning/multi_vla_efficiency_20261010_v1/active_node187.lock']
        for path in lock_paths:
            assert path.is_file(), path
            handle = path.open('r+')
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            handles.append(handle)
        receipt['locks'] = [str(x) for x in lock_paths]
        sockets = command(['ss', '-ltnp'], 20)
        assert sockets['returncode'] == 0
        assert PORT not in {int(x) for x in re.findall(r':(\d+)\s', sockets['stdout'])}
        module_spec = importlib.util.spec_from_file_location('strict_gate_v8_frozen', GATE)
        module = importlib.util.module_from_spec(module_spec)
        module_spec.loader.exec_module(module)
        module.gate(OUT, PROJECT, command, save)
        receipt['gate_sha256'] = digest(OUT / 'RESOURCE_PREFLIGHT.json')
        assert json.loads((OUT / 'RESOURCE_PREFLIGHT.json').read_text())['status'] == 'FRESH_FULL_HOST_GATE_PASSED'

        template = json.loads((FROZEN / 'node187/lane0_resume0007_segment1/LAUNCH_qwen.json').read_text())['argv']
        argv = list(template)
        replace_after(argv, '--cidfile', str(OUT / 'CID'))
        replace_after(argv, '--name', 'yuchen-fastrsi-report-loop-replay-' + identity[:12])
        for i, arg in enumerate(argv):
            if arg.startswith('io.fastrsi.owner='):
                argv[i] = 'io.fastrsi.owner=' + identity
            elif arg.startswith('io.fastrsi.intent='):
                argv[i] = 'io.fastrsi.intent=' + identity
            elif arg.startswith('io.fastrsi.plan='):
                argv[i] = 'io.fastrsi.plan=report_loop_candidate_v1'
            elif arg.startswith('FASTRSI_SERVICE_PORTS='):
                argv[i] = 'FASTRSI_SERVICE_PORTS=' + json.dumps(
                    {'policy': 27500, 'target': 27502, 'qwen': PORT, 'verifier': 27520}, separators=(',', ':'))
            elif arg.endswith(':/pilot:rw'):
                argv[i] = str(OUT) + ':/pilot:rw'
            elif arg.endswith(':/project/runs/system_joint_v3_pi05_P:rw'):
                argv[i] = str(CAND) + ':/project/runs/system_joint_v3_pi05_P:rw'
        assert argv.count(str(CAND) + ':/project/runs/system_joint_v3_pi05_P:rw') == 1
        assert argv.count(str(OUT) + ':/pilot:rw') == 1
        replace_after(argv, 'timeout', '1500')
        launch = command(argv, 60)
        assert launch['returncode'] == 0, launch['stderr'][-1000:]
        cid = launch['stdout'].strip()
        assert re.fullmatch(r'[0-9a-f]{64}', cid), cid
        receipt['cid'] = cid
        receipt['launch_at'] = time.time()
        save(OUT / 'STATUS.json', receipt)
        url = 'http://127.0.0.1:' + str(PORT)
        deadline = time.monotonic() + 360
        while time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(url + '/healthz', timeout=3) as reply:
                    health = json.load(reply)
                if health.get('ok') is True:
                    receipt['health'] = health
                    break
            except (OSError, ValueError):
                time.sleep(3)
        else:
            raise TimeoutError('Qwen service did not become healthy')
        cases = [('lane0_resume0007_segment0', 'case009_t0_i33', 'call_8181377f61a248d6ae8f0cc15eb03f46'),
                 ('lane0_resume0007_segment1', 'case002_t0_i35', 'call_881784e75c0147968a16d10b4db17c4c')]
        receipt['replays'] = []
        for segment, case, call_id in cases:
            request_path = FROZEN / 'node187' / segment / 'service_calls' / call_id / 'REQUEST.json'
            body = request_path.read_bytes()
            receipt['original_request_sha256'][case] = digest(request_path)
            started = time.time()
            req = urllib.request.Request(url + '/v1/chat/completions', data=body,
                                         headers={'Content-Type': 'application/json'}, method='POST')
            try:
                with urllib.request.urlopen(req, timeout=480) as reply:
                    response = reply.read()
                    http_status = reply.status
            except urllib.error.HTTPError as exc:
                response = exc.read()
                http_status = exc.code
            (OUT / (case + '_RESPONSE.json')).write_bytes(response)
            item = {'case': case, 'http_status': http_status,
                    'response_sha256': hashlib.sha256(response).hexdigest(),
                    'seconds': time.time() - started}
            if http_status == 200:
                parsed = json.loads(response)
                choice = parsed['choices'][0]
                item['finish_reason'] = choice['finish_reason']
                item['tool_names'] = [x['function']['name'] for x in choice['message'].get('tool_calls', [])]
                item['usage'] = parsed.get('usage')
            else:
                item['error'] = json.loads(response).get('error')
            receipt['replays'].append(item)
            save(OUT / 'STATUS.json', receipt)
        receipt['status'] = ('NEURAL_REPLAY_PASSED' if all(
            r['http_status'] == 200 and r['finish_reason'] == 'tool_calls' and 'write_file' in r['tool_names']
            for r in receipt['replays']) else 'NEURAL_REPLAY_FAILED')
    except BaseException as exc:
        receipt['status'] = 'REPLAY_INTERRUPTED_OR_REFUSED'
        receipt['error'] = repr(exc)
        raise
    finally:
        if cid:
            inspect = command(['docker', 'inspect', '--format', '{{index .Config.Labels "io.fastrsi.owner"}}', cid], 15)
            if inspect['returncode'] == 0 and inspect['stdout'].strip() == identity:
                receipt['stop'] = command(['docker', 'stop', '-t', '15', cid], 35)
                receipt['remove'] = command(['docker', 'rm', '-f', cid], 30)
            else:
                receipt['cleanup_identity_mismatch'] = inspect
        for handle in reversed(handles):
            handle.close()
        receipt['finished_at'] = time.time()
        save(OUT / 'STATUS.json', receipt)
        print(json.dumps({'status': receipt['status'], 'replays': receipt.get('replays'),
                          'cleanup_identity_mismatch': receipt.get('cleanup_identity_mismatch')}))


if __name__ == '__main__':
    main()
