"""Bounded 16-call saved-input public-adapter experiment on one idle Qwen GPU."""
import fcntl
import hashlib
import json
import shutil
import signal
import socket
import time
import traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen
from orchestrate_resident_services import configure_context, gpu
from orchestrate_resident_full_os import freeze, tree, read

NAME = 'verifier_semantics_v2_20261003'


def main():
    assert socket.gethostname() == 'j03r4n13'
    lock = open(life.PROJECT / 'runs/full_os_v1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = life.PROJECT / 'runs' / NAME
    run.mkdir(exist_ok=False)
    started = time.time()
    deadline = started + 1000
    configure_context(run, deadline)
    stream = run / 'stream'; stream.mkdir()
    cpu = run / 'cpu'; cpu.mkdir()
    state = {'status': 'RUNNING', 'started_at': started, 'deadline': deadline,
             'scope': 'Public verification input correction only; zero robot episodes, no experience updates'}
    life.save(run / 'STATUS.json', state)
    def interrupt(n, f):
        raise KeyboardInterrupt('Interrupted ' + str(n))
    signal.signal(signal.SIGINT, interrupt); signal.signal(signal.SIGTERM, interrupt)
    try:
        for rel in ('scripts', 'src/fastrsi', 'tests', 'configs'):
            freeze(life.PROJECT / rel, run / 'code' / rel)
        life.save(run / 'CODE_MANIFEST.json', tree(run / 'code'))
        source = life.PROJECT / 'runs/structural_window_v1_20261003'
        assert read(source / 'STATUS.json')['status'] == 'COMPLETED'
        for rel in ('MODEL_CONFIG.json', 'MODEL_BINDING.json', 'CORE_SOURCE_BINDING.json'):
            shutil.copyfile(source / rel, run / rel)
        protocol = read(life.PROJECT / 'planning' / NAME / 'PROTOCOL.json')
        protocol['deadline'] = deadline
        life.save(run / 'PROTOCOL.json', protocol)
        (run / 'inputs').mkdir()
        for spec in protocol['cases']:
            data = (life.PROJECT / spec['source']).read_bytes()
            assert hashlib.sha256(data).hexdigest() == spec['sha256']
            (run / 'inputs' / (spec['name'] + '.json')).write_bytes(data)
        life.gate(cpu)
        tests = 'test_libero_verification_semantics test_verifier_semantics_replay test_session_image_labels test_verifier_schema_contract test_experience_admission test_native_isolation'
        command = 'source /opt/dtk/env.sh && cd ' + life.BASE + '/code/tests && /project/envs/os-v1/bin/python -m unittest ' + tests + ' -v'
        args = life.cpu_args('yuchen-fastrsi-'+NAME+'-cpu', cpu)
        args += ['-e', 'FASTRSI_TEST_PROJECT=/project', '-e', 'FASTRSI_NATIVE_RUN=/tmp/fastrsi-semantics-cpu/stream', life.IMAGE, 'timeout', '180', 'bash', '-lc', command]
        cid = life.launch('cpu', args, cpu)
        life.wait_exit(cid, 190); life.save(cpu / 'STOP_RECEIPT.json', life.stop_all())
        state['cpu_status'] = 'PASSED'
        load_start = time.time()
        argv = read(life.OLD / 'LAUNCH_qwen.json')['argv']
        argv = [x.replace(str(life.OLD), str(stream)) for x in argv]
        argv[argv.index('--name')+1] = 'yuchen-fastrsi-'+NAME+'-qwen'
        argv = gpu(configure_qwen(argv, read(run / 'MODEL_CONFIG.json'), life), 0)
        ix = argv.index(life.IMAGE)
        assert argv[ix+1] == 'timeout'
        argv[ix+2] = str(int(deadline-time.time()))
        life.launch('qwen', argv, stream); life.health(49510, limit=300)
        state['model_startup_seconds'] = time.time()-load_start
        life.save(run / 'STATUS.json', state)
        remaining = int(deadline-time.time()-20); assert remaining > 60
        cmd = 'source /opt/dtk/env.sh && exec /project/envs/os-v1/bin/python ' + life.BASE + '/code/scripts/replay_verifier_semantics.py --run ' + life.BASE
        cid = life.launch('runner', life.cpu_args('yuchen-fastrsi-'+NAME+'-runner', stream) + [life.IMAGE, 'timeout', str(remaining), 'bash', '-lc', cmd], stream)
        life.wait_exit(cid, remaining+5)
        result = read(stream / 'RESULT.json')
        count = len(list((stream / 'service_calls').glob('*/REQUEST.json')))
        assert result['status'] == 'COMPLETED' and len(result['cases']) == 16 and count == 16, (result['status'], count)
        state.update(status='COMPLETED', replay_seconds=result['seconds'], model_calls=count)
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc())
    finally:
        stopped = life.stop_all()
        life.save(stream / 'STOP_RECEIPT.json', stopped)
        if not all(not c['running'] for c in stopped):
            state.update(status='FAILED', cleanup_attention='An owned container is still running')
        state.update(finished_at=time.time(), full_wall_seconds=time.time()-started)
        life.save(run / 'STATUS.json', state)
        print(json.dumps(state), flush=True)
    if state['status'] != 'COMPLETED':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
