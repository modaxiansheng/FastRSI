"""One-card sparse Qwen fact diagnosis, reusing the accepted common service."""
import fcntl
import argparse
import hashlib
import json
import signal
import subprocess
import time
import traceback
import orchestrate_full_os as life
from native_service_config import configure_qwen

NAME = 'object_stage_teacher_v1_20261006'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', choices=(NAME, 'object_stage_teacher_v2_20261006', 'object_stage_teacher_v3_20261006'), default=NAME)
    args = parser.parse_args(); name = args.name
    run = life.PROJECT / 'runs' / name
    reg = read(run / 'REGISTRATION.json')
    if reg['name'] != name or sha(run / 'MODEL_CONFIG.json') != reg['model_config_sha256']:
        raise ValueError('Unbound diagnosis')
    for rel, expected in read(run / 'CODE_MANIFEST.json').items():
        if sha(run / 'code' / rel) != expected: raise ValueError('Frozen source changed')
    for rel, expected in read(run / 'inputs/SOURCE_MANIFEST.json').items():
        if sha(life.PROJECT / rel) != expected: raise ValueError('Saved image changed')
    lock = open(life.PROJECT / 'runs/full_os_v1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (run / 'STATUS.json').exists(): raise RuntimeError('Already dispatched')
    life.CAMPAIGN = run; life.BASE = '/project/runs/' + name
    start = time.time(); life.deadline = start + 900
    def interrupted(n, f): raise KeyboardInterrupt('Interrupted diagnosis ' + str(n))
    signal.signal(signal.SIGTERM, interrupted); signal.signal(signal.SIGINT, interrupted)
    state = dict(status='RUNNING', started_at=start, new_robot_episodes=0, persistent_updates=0)
    life.save(run / 'STATUS.json', state)
    try:
        life.gate(run)
        cfg = read(run / 'MODEL_CONFIG.json')
        argv = read(life.OLD / 'LAUNCH_qwen.json')['argv']
        argv[argv.index('--name') + 1] = 'yuchen-fastrsi-' + name + '-qwen'
        argv = [x.replace(str(life.OLD), str(run)).replace('/project/runs/pilot_20261001', life.BASE) for x in argv]
        life.launch('qwen', configure_qwen(argv, cfg, life), run)
        life.health(49510, limit=240)
        state['startup_seconds'] = time.time() - start
        with (run / 'runner.log').open('x') as log:
            subprocess.run(['python3', str(run / 'code/scripts/run_object_stage_teacher_v1.py'),
                '--run', str(run), '--project', str(life.PROJECT)],
                stdout=log, stderr=subprocess.STDOUT, check=True, timeout=620,
                env={**__import__('os').environ, 'PYTHONPATH': str(run / 'code/src')})
        result = read(run / 'diagnosis/RESULT.json')
        state.update(status='COMPLETED', teacher_calls=result['teacher_calls'],
            diagnostic_seconds=result['seconds'], typed_calls=sum(r['status'] == 'TYPED_DIAGNOSIS' for r in result['rows']))
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc()); raise
    finally:
        life.save(run / 'STOP_RECEIPT.json', life.stop_all())
        state.update(seconds=time.time() - start, finished_at=time.time())
        state['reserved_card_seconds'] = state['seconds']
        state['cost_scope'] = 'One service reservation including startup and teardown; no speedup or robot result.'
        life.save(run / 'STATUS.json', state); print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
