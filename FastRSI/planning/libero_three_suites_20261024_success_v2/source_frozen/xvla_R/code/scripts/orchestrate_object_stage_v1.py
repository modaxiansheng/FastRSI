"""Bounded CPU annotation and one-card frozen visual-head learning campaign.

Only source actions already executed in development are replayed. New live
Agent/robot trials are a separate phase, not a claim made by this orchestrator.
"""
import argparse
import fcntl
import hashlib
import json
import shutil
import time
import traceback
from pathlib import Path

import orchestrate_full_os as life

NAME = 'object_stage_memory_v2_20261006'


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pp(path):
    return str(path).replace(str(life.PROJECT), '/project')


def run_replays(run):
    out = run / 'replays'
    out.mkdir(exist_ok=False)
    state = dict(status='RUNNING', started_at=time.time(), phase='EXACT_DEVELOPMENT_REPLAY',
                 new_policy_calls=0, new_model_calls=0, new_online_episodes=0)
    life.save(out / 'STATUS.json', state)
    try:
        life.gate(out)
        workers = []
        for source in read(run / 'REGISTRATION.json')['sources']:
            stream = life.PROJECT / source['stream']
            if (sha(stream / 'RESULT.json') != source['result_sha256']
                    or sha(stream / 'LAUNCH_target.json') != source['launch_sha256']
                    or sha(life.PROJECT / source['trace']) != source['trace_sha256']):
                raise ValueError('Preregistered source changed')
            folder = out / ('init' + str(source['init']))
            folder.mkdir()
            old = read(stream / 'LAUNCH_target.json')['argv']
            image_index = next(i for i, v in enumerate(old) if v.startswith('sha256:'))
            prefix = []
            i = 0
            while i < image_index:
                value = old[i]
                if value == '-e' and old[i + 1].startswith('FASTRSI_'):
                    i += 2
                    continue
                if value == '-e' and old[i + 1].startswith('PYTHONPATH='):
                    prefix += ['-e', 'PYTHONPATH=/workspace/openpi/third_party/libero:'
                               + pp(run / 'code/scripts') + ':' + pp(run / 'code/src')]
                    i += 2
                    continue
                if value == '--name':
                    prefix += [value, 'yuchen-fastrsi-object-stage-v2-replay-i' + str(source['init'])]
                    i += 2
                    continue
                if value == '--network':
                    prefix += [value, 'none']
                    i += 2
                    continue
                if value == '-v' and old[i + 1] == str(life.PROJECT) + ':/project:rw':
                    prefix += [value, str(life.PROJECT) + ':/project:ro']
                    i += 2
                    continue
                prefix.append(value)
                i += 1
            argv = prefix + ['-v', str(folder) + ':/output:rw', old[image_index],
                'timeout', '360', 'python', pp(run / 'code/scripts/replay_object_stage_supervision_v1.py'),
                '--trace', pp(life.PROJECT / source['trace']), '--result', pp(stream / 'RESULT.json'),
                '--init', str(source['init']), '--output', '/output/REPLAY.json']
            workers.append((source, folder, life.launch('replay', argv, folder)))
        for source, folder, cid in workers:
            life.wait_exit(cid, 380)
            result = read(folder / 'REPLAY.json')
            if result['status'] != 'EXACT_IMAGE_REPLAY':
                raise ValueError('Exact replay not established')
        state['status'] = 'COMPLETED'
        state['contexts'] = len(workers)
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        life.save(out / 'STOP_RECEIPT.json', life.stop_all())
        state.update(seconds=time.time() - state['started_at'], finished_at=time.time())
        life.save(out / 'STATUS.json', state)


def run_learner(run):
    state = dict(status='RUNNING', started_at=time.time(), phase='FROZEN_ENCODER_STAGE_HEAD',
                 new_online_episodes=0, live_controller_changed=False)
    out = run / 'model_job'
    out.mkdir(exist_ok=False)
    life.save(out / 'STATUS.json', state)
    try:
        life.gate(out)
        revision = read(run / 'REGISTRATION.json')['model_revision']
        model = life.PROJECT / 'models/clef-flash' / revision
        argv = ['docker', 'run', '-d', '--name', 'yuchen-fastrsi-object-stage-head-v2',
            '--network', 'none', '--cpus', '8', '--memory', '64g', '--shm-size', '2g',
            '--device', '/dev/kfd', '--device', '/dev/mkfd', '--device', '/dev/dri',
            '--group-add', 'video', '--group-add', 'render',
            '-e', 'HIP_VISIBLE_DEVICES=0', '-e', 'CUDA_VISIBLE_DEVICES=0',
            '-e', 'HF_HUB_OFFLINE=1', '-e', 'PYTHONDONTWRITEBYTECODE=1',
            '-e', 'OMP_NUM_THREADS=4', '-e', 'MKL_NUM_THREADS=4',
            '-e', 'PYTHONPATH=' + pp(run / 'code/src'),
            '-v', '/opt/hyhal:/opt/hyhal:ro', '-v', '/etc/hfm:/etc/hfm:ro',
            '-v', str(life.PROJECT) + ':/project:ro', '-v', str(out) + ':/output:rw',
            life.IMAGE, 'timeout', '900', 'bash', '-lc',
            'source /opt/dtk/env.sh && source /opt/hyhal/env.sh && exec '
            '/project/envs/clef-decision-v1/bin/python '
            + pp(run / 'code/scripts/run_object_stage_learning_v1.py')
            + ' --project /project --model ' + pp(model)
            + ' --inputs ' + pp(run / 'inputs') + ' --output /output/learning']
        cid = life.launch('learner', argv, out)
        life.wait_exit(cid, 920)
        learned = read(out / 'learning/RESULT.json')
        if learned['status'] != 'COMPLETED':
            raise ValueError('Model extraction/learning incomplete')
        state['status'] = 'COMPLETED'
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        life.save(out / 'STOP_RECEIPT.json', life.stop_all())
        state.update(seconds=time.time() - state['started_at'], finished_at=time.time())
        state['reserved_card_seconds'] = state['seconds']
        state['cost_scope'] = 'One reserved card wall span, not GPU busy time; CPU replay accounted separately.'
        life.save(out / 'STATUS.json', state)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('replay', 'learn'))
    args = parser.parse_args()
    run = life.PROJECT / 'runs' / NAME
    life.deadline = time.time() + (600 if args.phase == 'replay' else 1000)
    lock = open(life.PROJECT / 'runs/full_os_v1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    reg = read(run / 'REGISTRATION.json')
    if reg['name'] != NAME or reg['online_hidden_facts'] is not False:
        raise ValueError('Bound phase registration missing')
    manifest = read(run / 'CODE_MANIFEST.json')
    for relative, expected in manifest.items():
        if sha(run / 'code' / relative) != expected:
            raise ValueError('Frozen campaign code changed')
    run_replays(run) if args.phase == 'replay' else run_learner(run)


if __name__ == '__main__':
    main()
