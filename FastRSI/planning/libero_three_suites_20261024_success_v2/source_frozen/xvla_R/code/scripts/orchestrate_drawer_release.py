"""Three bounded CPU-only exact replays. Preserve every source trajectory."""
import fcntl
import json
import time
import traceback
import orchestrate_full_os as life

NAME = 'drawer_release_v1_20261004'


def main():
    run = life.PROJECT / 'runs' / NAME
    lock = open(life.PROJECT / 'runs/full_os_v1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert not (run / 'STATUS.json').exists(), 'Already dispatched'
    life.deadline = time.time() + 360
    started = time.time()
    status = dict(status='RUNNING', started_at=started, policy_rollouts=0, model_calls=0, training=False)
    life.save(run / 'STATUS.json', status)
    workers = []
    try:
        life.gate(run)
        for arm in ('N', 'F', 'R'):
            source = life.PROJECT / 'runs/short_exit_v1_20261003/pair1' / arm / 'stream'
            folder = run / arm
            folder.mkdir()
            old = json.loads((source / 'LAUNCH_target.json').read_text())['argv']
            image = next(x for x in old if x.startswith('sha256:'))
            argv = old[:old.index(image)]
            argv[argv.index('--name') + 1] = 'yuchen-fastrsi-' + NAME + '-' + arm.lower()
            cleaned = []
            i = 0
            while i < len(argv):
                if argv[i] == '-e' and argv[i + 1].startswith('FASTRSI_'):
                    i += 2
                    continue
                cleaned.append(argv[i])
                i += 1
            project_path = lambda p: str(p).replace(str(life.PROJECT), '/project')
            command = cleaned + [image, 'timeout', '300', 'python',
                project_path(run / 'code/scripts/replay_drawer_release.py'),
                '--trace', project_path(source / 'stage_trace/net_t3_i38/TRACE.jsonl'),
                '--output', project_path(folder / 'REPLAY.json')]
            workers.append(life.launch('replay', command, folder))
        for cid in workers:
            life.wait_exit(cid, 310)
        status['status'] = 'COMPLETED'
    except BaseException:
        status.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        life.save(run / 'STOP_RECEIPT.json', life.stop_all())
        status.update(seconds=time.time() - started, finished_at=time.time())
        life.save(run / 'STATUS.json', status)
        print(json.dumps(status), flush=True)


if __name__ == '__main__':
    main()
