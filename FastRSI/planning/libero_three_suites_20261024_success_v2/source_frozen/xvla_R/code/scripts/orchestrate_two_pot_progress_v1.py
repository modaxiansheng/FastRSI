"""Two sequential CPU exact replays of existing successful development traces."""
import fcntl
import hashlib
import json
import shutil
import time
import traceback
import orchestrate_full_os as life


def main():
    project = life.PROJECT
    registration = project/'planning/two_pot_progress_v1_20261007/REGISTRATION.json'
    reg = json.loads(registration.read_text())
    for rel, digest in reg['code_bindings'].items():
        assert hashlib.sha256((project/rel).read_bytes()).hexdigest() == digest
    lock = (project/'runs/full_os_v1.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = project/'runs/two_pot_progress_v1_20261007'
    run.mkdir(exist_ok=False)
    started = time.time()
    life.deadline = started+400
    state = dict(status='RUNNING', started_at=started)
    life.save(run/'STATUS.json', state)
    try:
        life.gate(run)
        shutil.copyfile(registration, run/'REGISTRATION.json')
        for rel in ('scripts/replay_two_pot_progress_v1.py', 'scripts/stage_trace.py'):
            dest = run/'code'/rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(project/rel, dest)
        for source in reg['sources']:
            stream = project/source['stream']
            for rel, digest in source['bindings'].items():
                assert hashlib.sha256((project/rel).read_bytes()).hexdigest() == digest
            group = run/source['id']
            group.mkdir()
            old = json.loads((stream/'LAUNCH_target.json').read_text())['argv']
            index = next(i for i, value in enumerate(old) if value.startswith('sha256:'))
            argv = old[:index]
            argv[argv.index('--name')+1] = 'yuchen-fastrsi-pot-progress-v1-'+source['id']
            clean, i = [], 0
            while i < len(argv):
                if argv[i] == '-e' and argv[i+1].startswith('FASTRSI_'):
                    i += 2
                    continue
                clean.append(argv[i])
                i += 1
            base = str(run).replace(str(project), '/project')
            cid = life.launch('replay', clean+[old[index], 'timeout', '180', 'python',
                base+'/code/scripts/replay_two_pot_progress_v1.py', '--stream',
                str(stream).replace(str(project), '/project'), '--output',
                base+'/'+source['id']+'/REPLAY.json'], group)
            life.wait_exit(cid, 185)
            life.save(group/'STOP_RECEIPT.json', life.stop_all())
            assert json.loads((group/'REPLAY.json').read_text())['status'] == 'EXACT_IMAGE_REPLAY'
        state['status'] = 'COMPLETED'
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc())
        raise
    finally:
        life.save(run/'STOP_RECEIPT.json', life.stop_all())
        state.update(seconds=time.time()-started, finished_at=time.time())
        life.save(run/'STATUS.json', state)
        print(json.dumps(state), flush=True)


if __name__ == '__main__':
    main()
