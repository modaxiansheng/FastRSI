"""Two CPU-only exact replays of the finished operation-value pair."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import time
import traceback
import orchestrate_full_os as life

NAME = 'remaining_goal_physical_audit_v1_20261006'
SOURCE = 'remaining_goal_value_probe_v1_20261006'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def pp(p): return str(p).replace(str(life.PROJECT), '/project')


def main():
    lock = open(life.PROJECT / 'runs/full_os_v1.lock', 'a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = life.PROJECT / 'runs' / NAME; run.mkdir(exist_ok=False)
    started = time.time(); life.deadline = started + 360
    state = dict(status='RUNNING', started_at=started, new_robot_episodes=0,
                 new_model_calls=0, verified_training_labels=0, authority='PHYSICAL_DIAGNOSIS_ONLY')
    workers = []
    try:
        life.gate(run)
        for name in ('replay_object_stage_supervision_v2.py', 'stage_trace.py'):
            dst = run / 'code/scripts' / name; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(life.PROJECT / 'scripts' / name, dst)
        life.save(run / 'CODE_MANIFEST.json', {str(p.relative_to(run / 'code')): sha(p)
                   for p in (run / 'code').rglob('*') if p.is_file()})
        sources = []; pending = []
        for arm in ('N', 'F'):
            stream = life.PROJECT / 'runs' / SOURCE / arm / 'stream'
            result = read(stream / 'RESULT.json')
            if result['status'] != 'COMPLETED' or len(result['cases']) != 1:
                raise ValueError('Incomplete paired source')
            row = result['cases'][0]
            if row['chain_status'] != 'PASSED': raise ValueError('Source chain failed')
            trace = stream / 'stage_trace' / row['case']['id'] / 'TRACE.jsonl'
            sources.append(dict(arm=arm, stream=pp(stream), trace=pp(trace),
                                result_sha256=sha(stream / 'RESULT.json'), trace_sha256=sha(trace)))
            out = run / arm; out.mkdir()
            old = read(stream / 'LAUNCH_target.json')['argv']
            image = next(i for i, v in enumerate(old) if v.startswith('sha256:'))
            prefix = []; i = 0
            while i < image:
                value = old[i]
                if value == '-e' and old[i + 1].startswith('FASTRSI_'): i += 2; continue
                if value == '-e' and old[i + 1].startswith('PYTHONPATH='):
                    prefix += ['-e', 'PYTHONPATH=/workspace/openpi/third_party/libero:' + pp(run / 'code/scripts')]
                    i += 2; continue
                if value == '--name':
                    prefix += [value, 'yuchen-fastrsi-' + NAME + '-' + arm.lower()]; i += 2; continue
                if value == '--network': prefix += [value, 'none']; i += 2; continue
                if value == '-v' and old[i + 1] == str(life.PROJECT) + ':/project:rw':
                    prefix += [value, str(life.PROJECT) + ':/project:ro']; i += 2; continue
                prefix.append(value); i += 1
            argv = prefix + ['-v', str(out) + ':/output:rw', old[image], 'timeout', '300', 'python',
                pp(run / 'code/scripts/replay_object_stage_supervision_v2.py'), '--trace', pp(trace),
                '--result', pp(stream / 'RESULT.json'), '--init', '26', '--diagnostic-only', '--output', '/output/REPLAY.json']
            pending.append((arm, out, argv))
        life.save(run / 'REGISTRATION.json', dict(sources=sources, new_model_calls=0,
                  new_policy_episodes=0, learning_disabled=True, max_exact_replays=2))
        for arm, out, argv in pending:
            workers.append((arm, out, life.launch('replay', argv, out)))
        for arm, out, cid in workers:
            life.wait_exit(cid, 320)
            replay = read(out / 'REPLAY.json')
            if replay['status'] != 'EXACT_IMAGE_REPLAY': raise ValueError('Replay did not match source')
        state['status'] = 'COMPLETED'
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc()); raise
    finally:
        life.save(run / 'STOP_RECEIPT.json', life.stop_all())
        state.update(seconds=time.time() - started, finished_at=time.time(),
                     cost_scope='Two overlapping CPU replays of old actions; no policy/GPU/model inference. Project diagnostic expense, not speedup.')
        life.save(run / 'STATUS.json', state); print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
