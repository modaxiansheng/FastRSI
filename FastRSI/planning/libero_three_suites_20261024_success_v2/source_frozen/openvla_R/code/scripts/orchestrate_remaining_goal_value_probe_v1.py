"""One paired known-development operation-value diagnosis, no selector training."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import signal
import socket
import struct
import subprocess
import time
import traceback
import orchestrate_full_os as life

NAME = 'remaining_goal_value_probe_v1_20261006'
def read(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def tree(p): return {str(f.relative_to(p)): sha(f) for f in p.rglob('*') if f.is_file()}
def freeze(s, d): shutil.copytree(s, d, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))


def main():
    from fastrsi.decision_corpus import png_pixels
    from fastrsi.object_stage_memory_v1 import GOAL, REMAINING
    from fastrsi.remaining_goal_value_probe_v1 import AUTHORITY
    p = life.PROJECT; run = p / 'runs' / NAME
    lock = open(p / 'runs/full_os_v1.lock', 'a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run.mkdir(exist_ok=False); started = time.time(); life.deadline = started + 1200
    life.CAMPAIGN = run; life.BASE = '/project/runs/' + NAME
    workers = []; state = dict(status='PREPARING', started_at=started, authority=AUTHORITY)
    def interrupt(n, f): raise KeyboardInterrupt('Interrupted paired operation diagnosis ' + str(n))
    signal.signal(signal.SIGTERM, interrupt); signal.signal(signal.SIGINT, interrupt)
    try:
        life.gate(run)
        for port in [b + s for b in (49500, 49502, 49510, 49520) for s in (0, 100)]:
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1', port)) == 0: raise ValueError('Pair port occupied')
        source = p / 'runs/task4_informative_pair_dev_187_v8_20261006/native10a/init26'
        replay_path = p / 'runs/object_stage_memory_v2_20261006/replays/init26/REPLAY.json'
        replay = read(replay_path)
        if replay['status'] != 'EXACT_IMAGE_REPLAY' or replay['case']['mode'] != 'development':
            raise ValueError('No verified developmental source')
        frame = next(r for r in replay['rows'] if r['step'] == 120)
        facts = frame['offline_supervision']
        if not facts['white_mug']['released_support_proxy'] or facts['yellow_white_mug']['native_subgoal']:
            raise ValueError('Prerecorded developer boundary does not have the proposed remaining object')
        # These private facts select a developer intervention; they never enter the runner/model.
        life.save(run / 'OFFLINE_DIAGNOSTIC_BASIS.json', dict(source=str(replay_path.relative_to(p)),
            sha256=sha(replay_path), facts=facts, authority='Retrospective developer design only; not online evidence'))
        for rel in ('scripts', 'src/fastrsi'): freeze(p / rel, run / 'code' / rel)
        for rel in ('MODEL_CONFIG.json', 'FLOW.json', 'EXPERIENCE_BINDING.json'):
            shutil.copyfile(source / rel, run / rel)
        freeze(source / 'initial_experience', run / 'initial_experience')
        if tree(run / 'initial_experience') != read(run / 'EXPERIENCE_BINDING.json'):
            raise ValueError('Initial fixed experience changed')
        prefix = {}
        folder = source / 'F/stream/stage_trace/net_t4_i26'
        events_path = source / 'F/stream/OS_COST_EVENTS.jsonl'
        policy_events = [json.loads(line) for line in events_path.read_text().splitlines()
                         if json.loads(line).get('event') == 'actual_policy_input']
        if len(policy_events) != 23:
            raise ValueError('Registered source policy-call sequence changed')
        for row in replay['rows']:
            if row['step'] > 120: break
            actual = policy_events[row['step'] // 10]
            if actual.get('case') != 'net_t4_i26' or actual.get('prompt') != GOAL:
                raise ValueError('Source call belongs to another original task')
            arrays = {}
            for camera, key in [('agentview_image', 'observation/image'), ('robot0_eye_in_hand_image', 'observation/wrist_image')]:
                name = f'{row["step"]:04d}_{camera}.png'; path = folder / name
                if sha(path) != row['images'][name]: raise ValueError('Saved prefix pixels changed')
                pixels, shape = png_pixels(path.read_bytes())
                arrays[key] = dict(sha256=hashlib.sha256(pixels).hexdigest(), shape=shape, dtype='uint8')
            # Replay annotation uses a different quaternion conversion. Bind the
            # exact recorded public model input, never approximate replay robot8.
            arrays['observation/state'] = dict(sha256=hashlib.sha256(struct.pack('<8f', *actual['robot_state'])).hexdigest(), shape=[8], dtype='float32')
            if arrays != actual['array_bindings']:
                raise ValueError('Source pixels or exact public state differ from actual policy request')
            prefix[str(row['step'])] = arrays
        life.save(run / 'SOURCE_INPUT_BINDING.json', dict(
            events_source=str(events_path.relative_to(p)), events_sha256=sha(events_path),
            replay_source=str(replay_path.relative_to(p)), replay_sha256=sha(replay_path),
            steps=list(range(0, 121, 10)), exact_actual_array_bindings=True,
            replay_robot8_used_as_policy_input=False))
        life.save(run / 'MANUAL_REMAINING.json', dict(version=1, authority=AUTHORITY, start_step=120,
            end_step=220, remaining_goal=REMAINING['white_mug'], model_sha256=sha(run / 'MODEL_CONFIG.json'), prefix_bindings=prefix))
        case = dict(suite='libero_10', task_id=4, init_state_id=26, mode='development', max_steps=520,
            native_experience_maintenance=True, task_description=GOAL)
        life.save(run / 'PLAN.json', dict(cases=[case]))
        life.save(run / 'PROTOCOL.json', dict(slot=0, absolute_deadline=life.deadline, diagnostic_trace=True,
            control_kind='feedback_window', arm_modes=dict(N='fixed10', F='fixed10'),
            terminal_evidence_view=True, planning_entries=dict(N='public_session_flow', F='public_session_flow'),
            resource_scope='two cards per arm, four cards in parallel; equal seeds/models/limits',
            scope='Known developer fixed intervention 120<=step<220; no visual/model stage authorization or learned benefit claim',
            stopping='One pair only. No boundary, duration or wording sweep; no quality regression or no actual work reduction stops this control point.'))
        worker = run / 'code/scripts/run_placement_worker.py'
        text = worker.read_text()
        needle = "        case['terminal_prefetch']=False"
        if text.count(needle) != 1: raise ValueError('Worker injection point changed')
        text = text.replace(needle, needle + "\n        manual=campaign/'MANUAL_REMAINING.json'\n        case['remaining_goal_value_probe']=dict(artifact=str(manual).replace(str(life.PROJECT),'/project'),sha256=hashlib.sha256(manual.read_bytes()).hexdigest(),enabled=arm=='F')")
        needle = "runner_script='run_native_baseline.py'"
        if text.count(needle) != 1: raise ValueError('Runner selection point changed')
        worker.write_text(text.replace(needle, "runner_script='run_remaining_goal_value_probe_v1.py'"))
        life.save(run / 'CODE_MANIFEST.json', tree(run / 'code'))
        for arm in ('N', 'F'):
            group = run / arm; freeze(run / 'code', group / 'code')
            shift = 100 if arm == 'F' else 0
            if shift:
                for rel in ('scripts/run_native_baseline.py', 'scripts/qwen_agent_server.py'):
                    path = group / 'code' / rel; text = path.read_text()
                    for port in (49500, 49502, 49510, 49520): text = text.replace(str(port), str(port + shift))
                    path.write_text(text)
            life.save(group / 'CODE_MANIFEST.json', tree(group / 'code'))
            with (group / 'controller.log').open('x') as log:
                proc = subprocess.Popen(['python3', str(run / 'code/scripts/run_placement_worker.py'), NAME, arm],
                    stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            workers.append((arm, proc)); life.save(group / 'DISPATCH.json', dict(pid=proc.pid, started_at=time.time()))
        state.update(status='RUNNING', registration='PROTOCOL.json'); life.save(run / 'STATUS.json', state)
        while any(proc.poll() is None for _, proc in workers):
            if time.time() > life.deadline: raise TimeoutError('Single pair deadline')
            if any(proc.poll() not in (None, 0) for _, proc in workers): raise RuntimeError('One branch failed; retain both receipts')
            time.sleep(2)
        state['status'] = 'COMPLETED'
        state['arms'] = {arm: read(run / arm / 'STATUS.json') for arm, _ in workers}
    except BaseException:
        state.update(status='FAILED', error=traceback.format_exc()); raise
    finally:
        for _, proc in workers:
            if proc.poll() is None: proc.send_signal(signal.SIGTERM)
        for _, proc in workers:
            try: proc.wait(timeout=90)
            except subprocess.TimeoutExpired: state['cleanup_attention'] = 'Inspect exact owned worker IDs'
        state.update(seconds=time.time() - started, finished_at=time.time())
        life.save(run / 'STATUS.json', state); print(json.dumps(state), flush=True)


if __name__ == '__main__': main()
