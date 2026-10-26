"""Exact paid-development replay; hidden predicates are diagnosis, not policy input."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import time
from stage_trace import png_bytes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stream', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    roster = json.loads((args.stream.parent/'ROSTER.json').read_text())
    assert len(roster) == 1
    case = roster[0]
    assert case['mode'] == 'development' and case['suite'] == 'libero_10'
    assert case['task_id'] == 8 and case['init_state_id'] in (14, 47)
    assert case['task_description'] == 'put both moka pots on the stove'
    assert case['max_steps'] == 520 and case['feedback_window_mode'] == 'fixed10'
    trace = args.stream/'stage_trace'/case['id']/'TRACE.jsonl'
    raw = trace.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines()]
    assert sum(row['method'] == 'reset' for row in rows) == 1
    spec = importlib.util.spec_from_file_location('runtime', '/core/PhyAgentOS/runtime/targets/remote/libero/server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    runtime = module.LiberoRealRuntime(dict(benchmark_name='libero_10', task_id=8,
        init_state_id=case['init_state_id'], seed=7, max_steps=520, control_mode='relative'))
    started = time.time()
    result = dict(status='RUNNING', case=case, trace_sha256=hashlib.sha256(raw).hexdigest(),
        scope='Offline native predicate diagnostic only; no model, alternate action, or learning labels', rows=[])
    try:
        for row in rows:
            if row['method'] == 'reset':
                runtime.reset({'session_id': row['session_id']})
            else:
                assert row['method'] == 'action_chunk'
                runtime.action_chunk({'actions': row['actions']})
            assert runtime.step_idx == row['step']
            for camera in ('agentview_image', 'robot0_eye_in_hand_image'):
                digest = hashlib.sha256(png_bytes(runtime._last_obs[camera])).hexdigest()
                assert digest == row['images'][f"{row['step']:04d}_{camera}.png"]
            env = runtime.env.env
            predicates = [dict(predicate=g, value=bool(env._eval_predicate(g)))
                          for g in env.parsed_problem['goal_state']]
            result['rows'].append(dict(step=row['step'], images=row['images'], predicates=predicates,
                native_success=bool(runtime.env.check_success())))
        assert result['rows'][-1]['native_success']
        result['status'] = 'EXACT_IMAGE_REPLAY'
    except BaseException as error:
        result.update(status='FAILED', error=repr(error))
        raise
    finally:
        result['seconds'] = time.time()-started
        with args.output.open('x') as handle:
            json.dump(result, handle, indent=2)
        if runtime.env is not None:
            runtime.env.close()


if __name__ == '__main__':
    main()
