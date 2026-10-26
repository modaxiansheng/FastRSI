"""CPU-only first-intervention screen for a frozen baseline root-session trace."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fastrsi.action_memory import ActionMemory, load_bound
from fastrsi.prefix_screen import bound_observation, first_intervention


def screen(run, session):
    start = time.perf_counter()
    run = run.resolve()
    stream = run / 'N' / 'stream'
    trace_dir = (stream / 'stage_trace' / session).resolve()
    if trace_dir.parent != (stream / 'stage_trace').resolve():
        raise ValueError('Invalid session path')
    protocol = json.loads((run / 'PROTOCOL.json').read_text())
    artifact = load_bound(run / 'MEMORY.json', protocol['action_memory_probe']['sha256'])
    sources = {}

    def recorded(path):
        data = path.read_bytes()
        sources[str(path.relative_to(run))] = hashlib.sha256(data).hexdigest()
        return [json.loads(line) for line in data.splitlines() if line.strip()]

    traces = recorded(trace_dir / 'TRACE.jsonl')
    indexed = {t['observation_id']: t for t in traces}
    if len(indexed) != len(traces):
        raise ValueError('Duplicate observation IDs')
    events = recorded(stream / 'OS_COST_EVENTS.jsonl')

    def records():
        seen = set()
        previous_step = -1
        for event in events:
            if event.get('event') != 'action_memory_input' or event['runtime_session'] != session:
                continue
            oid = event['observation_id']
            trace = indexed[oid]
            if oid in seen or trace['step'] <= previous_step:
                raise ValueError('Non-monotone or duplicate policy input')
            seen.add(oid)
            previous_step = trace['step']

            def image_bytes(name):
                path = trace_dir / name
                blob = path.read_bytes()
                sources[str(path.relative_to(run))] = hashlib.sha256(blob).hexdigest()
                return blob

            obs = bound_observation(event, trace, image_bytes)
            yield dict(observation_id=oid, step=trace['step']), obs
        if not seen:
            raise ValueError('No bound policy inputs')

    result = first_intervention(records(), ActionMemory(artifact))
    sources['MEMORY.json'] = protocol['action_memory_probe']['sha256']
    sources['PROTOCOL.json'] = hashlib.sha256((run / 'PROTOCOL.json').read_bytes()).hexdigest()
    result.update(seconds=time.perf_counter() - start, source_sha256=sources,
                  scope='Retrospective baseline root prefix only; no new execution, training or counterfactual outcome.',
                  new_robot_steps=0, new_model_calls=0, session=session)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('session')
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = screen(args.run, args.session)
    with args.output.open('x') as f:
        json.dump(result, f, indent=2)
    print(json.dumps({k: v for k, v in result.items() if k != 'source_sha256'}, indent=2))
