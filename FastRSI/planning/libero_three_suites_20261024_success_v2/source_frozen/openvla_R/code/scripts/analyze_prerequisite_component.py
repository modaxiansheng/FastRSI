"""Bind a component-only intervention to its full-goal and offline replay outcomes."""
import collections
import hashlib
import json
from pathlib import Path


def main():
    p=Path(__file__).resolve().parents[1]
    native=p/'runs/prerequisite_native_v2_20261004'
    run=p/'runs/prerequisite_component_v1_20261004'
    replay_run=p/'runs/prerequisite_component_progress_v1_20261004'
    stream=run/'N/stream'
    read=lambda f:json.loads(f.read_text())
    lines=lambda f:[json.loads(x) for x in f.read_text().splitlines()]
    result=read(stream/'RESULT.json')
    assert result['status']=='FAILED' and 'validity_observer.summary()' in result['error']
    case=read(run/'N/ROSTER.json')[0]
    folder=stream/case['id'];workspace=folder/'workspace'
    evidence_file=next((workspace/'reports/evidence').glob('*.json'))
    evidence=read(evidence_file)
    sessions=next(s for s in evidence['sources'] if s['path']=='SESSIONS.md')
    assert hashlib.sha256((workspace/'SESSIONS.md').read_bytes()).hexdigest()==sessions['raw_sha256']
    session=next(s for s in sessions['data'] if s['session_id']==case['id'])
    verification=session['result']['metadata']['verification']['attempts']
    episode_file=workspace/'artifacts/runtime'/case['id']/'episode.json'
    episode=read(episode_file)
    events=lines(stream/'OS_COST_EVENTS.jsonl')
    old_events=lines(native/'N/stream/OS_COST_EVENTS.jsonl')
    inputs=[e for e in events if e['event']=='actual_policy_input']
    old_inputs=[e for e in old_events if e['event']=='actual_policy_input']
    overrides=[e for e in events if e['event']=='prerequisite_component_input']
    trace=lines(stream/'stage_trace'/case['id']/'TRACE.jsonl')
    old_trace=lines(native/'N/stream/stage_trace'/case['id']/'TRACE.jsonl')
    replay=read(replay_run/'REPLAY.json'); assert replay['status']=='EXACT_IMAGE_REPLAY'
    assert trace[0]['images']==old_trace[0]['images']
    assert inputs[0]['array_bindings']==old_inputs[0]['array_bindings']
    assert (run/'MODEL_CONFIG.json').read_bytes()==(native/'MODEL_CONFIG.json').read_bytes()
    for file in ('MODEL_BINDING.json','CORE_SOURCE_BINDING.json'):
        assert (run/file).read_bytes()==(native/file).read_bytes()
    for f in (native/'initial_experience').rglob('*'):
        if f.is_file():assert f.read_bytes()==(run/'initial_experience'/f.relative_to(native/'initial_experience')).read_bytes()
    assert inputs and len(inputs)==len(overrides)
    assert all(e['prompt']=='open the top drawer of the cabinet' for e in inputs)
    assert read(folder/'PLANNED_SESSION.json')['task_description']=='open the top drawer of the cabinet and put the bowl in it'
    choices=[e for e in events if e['event']=='feedback_window_choice']
    assert choices and all(e['mode']=='fixed10' for e in choices)
    spans=collections.defaultdict(float); counts=collections.Counter()
    for e in events:
        if e['event']=='end':spans[e['name']]+=e['duration_seconds'];counts[e['name']]+=1
    progress=[dict(predicate=g['predicate'],observed_true_steps=[r['step'] for r in replay['rows'] if r['predicates'][i]['value']])
              for i,g in enumerate(replay['rows'][0]['predicates'])]
    source_files=[stream/'RESULT.json',stream/'OS_COST_EVENTS.jsonl',stream/'stage_trace'/case['id']/'TRACE.jsonl',
                  replay_run/'REPLAY.json',run/'STATUS.json',run/'N/STATUS.json',replay_run/'STATUS.json',
                  evidence_file,workspace/'SESSIONS.md',episode_file,folder/'AGENT_EVENTS.jsonl']
    out=dict(scope='Component-only feasibility, not full task strategy comparison or learned gain',
             binding='PASSED: identical initial pixels/state, model config and accepted initial experience',
             case=case,whole_task_success=episode['success'],verification_status=[r['details']['verifier_status'] for r in verification],
             verdicts=[r['verdict'] for r in verification],chain_status='FAILED_OPTIONAL_HOOK_CLEANUP',
             admission='NOT_REACHED',root_steps=trace[-1]['step'],
             vla_calls=len(inputs),component_override_calls=len(overrides),goal_progress=progress,
             exact_replay_frames=len(replay['rows']),spans_seconds_nested_not_additive=dict(spans),span_counts=dict(counts),
             run_status=read(run/'STATUS.json'),worker_status=read(run/'N/STATUS.json'),replay_status=read(replay_run/'STATUS.json'),
             new_accepted_skills=0,new_model_updates=0,
             sources={str(f.relative_to(p)):hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files})
    dest=p/'planning/prerequisite_component_v1_20261004';dest.mkdir(exist_ok=True)
    with (dest/'ANALYSIS.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps({k:out[k] for k in ('binding','whole_task_success','chain_status','root_steps','vla_calls','goal_progress','spans_seconds_nested_not_additive')},indent=2))


if __name__=='__main__':main()
