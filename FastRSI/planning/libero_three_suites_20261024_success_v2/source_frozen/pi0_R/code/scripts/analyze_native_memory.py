"""Classify a memory intervention with the frozen SessionSpec, not raw JSON diffs."""
import argparse
import hashlib
import json
from pathlib import Path
from PhyAgentOS.runtime.schemas import SessionSpec


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project-root',default='/project')
    ap.add_argument('--run-id',default='native_memory_v3_20261002');ap.add_argument('--out',required=True)
    args=ap.parse_args();root=Path(args.project_root);run=root/'runs'/args.run_id;stream=run/'stream'
    load=lambda p:json.loads(p.read_text())
    expected=SessionSpec.model_validate(load(stream/'fresh_session/workspace/REQUEST.json')['session']).model_dump(mode='json')
    rows={}
    for label in ['with_memory_1','without_memory','with_memory_2']:
        source=stream/'decision_counterfactual'/(label+'.json');row=load(source)
        assert len(row['decision'])==1 and row['decision'][0]['name']=='append_yaml_item'
        actual=SessionSpec.model_validate(row['decision'][0]['arguments']['item']).model_dump(mode='json')
        differences={k:{'expected':v,'actual':actual[k]} for k,v in expected.items() if v!=actual[k]}
        rows[label]={'native_session_differences':differences,'would_pass_fixed_session_validation':not differences,
            'source':str(source.relative_to(root)),'sha256':hashlib.sha256(source.read_bytes()).hexdigest()}
    controls=load(stream/'decision_counterfactual/RESULT.json')['controls_match_actual_decision']
    events=[json.loads(l) for l in (stream/'fresh_session/AGENT_EVENTS.jsonl').read_text().splitlines()]
    rejected=[e for e in events if e['event']=='queue_not_accepted']
    edits=[e for e in events if e['event']=='tool' and e.get('name')=='edit_file']
    final=SessionSpec.model_validate(load(stream/'fresh_session/RESULT.json')['planned_session']).model_dump(mode='json')
    assert controls and final==expected
    assert rows['with_memory_1']['native_session_differences'].keys()=={'updated_at'}
    assert rows['without_memory']['would_pass_fixed_session_validation']
    assert len(rejected)==1 and any('updated_at' in e['params'].get('old_text','') for e in edits)
    spans=[json.loads(l) for l in (stream/'OS_COST_EVENTS.jsonl').read_text().splitlines()]
    calls=[s for s in spans if s['event']=='end' and s['name']=='agent_model_call']
    assert len(calls)==8
    result={'status':'VERIFIED_NEGATIVE_DECISION_EFFECT','run_id':args.run_id,
        'native_memory_formed':load(stream/'rollover/RESULT.json')['status']=='PASSED',
        'new_session_actually_received_memory':load(stream/'fresh_session/RESULT.json')['actual_memory_in_model_request'],
        'controls_match_actual_rejected_draft':controls,'counterfactuals':rows,
        'observed_rejections':rejected,'observed_repair_tool':edits,
        'observed_repair_model_call_seconds':calls[6]['duration_seconds'],
        'final_corrected_session_equals_request':True,
        'interpretation':'In this saved decision, native long-term memory caused an extra updated_at value. The real Agent draft was rejected and required an edit. Removing only long-term memory produced a SessionSpec matching the request; both memory-present controls reproduce the rejected draft.',
        'limits':['One decision, not a population-level effect.','Replay drafts were not dispatched to the robot.','Only long-term memory was ablated; other native feedback remained.','The exact responsible sentence was not separately ablated.','The repair-call duration is observed work, not a measured full-run counterfactual saving.']}
    out=Path(args.out);out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps({'status':result['status'],'repair_model_call_seconds':result['observed_repair_model_call_seconds']},indent=2))


if __name__=='__main__':main()
