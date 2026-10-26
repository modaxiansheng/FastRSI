"""Materialize paid development evidence; no teacher-generated decision labels."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from fastrsi.decision_corpus import digest, project_input, checked_image, grouped_folds, training_examples, observed_relation


def build(root, output):
    root=root.resolve();output.mkdir(parents=True,exist_ok=False);started=time.perf_counter()
    sources={};inputs=[];metadata=[];outcomes=[];pairs=[];decisions=[];raw_count=0
    def bind(relative, expected=None):
        path=Path(relative)
        if path.is_absolute() or not (root/path).resolve().is_relative_to(root):raise ValueError('Source outside project')
        data=(root/path).read_bytes();h=hashlib.sha256(data).hexdigest()
        if expected is not None and h!=expected:raise ValueError('Source changed: '+str(path))
        sources[str(path)]=h;return data
    def read(relative, expected=None):return json.loads(bind(relative,expected))
    catalog=read('planning/budget_history_refresh_v1_20261003/CATALOG.json')
    episode_ids=set()
    for pair in catalog['records']:
        arm_rows=[]
        for arm in pair['arms']:
            episode=arm['path']
            if episode in episode_ids:raise ValueError('Duplicate episode')
            episode_ids.add(episode)
            for f,h in arm['sources'].items():bind(f,h)
            stream=Path(episode)/'stream';result=read(stream/'RESULT.json')
            if result['status']!='COMPLETED' or len(result['cases'])!=1:raise ValueError('Unexpected historical execution')
            row=result['cases'][0];case=row['case']
            if case['mode']!='development':raise ValueError('Evaluation data cannot enter corpus')
            log=stream/'OS_COST_EVENTS.jsonl';events=[json.loads(x) for x in bind(log).splitlines() if x]
            events=[e for e in events if e['event']=='actual_policy_input']
            trace_path=stream/'stage_trace'/case['id']/'TRACE.jsonl'
            trace=[json.loads(x) for x in bind(trace_path).splitlines() if x]
            if len(trace)!=len(events)+1 or trace[0]['method']!='reset' or any(r['method']!='action_chunk' for r in trace[1:]):
                raise ValueError('Cannot bind calls to pre-action frames: '+episode)
            task_group=case['suite']+'/'+digest(case['task_description'])[:16]
            episode_id=digest(episode)
            start_input=None
            for i,(event,frame) in enumerate(zip(events,trace)):
                raw_count+=1
                if event['case']!=case['id'] or frame['session_id']!=case['id']:raise ValueError('Cross-session source')
                projected,missing=project_input(event,arm['mode']);record_id=digest([episode,i]);images=[]
                if projected is not None:
                    b=projected['array_bindings']['observation/state']
                    if b['dtype']!='float32' or b['shape']!=[8] or hashlib.sha256(struct.pack('<8f',*projected['robot_state'])).hexdigest()!=b['sha256']:
                        raise ValueError('Robot state hash mismatch')
                    for camera,role in (('agentview_image','observation/image'),('robot0_eye_in_hand_image','observation/wrist_image')):
                        filename=f"{frame['step']:04d}_{camera}.png";relative=trace_path.parent/filename
                        blob=bind(relative,frame['images'][filename])
                        pixel_sha=checked_image(blob,frame['images'][filename],projected['array_bindings'][role])
                        images.append(dict(path=str(relative),role=role,sha256=sources[str(relative)],pixel_sha256=pixel_sha))
                    projected['elapsed_steps']=frame['step']
                    signature=digest({k:v for k,v in projected.items() if k!='current_execution_rule'})
                    if i==0:start_input=signature
                else:signature=None
                inputs.append(dict(record_id=record_id,model_input=projected,images=images))
                metadata.append(dict(record_id=record_id,episode_id=episode_id,source=episode,source_event_index=i,
                    source_log_sha256=sources[str(log)],source_trace_sha256=sources[str(trace_path)],
                    task_group=task_group,instance_group=pair['independent_instance_key'],
                    observation_id=frame['observation_id'],step=frame['step'],input_signature=signature,
                    input_status='COMPLETE_EXACT_PIXELS_AND_STATE' if projected is not None else 'INCOMPLETE_LEGACY_INPUT',
                    missing=missing,usage='seen_development_only'))
                decisions.append(dict(record_id=record_id,label=None,label_status='UNLABELLED',
                    reason='Recorded continuation and final success do not establish the best alternative decision'))
            end_events=[json.loads(x) for x in (root/log).read_text().splitlines()]
            counts=Counter(e['name'] for e in end_events if e['event']=='end')
            outcome=dict(episode_id=episode_id,source=episode,task_group=task_group,instance_group=pair['independent_instance_key'],
                mode=arm['mode'],native_success=row['environment_success'],semantic_verdicts=[o['verdict'] for o in row['outcomes']],
                admission=arm['admission'],steps=arm['steps'],vla_calls=arm['vla_calls'],call_counts=dict(counts),
                full_wall_seconds=arm['full_wall_seconds'],protocol_group=pair['spec']['protocol_group'],
                current_cost_compatible=pair['cost_compatible_with_current_continuous_F'],
                semantic_alignment=arm['semantic_alignment'],source_result_sha256=sources[str(stream/'RESULT.json')])
            outcomes.append(outcome);arm_rows.append(dict(episode_id=episode_id,mode=arm['mode'],success=outcome['native_success'],initial_input_signature=start_input))
        if len(arm_rows)!=2:raise ValueError('Expected paid two-arm comparison')
        pairs.append(dict(pair_id=digest(pair['evidence_id']),source=pair['evidence_id'],instance_group=pair['independent_instance_key'],
            arms=arm_rows,observed_episode_relation=observed_relation(arm_rows[0]['success'],arm_rows[1]['success']),
            initial_inputs_fully_bound=all(a['initial_input_signature'] for a in arm_rows),
            exact_pre_action_input_match=bool(arm_rows[0]['initial_input_signature'] and arm_rows[0]['initial_input_signature']==arm_rows[1]['initial_input_signature']),
            legacy_initial_images_match=pair['same_initial_images'],models_match=pair['same_model_config'],
            current_cost_compatible=pair['cost_compatible_with_current_continuous_F'],repeat_of=pair['repeat_of'],
            decision_label=None,scope='Whole-episode execution-window comparison only; not continue/inspect/escalate labels'))
    # Retain actual teacher requests independently, with their original null labels.
    request_base=Path('planning/decision_validity_v1_20261004/requests_v2')
    manifest=read(request_base/'MANIFEST.json')
    proposals=read('runs/decision_validity_teacher_v2_20261004/proposals/RESULT.json')
    by_name={r['input']:r for r in proposals['rows']};teachers=[]
    for entry in manifest['records']:
        request=read(request_base/entry['file'],entry['sha256'])
        for img in request['images']:
            path=Path(img['path']);relative=path.relative_to(root);bind(relative,img['sha256'])
        teachers.append(dict(source=str(request_base/entry['file']),request_sha256=entry['sha256'],
            proposed_choice=by_name[entry['file']]['proposal']['choice'],label=None,label_status='TEACHER_PROPOSAL_ONLY',
            request=request['request'],evidence='Original request and response retained; no executed alternative outcome'))
    labels=training_examples(inputs,decisions)
    complete=[r for r in metadata if r['input_signature']]
    summary=dict(status='BUILT',raw_pre_action_records=raw_count,complete_input_records=len(complete),
        incomplete_input_records=raw_count-len(complete),unique_complete_input_signatures=len({r['input_signature'] for r in complete}),
        paid_episodes=len(outcomes),distinct_initial_conditions=len({r['instance_group'] for r in metadata}),
        task_groups=len({r['task_group'] for r in metadata}),complete_input_task_groups=len({r['task_group'] for r in complete}),
        paired_episode_comparisons=len(pairs),exact_bound_initial_pairs=sum(p['exact_pre_action_input_match'] for p in pairs),
        current_cost_compatible_pairs=sum(p['current_cost_compatible'] for p in pairs),
        paired_relations=dict(Counter(r['observed_episode_relation'] for r in pairs)),
        validated_three_way_decision_labels=len(labels),teacher_proposals=len(teachers),new_model_calls=0,new_robot_episodes=0,
        learning_gain_established=False,measured_saved_seconds=None,wall_seconds=time.perf_counter()-started,
        scope='Source-bound retrospective development corpus. No unexecuted counterfactual, no random-frame final-test claim, no training or runtime deployment.')
    for filename,records in (('INPUTS.jsonl',inputs),('METADATA.jsonl',metadata),('EPISODE_OUTCOMES.jsonl',outcomes),
                             ('PAIRS.jsonl',pairs),('DECISION_LABELS.jsonl',decisions),('TEACHER_PROPOSALS.jsonl',teachers)):
        with (output/filename).open('x') as f:
            for record in records:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    for filename,value in (('SUMMARY.json',summary),('SOURCES.json',sources),('DEVELOPMENT_FOLDS.json',grouped_folds(metadata))):
        (output/filename).write_text(json.dumps(value,indent=2,ensure_ascii=False))
    (output/'MANIFEST.json').write_text(json.dumps({f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in output.iterdir() if f.is_file()},indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();print(json.dumps(build(a.root,a.output),indent=2))
