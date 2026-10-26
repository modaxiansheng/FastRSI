"""Source-bound numeric feedback, independent from narrative admission."""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fastrsi.window_learning import empty, update


def read(p):
    return json.loads(p.read_text())


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def collect(root, group, goal_binding):
    result_path=group/'stream/RESULT.json'
    result=read(result_path); row=result['cases'][0]; c=row['case']
    status_path=group/'STATUS.json'; status=read(status_path)
    assert result['status']=='COMPLETED' and status['status']=='COMPLETED'
    assert c['mode']=='development' and c['suite']=='libero_10' and row['chain_status']=='PASSED'
    assert len(result['cases'])==1 and len(row['native_terminal_by_session'])==1
    ep_path=group/'stream'/c['id']/'workspace/artifacts/runtime'/c['id']/'episode.json'
    ep=read(ep_path); b=ep['benchmark']; native=row['native_terminal_by_session'][c['id']]
    assert b['task_description']==c['task_description'] and b['task_id']==c['task_id'] and b['init_state_id']==c['init_state_id']
    assert b['success']==ep['success']==native['native_success']==row['environment_success']
    assert b['target_step_index']==ep['num_steps']==native['steps']
    events_path=group/'stream/OS_COST_EVENTS.jsonl'
    events=[json.loads(line) for line in events_path.read_text().splitlines() if line.strip()]
    choices=[e for e in events if e.get('event')=='feedback_window_choice']
    inputs=[e for e in events if e.get('event')=='actual_policy_input']
    assert len(inputs)==len(choices)==b['num_action_chunks'] and choices
    assert all(e['prompt']==c['task_description'] for e in inputs)
    policy=row.get('window_learning_decision',{}).get('selected_mode',c['feedback_window_mode'])
    assert all(e['mode']==policy for e in choices)
    binding=read(root/goal_binding)
    audited_goals={2:'491835cc2eb6956f7ce3d1ee4e266377116168fe453e6529fe105e3b0333635d',
                   3:'5255fe54d7f25fad4dee8fa30a30033d8cb908f1708d953c91ab609264fb4fb8'}
    assert binding['sha256']==audited_goals[c['task_id']]
    expected={2:'turn on the stove and put the moka pot on it',3:'put the black bowl in the bottom drawer of the cabinet and close it'}
    assert expected[c['task_id']]==c['task_description']
    files=[result_path,status_path,ep_path,events_path,root/goal_binding]
    images=sorted((group/'stream'/c['id']/'workspace/artifacts/runtime'/c['id']/'rgb').glob('initial_*.png'))
    assert images
    model=group/'MODEL_CONFIG.json'
    signature={'goal_sha':binding['sha256'],'task':c['task_id'],'init':c['init_state_id'],
               'max_steps':c['max_steps'],'model_config_sha':digest(model),
               'initial_images':{p.name:digest(p) for p in images},
               'seed':ep['metadata']['final_status']['libero']['seed']}
    files += [model]+images
    return dict(evidence_id=str(group.relative_to(root)), mode=c['mode'], policy=policy,
                block_id=str(group.parent.relative_to(root)),
                comparison_signature=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),
                task=c['task_id'], init=c['init_state_id'], goal=c['task_description'],
                goal_definition_sha256=binding['sha256'], native_goal_success=b['success'],
                steps=native['steps'], vla_calls=len(inputs), full_wall_seconds=status['full_wall_seconds'],
                semantic_verdicts=[o.get('verdict') for o in row['outcomes']],
                native_experience_admission=row['experience_admission']['status'],
                chain_status=row['chain_status'], native_facts_consistent=True,
                sources={str(p.relative_to(root)):digest(p) for p in files})


def prepare(root, out):
    prior=root/'planning/feedback_window_structures_v1_20261003/VERIFIED_EPISODE_FACTS.json'
    original=read(prior); records=[]
    for row in original['records']:
        for rel, sha in row['sources'].items():
            assert digest(root/rel)==sha, 'Historical source changed: '+rel
        result=next(rel for rel in row['sources'] if rel.endswith('/stream/RESULT.json'))
        binding=next(rel for rel in row['sources'] if rel.endswith('_GOAL_BINDING.json'))
        record=collect(root,(root/result).parent.parent,binding)
        assert record['native_goal_success']==row['native_goal_success'] and record['vla_calls']==row['vla_calls']
        records.append(record)
    initial=empty(); trained=update(initial,records,phase='development')
    out.mkdir(parents=True,exist_ok=True)
    for name,value in [('FROZEN.json',initial),('LEARNED.json',trained)]:
        (out/name).write_text(json.dumps(value,indent=2))
    return {'historical_source_sha256':digest(prior),'episodes':len(records),
            'paid_history_lifecycle_seconds':sum(r['full_wall_seconds'] for r in records),
            'frozen_sha256':digest(out/'FROZEN.json'),'learned_sha256':digest(out/'LEARNED.json'),
            'scope':'Mechanism diagnostic using paid historical development data; not a cold-start comparison',
            'narrative_admission_changed':False}


def prepare_paired(root,out):
    from fastrsi.paired_window_learning import empty as paired_empty, update as paired_update
    initial_receipt=prepare(root,out/'original_history')
    first=read(out/'original_history/LEARNED.json')['records']
    state=paired_update(paired_empty(),first,phase='development')
    # Fixed, predeclared historical batches; never look at new-run results here.
    additions=[]
    bindings={2:'planning/feedback_window_structures_v1_20261003/STOVE_GOAL_BINDING.json',
              3:'planning/feedback_window_drawer_retry_v1_20261003/DRAWER_GOAL_BINDING.json'}
    for index,task in [(0,2),(1,3)]:
        for arm in ('N','F','R'):
            additions.append(collect(root,root/f'runs/window_learning_v1_20261003/pair{index}'/arm,bindings[task]))
    before=state
    state=paired_update(state,additions,phase='development')
    for name in ('FROZEN.json','LEARNED.json'):(out/name).write_text(json.dumps(state,indent=2))
    (out/'HISTORICAL_UPDATE_AUDIT.json').write_text(json.dumps({'before':before['incumbent'],'after':state['incumbent'],'update':state['last_update']},indent=2))
    return {'episodes':len(state['records']),'paid_history_lifecycle_seconds':sum(r['full_wall_seconds'] for r in state['records']),
            'frozen_sha256':digest(out/'FROZEN.json'),'learned_sha256':digest(out/'LEARNED.json'),
            'initial_snapshots_identical':True,'initial_updates':state['updates'],
            'initial_incumbent':state['incumbent'],'scope':'Matched evidence diagnostic, common trained starting snapshot; history and probe costs retained',
            'narrative_admission_changed':False,'first_history_receipt':initial_receipt}
