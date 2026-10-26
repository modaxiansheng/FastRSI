"""Read complete native results without turning one transfer into a skill claim."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def analyze(run):
    status=read(run/'STATUS.json')
    if status['status']!='COMPLETED':raise ValueError('Incomplete paired run')
    artifact=read(run/'PREFIX.json');arms={};inputs={};windows={};cases={};inheritance={}
    for arm in ('N','F'):
        stream=run/arm/'stream';report=read(stream/'RESULT.json')
        if report['status']!='COMPLETED' or len(report['cases'])!=1:
            raise ValueError('Expected one completed case')
        row=report['cases'][0];receipt=read(stream/'PREFIX_RECEIPT.json')
        cases[arm]=row
        inheritance[arm]=read(stream/row['case']['id']/'INHERITANCE.json')
        if not inheritance[arm]['preserved_before_agent']:
            raise ValueError('Experience was not preserved')
        events=[json.loads(s) for s in (stream/'OS_COST_EVENTS.jsonl').read_text().splitlines()]
        inputs[arm]=[e for e in events if e.get('event')=='actual_policy_input']
        windows[arm]=[e for e in events if e.get('event')=='feedback_window_choice']
        choices=[e for e in events if e.get('event')=='prefix_memory_choice']
        native=[e for e in events if e.get('event')=='native_policy_request']
        stages=defaultdict(float)
        for e in events:
            if e.get('event')=='end':stages[e['name']]+=e['duration_seconds']
        if len(inputs[arm])!=receipt['requests'] or len(windows[arm])!=receipt['requests']:
            raise ValueError('Input/action accounting mismatch')
        if len(native)!=receipt['native_policy_requests'] or sum(c['reuse'] for c in choices)!=receipt['reused_prefixes']:
            raise ValueError('Reuse/request accounting mismatch')
        expected=0 if arm=='N' else 1
        if receipt['reused_prefixes']!=expected:raise ValueError('Intervention not exercised as declared')
        if any(w['mode']!='fixed10' for w in windows[arm]):raise ValueError('Execution window changed')
        if any(w['step']!=10*i for i,w in enumerate(windows[arm])):
            raise ValueError('Unexpected reset/replan or step sequence')
        arms[arm]=dict(environment_success=row['environment_success'],chain_status=row['status'],
            semantic_verdicts=[o.get('verdict') for o in row['outcomes']],
            admission=row['experience_admission'],
            steps=sum(t['steps'] for t in row['native_terminal_by_session'].values()),
            policy_boundaries=receipt['requests'],native_vla_requests=len(native),
            reused_prefixes=receipt['reused_prefixes'],
            agent_calls=len(list((stream/row['case']['id']).glob('agent_call_*'))),
            case_seconds=row['seconds'],controller_seconds=report['seconds'],
            startup_seconds=status['workers'][arm]['startup_seconds'],
            full_wall_seconds=status['workers'][arm]['full_wall_seconds'],
            stage_seconds=dict(stages),
            selection_seconds=sum(c['selection_seconds'] for c in choices))
    if inputs['N'][0]['array_bindings']!=inputs['F'][0]['array_bindings']:
        raise ValueError('Paired initial inputs differ')
    for key in ('suite','task_id','init_state_id','task_description','max_steps','mode'):
        if cases['N']['case'][key]!=cases['F']['case'][key]:
            raise ValueError('Task/protocol differs: '+key)
    for key in ('source_seal_sha256','inherited_learning_files'):
        if inheritance['N'][key]!=inheritance['F'][key]:
            raise ValueError('Initial experience differs')
    if windows['F'][0]['predicted_actions']!=artifact['actions']:
        raise ValueError('Stored prefix not submitted unchanged')
    if inputs['N'][0]['array_bindings']==artifact['source_policy_input']['array_bindings']:
        raise ValueError('No transfer to a changed input was tested')
    n,f=arms['N'],arms['F']
    return dict(status='EVIDENCE_BOUND',arms=arms,
        same_initial_policy_input=True,stored_prefix_used_exactly=True,source_input_differs=True,
        batch_wall_seconds=status['seconds'],
        native_vla_request_difference=n['native_vla_requests']-f['native_vla_requests'],
        case_seconds_difference=n['case_seconds']-f['case_seconds'],
        full_arm_seconds_difference=n['full_wall_seconds']-f['full_wall_seconds'],
        learned_applicability=False,learning_net_benefit_established=False,default_changed=False,
        rng_caveat=read(run/'PROTOCOL.json')['rng_caveat'],
        scope='One paired full-method development intervention, not independent generalization or noninferiority. Summed stage spans overlap and must not be added as wall time.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('run');ap.add_argument('output');a=ap.parse_args()
    run=Path(a.run);result=analyze(run)
    result['sources']={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(run.rglob('*')) if p.is_file() and 'code' not in p.relative_to(run).parts
        and p.name not in ('tokenizer.model',)}
    with open(a.output,'x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps({k:v for k,v in result.items() if k not in ('sources','arms')},indent=2))


if __name__=='__main__':main()
