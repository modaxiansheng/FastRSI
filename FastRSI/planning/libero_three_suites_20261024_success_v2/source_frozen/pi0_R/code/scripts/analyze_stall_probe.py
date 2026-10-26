"""Actual recovery consequences; distinguish actuator effects from task benefit."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


def read(p):return json.loads(p.read_text())
def lines(p):return [json.loads(s) for s in p.read_text().splitlines()]


def analyze(run):
    status=read(run/'STATUS.json')
    if status['status']!='COMPLETED':raise ValueError('Incomplete capability comparison')
    data={}; rows={}; traces={}; inputs={}; events_by={}
    for arm in ('N','F'):
        stream=run/arm/'stream';result=read(stream/'RESULT.json')
        if result['status']!='COMPLETED' or len(result['cases'])!=1:raise ValueError('Bad full OS result')
        row=result['cases'][0];rows[arm]=row;receipt=read(stream/'STALL_RECEIPT.json')
        events=lines(stream/'OS_COST_EVENTS.jsonl');events_by[arm]=events
        choices=[e for e in events if e['event']=='stall_recovery_choice']
        inputs[arm]=[e for e in events if e['event']=='actual_policy_input']
        windows=[e for e in events if e['event']=='feedback_window_choice']
        if len(inputs[arm])!=receipt['policy_requests'] or len(choices)!=len(inputs[arm]):
            raise ValueError('Inference count mismatch')
        if any(e['mode']!='fixed10' for e in windows):raise ValueError('Window changed')
        if len(windows)!=len(inputs[arm]):raise ValueError('Policy/action boundary mismatch')
        if sum(c['applied'] for c in choices)!=receipt['replaced_chunks']:
            raise ValueError('Replacement mismatch')
        if arm=='N' and receipt['replaced_chunks']:raise ValueError('Control was changed')
        traces[arm]=lines(stream/'stage_trace'/row['case']['id']/'TRACE.jsonl')
        if len(traces[arm])!=len(inputs[arm])+1:raise ValueError('Unexpected reset or missing action trace')
        inheritance=read(stream/row['case']['id']/'INHERITANCE.json')
        if not inheritance['preserved_before_agent']:raise ValueError('Experience was not preserved')
        seconds=defaultdict(float)
        for e in events:
            if e['event']=='end':seconds[e['name']]+=e['duration_seconds']
        data[arm]=dict(native_success=row['environment_success'],
            semantic_verdicts=[o.get('verdict') for o in row['outcomes']],
            chain_status=row['status'],admission=row['experience_admission']['status'],
            steps=traces[arm][-1]['step'],policy_requests=receipt['policy_requests'],
            replacements=receipt['replaced_chunks'],trigger_step=receipt['trigger_step'],
            case_seconds=row['seconds'],worker_seconds=status['workers'][arm]['full_wall_seconds'],
            startup_seconds=status['workers'][arm]['startup_seconds'],stage_seconds=dict(seconds),
            agent_calls=len(list((stream/row['case']['id']).glob('agent_call_*'))),
            selector_seconds=sum(c['selector_seconds'] for c in choices),
            experience_seal=inheritance['source_seal_sha256'])
    for key in ('suite','task_id','init_state_id','task_description','mode','max_steps'):
        if rows['N']['case'][key]!=rows['F']['case'][key]:raise ValueError('Different task/protocol')
    if data['N']['experience_seal']!=data['F']['experience_seal']:raise ValueError('Different initial experience')
    trigger=data['F']['trigger_step']
    until=trigger if trigger is not None else min(data['N']['steps'],data['F']['steps'])
    # Includes the input at intervention time, never the changed output at that time.
    checked=0
    for n,f in zip(traces['N'],traces['F']):
        if n['step']>until:break
        if n!=f:raise ValueError('Before-intervention physical trajectory differs')
        checked+=1
    for n,f in zip(inputs['N'][:checked],inputs['F'][:checked]):
        if n['array_bindings']!=f['array_bindings']:raise ValueError('Before-intervention policy input differs')
    effect={}
    if trigger is not None:
        by_step={e['step']:i for i,e in enumerate([e for e in events_by['F'] if e['event']=='stall_recovery_choice'])}
        for step in (trigger,trigger+10,trigger+20,trigger+30):
            if step in by_step:effect[str(step)]=inputs['F'][by_step[step]]['robot_state']
        if str(trigger+30) in effect:
            effect['net_eef_displacement_m']=math.dist(effect[str(trigger)][:3],effect[str(trigger+30)][:3])
    f=data['F'];n=data['N']
    feasible=(trigger is not None and f['native_success'] and
        f['semantic_verdicts']==['success'] and f['chain_status']=='PASSED' and f['admission']=='ADMITTED')
    return dict(status='ANALYZED',arms=data,pre_intervention_trace_boundaries_equal=checked,
        physical_effect=effect,capability_candidate='FEASIBLE_DEVELOPMENT' if feasible else 'NOT_ACCEPTED',
        policy_request_difference=n['policy_requests']-f['policy_requests'],
        action_step_difference=n['steps']-f['steps'],batch_wall_seconds=status['seconds'],
        learned_applicability=False,learning_net_gain_established=False,
        interpretation='A manually proposed recovery capability comparison on one seen failure. No training, held-out transfer or default promotion. Timings include overlaps; stage spans are not additive.',
        source_sha256={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (run/'STATUS.json',run/'PROTOCOL.json',run/'STALL.json',
                      run/'N/stream/RESULT.json',run/'F/stream/RESULT.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    result=analyze(a.run)
    with a.output.open('x') as f:json.dump(result,f,indent=2,ensure_ascii=False)
    print(json.dumps(result,indent=2,ensure_ascii=False))
