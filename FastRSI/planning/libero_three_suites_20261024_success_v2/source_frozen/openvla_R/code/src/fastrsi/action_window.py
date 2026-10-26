"""Development probe: execute five predicted actions before observing again.

This is a fixed action-level control candidate, not a learned policy or a
change to diffusion sampling. No hidden task predicates are consumed.
"""
import copy
import hashlib
import json
import re
from pathlib import Path
from fastrsi.placement_recovery import GOAL


def proposal(model_sha):
    return {'version':1,'operation':'execute_prefix','steps':5,'goal':GOAL,
            'status':'PROPOSED','scope':{'suite':'libero_10','task_id':4,'max_steps':520,'model_sha':model_sha},
            'origin':'Manually specified feedback-frequency feasibility probe; no learned discovery.', 'evidence':[]}


def validate(state):
    if (state.get('version'),state.get('operation'),state.get('steps'),state.get('goal'))!=(1,'execute_prefix',5,GOAL):
        raise ValueError('Unbound action-window candidate')
    if state.get('status') not in ('PROPOSED','ACCEPTED_DEVELOPMENT','REJECTED'):
        raise ValueError('Unknown candidate status')
    if state['status']=='ACCEPTED_DEVELOPMENT' and not state.get('evidence'):
        raise ValueError('Missing development evidence')
    return state


def decision(state,case,model_sha,probe=False):
    validate(state)
    if case.get('task_description')!=GOAL or state['scope']['model_sha']!=model_sha or any(case.get(k)!=state['scope'][k] for k in ('suite','task_id','max_steps')):
        return 'native'
    if probe:
        if case['mode']!='development':raise ValueError('Evaluation cannot probe an unaccepted candidate')
        return 'short_window'
    return 'short_window' if state['status']=='ACCEPTED_DEVELOPMENT' else 'native'


def apply(chunk,context,session_id,choice):
    if choice!='short_window' or context.get('session_id')!=session_id or context.get('task_description')!=GOAL:
        return chunk,None
    match=re.fullmatch(r'libero_obs_(\d+)',str(context.get('source_observation_id','')))
    if not match:raise ValueError('Cannot bind control action to current observation')
    step=int(match.group(1))
    if not 0<=step<520:raise ValueError('Action outside step budget')
    count=min(5,len(chunk['actions']),520-step)
    if count<1:raise ValueError('Empty policy action buffer')
    out={**chunk,'actions':chunk['actions'][:count].copy(),
         'timing':{**chunk['timing'],'horizon_steps':count},
         'action_contract':{**chunk['action_contract'],'shape':[count,7]}}
    return out,{'event':'action_window_choice','step':step,'original_steps':len(chunk['actions']),
                'submitted_steps':count,'source_observation_id':context['source_observation_id'],
                'scope':'Current task only; original prompt and action values preserved.'}


def update(state,native,candidate):
    validate(state)
    for r in (native,candidate):
        if r['mode']!='development' or r['goal']!=GOAL or r['steps']>520 or r['resets']!=1 or not r.get('sources'):
            raise ValueError('Unbound development feedback')
    if not native['initial_images'] or native['initial_images']!=candidate['initial_images']:
        raise ValueError('Initial state mismatch')
    good=all(r['chain_passed'] and r['admitted'] for r in (native,candidate))
    accept=good and not native['success'] and candidate['success'] and candidate['verdict']=='success'
    out=copy.deepcopy(state);out['status']='ACCEPTED_DEVELOPMENT' if accept else 'REJECTED'
    out['evidence'].append({'native':native,'candidate':candidate})
    out['acceptance_scope']='One known-development capability screen, no general quality or net benefit claim.'
    return out


def load(path,sha):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Candidate changed')
    return validate(json.loads(data))
