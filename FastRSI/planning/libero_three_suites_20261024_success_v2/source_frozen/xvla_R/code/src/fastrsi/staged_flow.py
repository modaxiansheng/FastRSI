"""One task-scoped, paid stage-flow probe; original goal and scorer stay intact."""
import copy
import hashlib
import json
import re
from pathlib import Path

GOAL = 'put the white mug on the left plate and put the yellow and white mug on the right plate'
FIRST = 'put the white mug on the left plate'
SPLIT = 260


def proposal(model_sha):
    return dict(version=1, status='PROPOSED', goal=GOAL, first=FIRST, split=SPLIT,
                scope=dict(suite='libero_10', task_id=4, max_steps=520, model_sha=model_sha),
                origin='Human-authored staged flow motivated by observed task4/init31 failure; not learned discovery.',
                evidence=[])


def validate(state):
    if (state.get('version'),state.get('goal'),state.get('first'),state.get('split')) != (1,GOAL,FIRST,SPLIT):
        raise ValueError('Unbound stage flow')
    if state.get('status') not in ('PROPOSED','ACCEPTED_DEVELOPMENT','REJECTED'):
        raise ValueError('Unknown flow status')
    if state['status']=='ACCEPTED_DEVELOPMENT' and not state.get('evidence'):
        raise ValueError('No development evidence')
    return state


def decision(state, case, model_sha, probe=False):
    validate(state)
    if (case.get('task_description') != GOAL or state['scope']['model_sha'] != model_sha
        or any(case.get(k)!=state['scope'][k] for k in ('suite','task_id','max_steps'))):
        return 'native'
    if probe:
        if case['mode']!='development':raise ValueError('Unaccepted probe cannot enter evaluation')
        return 'staged'
    return 'staged' if state['status']=='ACCEPTED_DEVELOPMENT' else 'native'


def policy_input(payload, observation, context, session_id, action):
    """Only already-visible elapsed steps decide phase; no hidden subgoal labels."""
    if action!='staged' or context.get('session_id')!=session_id or context.get('task_description')!=GOAL:
        return payload, None
    obs_id=str(observation.get('observation_id',''))
    matched=re.fullmatch(r'libero_obs_(\d+)',obs_id)
    if not matched:return payload, {'reason':'unknown_observation_id','action':'native'}
    step=int(matched.group(1))
    if step>=520:return payload, {'reason':'outside_step_scope','action':'native','step':step}
    result=dict(payload);result['prompt']=FIRST if step<SPLIT else GOAL
    return result, {'action':'staged','step':step,'phase':'white_first' if step<SPLIT else 'full_goal',
                    'prompt':result['prompt'],'original_goal':GOAL,'observation_id':obs_id}


def update(state, native, candidate):
    validate(state)
    for row in (native,candidate):
        if row['mode']!='development':raise ValueError('No evaluation feedback')
        if row['goal']!=GOAL or row['steps']>520 or row['resets']!=1:raise ValueError('Goal/budget changed')
        if not row.get('sources'):raise ValueError('Missing provenance')
    if native['initial_images']!=candidate['initial_images'] or not native['initial_images']:
        raise ValueError('Initial state mismatch')
    good=all(r['chain_passed'] and r['admitted'] for r in (native,candidate))
    # First screen is capability improvement, not a post-hoc 1% timing threshold.
    accept=good and not native['success'] and candidate['success'] and candidate['verdict']=='success'
    new=copy.deepcopy(state);new['status']='ACCEPTED_DEVELOPMENT' if accept else 'REJECTED'
    new['evidence'].append({'native':native,'candidate':candidate})
    new['acceptance_scope']='One paid matched development win; tentative reuse, not generalized skill proof.'
    return new


def load(path, sha):
    data=Path(path).read_bytes()
    if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Flow artifact changed')
    return validate(json.loads(data))
