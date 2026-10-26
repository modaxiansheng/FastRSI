"""One bounded instruction-experience decision; no LLM, VLA training or queue writer."""
import hashlib
import json
import math
from pathlib import Path

GOAL='put both the alphabet soup and the tomato sauce in the basket'
CANDIDATE='put the alphabet soup in the basket and put the tomato sauce in the basket'
REL='skills/instruction-reuse/EXPERIENCE.json'


def proposal(source, model_config_sha):
    from .experience_admission import validate_snapshot, SEAL
    seal=validate_snapshot(source)
    # Provenance for the task, not validation of the new wording.
    if seal['source_case']!='native_long_t0_i8':raise ValueError('Unexpected task source')
    return {'version':1,'status':'PROPOSED','original_goal':GOAL,'candidate_instruction':CANDIDATE,
            'scope':{'suite':'libero_10','task_id':0,'max_steps':520,'model_config_sha256':model_config_sha},
            'source_case':seal['source_case'],'source_seal_sha256':hashlib.sha256((Path(source)/SEAL).read_bytes()).hexdigest(),
            'origin':'Engineering-authored equivalent conjunction expansion. Historical success validates original task only, not this candidate.',
            'goal_predicates':['alphabet soup in basket','tomato sauce in basket'],
            'preference':'native','observations':[],'screening_min_savings_fraction':0.05}


def validate(state):
    if state['version']!=1 or state['original_goal']!=GOAL or state['candidate_instruction']!=CANDIDATE:
        raise ValueError('Unbound instruction or goal')
    if state['status'] not in ('PROPOSED','VALIDATED_DEVELOPMENT','REJECTED'):raise ValueError('Unknown status')
    if state['preference'] not in ('native','reuse'):raise ValueError('Unknown preference')
    if state['preference']=='reuse' and (state['status']!='VALIDATED_DEVELOPMENT' or not state['observations']):
        raise ValueError('Cannot reuse unvalidated experience')
    return state


def select(state, case, config_sha):
    validate(state)
    matches=case['task_description']==GOAL and config_sha==state['scope']['model_config_sha256'] and all(case.get(k)==state['scope'][k] for k in ('suite','task_id','max_steps'))
    if not matches:return {'action':'native','instruction':case['task_description'],'reason':'outside_validated_scope','version':state['version']}
    action='reuse' if state['status']=='VALIDATED_DEVELOPMENT' and state['preference']=='reuse' else 'native'
    return {'action':action,'instruction':CANDIDATE if action=='reuse' else GOAL,'reason':'paid_development_evidence' if action=='reuse' else 'no_validated_cost_advantage','version':state['version']}


def update(state, rows):
    """Only paid matched development observations; evaluation is never ingested."""
    validate(state)
    if set(rows)!= {'native','candidate'}:raise ValueError('Both observed actions required')
    for action,r in rows.items():
        if r['mode']!='development':raise ValueError('Evaluation cannot update policy')
        if r['goal']!=GOAL or r['instruction']!=(GOAL if action=='native' else CANDIDATE):raise ValueError('Goal or action mismatch')
        if not isinstance(r['seconds'],(int,float)) or not math.isfinite(r['seconds']) or r['seconds']<=0:raise ValueError('Missing cost')
    a,b=rows['native'],rows['candidate']
    if a['initial_images']!=b['initial_images'] or not a['initial_images'] or a['init_state_id']!=b['init_state_id']:
        raise ValueError('Development pair not aligned')
    good=all(r['chain_passed'] and r['feedback_admitted'] and r['environment_success'] is True and r['verification_success'] and r['prompt_verified'] for r in rows.values())
    new=json.loads(json.dumps(state));new['status']='VALIDATED_DEVELOPMENT' if good else 'REJECTED'
    new['observations'].append(rows)
    savings=1-b['seconds']/a['seconds']
    new['preference']='reuse' if good and savings>=state['screening_min_savings_fraction'] else 'native'
    new['observed_savings_fraction']=savings
    new['scope_note']='One matched development pair; tentative task-specific evidence, not statistical quality or generalization proof.'
    new['acquisition_controller_seconds']=a['seconds']+b['seconds']
    return new


def install(workspace, artifact, expected_sha):
    data=Path(artifact).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected_sha:raise ValueError('Experience artifact changed')
    state=validate(json.loads(data));p=Path(workspace)/REL;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data)
    (p.parent/'SKILL.md').write_text('---\nname: instruction-reuse\ndescription: Inspect task-scoped instruction experience and its measured cost before reuse.\n---\nRead EXPERIENCE.json in this skill directory. It distinguishes a proposed candidate from validated development evidence. Never infer new object positions or change the task goal. Unknown scope uses native planning.\n')
    return state


def intent_request(spec, goal, decision):
    return {'goal':goal,'session_id':spec['session_id'],'goal_id':spec.get('goal_id'),'benchmark':spec['benchmark'],
            'execution_constraints':{k:spec[k] for k in ['execution','timeouts','safety_profile','verification_profile']},
            'instruction_decision':decision,'experience_file':REL,
            'construction':'Construct the Session using native benchmarking skill and current registries. No complete Session answer is provided.'}
