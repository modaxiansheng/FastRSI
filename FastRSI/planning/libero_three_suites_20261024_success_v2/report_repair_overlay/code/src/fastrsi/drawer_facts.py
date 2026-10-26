"""Concrete observable questions for a candidate place-then-close skill.

Facts are model estimates, not native goal truth. No predicate values from the
simulator enter these requests. The proposed skill is not yet accepted.
"""
from .decision_validity import validate_choice

GOAL = 'put the black bowl in the bottom drawer of the cabinet and close it'
DESCRIPTIONS = {
    'drawer_open': 'The bottom drawer of the white cabinet is visibly pulled open, with its interior accessible.',
    'bowl_inside': 'The black bowl is placed inside the bottom drawer, rather than beside it, above it in the gripper, or on the tabletop.',
    'drawer_closed': 'The bottom drawer is fully pushed in, rather than still pulled out or only partly closed.',
}
OPTIONS = {'true': 'The current evidence supports the stated fact.',
           'false': 'The current evidence contradicts the stated fact.',
           'unknown': 'Occlusion, missing depth or ambiguous evidence prevents a reliable judgment.'}


def request(state):
    allowed={'goal','observation_id','robot_state','execution_window_rule'}
    if set(state)!=allowed or state['goal']!=GOAL or len(state['robot_state'])!=8:
        raise ValueError('Unbound factual input')
    return dict(model='clef-flash',state=dict(state),questions={name:dict(type='choice',
        instructions=description+' Judge the current front and wrist images. The task instruction is not evidence that it has been achieved.',
        criteria=dict(OPTIONS)) for name,description in DESCRIPTIONS.items()})


def encode(encoder,processor,record):
    if set(record['questions'])!=set(DESCRIPTIONS):raise ValueError('Missing factual questions')
    if any(q['criteria']!=OPTIONS for q in record['questions'].values()):raise ValueError('Changed choices')
    encoded=encoder(processor.tokenizer,record,max_length=1000000,processor=processor)
    if len(encoded.input_ids)>8192:raise ValueError('Full input too long; never truncate')
    return encoded


def validate(response):
    answers=response.get('answers',{})
    if set(answers)!=set(DESCRIPTIONS):raise ValueError('Missing or extra facts')
    for answer in answers.values():validate_choice(answer,list(OPTIONS))
    return response


def proposed_operation(facts):
    """Diagnostic-only mapping; not authority to modify a robot's instruction.

    Containment and even a positive placement answer do not independently bind
    release or clearance. Exact replay shows containment while still grasped.
    This interface cannot authorize closing from its three current questions.
    """
    if set(facts)!=set(DESCRIPTIONS) or any(v not in OPTIONS for v in facts.values()):
        raise ValueError('Invalid fact values')
    if facts['drawer_open']=='true' and facts['drawer_closed']=='true':return 'native_unresolved'
    if facts['bowl_inside']=='true' and facts['drawer_closed']=='false':return 'needs_release_and_clearance_evidence'
    return 'native_continue'


class InspectionLedger:
    """A repeated unresolved question on identical evidence cannot buy a new call.

    This is shared deterministic accounting, not a learned sparsity result.
    A changed image/state hash is new evidence. The caller owns episode scope.
    """
    def __init__(self,max_calls):
        if type(max_calls) is not int or max_calls<1:raise ValueError('Invalid inspection budget')
        self.max_calls=max_calls;self.seen=set();self.calls=0

    def request(self,evidence_sha,question_version):
        key=(evidence_sha,question_version)
        if key in self.seen:return 'reuse_or_native'
        if self.calls>=self.max_calls:return 'native_budget_exhausted'
        self.seen.add(key);self.calls+=1
        return 'inspect_once'
