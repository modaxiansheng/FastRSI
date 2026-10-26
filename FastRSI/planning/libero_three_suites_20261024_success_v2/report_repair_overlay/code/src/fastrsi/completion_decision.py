"""Experimental visual completion decision; never approves OS memory or execution.

Native development predicates supervise a completion estimate, not narrative
correctness, recovery quality, or the best next action. Unknown is a fallback.
"""
import hashlib
import json

OPTIONS = {'complete': 'All conditions of the given task are currently satisfied.',
           'incomplete': 'At least one required condition is currently not satisfied.',
           'unknown': 'The available views do not support a reliable completion judgment.'}


def request(goal):
    if not isinstance(goal, str) or not goal.strip(): raise ValueError('Missing task')
    return {'model':'clef-flash','state':{'goal':goal},'questions':{'completion':{
        'type':'choice','instructions':'Judge the task from the current front and wrist images. '
        'The instruction is a goal, not evidence of success. Occlusion is not success. '
        'Do not impose extra unstated requirements.', 'criteria':dict(OPTIONS)}}}


def encode(encoder, processor, record):
    if set(record) != {'model','state','questions','images'} or set(record['state']) != {'goal'}:
        raise ValueError('Unexpected model input; labels and metadata are forbidden')
    expected=request(record['state']['goal'])
    if any(record[k]!=expected[k] for k in ('model','state','questions')) or len(record['images'])!=2:
        raise ValueError('Changed visual decision contract')
    value=encoder(processor.tokenizer,record,max_length=1000000,processor=processor)
    if len(value.input_ids)>8192:raise ValueError('Reject rather than truncate evidence')
    return value


def partition(groups):
    """Per goal: first init seeds, last init tests, intermediate inits update.

    Ordered by observed acquisition time, never by success or measured cost.
    Singleton goals provide training only. All arms/repeats stay in one group.
    """
    by_goal={}
    for r in groups:
        by_goal.setdefault(r['goal'],[]).append(r)
    result={}
    for goal,items in by_goal.items():
        items=sorted(items,key=lambda r:(r['observed_at'],r['group']))
        if len({r['group'] for r in items})!=len(items):raise ValueError('Duplicate initial group')
        for index,r in enumerate(items):
            result[r['group']]='seed' if index==0 else 'test' if index==len(items)-1 else 'update'
    return result


def signature(goal, images):
    return hashlib.sha256(json.dumps([goal,images],sort_keys=True).encode()).hexdigest()


def route(probability, threshold=.99):
    if type(probability) not in (float,int) or not 0<=probability<=1:
        raise ValueError('Invalid completion estimate')
    return 'propose_complete' if probability>=threshold else 'escalate_qwen'
