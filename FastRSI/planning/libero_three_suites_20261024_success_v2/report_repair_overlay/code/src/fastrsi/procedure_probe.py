"""Bounded developer probe of official remaining subgoals, not a learned policy."""
import hashlib
import json

GOALS = {
  2: 'turn on the stove and put the moka pot on it',
  9: 'put the yellow and white mug in the microwave and close it',
}
FACTS = {2: ('stove_on', 'pot_on_stove'), 9: ('mug_inside', 'mug_released', 'door_closed')}
PROMPTS = {2: {'place_pot': 'put the moka pot on the stove', 'turn_on': 'turn on the stove'},
           9: {'close_door': 'close the microwave'}}
CHECK_STEPS = (100, 200, 300, 400)


def artifact(task):
    return {'version': 1, 'task_id': task, 'goal': GOALS[task],
            'facts': list(FACTS[task]), 'remaining_prompts': PROMPTS[task],
            'check_steps': list(CHECK_STEPS), 'valid_for_steps': 100,
            'status': 'UNVALIDATED_DEVELOPMENT_CANDIDATE'}


def bind(case, content, digest):
    if hashlib.sha256(content).hexdigest() != digest:
        raise ValueError('Changed procedure artifact')
    spec = json.loads(content)
    if spec != artifact(case['task_id']):
        raise ValueError('Unbound procedure definition')
    if (case['mode'] != 'development' or case['suite'] != 'libero_10'
        or case['task_description'] != spec['goal'] or case['max_steps'] != 520
        or case['feedback_window_mode'] != 'fixed10'):
        raise ValueError('Procedure only allowed in the bound development case')
    return spec


def choose(task, facts):
    if set(facts) != set(FACTS[task]) or any(v not in ('true', 'false', 'unknown') for v in facts.values()):
        raise ValueError('Illegal or incomplete facts')
    if task == 2:
        if facts == {'stove_on': 'true', 'pot_on_stove': 'false'}:return 'place_pot'
        if facts == {'stove_on': 'false', 'pot_on_stove': 'true'}:return 'turn_on'
    elif facts == {'mug_inside': 'true', 'mug_released': 'true', 'door_closed': 'false'}:
        return 'close_door'
    return 'full_goal'


class Controller:
    def __init__(self, spec):
        self.spec = spec
        self.done = set()
        self.choice = 'full_goal'
        self.expires = 0

    def needs_check(self, step):
        return step in self.spec['check_steps'] and step not in self.done

    def update(self, step, facts):
        if not self.needs_check(step):raise ValueError('Unscheduled or duplicate teacher query')
        self.done.add(step)
        self.choice = choose(self.spec['task_id'], facts)
        self.expires = step + self.spec['valid_for_steps']
        return self.choice

    def prompt(self, step):
        choice = self.choice if step < self.expires else 'full_goal'
        return self.spec['remaining_prompts'].get(choice, self.spec['goal']), choice
