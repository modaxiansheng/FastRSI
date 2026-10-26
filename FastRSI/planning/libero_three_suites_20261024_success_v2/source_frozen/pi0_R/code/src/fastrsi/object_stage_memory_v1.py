"""Task4 object-stage predictions and a private, opt-in remaining-goal memory.

Only legal RGB/proprio observations enter the model. Offline native predicates
may supervise a model elsewhere; this consumer never accepts those predicates.
Two consecutive estimates are a calibrated-experiment candidate, not a physical
certificate. Original H10/W10 execution and final task verification are retained.
"""
import copy
import hashlib
import json
import math

GOAL = 'put the white mug on the left plate and put the yellow and white mug on the right plate'
SUBJECTS = ('white_mug', 'yellow_white_mug')
ROLES = {
    'white_mug': 'the white mug and its left-plate target in the task instruction',
    'yellow_white_mug': 'the yellow and white mug and its right-plate target in the task instruction',
}
REMAINING = {
    'white_mug': 'put the yellow and white mug on the right plate',
    'yellow_white_mug': 'put the white mug on the left plate',
}
IMAGE_ORDER = ('previous_front', 'previous_wrist', 'current_front', 'current_wrist')
OPTIONS = {
    'complete': 'The correct mug is supported on its correct goal plate and released from the gripper.',
    'incomplete': 'At least one of correct object, correct target, support or release is visibly unsatisfied.',
    'unknown': 'Object/target identity, support or release is ambiguous or occluded in the available views.',
}
AUTHORITY = 'MODEL_PREDICTION_NOT_PHYSICAL_CERTIFICATE'
OBSERVATION_FIELDS = {'episode_id', 'observation_id', 'step', 'robot_state', 'images'}


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
        allow_nan=False).encode()).hexdigest()


def _version(value):
    if (not isinstance(value, str) or len(value) != 64 or
            any(c not in '0123456789abcdef' for c in value)):
        raise ValueError('An explicit model artifact SHA is required')
    return value


def _robot(value):
    if (not isinstance(value, (list, tuple)) or len(value) != 8 or
            any(type(x) not in (int, float) or not math.isfinite(x) for x in value)):
        raise ValueError('Only the existing finite public robot8 is allowed')
    return list(value)


def request(subject, robot_state, previous_robot_state):
    """One typed estimate, with images supplied separately in IMAGE_ORDER.

    Neither episode/init IDs nor outcome, predicate or object-pose fields can be
    added to this schema. Robot state and command signs never prove completion.
    """
    if subject not in SUBJECTS:
        raise ValueError('Unknown task4 object role')
    state = dict(goal=GOAL, subject=subject, robot_state=_robot(robot_state),
        previous_robot_state=_robot(previous_robot_state), execution_window_rule='H10/W10')
    instructions = (
        'Estimate CURRENT completion of ' + ROLES[subject] + '. '
        'Images are previous front, previous wrist, current front, current wrist. '
        'Use the previous views as context, not as evidence that a past fact persists. '
        'Complete requires correct mug identity, correct task-target identity, '
        'support on that plate, and release from the gripper. A mug merely above '
        'or inside a plate region while held is incomplete. Closed/open gripper '
        'commands, elapsed steps and arm lift do not prove release or completion. '
        'Do not replace task-target identity by left/right pixel position. '
        'Occluded or ambiguous identity, support or release requires unknown. '
        'This is a fallible model estimate, not native task truth.')
    return dict(model='clef-flash', state=state, questions={'stage': dict(
        type='choice', instructions=instructions, criteria=dict(OPTIONS))})


def encode(encoder, processor, record, max_tokens=8192):
    """Validate legal input before one full vendor encoding; never truncate."""
    if not isinstance(record, dict) or set(record) != {'model', 'state', 'questions', 'images'}:
        raise ValueError('Only model/state/questions/four images may be encoded')
    state = record['state']
    if not isinstance(state, dict) or set(state) != {
            'goal', 'subject', 'robot_state', 'previous_robot_state', 'execution_window_rule'}:
        raise ValueError('Labels, hidden poses and episode/init identifiers are forbidden')
    expected = request(state['subject'], state['robot_state'], state['previous_robot_state'])
    if any(record[k] != expected[k] for k in ('model', 'state', 'questions')):
        raise ValueError('Object-stage request or legal choices changed')
    images = record['images']
    if (not isinstance(images, list) or len(images) != 4 or
            any(not callable(getattr(im, 'convert', None)) for im in images)):
        raise ValueError('Four decoded images in declared previous/current order are required')
    if type(max_tokens) is not int or not 1 <= max_tokens <= 65536:
        raise ValueError('Invalid complete-input context budget')
    encoded = encoder(processor.tokenizer, record, max_length=1000000, processor=processor)
    if len(encoded.input_ids) > max_tokens:
        raise ValueError('Reject oversize complete evidence rather than silently truncating it')
    media = getattr(encoded, 'media', None)
    if not isinstance(media, dict) or not {'pixel_values', 'image_grid_thw'} <= set(media):
        raise ValueError('Vendor encoding omitted actual image tensors')
    grid, pixels = media['image_grid_thw'], media['pixel_values']
    if tuple(getattr(grid, 'shape', ())) != (4, 3):
        raise ValueError('Exactly four actual image grids are required')
    cells = grid.tolist()
    if any(len(row) != 3 or row[0] != 1 or any(type(v) is not int or v <= 0 for v in row)
           for row in cells):
        raise ValueError('Malformed or non-image vision grid')
    shape = tuple(getattr(pixels, 'shape', ()))
    if len(shape) != 2 or shape[0] != sum(t * h * w for t, h, w in cells) or shape[1] <= 0:
        raise ValueError('Image patches and grids do not match')
    return encoded


def validate_answer(response, subject):
    """Return one legal model label; this does not admit a skill or outcome."""
    if subject not in SUBJECTS or not isinstance(response, dict) or set(response) != {'answers'}:
        raise ValueError('One subject-bound typed model answer is required')
    answers = response['answers']
    if (not isinstance(answers, dict) or set(answers) != {'stage'} or
            not isinstance(answers['stage'], str) or answers['stage'] not in OPTIONS):
        raise ValueError('Missing, extra or illegal completion choice')
    return answers['stage']


def validate_observation(observation):
    """Private provenance IDs stay out of model requests; images are saved PNGs."""
    if not isinstance(observation, dict) or set(observation) != OBSERVATION_FIELDS:
        raise ValueError('Only current legal observation fields are allowed')
    sid, step = observation['episode_id'], observation['step']
    if not isinstance(sid, str) or not sid:
        raise ValueError('Missing episode identity')
    if type(step) is not int or not 0 <= step <= 520 or step % 10:
        raise ValueError('Existing task4 H10/W10 observation boundary required')
    if observation['observation_id'] != 'libero_obs_' + str(step):
        raise ValueError('Observation does not match its actual existing boundary')
    _robot(observation['robot_state'])
    images = observation['images']
    if (not isinstance(images, (list, tuple)) or len(images) != 2 or
            any(not isinstance(im, bytes) or not im.startswith(b'\x89PNG\r\n\x1a\n') for im in images)):
        raise ValueError('Current existing front/wrist PNG bytes required')
    return observation


def observation_binding(observation):
    validate_observation(observation)
    return dict(episode_id=observation['episode_id'], observation_id=observation['observation_id'],
        step=observation['step'], robot_state=_robot(observation['robot_state']),
        image_sha256=[hashlib.sha256(im).hexdigest() for im in observation['images']])


class ObjectStageMemory:
    """Task-local predicted conditions; trial_enabled is the only input mutation gate.

    The caller must record actual model requests/replies and charge their cost.
    An unknown current prediction invalidates a past completion. An omitted
    prediction does not count as a second positive and cannot extend its age.
    No persistent learning or benchmark-success decision happens in this class.
    """
    def __init__(self, episode_id, *, trial_enabled=False, max_age_steps=20):
        if not isinstance(episode_id, str) or not episode_id or type(trial_enabled) is not bool:
            raise ValueError('Explicit episode identity and boolean trial permission required')
        if type(max_age_steps) is not int or max_age_steps < 10 or max_age_steps > 60 or max_age_steps % 10:
            raise ValueError('Bounded whole-H10 evidence lifetime required')
        self.episode_id, self.trial_enabled = episode_id, trial_enabled
        self.max_age_steps, self.model_version, self.previous_step = max_age_steps, None, None
        self.objects = {subject: dict(version=0, consecutive=0, last_positive=None,
            contract=None, last_label='unknown', reason='no_model_evidence') for subject in SUBJECTS}
        self.last_receipt = None

    def _invalidate(self, subject, reason):
        item = self.objects[subject]
        if item['contract'] is not None or item['consecutive']:
            item['version'] += 1
        item.update(consecutive=0, last_positive=None, contract=None, last_label='unknown', reason=reason)

    def _prompt(self, step):
        active = []
        for subject, item in self.objects.items():
            contract = item['contract']
            if contract is not None and step >= contract['expires_step']:
                self._invalidate(subject, 'evidence_expired')
                contract = None
            if contract is not None:
                active.append(subject)
        if len(active) == 1:
            return REMAINING[active[0]], active
        # Both-complete estimates never skip final verification or terminate the task.
        return GOAL, active

    def observe(self, observation, predictions, *, model_version, invalidated=()):
        validate_observation(observation)
        if observation['episode_id'] != self.episode_id:
            raise ValueError('Object memory cannot cross episode/reset identities')
        step = observation['step']
        if self.previous_step is not None and step <= self.previous_step:
            raise ValueError('Only advancing actual observations are accepted')
        _version(model_version)
        if (predictions is not None and (not isinstance(predictions, dict) or
                set(predictions) != set(SUBJECTS) or
                any(not isinstance(v, str) or v not in OPTIONS for v in predictions.values()))):
            raise ValueError('Exactly two legal MODEL PREDICTIONS are required; no outcomes or poses')
        if (not isinstance(invalidated, (tuple, list)) or any(not isinstance(x, str) for x in invalidated)
                or len(set(invalidated)) != len(invalidated)
                or not set(invalidated) <= set(SUBJECTS)):
            raise ValueError('Invalidated dependencies must name existing object roles')
        if self.model_version is not None and model_version != self.model_version:
            for subject in SUBJECTS:
                self._invalidate(subject, 'model_version_changed')
        self.model_version = model_version
        binding = observation_binding(observation)
        self._prompt(step)  # Expiry occurs before a possible new confirmation.
        for subject in SUBJECTS:
            item = self.objects[subject]
            if subject in invalidated:
                self._invalidate(subject, 'explicit_dependency_invalidated')
                continue
            if predictions is None:
                # An absent classifier call may read valid memory, but never refresh it.
                item['consecutive'] = 0
                item['last_positive'] = None
                continue
            label = predictions[subject]
            if label != 'complete':
                self._invalidate(subject, 'current_' + label)
                item['last_label'] = label
                continue
            consecutive = item['last_positive'] is not None and step - item['last_positive'] == 10
            item['consecutive'] = item['consecutive'] + 1 if consecutive else 1
            item.update(last_positive=step, last_label=label, reason='await_second_current_estimate')
            if item['consecutive'] >= 2:
                if item['contract'] is None:
                    item['version'] += 1
                item['contract'] = dict(kind='predicted_object_stage_dependency', authority=AUTHORITY,
                    episode_id=self.episode_id, object_role=subject, object_version=item['version'],
                    goal_sha256=hashlib.sha256(GOAL.encode()).hexdigest(), model_version=model_version,
                    support_steps=[step - 10, step], observed_input=binding,
                    expires_step=min(520, step + self.max_age_steps),
                    depends_on=['correct_object_identity', 'correct_goal_target_identity',
                        'support_on_goal_target', 'released_from_gripper'])
                item['reason'] = 'two_consecutive_model_estimates'
        proposed, active = self._prompt(step)
        self.previous_step = step
        self.last_receipt = dict(kind='object_stage_memory_v1', authority=AUTHORITY,
            input_binding=binding, model_version=model_version, predictions=copy.deepcopy(predictions),
            objects=copy.deepcopy(self.objects), active_objects=active, proposed_prompt=proposed,
            applied_prompt=proposed if self.trial_enabled else GOAL,
            trial_enabled=self.trial_enabled, inference_horizon=10, commit_steps=10,
            native_success_label=None, persistent_update=False, task_complete_claimed=False)
        return copy.deepcopy(self.last_receipt)

    def policy_input(self, payload, observation):
        """Read memory at the exact observed boundary; default returns input unchanged."""
        validate_observation(observation)
        forbidden = {'native_success', 'environment_success', 'init_state_id', 'result',
            'probe_label', 'object_pose', 'target_pose', 'privileged'}
        if not isinstance(payload, dict) or forbidden.intersection(payload):
            raise ValueError('Results, hidden poses and init identifiers are not policy input')
        if observation['episode_id'] != self.episode_id or payload.get('prompt') != GOAL:
            raise ValueError('Original episode and complete benchmark goal must remain bound')
        if self.last_receipt is None or observation_binding(observation) != self.last_receipt['input_binding']:
            raise ValueError('A current observation must pass through observe before dispatch')
        receipt = copy.deepcopy(self.last_receipt)
        receipt['input_changed'] = self.trial_enabled and receipt['proposed_prompt'] != GOAL
        if not receipt['input_changed']:
            return payload, receipt
        return {**payload, 'prompt': receipt['proposed_prompt']}, receipt
