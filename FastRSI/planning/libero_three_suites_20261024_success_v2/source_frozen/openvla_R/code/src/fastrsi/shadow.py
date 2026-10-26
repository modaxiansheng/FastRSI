"""Evidence-aware legal actions for a text-only, non-controlling Laya probe."""
import hashlib
from pathlib import Path
from .contracts import reuse_admissible


def context_from_bundle(bundle, workspace, rules):
    missing = [key for key in ('session', 'runtime_result', 'initial_rgb_paths',
               'final_rgb_paths', 'lessons_md', 'history_md') if key not in bundle]
    images = []
    root = Path(workspace).resolve()
    for phase in ('initial_rgb_paths', 'final_rgb_paths'):
        if not bundle.get(phase):
            missing.append(phase)
        for rel in bundle.get(phase, []):
            path = (root / rel).resolve()
            exists = path.is_relative_to(root) and path.is_file()
            images.append({'path': rel, 'phase': phase, 'exists': exists,
                           'sha256': hashlib.sha256(path.read_bytes()).hexdigest() if exists else None})
            if not exists:
                missing.append(rel)
    context = {'task': bundle.get('session', {}).get('task_description'),
               'session_id': bundle.get('session', {}).get('session_id'),
               'prior_lessons': bundle.get('lessons_md', ''),
               'history': bundle.get('history_md', ''),
               'runtime_result': bundle.get('runtime_result', {}),
               'environment': bundle.get('environment_md', ''),
               'evidence_version': bundle.get('created_at'), 'images': images,
               'visual_summary': None, 'visual_summary_available': False,
               'rules': rules, 'missing_required_fields': sorted(set(missing)),
               'task_semantics_available_to_text_model': False}
    if not context['task']:
        context['missing_required_fields'].append('task_description')
    return context


def legal_options(context):
    options = ['escalate']
    if context['missing_required_fields'] or not context['visual_summary_available']:
        return options
    for rule_id in sorted(context['rules']):
        if reuse_admissible(rule_id, context['rules'], context['evidence_version'])[0]:
            options.append('reuse:' + rule_id)
    return options


def validate_choice(choice, context, offered):
    # Recompute legality: a candidate may have expired while the model ran.
    current = legal_options(context)
    if choice not in offered or choice not in current:
        return 'escalate', 'illegal_or_stale_output'
    return choice, 'validated'


def run_shadow(context, router=None):
    offered = legal_options(context)
    if len(offered) == 1:
        return {'offered': offered, 'raw_choice': None, 'validated_choice': 'escalate',
                'called': False, 'reason': 'single_legal_option', 'controls_execution': False,
                'input_completeness': {'missing': context['missing_required_fields'],
                 'visual_summary_available': context['visual_summary_available']}}
    if router is None:
        raise ValueError('Multiple legal options require a configured shadow router')
    answer = router.predict(context, {'route': {'type': 'choice',
         'instructions': 'Choose only an offered action using current evidence.',
         'criteria': {key: key for key in offered}}}, model='typed-decisions')
    choice = answer.get('answers', {}).get('route', {}).get('choice')
    validated, reason = validate_choice(choice, context, offered)
    return {'offered': offered, 'raw_choice': choice, 'validated_choice': validated,
            'called': True, 'reason': reason, 'answer': answer, 'controls_execution': False}
